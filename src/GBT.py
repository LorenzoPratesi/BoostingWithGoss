import sys
import time
import numpy as np
from sklearn.metrics import mean_squared_error

from .Tree import Tree
from .subsample import goss_sampling
from .Dataset import Dataset
from .binning import BinMapper

try:
    # For python2
    from itertools import izip as zip

    LARGE_NUMBER = sys.maxint
except ImportError:
    # For python3
    LARGE_NUMBER = sys.maxsize

class GBT(object):
    def __init__(self, run_goss, top_rate, other_rate):
        self.params = {'gamma': 0.,
                       'lambda': 1.,
                       'min_split_gain': 0.1,
                       'max_depth': 5,
                       'learning_rate': 0.3,
                       }
        self.best_iteration = None,
        self.run_goss = run_goss,
        self.top_rate, self.other_rate = top_rate, other_rate  # goss sampling rate
        self.goss_data = {}

    def _evaluate(self, X_test, num_iteration):
        return [self.predict(x, num_iteration=num_iteration) for x in X_test]

    def _calc_training_data_scores(self, train_set, models):
        if len(models) == 0:
            return None
        X = train_set.X
        scores = np.zeros(len(X))
        for i in range(len(X)):
            scores[i] = self.predict(X[i], models=models)
        return scores

    def _calc_l2_gradient(self, train_set, scores):
        labels = train_set.y
        hessian = np.full(len(labels), 2)
        if scores is None:
            grad = np.random.uniform(size=len(labels))
        else:
            grad = np.array([2 * (labels[i] - scores[i]) for i in range(len(labels))])
        return grad, hessian

    def _calc_gradient(self, train_set, scores):
        """For now, only L2 loss is supported"""
        return self._calc_l2_gradient(train_set, scores)

    def _calc_l2_loss(self, models, data_set):
        errors = []
        for x, y in zip(data_set.X, data_set.y):
            errors.append(y - self.predict(x, models))
        return np.mean(np.square(errors))

    def _calc_loss(self, models, data_set):
        """For now, only L2 loss is supported"""
        return self._calc_l2_loss(models, data_set)

    def _build_learner(self, train_set, grad, hessian, shrinkage_rate):
        learner = Tree()
        learner.build(train_set.X, grad, hessian, shrinkage_rate, self.params)
        return learner

    def train(self, params, train_set, num_boost_round=20, valid_set=None, early_stopping_rounds=5):
        self.params.update(params)
        models = []
        shrinkage_rate = 1.
        best_iteration = None
        eval_scores = []
        best_val_loss = LARGE_NUMBER
        train_start_time = time.time()

        train_set_full = Dataset(train_set.X, train_set.y)

        print(f"Training until validation scores do not improve for {early_stopping_rounds} rounds.")
        for iter_cnt in range(num_boost_round):
            iter_start_time = time.time()
            scores = self._calc_training_data_scores(train_set_full, models)
            grad, hessian = self._calc_gradient(train_set_full, scores)

            # goss subsample
            if self.run_goss[0]:
                self.goss_data = goss_sampling(grad, hessian, self.top_rate, self.other_rate)
                self.params['goss_data'] = self.goss_data
                train_set.X = train_set_full.X[self.goss_data['idx']]
                train_set.y = train_set_full.y[self.goss_data['idx']]
                grad = self.goss_data['grad']
                hessian = self.goss_data['hess']

            learner = self._build_learner(train_set, grad, hessian, shrinkage_rate)

            if iter_cnt > 0:
                shrinkage_rate *= self.params['learning_rate']

            models.append(learner)

            # eval_scores.append(mean_squared_error(valid_set.y, self.predict(valid_set.X)))
            # scores = self._evaluate(train_set.X, models)
            # train_loss = np.sqrt(mean_squared_error(scores, y_test))

            train_loss = self._calc_loss(models, train_set)
            val_loss = self._calc_loss(models, valid_set) if valid_set else None
            val_loss_str = '{:.10f}'.format(val_loss) if val_loss else '-'
            print("Iter {:>3}, Train's L2: {:.10f}, Valid's L2: {}, Elapsed: {:.2f} secs"
                  .format(iter_cnt, train_loss, val_loss_str, time.time() - iter_start_time))

            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_iteration = iter_cnt

            if iter_cnt - best_iteration >= early_stopping_rounds:
                print("Early stopping, best iteration is:")
                print("Iter {:>3}, Train's L2: {:.10f}".format(best_iteration, best_val_loss))
                break

        self.models = models
        self.best_iteration = best_iteration
        print("Training finished. Elapsed: {:.2f} secs".format(time.time() - train_start_time))

    def predict(self, x, models=None, num_iteration=None):
        if models is None:
            models = self.models
        assert models is not None
        return np.sum(m.predict(x) for m in models[:num_iteration])
