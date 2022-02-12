import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import pandas as pd

from src.Dataset import Dataset
from src.GBT import GBT

def get_fold(data):
    X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
    n_training = 463715
    n_training = int(X.shape[0] * 89.98 / 100)
    return X[:n_training], X[n_training:], y[:n_training], y[n_training:]


params = {'gamma': 0.,
          'lambda': 1.,
          'min_split_gain': 0.1,
          'max_depth': 5,
          'learning_rate': 0.3,
          }

# %% LightGBM specific
method = 'lightgbm'
params['lambda'] = 1
params['device'] = 'cpu'
params['objective'] = 'regression'
params['metric'] = 'rmse'
params['num_leaves'] = 8
params['bagging_freq'] = 1
params['min_data_in_bin'] = 1
# %% Loop
datasets = ['msd']
base_estimators = 2000
df = pd.DataFrame(
    columns=['method', 'dataset', 'fold', 'device', 'validation_estimators', 'test_estimators', 'rmse_test',
             'crps_test', 'validation_time'])

for i, dataset in enumerate(datasets):
    params['bagging_fraction'] = 0.1
    n_folds = 1

    data = pd.read_feather('../datasets/msd.feather')
    for fold in range(n_folds):
        print(f'{dataset}: fold {fold + 1}/{n_folds}')

        # Get data
        X_train, X_test, y_train, y_test = get_fold(data)
        X_train_val, X_val, y_train_val, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=fold)

        # Build datasets
        train_data = Dataset(X_train, y_train)
        train_val_data = Dataset(X_train_val, y_train_val)
        valid_data = Dataset(X_val, y_val)
        test_data = Dataset(X_test, y_test)
        params['n_estimators'] = base_estimators

        # Train to retrieve best iteration
        print('Validating...')
        start = time.perf_counter()
        model = GBT(run_goss=False, top_rate=0.2, other_rate=0.1)
        model.train(params, train_val_data, num_boost_round=100, valid_set=valid_data, early_stopping_rounds=100)
        end = time.perf_counter()
        validation_time = end - start
        print(f'Fold time: {validation_time:.2f}s')

        # Set iterations to best iteration
        params['n_estimators'] = model.best_iteration + 1

        # Predictions
        print('Prediction...')
        yhat_point = []
        for x in X_test:
            yhat_point.append(model.predict(x, num_iteration=model.best_iteration))

        # Scoring
        rmse = np.sqrt(mean_squared_error(yhat_point, y_test))
        crps = 0

        # Save data
        df = df.append({'method': method, 'dataset': dataset, 'fold': fold, 'device': params['device'],
                        'validation_estimators': base_estimators, 'test_estimators': params['n_estimators'],
                        'rmse_test': rmse, 'crps_test': crps, 'validation_time': validation_time}, ignore_index=True)
# %% Save
filename = f"{method}_{params['device']}.csv"
df.to_csv(f'{filename}')
