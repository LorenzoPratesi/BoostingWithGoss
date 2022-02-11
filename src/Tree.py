import numpy as np
from tqdm import tqdm


class Tree(object):
    ''' Classification and regression tree for tree ensemble '''

    def __init__(self):
        self.root = None

    def build(self, instances, grad, hessian, shrinkage_rate, param):
        assert len(instances) == len(grad) == len(hessian)
        self.root = TreeNode()
        current_depth = 0
        self.root.build(instances, grad, hessian, shrinkage_rate, current_depth, param)

    def predict(self, x):
        return self.root.predict(x)


class TreeNode(object):
    def __init__(self):
        self.is_leaf = False
        self.left_child = None
        self.right_child = None
        self.split_feature_id = None
        self.split_val = None
        self.weight = None

    def _calc_split_gain(self, G, H, G_l, H_l, G_r, H_r, lambd):
        """
        Loss reduction
        (Refer to Eq7 of XGBoost paper)
        """

        def calc_term(g, h):
            return np.square(g) / (h + lambd)

        return calc_term(G_l, H_l) + calc_term(G_r, H_r) - calc_term(G, H)

    def _calc_goss_split_gain(self, num_grad_branch,
                              sum_large_grad_left, sum_small_grad_left,
                              sum_large_grad_right, sum_small_grad_right,
                              amplify_weights, split_idx):

        weight_left = sum_large_grad_left + (amplify_weights * sum_small_grad_left)
        gain_left = np.square(weight_left) / split_idx

        weight_right = sum_large_grad_right + (amplify_weights * sum_small_grad_right)
        gain_right = np.square(weight_right) / (num_grad_branch - split_idx)

        return (gain_left + gain_right) / num_grad_branch

    def _calc_leaf_weight(self, grad, hessian, lambd):
        """
        Calculate the optimal weight of this leaf node.
        (Refer to Eq5 of XGBoost paper)
        """
        return np.sum(grad) / (np.sum(hessian) + lambd)

    def build(self, instances, grad, hessian, shrinkage_rate, depth, param):
        """
        Exact Greedy Alogirithm for Split Finidng
        (Refer to Algorithm1 of XGBoost paper)
        """
        assert instances.shape[0] == len(grad) == len(hessian)
        if depth > param['max_depth']:
            self.is_leaf = True
            self.weight = self._calc_leaf_weight(grad, hessian, param['lambda']) * shrinkage_rate
            return

        G = np.sum(grad)
        H = np.sum(hessian)

        if 'goss_data' in param:
            a_part_num = param['goss_data']['a_part_num']
            b_part_num = param['goss_data']['b_part_num']
            amplify_weights = param['goss_data']['amplify_weights']

            G_large = np.sum(grad[:a_part_num])
            G_small = np.sum(grad[a_part_num:])


        best_gain = 0.
        best_feature_id = None
        best_val = 0.
        best_left_instance_ids = None
        best_right_instance_ids = None
        for feature_id in tqdm(range(instances.shape[1])):
            sample_num = instances.shape[0]

            G_l, H_l = 0., 0.
            sum_large_grad_left, sum_small_grad_left, sum_large_grad_right, sum_small_grad_right = 0., 0., 0., 0.
            sorted_instance_ids = instances[:, feature_id].argsort()
            for j in range(sorted_instance_ids.shape[0] - 1):
                if 'goss_data' in param:
                    inst = sorted_instance_ids[j]
                    num_grad_branch = sample_num

                    sum_large_grad_left += grad[inst] if inst < a_part_num else 0.
                    sum_small_grad_left += grad[inst] if inst >= a_part_num else 0.

                    sum_large_grad_right = G_large - sum_large_grad_left
                    sum_small_grad_right = G_small - sum_small_grad_left

                    current_gain = self._calc_goss_split_gain(num_grad_branch,
                                                              sum_large_grad_left, sum_small_grad_left,
                                                              sum_large_grad_right, sum_small_grad_right,
                                                              amplify_weights, j + 1)

                else:
                    G_l += grad[sorted_instance_ids[j]]
                    H_l += hessian[sorted_instance_ids[j]]
                    G_r = G - G_l
                    H_r = H - H_l
                    current_gain = self._calc_split_gain(G, H, G_l, H_l, G_r, H_r, param['lambda'])

                if current_gain > best_gain:
                    best_gain = current_gain
                    best_feature_id = feature_id
                    best_val = instances[sorted_instance_ids[j]][feature_id]
                    best_left_instance_ids = sorted_instance_ids[:j + 1]
                    best_right_instance_ids = sorted_instance_ids[j + 1:]
        if best_gain < param['min_split_gain']:
            self.is_leaf = True
            self.weight = self._calc_leaf_weight(grad, hessian, param['lambda']) * shrinkage_rate
        else:
            self.split_feature_id = best_feature_id
            self.split_val = best_val

            self.left_child = TreeNode()
            self.left_child.build(instances[best_left_instance_ids],
                                  grad[best_left_instance_ids],
                                  hessian[best_left_instance_ids],
                                  shrinkage_rate,
                                  depth + 1, param)

            self.right_child = TreeNode()
            self.right_child.build(instances[best_right_instance_ids],
                                   grad[best_right_instance_ids],
                                   hessian[best_right_instance_ids],
                                   shrinkage_rate,
                                   depth + 1, param)

    def predict(self, x):
        if self.is_leaf:
            return self.weight
        else:
            if x[self.split_feature_id] <= self.split_val:
                return self.left_child.predict(x)
            else:
                return self.right_child.predict(x)
