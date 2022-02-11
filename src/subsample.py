import numpy as np


def goss_sampling(grad, hessian, top_rate, other_rate):
    """
    sampling method introduced in LightGBM
    """
    assert len(grad) == len(hessian)
    sample_num = len(grad)
    id_list = np.arange(sample_num)
    id_type = type(id_list[0])

    g_arr = grad.ravel().astype(np.float64)
    h_arr = hessian.ravel().astype(np.float64)
    abs_g_list_arr = np.abs(g_arr)
    sorted_idx = np.argsort(-abs_g_list_arr, kind='stable')

    a_part_num = int(sample_num * top_rate)
    b_part_num = int(sample_num * other_rate)

    if a_part_num == 0 or b_part_num == 0:
        raise ValueError(f"subsampled result is 0: top sample {a_part_num}, other sample {b_part_num}")

    # index of a part
    a_sample_idx = sorted_idx[:a_part_num]

    # index of b part
    rest_sample_idx = sorted_idx[a_part_num:]
    b_sample_idx = np.random.choice(rest_sample_idx, size=b_part_num, replace=False)

    # small gradient sample weights
    amplify_weights = (1 - top_rate) / other_rate
    g_arr[b_sample_idx] *= amplify_weights
    h_arr[b_sample_idx] *= amplify_weights

    # get selected sample
    selected_idx = np.r_[a_sample_idx, b_sample_idx]
    selected_g, selected_h = g_arr[selected_idx], h_arr[selected_idx]
    selected_id = id_list[selected_idx]

    data = {
        'idx': selected_id,
        'grad': selected_g,
        'hess': selected_h,
        'a_part_num': a_part_num,
        'b_part_num': b_part_num,
        'amplify_weights': amplify_weights
    }

    return data
