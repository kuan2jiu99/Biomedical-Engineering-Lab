import numpy as np

RIGHT_MAX = 912
RIGHT_MIN = 72
LEFT_MAX = 852
LEFT_MIN = 0

def emg_utils(right_raw_array, left_raw_array, threshold=0.45):
    # Input: 
    # (1) right raw emg data.
    # (2) left raw emg data.

    # Output:
    # (1) processed right emg data.
    # (2) processed left emg data.
    # (3) whether it is balance, true for balance, false for unbalanced.

    # get window size.
    window_size = right_raw_array.shape[0]

    # normalization.
    right_norm_array = (right_raw_array - RIGHT_MAX) / (RIGHT_MAX - RIGHT_MIN)
    left_norm_array = (left_raw_array - LEFT_MAX) / (LEFT_MAX - LEFT_MIN)

    # p2p value of right.
    peak_max = max(right_norm_array)
    peak_min = min(right_norm_array)
    peak2peak_right = peak_max - peak_min

    # p2p value of left.
    peak_max = max(left_norm_array)
    peak_min = min(left_norm_array)
    peak2peak_left = peak_max - peak_min

    print(peak2peak_right, peak2peak_left)

    if abs(peak2peak_right - peak2peak_left) < threshold:
        is_balanced = True
    else:
        is_balanced = False


    return peak2peak_right, peak2peak_left, is_balanced