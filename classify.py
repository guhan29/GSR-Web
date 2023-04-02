import numpy as np
from dtaidistance import dtw

def dtw_algorithm(gsr_value_without_connection, sample_array):
    dataSize = len(sample_array)

    print(gsr_value_without_connection, sample_array)

    no_dehydration = gsr_value_without_connection
    severe_dehydration = (no_dehydration - 33)/2
    some_dehydration = (no_dehydration + severe_dehydration)/2
    compensated = (severe_dehydration + 33)/2
    decompensated = 33
    print(no_dehydration, some_dehydration, severe_dehydration, compensated, decompensated)

    arr_no_dehydration = np.array([(no_dehydration+some_dehydration)/2]*dataSize)
    arr_some_dehydration = np.array([(some_dehydration+severe_dehydration)/2]*dataSize)
    arr_severe_dehydration = np.array([(severe_dehydration+compensated)/2]*dataSize)
    arr_compensated = np.array([(compensated+decompensated)/2]*dataSize)
    arr_decompensated = np.array([33]*dataSize)

    # print(arr_no_dehydration.head())
    # print(arr_some_dehydration.head())
    # print(arr_severe_dehydration.head())
    # print(arr_compensated.head())
    # print(arr_decompensated.head())


    distance_no_dehydration, paths_no_dehydration = dtw.warping_paths(sample_array, arr_no_dehydration)
    best_path_no_dehydration = dtw.best_path(paths_no_dehydration)

    distance_some_dehydration, paths_some_dehydration = dtw.warping_paths(sample_array, arr_some_dehydration)
    best_path_some_dehydration = dtw.best_path(paths_some_dehydration)

    distance_severe_dehydration, paths_severe_dehydration = dtw.warping_paths(sample_array, arr_severe_dehydration)
    best_path_severe_dehydration = dtw.best_path(paths_severe_dehydration)

    distance_compensated, paths_compensated = dtw.warping_paths(sample_array, arr_compensated)
    best_path_compensated = dtw.best_path(paths_compensated)

    distance_decompensated, paths_decompensated = dtw.warping_paths(sample_array, arr_decompensated)
    best_path_decompensated = dtw.best_path(paths_decompensated)

    print(distance_no_dehydration, distance_some_dehydration, distance_severe_dehydration, distance_compensated, distance_decompensated)

    cl_tuple = [
        (distance_no_dehydration, "No Dehydration"), 
        (distance_some_dehydration, "Some Dehydration"), 
        (distance_severe_dehydration, "Severe Dehydration"),
        (distance_compensated, "Compenstated"),
        (distance_decompensated, "Decompenstated"),
    ]

    ans = sorted(cl_tuple, key=lambda x: x[0])

    return (ans[0], distance_no_dehydration, distance_some_dehydration, distance_severe_dehydration, distance_compensated, distance_decompensated)
