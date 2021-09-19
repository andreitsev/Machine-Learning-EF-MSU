import numpy as np



compute_criterion_tests = [
    {
         "target_vector": np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]),
         "feature_vector": np.arange(9),
         "threshold": 4,
         "criterion": 'gini',
         "true_result": 0.23333333333333345,
    },#1

    {
        "target_vector": np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]),
        "feature_vector": np.arange(9),
        "threshold": 4,
        "criterion": 'entropy',
        "true_result": 0.6849774484867257,
    },#2

    {
        "target_vector": np.array([1, 1, 1, 2, 2, 2]),
        "feature_vector": np.arange(6),
        "threshold": 2.5,
        "criterion": 'gini',
        "true_result": 0.5,
    },#3

    {
        "target_vector": np.array([1, 1, 1, 2, 2, 2]),
        "feature_vector": np.arange(6),
        "threshold": 2.5,
        "criterion": 'entropy',
        "true_result": 1.0,
    },#4

    {
        "target_vector": np.array([1, 1, 1, 2, 2, 2]),
        "feature_vector": np.arange(6),
        "threshold": 3.5,
        "criterion": 'entropy',
        "true_result": 0.4591479170272448,
    },#5
]



find_best_split_tests = [
    {
        "feature_vector": np.array([1, 2, 3]),
        "target_vector": np.array([0, 1, 1]),
        "criterion": "gini",
        "true_result": (np.array([1.5, 2.5]), np.array([0.44, 0.11]), 1.5, 0.444)
    }, #1

    {
        "feature_vector": np.array([1, 2, 3, 4, 5, 6, 7]),
        "target_vector": np.array([0, 1, 1, 2, 2, 2, 1]),
        "criterion": "gini",
        "true_result": (
            np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5]),
            np.array([0.184, 0.127, 0.207, 0.065, 0.012, 0.088]),
            3.5,
            0.207
        )
    }, #2

    {
        "feature_vector": np.array([1, 1, 2, 2, 2, 3, 3, 3, 4, 5, 1, 1, 2, 2]),
        "target_vector": np.array([0, 1, 1, 1, 0, 0, 2, 2, 2, 1, 1, 1, 1, 2]),
        "criterion": "entropy",
        "true_result": (
            np.array([1.5, 2.5, 3.5, 4.5]),
            np.array([0.174, 0.216, 0.064, 0.075]),
            2.5,
            0.216,
        )
    }, #3
]