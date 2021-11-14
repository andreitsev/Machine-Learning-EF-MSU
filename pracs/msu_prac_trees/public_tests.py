import numpy as np
import pandas as pd
from numpy import array


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


DecisionTree_fit_tests= [
    {
        'ADD_SCORE_FOR_THIS_TEST': 1,
        "X": pd.read_csv('./data/moon_data.csv', header=None).drop([2], axis=1).values,
        'y': pd.read_csv('./data/moon_data.csv', header=None)[2].values.astype(int),
        "tree_params": {
            "feature_types": ['real', 'real'],
            "criterion": 'gini',
            "max_depth": 2,
        },
        "true_result": {
                'type': 'nonterminal',
                 'feature_type': 'real',
                 'feature_number': 1,
                 'threshold': 0.17122713966611064,
                 'left_child': {'type': 'terminal', 'classes_distribution': np.array([ 66, 381])},
                 'right_child': {'type': 'terminal', 'classes_distribution': np.array([434, 119])}
            }
    },#1

    {
        'ADD_SCORE_FOR_THIS_TEST': 2,
        "X": pd.read_csv('./data/moon_data.csv', header=None).drop([2], axis=1).values,
        'y': pd.read_csv('./data/moon_data.csv', header=None)[2].values.astype(int),
        "tree_params": {
            "feature_types": ['real', 'real'],
            "criterion": 'gini',
            "min_samples_split": 100,
            'max_depth': 4,
        },
        'true_result': {
            'type': 'nonterminal',
             'feature_type': 'real',
             'feature_number': 1,
             'threshold': 0.17122713966611064,
             'left_child': {
                 'type': 'nonterminal',
                  'feature_type': 'real',
                  'feature_number': 0,
                  'threshold': -0.6194477171724573,
                  'left_child': {'type': 'terminal', 'classes_distribution': array([39,  1])},
                  'right_child': {
                      'type': 'nonterminal',
                       'feature_type': 'real',
                       'feature_number': 1,
                       'threshold': -0.10561213615249625,
                       'left_child': {'type': 'terminal', 'classes_distribution': array([  7, 275])},
                       'right_child': {'type': 'terminal', 'classes_distribution': array([ 20, 105])}
                  }
             },
             'right_child': {
                 'type': 'nonterminal',
                  'feature_type': 'real',
                  'feature_number': 0,
                  'threshold': 1.5082666540295038,
                  'left_child': {
                      'type': 'nonterminal',
                       'feature_type': 'real',
                       'feature_number': 1,
                       'threshold': 0.3373510036902131,
                       'left_child': {'type': 'terminal', 'classes_distribution': array([56, 25])},
                       'right_child': {'type': 'terminal', 'classes_distribution': array([376,  39])}
                  },
                  'right_child': {'type': 'terminal', 'classes_distribution': array([ 2, 55])}
             }
        }
    }, #2

    {
        'ADD_SCORE_FOR_THIS_TEST': 1,
        "X": pd.read_csv('./data/agaricus_lepiota.csv').values[:, 1:],
        'y': pd.read_csv('./data/agaricus_lepiota.csv').values[:, 0].astype(int),
        "tree_params": {
            "feature_types": ['categorical']*22,
            "criterion": 'entropy',
            'max_depth': 4,
        },
        'true_result': {
            'type': 'nonterminal',
             'feature_type': 'categorical',
             'feature_number': 4,
             'threshold': 0.4829931972789117,
             'left_child': {'type': 'terminal', 'classes_distribution': array([3796,    0])},
             'right_child': {
                 'type': 'nonterminal',
                  'feature_type': 'categorical',
                  'feature_number': 19,
                  'threshold': 0.014705882352941176,
                  'left_child': {'type': 'terminal', 'classes_distribution': array([72,  0])},
                  'right_child': {
                      'type': 'nonterminal',
                       'feature_type': 'categorical',
                       'feature_number': 14,
                       'threshold': 0.21634615384615385,
                       'left_child': {'type': 'terminal', 'classes_distribution': array([40, 64])},
                       'right_child': {'type': 'terminal', 'classes_distribution': array([   8, 4144])}
                  }
             }
        }
    }, #3

    {
        'ADD_SCORE_FOR_THIS_TEST': 1,
        "X": pd.read_csv('./data/cars.csv').values[:, :-1],
        'y': pd.read_csv('./data/cars.csv')['target'].astype('category').cat.codes.values.astype(int),
        "tree_params": {
            "feature_types": ['categorical']*6,
            "criterion": 'entropy',
            'max_depth': 5,
        },
        'true_result': {
            'type': 'nonterminal',
             'feature_type': 'categorical',
             'feature_number': 3,
             'threshold': 1.6788194444444444,
             'left_child': {
                 'type': 'nonterminal',
                  'feature_type': 'categorical',
                  'feature_number': 5,
                  'threshold': 1.6762152777777777,
                  'left_child': {
                      'type': 'nonterminal',
                       'feature_type': 'categorical',
                       'feature_number': 1,
                       'threshold': 1.5358796296296295,
                       'left_child': {
                           'type': 'nonterminal',
                            'feature_type': 'categorical',
                            'feature_number': 5,
                            'threshold': 1.3298611111111112,
                            'left_child': {'type': 'terminal', 'classes_distribution': array([105,  39,  48,   0])},
                            'right_child': {'type': 'terminal', 'classes_distribution': array([102,  30,   8,  52])}
                       },
                       'right_child': {
                           'type': 'nonterminal',
                            'feature_type': 'categorical',
                            'feature_number': 0,
                            'threshold': 1.619212962962963,
                            'left_child': {'type': 'terminal', 'classes_distribution': array([177,   0,  98,  13])},
                            'right_child': {'type': 'terminal', 'classes_distribution': array([ 0,  0, 96,  0])}
                       }
                  },
                  'right_child': {'type': 'terminal', 'classes_distribution': array([  0,   0, 384,   0])}
             },
             'right_child': {'type': 'terminal', 'classes_distribution': array([  0,   0, 576,   0])}
        }
    }, #4

]

DecisionTree_predict_proba_tests = [
    {
        'ADD_SCORE_FOR_THIS_TEST': 0.5,
        "X": pd.read_csv('./data/moon_data.csv', header=None).drop([2], axis=1).values,
        'y': pd.read_csv('./data/moon_data.csv', header=None)[2].values.astype(int),
        "tree_params": {
            "feature_types": ['real', 'real'],
            "criterion": 'gini',
            "max_depth": 3,
        },
        "true_result": 0.906
    }, #1

    {
        'ADD_SCORE_FOR_THIS_TEST': 1,
        "X": pd.read_csv('./data/moon_data.csv', header=None).drop([2], axis=1).values,
        'y': pd.read_csv('./data/moon_data.csv', header=None)[2].values.astype(int),
        "tree_params": {
            "feature_types": ['real', 'real'],
            "criterion": 'gini',
            "min_samples_split": 100,
            'max_depth': 4,
        },
        'true_result': 0.906
    }, #2

    {
        'ADD_SCORE_FOR_THIS_TEST': 0.5,
        "X": pd.read_csv('./data/agaricus_lepiota.csv').values[:, 1:],
        'y': pd.read_csv('./data/agaricus_lepiota.csv').values[:, 0].astype(int),
        "tree_params": {
            "feature_types": ['categorical']*22,
            "criterion": 'entropy',
            'max_depth': 4,
        },
        'true_result': 0.517971442639094
    }, #3

    {
        'ADD_SCORE_FOR_THIS_TEST': 0.5,
        "X": pd.read_csv('./data/cars.csv').values[:, :-1],
        'y': pd.read_csv('./data/cars.csv')['target'].astype('category').cat.codes.values.astype(int),
        "tree_params": {
            "feature_types": ['categorical']*6,
            "criterion": 'entropy',
            'max_depth': 5,
        },
        'true_result': 0.3888888888888889
    }, #4
]