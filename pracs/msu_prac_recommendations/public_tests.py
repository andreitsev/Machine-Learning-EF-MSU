import numpy as np
import pandas as pd

from utils.models_solved import (
    ConstantRecommender
)
# from utils.models import ConstantRecommender


_compute_binary_relevance_test_cases = [
    {
        "args": {
            "recommended_items_list": [1],
            "true_items_list": [1, 2, 3]
        },
        "expected_output": [1]
    },
    {
        "args": {
            "recommended_items_list": [1, 2],
            "true_items_list": [1, 2, 3]
        },
        "expected_output": [1, 1]
    },
    {
        "args": {
            "recommended_items_list": [0, 1],
            "true_items_list": [1, 2, 3]
        },
        "expected_output": [0, 1]
    },
    {
        "args": {
            "recommended_items_list": [5, 2],
            "true_items_list": [1, 2, 3]
        },
        "expected_output": [0, 1]
    },
    {
        "args": {
            "recommended_items_list": [1, 1],
            "true_items_list": [1, 2, 3]
        },
        "expected_output": [1, 1]
    },
    {
        "args": {
            "recommended_items_list": [3, 4, 5, 1],
            "true_items_list": [1, 2, 3]
        },
        "expected_output": [1, 0, 0, 1]
    },
    {
        "args": {
            "recommended_items_list": [1],
            "true_items_list": [2, 3, 4, 5, 6, 7]
        },
        "expected_output": [0]
    },
    {
        "args": {
            "recommended_items_list": [2, 3, 5, 6, 7],
            "true_items_list": [1, 2, 3, 4],
        },
        "expected_output": [1, 1, 0, 0, 0]
    },
]

ap_at_k_test_cases = [
      {
          "args": {
              "recommended_items_list": [1],
              "true_items_list": [1, 2, 3],
              "k": 1
          },
          "expected_output": 1
      },
      {
          "args": {
              "recommended_items_list": [2, 3, 5, 6, 7],
              "true_items_list": [1, 2, 3, 4],
              "k": 4
          },
          "expected_output": 0.75/4
          # binary_rel = [1, 1, 0, 0, 0]
          # cumrel = [1, 2, 2, 2, 2] / 4 = [0.25, 0.5, 0.5, 0.5, 0.5]
          # res = 1/min(4, 4) * (0.25*1 + 0.5*1) = 0.75/4
      },
      {
          "args": {
              "recommended_items_list": [4, 5, 6],
              "true_items_list": [1, 2, 3],
              "k": 3
          },
          "expected_output": 0
          # binary_rel = [0, 0, 0]
          # cumrel = [0, 0, 0] / 3 = [0, 0, 0]
          # res = 1/min(3, 3) * 0 = 0
      },
      {
          "args": {
              "recommended_items_list": [4, 5, 3],
              "true_items_list": [1, 2, 3],
              "k": 3
          },
          "expected_output": 1/9
          # binary_rel = [0, 0, 1]
          # cumrel = [0, 0, 1] / 3 = [0, 0, 1/3]
          # res = 1/min(3, 3) * 1/3 = 1/9
      },
      {
          "args": {
              "recommended_items_list": [4, 5, 3],
              "true_items_list": [0, 1, 2, 3],
              "k": 3
          },
          "expected_output": 1/9
          # binary_rel = [0, 0, 1]
          # cumrel = [0, 0, 1] / 3 = [0, 0, 1/3]
          # res = 1/min(4, 3) * 1/3 = 1/9
      },
      {
          "args": {
              "recommended_items_list": [4, 5, 3],
              "true_items_list": [0, 1, 2, 3],
              "k": 3
          },
          "expected_output": 1/9
          # binary_rel = [0, 0, 1]
          # cumrel = [0, 0, 1] / 3 = [0, 0, 1/3]
          # res = 1/min(4, 3) * 1/3 = 1/9
      },
      {
          "args": {
              "recommended_items_list": [4, 2, 3],
              "true_items_list": [1, 2, 3],
              "k": 3
          },
          "expected_output": 1/3
          # binary_rel = [0, 1, 1]
          # cumrel = [0, 1, 2] / 3 = [0, 1/3, 2/3]
          # res = 1/min(3, 3) * (1/3 + 2/3) = 1/3
      },
      {
          "args": {
              "recommended_items_list": [1, 2, 3, 4, 5],
              "true_items_list": [3, 4, 5],
              "k": 5
          },
          "expected_output": 0.4
          # binary_rel = [0, 0, 1, 1, 1]
          # cumrel = [0, 0, 1, 2, 3] / 5 = [0, 0, 1/5, 2/5, 3/5]
          # res = 1/min(3, 5) * (1/5 + 2/5 + 3/5) = 2/5 = 0.4
      },
      {
          "args": {
              "recommended_items_list": [1, 2, 3, 4, 5],
              "true_items_list": [3, 4, 5],
              "k": 3
          },
          "expected_output": 1/9
          # binary_rel = [0, 0, 1, 1, 1]
          # cumrel = [0, 0, 1] / 3 = [0, 0, 1/3]
          # res = 1/min(3, 5) * (1/3) = 1/9
      },
      {
          "args": {
              "recommended_items_list": [1, 2, 4, 5, 6],
              "true_items_list": [1, 2, 3],
              "k": 5
          },
          "expected_output": 1/5
          # binary_rel = [1, 1, 0, 0, 0]
          # cumrel = [1, 2, 0, 0, 0] / 5 = [1/5, 2/5, 0, 0, 0]
          # res = 1/min(3, 5) * (1/5 + 2/5) = (3/5) * (1/3) = 1/5
      },
  ]

map_at_k_test_cases = [
      {
          "args": {
              "recommended_items_lists": [[1]],
              "true_items_lists": [[1, 2, 3]],
              "k": 1
          },
          "expected_output": 1
      },
      {
          "args": {
              "recommended_items_lists": [
                    [4, 5, 6],
                    [4, 5, 3]
              ],
              "true_items_lists": [
                    [1, 2, 3],
                    [1, 2, 3]
              ],
              "k": 3
          },
          "expected_output": (0 + 1/9) / 2
      },
      {
          "args": {
              "recommended_items_lists": [
                    [4, 5, 6],
                    [4, 5, 3],
                    [4, 5, 3]
              ],
              "true_items_lists": [
                    [1, 2, 3],
                    [1, 2, 3],
                    [0, 1, 2, 3]
              ],
              "k": 3
          },
          "expected_output": (0 + 1/9 + 1/9) / 3
      },
  ]

# get_test_recommendations_test_cases = [
#     {   
#         "model_type": ConstantRecommender,
#         "init_args": {
#             "const": 1,
#             'ratings': (
#                 pd.DataFrame({
#                 #         train          test
#                 #        0  1  2  3     4  5  6     
#                     0: [[1, 1, 1, 1] + [2, 2, 2]],
#                 #        7  8  9  10 11   12 13 14 15 16
#                     1: [[2, 2, 2, 2, 2] + [1, 1, 2, 2, 3]],
#                 #       17 18 19   20 21 22 23 24 25    
#                     2: [1, 2, 3] + [3, 1, 5, 4, 2, 1],
#                 }).T
#                 .rename(columns={0: 'trackId'})
#                 .explode('trackId')
#                 .reset_index()
#                 .rename(columns={'index': 'userId'})
#             ),
#         },
#         "get_test_recommendations_args": {
#             "test_idxs": [4, 5, 6, 12, 13, 14, 15, 16, 20, 21, 22, 23, 24, 25],
#             "k": 4,
#         },
#         "expected_output": [
#             [],
#             [1]*4,
#             [1]*6
#         ]
#     }
# ]

jaccard_sim_test_cases = [
    {
        "args": {
            "ratings": np.array([
                [1, 0, 0],
                [1, 1, 0]
            ]),
            "user_vector": np.array([1, 0, 0])
        },
        "expected_output": np.array([1, 0.5]),
    },
    {
        "args": {
            "ratings": np.array([
                [1, 0, 0, 1],
                [1, 1, 0, 1],
            ]),
            "user_vector": np.array([1, 1, 1, 0]),
        },
        "expected_output": np.array([0.25, 0.5]),
    },
    {
        "args": {
            "ratings": np.array([
                [1, 0, 0, 0],
                [1, 0, 0, 1],
            ]),
            "user_vector": np.array([0, 1, 1, 0]),
        },
        "expected_output": np.array([0, 0]),
    },
    {
        "args": {
            "ratings": np.array([
                [1, 1, 1, 0],
                [0, 0, 0, 1],
            ]),
            "user_vector": np.array([1, 1, 1, 0]),
        },
        "expected_output": np.array([1, 0]),
    },
    {
        "args": {
            "ratings": np.array([
                [1, 1, 0, 1, 1],
            ]),
            "user_vector": np.array([1, 1, 1, 0, 0]),
        },
        "expected_output": np.array([2/5]),
    },
    {
        "args": {
            "ratings": np.array([
                [1, 1, 1, 1, 1],
            ]),
            "user_vector": np.array([1, 1, 1, 0, 0]),
        },
        "expected_output": np.array([3/5]),
    },
]



