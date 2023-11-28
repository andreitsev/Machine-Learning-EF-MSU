import numpy as np
import pandas as pd

from utils.models_solved import (
    ConstantRecommender
)
# from utils.models import ConstantRecommender

## TODO uncomment utils.distances and remove utils.distances_solved 
# from utils.distances import jaccard_sim
from utils.distances_solved import jaccard_sim


def _process_ratings_array(arr: np.array) -> pd.DataFrame:
    userId, trackId = np.where(arr > 0)
    processed_df = pd.DataFrame({'userId': userId, 'trackId': trackId})
    return processed_df


ratings_arr1 = np.array([
    #       \   trackId
    # userId \
             [0, 1, 1, 0, 1, 1, 1], # user0
             [1, 0, 1, 1, 0, 0, 0], # user1
             [0, 0, 0, 0, 1, 0, 1], # user2
             [1, 1, 1, 1, 0, 0, 0], # user3
             [1, 1, 0, 0, 1, 1, 0], # user4
             [1, 0, 0, 0, 0, 1, 0], # user5
])

ratings_arr2 = np.array([
    #       \   trackId
    # userId \
             [0, 1, 1, 0], # user0
             [1, 0, 1, 1], # user1
             [0, 1, 0, 0], # user2
             [1, 1, 1, 1], # user3
             [1, 1, 0, 0], # user4
])


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


user2user_similarity_output_length_test_cases = [
    {
        "init_args": {
            "ratings": _process_ratings_array(ratings_arr1), 
            "similarity_func": jaccard_sim,
            "alpha": 0.02,
        },
        "similarity_args": {
            "user_vector": np.array([0, 0, 1, 1, 1, 0, 0])
        },
        "expected_output": 6,
    },
    {
        "init_args": {
            "ratings": _process_ratings_array(ratings_arr1[:-1]), 
            "similarity_func": jaccard_sim,
            "alpha": 0.02,
        },
        "similarity_args": {
            "user_vector": np.array([0, 0, 1, 1, 1, 0, 0])
        },
        "expected_output": 5,
    },
    {
        "init_args": {
            "ratings": _process_ratings_array(ratings_arr1[1:]), 
            "similarity_func": jaccard_sim,
            "alpha": 0.02,
        },
        "similarity_args": {
            "user_vector": np.array([0, 0, 1, 1, 1, 0, 0])
        },
        "expected_output": 5,
    },
    {
        "init_args": {
            "ratings": _process_ratings_array(ratings_arr1[:, :-1]), 
            "similarity_func": jaccard_sim,
            "alpha": 0.02,
        },
        "similarity_args": {
            "user_vector": np.array([0, 0, 1, 1, 1, 0])
        },
        "expected_output": 6,
    },
    {
        "init_args": {
            "ratings": _process_ratings_array(ratings_arr1[:, 1:-1]), 
            "similarity_func": jaccard_sim,
            "alpha": 0.02,
        },
        "similarity_args": {
            "user_vector": np.array([0, 1, 1, 1, 0])
        },
        "expected_output": 6,
    },
    {
        "init_args": {
            "ratings": _process_ratings_array(ratings_arr1[1:, 1:-1]), 
            "similarity_func": jaccard_sim,
            "alpha": 0.02,
        },
        "similarity_args": {
            "user_vector": np.array([0, 1, 1, 1, 0])
        },
        "expected_output": 5,
    },
    {
        "init_args": {
            "ratings": _process_ratings_array(ratings_arr1[:3, 1:-1]), 
            "similarity_func": jaccard_sim,
            "alpha": 0.02,
        },
        "similarity_args": {
            "user_vector": np.array([0, 1, 1, 1, 0])
        },
        "expected_output": 3,
    },
    {
        "init_args": {
            "ratings": _process_ratings_array(ratings_arr1[1:3, [2, 3, 6]]), 
            "similarity_func": jaccard_sim,
            "alpha": 0.02,
        },
        "similarity_args": {
            "user_vector": np.array([0, 1, 1])
        },
        "expected_output": 2,
    },
]

user2user_similarity_test_cases = [
    {
        "init_args": {
            "ratings": _process_ratings_array(ratings_arr1), 
            "similarity_func": jaccard_sim,
            "alpha": 0.02,
        },
        "similarity_args": {
            "user_vector": np.array([0, 0, 1, 1, 1, 0, 0])
        },
        "expected_output": np.array([2/6, 2/4, 1/4, 2/5, 1/6, 0/5]),
        # Explanation:
        #
        # Ratings matrix in this test is:
        #     user0: [0, 1, 1, 0, 1, 1, 1],
        #     user1: [1, 0, 1, 1, 0, 0, 0],
        #     user2: [0, 0, 0, 0, 1, 0, 1],
        #     user3: [1, 1, 1, 1, 0, 0, 0],
        #     user4: [1, 1, 0, 0, 1, 1, 0],
        #     user5: [1, 0, 0, 0, 0, 1, 0],
        #
        # So, similarities are:
        #
        # jaccard_sim(
        #     request vector: [0, 0, 1, 1, 1, 0, 0],
        #     user0 vector:   [0, 1, 1, 0, 1, 1, 1]
        # ) = 
        # |request vector INTERSEC user0 vector| / |request vector UNION user0 vector| =
        # 2 / 6 = 0.33
        #
        # jaccard_sim(
        #     request vector: [0, 0, 1, 1, 1, 0, 0],
        #     user1 vector:   [1, 0, 1, 1, 0, 0, 0]
        # ) = 2 / 4 = 0.5
        #
        # ...
        #
        # jaccard_sim(
        #     request vector: [0, 0, 1, 1, 1, 0, 0],
        #     user5 vector:   [1, 0, 0, 0, 0, 1, 0]
        # ) = 0 / 5 = 0
    },
    {
        "init_args": {
            "ratings": _process_ratings_array(ratings_arr1), 
            "similarity_func": jaccard_sim,
            "alpha": 0.02,
        },
        "similarity_args": {
            "user_vector": np.array([0, 1, 1, 0, 1, 1, 1])
        },
        "expected_output": np.array([5/5, 1/7, 2/5, 2/7, 3/6, 1/6]),
    },
    {
        "init_args": {
            "ratings": _process_ratings_array(ratings_arr1), 
            "similarity_func": jaccard_sim,
            "alpha": 0.1,
        },
        "similarity_args": {
            "user_vector": np.array([0, 1, 1, 0, 1, 1, 1])
        },
        "expected_output": np.array([5/5, 1/7, 2/5, 2/7, 3/6, 1/6]),
    },
    {
        "init_args": {
            "ratings": _process_ratings_array(ratings_arr1[:3]), 
            "similarity_func": jaccard_sim,
            "alpha": 0.02,
        },
        "similarity_args": {
            "user_vector": np.array([1, 0, 1, 1, 0, 0, 0])
        },
        "expected_output": np.array([1/7, 3/3, 0/5]),
    },
    {
        "init_args": {
            "ratings": _process_ratings_array(ratings_arr2), 
            "similarity_func": jaccard_sim,
            "alpha": 0.02,
        },
        "similarity_args": {
            "user_vector": np.array([1, 0, 1, 1])
        },
        "expected_output": np.array([1/4, 3/3, 0/4, 3/4, 1/4]),
    },
    {
        "init_args": {
            "ratings": _process_ratings_array(ratings_arr2[1:-1]), 
            "similarity_func": jaccard_sim,
            "alpha": 0.02,
        },
        "similarity_args": {
            "user_vector": np.array([1, 1, 0, 1])
        },
        "expected_output": np.array([2/4, 1/3, 3/4]),
    },
]


user2user_get_items_scores_test_cases = [
    {
        "init_args": {
            "ratings": _process_ratings_array(ratings_arr1), 
            "similarity_func": jaccard_sim,
            "alpha": 0.3,
        },
        "get_items_scores_args": {
            "uid": 0
        },
        "expected_output": np.array([5/9, 5/9, 0, 0, 9/9, 5/9, 4/9]),
        # Explanation:
        # Ratings matrix:
        #      [0, 1, 1, 0, 1, 1, 1], # user0     jaccard_sim(0, 0) = 0 (similarity with myself should be 0)
        #      [1, 0, 1, 1, 0, 0, 0], # user1     jaccard_sim(0, 1) = 1/7 = 0.143
        #      [0, 0, 0, 0, 1, 0, 1], # user2     jaccard_sim(0, 2) = 2/5 = 0.4
        #      [1, 1, 1, 1, 0, 0, 0], # user3     jaccard_sim(0, 3) = 2/7 = 0.286
        #      [1, 1, 0, 0, 1, 1, 0], # user4     jaccard_sim(0, 4) = 3/6 = 0.5
        #      [1, 0, 0, 0, 0, 1, 0], # user5     jaccard_sim(0, 5) = 1/6 = 0.167

        # taking only users, that are closer than alpha (in this case 0.3). So here we consider 
        # user2 and user4 to be similar to user0, so predicted ratings (\hat{r}_{ui}) here would be:
        
        #   2/5 * [0, 0, 0, 0, 1, 0, 1] 
        # + 
        #   3/6 * [1, 1, 0, 0, 1, 1, 0]
        # / 
        #  (2/5 + 3/6)
        # =
        #  [0.5, 0.5, 0. , 0. , 0.9, 0.5, 0.4] / 0.9
        # =
        #  [5/9, 5/9, 0,   0,   9/9, 5/9, 4/9]
    },
    {
        "init_args": {
            "ratings": _process_ratings_array(ratings_arr1), 
            "similarity_func": jaccard_sim,
            "alpha": 0.288,
        },
        "get_items_scores_args": {
            "uid": 0
        },
        "expected_output": np.array([5/9, 5/9, 0, 0, 9/9, 5/9, 4/9]),
    },
    {
        "init_args": {
            "ratings": _process_ratings_array(ratings_arr1), 
            "similarity_func": jaccard_sim,
            "alpha": 0.28,
        },
        "get_items_scores_args": {
            "uid": 0
        },
        "expected_output": np.array([55/83, 55/83, 20/83, 20/83, 63/83, 35/83, 28/83]),
        # Explanation:
        # Ratings matrix:
        #      [0, 1, 1, 0, 1, 1, 1], # user0     jaccard_sim(0, 0) = 0 (similarity with myself should be 0)
        #      [1, 0, 1, 1, 0, 0, 0], # user1     jaccard_sim(0, 1) = 1/7 = 0.143
        #      [0, 0, 0, 0, 1, 0, 1], # user2     jaccard_sim(0, 2) = 2/5 = 0.4
        #      [1, 1, 1, 1, 0, 0, 0], # user3     jaccard_sim(0, 3) = 2/7 = 0.286
        #      [1, 1, 0, 0, 1, 1, 0], # user4     jaccard_sim(0, 4) = 3/6 = 0.5
        #      [1, 0, 0, 0, 0, 1, 0], # user5     jaccard_sim(0, 5) = 1/6 = 0.167
        # taking only users, that are closer than alpha (in this case 0.28). So here we consider 
        # user2, user3 and user4 to be similar to user0, so predicted ratings (\hat{r}_{ui}) here would be:
        #   2/5 * [0, 0, 0, 0, 1, 0, 1] 
        # + 
        #   3/6 * [1, 1, 0, 0, 1, 1, 0]
        # +
        #   2/7 * [1, 1, 1, 1, 0, 0, 0]
        # / 
        #  (2/5 + 3/6 + 2/7)
        # =
        #  [55/83, 55/83, 20/83, 20/83, 63/83, 35/83, 28/83]
    },
]


