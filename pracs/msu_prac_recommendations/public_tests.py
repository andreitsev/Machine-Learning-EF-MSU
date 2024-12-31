import array
from fractions import Fraction

import numpy as np
import pandas as pd

from utils.models import (
    user_col,
    item_col,
    rating_col,
)
from utils.distances import jaccard_sim


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
          "expected_output": round(
              float(
                  Fraction(1, 4)*(
                      Fraction(1, 1) * 1
                      + Fraction(2, 2) * 1
                      + Fraction(2, 3) * 0
                      + Fraction(2, 4) * 0
                  )
              ), 
              6
            )
      },
      {
          "args": {
              "recommended_items_list": [4, 5, 6],
              "true_items_list": [1, 2, 3],
              "k": 3
          },
          "expected_output": 0
      },
      {
          "args": {
              "recommended_items_list": [4, 5, 3],
              "true_items_list": [1, 2, 3],
              "k": 3
          },          
          "expected_output": round(
              float(
                  Fraction(1, 3)*(
                      Fraction(0, 1) * 0
                      + Fraction(0, 2) * 0
                      + Fraction(1, 3) * 1
                  )
              ), 
              6
            )
      },
      {
          "args": {
              "recommended_items_list": [4, 5, 3],
              "true_items_list": [0, 1, 2, 3],
              "k": 3
          },
          "expected_output": round(
              float(
                  Fraction(1, 3)*(
                      Fraction(0, 1) * 0
                      + Fraction(0, 2) * 0
                      + Fraction(1, 3) * 1
                  )
              ), 
              6
            )
      },
      {
          "args": {
              "recommended_items_list": [1, 5, 3],
              "true_items_list": [0, 1, 2, 3],
              "k": 3
          },
          "expected_output": round(
              float(
                  Fraction(1, 3)*(
                      Fraction(1, 1) * 1
                      + Fraction(1, 2) * 0
                      + Fraction(2, 3) * 1
                  )
              ), 
              6
            )
      },
      {
          "args": {
              "recommended_items_list": [4, 2, 3],
              "true_items_list": [1, 2, 3],
              "k": 3
          },
          "expected_output": round(
              float(
                  Fraction(1, 3)*(
                      Fraction(0, 1) * 0
                      + Fraction(1, 2) * 1
                      + Fraction(2, 3) * 1
                  )
              ), 
              6
            )
      },
      {
          "args": {
              "recommended_items_list": [1, 2, 3, 4, 5],
              "true_items_list": [3, 4, 5],
              "k": 5
          },
          "expected_output": round(
              float(
                  Fraction(1, min(3, 5))*(
                      Fraction(0, 1) * 0
                      + Fraction(0, 2) * 0
                      + Fraction(1, 3) * 1
                      + Fraction(2, 4) * 1
                      + Fraction(3, 5) * 1
                  )
              ), 
              6
            )
      },
      {
          "args": {
              "recommended_items_list": [1, 2, 3, 4, 5],
              "true_items_list": [3, 4, 5],
              "k": 3
          },
          "expected_output": round(
              float(
                  Fraction(1, 3)*(
                      Fraction(0, 1) * 0
                      + Fraction(0, 2) * 0
                      + Fraction(1, 3) * 1
                  )
              ), 
              6
            )
      },
      {
          "args": {
              "recommended_items_list": [1, 2, 4, 5, 6],
              "true_items_list": [1, 2, 3],
              "k": 5
          },
          "expected_output": round(
              float(
                  Fraction(1, min(3, 5))*(
                      Fraction(1, 1) * 1
                      + Fraction(2, 2) * 1
                      + Fraction(2, 3) * 0
                      + Fraction(2, 4) * 0
                      + Fraction(2, 5) * 0
                  )
              ), 
              6
            )
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
    {
        "init_args": {
            "ratings": _process_ratings_array(ratings_arr1), 
            "similarity_func": jaccard_sim,
            "alpha": 0.1,
        },
        "get_items_scores_args": {
            "uid": 2
        },
        "expected_output": np.array([1/3, 1, 2/3, 0, 1, 1, 2/3])
    },
    {
        "init_args": {
            "ratings": _process_ratings_array(ratings_arr1), 
            "similarity_func": jaccard_sim,
            "alpha": 0.15,
        },
        "get_items_scores_args": {
            "uid": 2
        },
        "expected_output": np.array([1/3, 1, 2/3, 0, 1, 1, 2/3])
    },
    {
        "init_args": {
            "ratings": _process_ratings_array(ratings_arr1), 
            "similarity_func": jaccard_sim,
            "alpha": 0.12,
        },
        "get_items_scores_args": {
            "uid": 5
        },
        "expected_output": np.array([0.8507, 0.7761, 0.5522, 0.403 , 0.597 , 0.597 , 0.1493])
    },
    {
        "init_args": {
            "ratings": _process_ratings_array(ratings_arr2), 
            "similarity_func": jaccard_sim,
            "alpha": 0.15,
        },
        "get_items_scores_args": {
            "uid": 0
        },
        "expected_output": np.array([0.6842, 0.8421, 0.4737, 0.4737])
    }
]

interactions_df = pd.DataFrame({
    "user_id": [1, 1, 2, 2, 3, 3],
    "item_id": [10, 20, 10, 30, 20, 30],
    "rating": [5, 4, 5, 3, 4, 2],
})


als_initialise_embeddings_test_cases = [
    {
        "init_args": {
            "embeddings_dim": 8, 
            "random_seed": 42,
        },
        "_initialise_embeddings_args": {
            "n_unq_users": 3, 
            "n_unq_items": 4,
        },
        "expected_output": {
            "users_embeddings_shape": (3, 8),
            "items_embeddings_shape": (4, 8),
        }
    },
    {
        "init_args": {
            "embeddings_dim": 15, 
            "random_seed": 12,
        },
        "_initialise_embeddings_args": {
            "n_unq_users": 10, 
            "n_unq_items": 4,
        },
        "expected_output": {
            "users_embeddings_shape": (10, 15),
            "items_embeddings_shape": (4, 15),
        }
    },
]

_als_user_step_test_cases = [
    {
        "args": {
            "items_embeddings": np.array([
                [0, 1.0, 0],
                [-1.0, 2.0, 2.1],
                [1.5, 1.1, 2.0],
            ]),
            "user_ratings": np.array([5.0, 1.0, 2.0]),
            "reg_coef": 1.0
        },
        "expected_output": np.array([0.81186175, 2.04128491, -0.81773635])
    },
    {
        "args": {
            "items_embeddings": np.array([
                [0, 1.0, 0, 1.0, 1.5, -1.5, 0],
                [-1.0, 2.0, 2.1, 3.1, 3.1, -2.0, 1.1],
                [1.5, 1.1, 2.0, 11.5, -3.1, 5.7, -0.5],
            ]),
            "user_ratings": np.array([5.0, 1.0, 2.0]),
            "reg_coef": 2.0
        },
        "expected_output": np.array([0.55809731, 0.43001109, -0.8480172, 0.57417105, 0.29982033, -0.64635934, -0.54156039])
    },
    {
        "args": {
            "items_embeddings": np.array([
                [0.0, 1.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 2.0, 3.0],
                [-1.0, 0.0, 1.0, 0.0],
                [1.0, 2.0, 3.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
            ]),
            "user_ratings": np.array([5.0, 4.0, 5.0, 5.0, 3.0, 4.0]),
            "reg_coef": 1.0
        },
        "expected_output": np.array([-2.50867052, 2.75337187, 0.00963391, 0.97109827])
    },
    {
        "args": {
            "items_embeddings": np.array([
                [0.0, 1.0, -1.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 2.0],
            ]),
            "user_ratings": np.array([5.0, 4.0, 5.0]),
            "reg_coef": 1.0
        },
        "expected_output": np.array([0.0, 3.05, 0.85])
    },
    {
        "args": {
            "items_embeddings": np.array([
                [1.0, 1.0],
                [1.0, 0.0],
                [3.0, 1.0],
                [-1.5, 1.0],
                [1.0, 2.0],
                [0.0, 1.0],
            ]),
            "user_ratings": np.array([5.0, 4.0, 5.0, 3.1, 2.5, 5.0]),
            "reg_coef": 1.5
        },
        "expected_output": np.array([0.80096618, 2.05217391])
    },
]

_als_item_step_test_cases = [
    {
        "args": {
            "users_embeddings": np.array([
                [0, 1.0, 0],
                [-1.0, 2.0, 2.1],
                [1.5, 1.1, 2.0],
            ]),
            "items_ratings": np.array([5.0, 1.0, 2.0]),
            "reg_coef": 1.0
        },
        "expected_output": np.array([0.81186175, 2.04128491, -0.81773635]),
    },
    {
        "args": {
            "users_embeddings": np.array([
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 2.0, 3.5, 0.0],
            ]),
            "items_ratings": np.array([5.0, 1.0, 2.0]),
            "reg_coef": 1.0
        },
        "expected_output": np.array([0.0, 1.24680851, -0.05106383, 1.87659574]),
    },
    {
        "args": {
            "users_embeddings": np.array([
                [1.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 3.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.1, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 3.0, 3.5, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.5],
            ]),
            "items_ratings": np.array([1.0, 1.0, 2.0, 2.0, 5.0, 5.0, 3.5, 2.5]),
            "reg_coef": 3.5
        },
        "expected_output": np.array([0.35468729, 0.36209406, 0.59609282, 0.07696039, 0.5599045, 0.74954861]),
    },
        {
        "args": {
            "users_embeddings": np.array([
                [0.0, 1.0, -1.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 2.0],
            ]),
            "items_ratings": np.array([5.0, 4.0, 5.0]),
            "reg_coef": 1.0
        },
        "expected_output": np.array([0.0, 3.05, 0.85])
    },
]


user_col = 'user_id'
item_col = 'item_id'
rating_col = 'rating'

fitted_als_test_cases = [
    {
        "init_args": {
            "embeddings_dim": 4,
            "reg_coef": 1.0,
            "random_seed": 59812,
        },
        "init_users_embeddings": {
            "u1": [1.0, 0.0, 1.0, -1.0],
            "u2": [-1.0, 2.0, 0.0, 1.5],
        },
        "init_items_embeddings": {
            "i1": [1.0, 0.0, 2.0, 0.0],
            "i2": [0.0, 1.0, -2.0, 0.0],
            "i3": [-1.0, 2.0, 3.0, 0.0],
            "i4": [-1.0, -2.0, -3.0, 1.0],
            "i5": [1.0, 0.0, 1.0, -1.5],
            "i6": [-1.5, 0.0, -4.2, 3.5],
        },
        "fit_args": {
            "interactions": pd.DataFrame([
               ['u1', 'i1', 5.0], 
               ['u1', 'i2', 3.5], 
               ['u1', 'i3', 1.4],
               ['u1', 'i5', 5.0],
               ['u1', 'i6', 5.0],
               ['u2', 'i1', 3.5],
               ['u2', 'i3', 2.0],
               ['u2', 'i4', 5.0],
               ['u2', 'i6', 5.0],
            ], columns=[user_col, item_col, rating_col]), 
            "epochs": 1, 
            "verbose": False,
            "embeddings_initialized": True,
        },
        "expected_output": {
            "users_embeddings": {
                'u1': [ 2.91189875,  2.1862228 , -0.24838483,  1.65364402],
                'u2': [-0.91152467, -1.65963185,  1.13801769,  2.35976282]
            },
            "items_embeddings": {
                'i1': [0.67642815, 0.12945758, 0.35367606, 1.50055343],
                'i2': [ 0.59757671,  0.44865427, -0.05097327,  0.33935903],
                'i3': [ 0.14716455, -0.08562527,  0.20065338,  0.66218629],
                'i4': [-0.39808673, -0.72480476,  0.49700218,  1.03057033],
                'i5': [ 0.85368101,  0.64093468, -0.07281895,  0.48479862],
                'i6': [ 0.61431736, -0.04976799,  0.50312324,  1.85648186],
            },
        }
    },
    {
        "init_args": {
            "embeddings_dim": 4,
            "reg_coef": 1.0,
            "random_seed": 59812,
        },
        "init_users_embeddings": {
            "u1": [1.0, 0.0, 1.0, -1.0],
            "u2": [-1.0, 2.0, 0.0, 1.5],
        },
        "init_items_embeddings": {
            "i1": [1.0, 0.0, 2.0, 0.0],
            "i2": [0.0, 1.0, -2.0, 0.0],
            "i3": [-1.0, 2.0, 3.0, 0.0],
            "i4": [-1.0, -2.0, -3.0, 1.0],
            "i5": [1.0, 0.0, 1.0, -1.5],
            "i6": [-1.5, 0.0, -4.2, 3.5],
        },
        "fit_args": {
            "interactions": pd.DataFrame([
               ['u1', 'i1', 5.0], 
               ['u1', 'i2', 3.5], 
               ['u1', 'i3', 1.4],
               ['u1', 'i5', 5.0],
               ['u1', 'i6', 5.0],
               ['u2', 'i1', 3.5],
               ['u2', 'i3', 2.0],
               ['u2', 'i4', 5.0],
               ['u2', 'i6', 5.0],
            ], columns=[user_col, item_col, rating_col]), 
            "epochs": 2, 
            "verbose": False,
            "embeddings_initialized": True,
        },
        "expected_output": {
            "users_embeddings": {
                'u1': [2.14076109, 1.36890338, 0.07652375, 1.91896533],
                'u2': [-0.24766539, -1.01162388,  0.91876417,  2.29541798],
            },
            "items_embeddings": {
                'i1': [0.7319243 , 0.20436535, 0.31280333, 1.4339933 ],
                'i2': [0.67228695, 0.42989191, 0.0240316 , 0.60263397],
                'i3': [ 0.10582747, -0.1205239 ,  0.20837931,  0.65010917],
                'i4': [-0.15105615, -0.61700995,  0.56037293,  1.40002204],
                'i5': [0.96040993, 0.6141313 , 0.03433086, 0.86090567],
                'i6': [ 0.58617234, -0.05702611,  0.49044212,  1.79956837],
            },
        }
    },
    {
        "init_args": {
            "embeddings_dim": 4,
            "reg_coef": 1.0,
            "random_seed": 59812,
        },
        "init_users_embeddings": {
            "u1": [1.0, 0.0, 1.0, -1.0],
            "u2": [-1.0, 2.0, 0.0, 1.5],
        },
        "init_items_embeddings": {
            "i1": [1.0, 0.0, 2.0, 0.0],
            "i2": [0.0, 1.0, -2.0, 0.0],
            "i3": [-1.0, 2.0, 3.0, 0.0],
            "i4": [-1.0, -2.0, -3.0, 1.0],
            "i5": [1.0, 0.0, 1.0, -1.5],
            "i6": [-1.5, 0.0, -4.2, 3.5],
        },
        "fit_args": {
            "interactions": pd.DataFrame([
               ['u1', 'i1', 5.0], 
               ['u1', 'i2', 3.5], 
               ['u1', 'i3', 1.4],
               ['u1', 'i5', 5.0],
               ['u1', 'i6', 5.0],
               ['u2', 'i1', 3.5],
               ['u2', 'i3', 2.0],
               ['u2', 'i4', 5.0],
               ['u2', 'i6', 5.0],
            ], columns=[user_col, item_col, rating_col]), 
            "epochs": 3, 
            "verbose": False,
            "embeddings_initialized": True,
        },
        "expected_output": {
            "users_embeddings": {
                'u1': [1.75301713, 0.94766214, 0.25106621, 2.08269283],
                'u2': [ 0.1271879 , -0.65229955,  0.80211268,  2.2784939 ],
            },
            "items_embeddings": {
                'i1': [0.78206907, 0.27024811, 0.2778307 , 1.3791674 ],
                'i2': [0.65468485, 0.35391557, 0.09376363, 0.77780612],
                'i3': [ 0.07316793, -0.14833306,  0.2147405 ,  0.64126503],
                'i4': [ 0.08739525, -0.44821784,  0.55115968,  1.56563286],
                'i5': [0.93526408, 0.50559367, 0.13394804, 1.1111516 ],
                'i6': [ 0.56764052, -0.05916737,  0.47922467,  1.75431175],
            },
        }
    },
    {
        "init_args": {
            "embeddings_dim": 4,
            "reg_coef": 1.0,
            "random_seed": 59812,
        },
        "init_users_embeddings": {
            "u1": [1.0, 0.0, 1.0, -1.0],
            "u2": [-1.0, 2.0, 0.0, 1.5],
        },
        "init_items_embeddings": {
            "i1": [1.0, 0.0, 2.0, 0.0],
            "i2": [0.0, 1.0, -2.0, 0.0],
            "i3": [-1.0, 2.0, 3.0, 0.0],
            "i4": [-1.0, -2.0, -3.0, 1.0],
            "i5": [1.0, 0.0, 1.0, -1.5],
            "i6": [-1.5, 0.0, -4.2, 3.5],
        },
        "fit_args": {
            "interactions": pd.DataFrame([
               ['u1', 'i1', 5.0], 
               ['u1', 'i2', 3.5], 
               ['u1', 'i3', 1.4],
               ['u1', 'i5', 5.0],
               ['u1', 'i6', 5.0],
               ['u2', 'i1', 3.5],
               ['u2', 'i3', 2.0],
               ['u2', 'i4', 5.0],
               ['u2', 'i6', 5.0],
            ], columns=[user_col, item_col, rating_col]), 
            "epochs": 7, 
            "verbose": False,
            "embeddings_initialized": True,
        },
        "expected_output": {
            "users_embeddings": {
                'u1': [1.32706701, 0.47945577, 0.44874116, 2.27865728],
                'u2': [ 0.56346785, -0.23356026,  0.66576623,  2.25722394],
            },
            "items_embeddings": {
                'i1': [0.81117504, 0.31479508, 0.25067559, 1.32873907],
                'i2': [0.55395805, 0.20013939, 0.18731818, 0.95118071],
                'i3': [ 0.06482416, -0.14853007,  0.20885604,  0.61862557],
                'i4': [ 0.40769843, -0.16899305,  0.48171665,  1.63321945],
                'i5': [0.79136864, 0.28591342, 0.2675974 , 1.35882958],
                'i6': [ 0.57249143, -0.033253  ,  0.45459747,  1.6913547 ],
            },
        }
    },

]

