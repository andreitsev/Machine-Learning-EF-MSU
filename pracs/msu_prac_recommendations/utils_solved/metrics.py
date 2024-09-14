from typing import List

import numpy as np

def _compute_binary_relevance(
    recommended_items_list: List[int],
    true_items_list: List[int],
) -> List[int]:
  # your code here:
  return [rec_item in true_items_list for rec_item in recommended_items_list]


def ap_at_k(
    recommended_items_list: List[int],
    true_items_list: List[int],
    k: int
) -> float:
  # your code here:
  binary_relevance_at_k = np.array(_compute_binary_relevance(
      recommended_items_list=recommended_items_list,
      true_items_list=true_items_list
  ))[:k]
  pu_at_k = np.array([cumrel / k_ for k_, cumrel in enumerate(np.cumsum(binary_relevance_at_k), start=1)])
  n_u = len(true_items_list)
  return (1 / min(k, n_u)) * (pu_at_k @ binary_relevance_at_k)



def map_at_k(
    recommended_items_lists: List[List[int]],
    true_items_lists: List[List[int]],
    k: int,
) -> float:
  """
  Computes ap@k for all buyers
  """
  assert len(recommended_items_lists) == len(true_items_lists), \
  'len(true_items_list) != len(recommended_items_list)'

  # your code here:
  ...