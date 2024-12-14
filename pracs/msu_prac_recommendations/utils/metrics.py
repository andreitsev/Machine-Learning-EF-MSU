from typing import List

def _compute_binary_relevance(
    recommended_items_list: List[int],
    true_items_list: List[int],
) -> List[int]:
  # your code here:
  ...


def ap_at_k(
    recommended_items_list: List[int],
    true_items_list: List[int],
    k: int
) -> float:
  # your code here:
  ...



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