try:
    from fabulous import color as fb_color
    color_print = lambda x, color='green': print(getattr(fb_color, color)(x)) if 'fb_color' in globals() else print(x)
except:
    color_print = lambda x, color='green': print(x)

# from utils.metrics import (
#    _compute_binary_relevance,
#    ap_at_k,
#    map_at_k
# )
from utils.metrics_solved import (
   _compute_binary_relevance,
   ap_at_k,
   map_at_k
)

def test__compute_binary_relevance(add_score_for_this_test: float=1.0) -> float:
  score = 0
  add_score_flag = True
  test_cases = [
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
  for i, test_case in enumerate(test_cases, start=1):
    print(f"Test {i}:")
    conputed_output = _compute_binary_relevance(**test_case['args'])
    decision = (
        'passed ✓' if conputed_output == test_case['expected_output'] else 'failed x'
    )
    color_print(decision, color='green' if decision == 'passed ✓' else 'red')
    if decision == 'failed x':
      add_score_flag = False
      print(test_case)
      print('got output:')
      print(conputed_output)
      print()
  if add_score_flag:
     score += add_score_for_this_test
  return score


def test_ap_at_k(add_score_for_this_test: float=1.0) -> float:
  score = 0
  add_score_flag = True
  test_cases = [
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
  for i, test_case in enumerate(test_cases, start=1):
    print(f"Test {i}:")
    computed_output = ap_at_k(**test_case['args'])
    decision = (
        'passed ✓' if computed_output == test_case['expected_output'] else 'failed x'
    )
    color_print(decision, color='green' if decision == 'passed ✓' else 'red')
    if decision == 'failed x':
      add_score_flag = False
      print(test_case)
      print('got output:')
      print(computed_output)
      print()
  if add_score_flag:
     score += add_score_for_this_test
  return score


if __name__ == '__main__':
    total_score = 0
    print()
    print(f"Testing _compute_binary_relevance...")
    total_score += test__compute_binary_relevance()
    color_print(f"Текущий скор: {round(total_score, 3):,}", color='magenta')
    print()
    print(f"Testing ap_at_k...")
    total_score += test_ap_at_k()
    color_print(f"Текущий скор: {round(total_score, 3):,}", color='magenta')
    print()
    color_print(f"Общий скор: {round(total_score, 3):,}", color='magenta')