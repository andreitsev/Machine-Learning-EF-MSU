try:
    from fabulous import color as fb_color
    color_print = lambda x, color='green': print(getattr(fb_color, color)(x)) if 'fb_color' in globals() else print(x)
except:
    color_print = lambda x, color='green': print(x)

from public_tests import (
   _compute_binary_relevance_test_cases,
   ap_at_k_test_cases,
   map_at_k_test_cases,
   get_test_recommendations_test_cases
)

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
  test_cases = _compute_binary_relevance_test_cases
  for i, test_case in enumerate(test_cases, start=1):
    try:
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
    except Exception as e:
       color_print(f"Failed to test test__compute_binary_relevance for test {i}!", color='red')
       print(e, end='\n'*2)
  if add_score_flag:
     score += add_score_for_this_test
  return score


def test_ap_at_k(add_score_for_this_test: float=1.0) -> float:
  score = 0
  add_score_flag = True
  test_cases = ap_at_k_test_cases
  for i, test_case in enumerate(test_cases, start=1):
    print(f"Test {i}:")
    computed_output = ap_at_k(**test_case['args'])
    decision = (
        'passed ✓' if abs(computed_output - test_case['expected_output']) < 1e-2 else 'failed x'
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


def test_map_at_k(add_score_for_this_test: float=1.0) -> float:
  score = 0
  add_score_flag = True
  test_cases = map_at_k_test_cases
  for i, test_case in enumerate(test_cases, start=1):
    print(f"Test {i}:")
    computed_output = map_at_k(**test_case['args'])
    decision = (
        'passed ✓' if abs(computed_output - test_case['expected_output']) < 1e-2 else 'failed x'
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


def test_get_test_recommendations(add_score_for_this_test: float=1.0) -> float:
  score = 0
  add_score_flag = True
  test_cases = get_test_recommendations_test_cases
  for i, test_case in enumerate(test_cases, start=1):
    try:
      print(f"Test {i}:")
      recommender_model = test_case['model_type'](**test_case['init_args'])
      recommendations = recommender_model.get_test_recommendations(**test_case['get_test_recommendations_args'])
      decision = (
          'passed ✓' if conputed_output == test_case['expected_output'] else 'failed x'
      )
      color_print(decision, color='green' if decision == 'passed ✓' else 'red')
      # if decision == 'failed x':
      #   add_score_flag = False
      #   print(test_case)
      #   print('got output:')
      #   print(conputed_output)
      #   print()
    except Exception as e:
      color_print(f"Failed to test test__compute_binary_relevance for test {i}!", color='red')
      print(e, end='\n'*2)
  if add_score_flag:
    score += add_score_for_this_test
  return score


if __name__ == '__main__':
    total_score = 0
    print()
    print(f"Testing _compute_binary_relevance...")
    try:
      total_score += test__compute_binary_relevance()
    except Exception as e:
       color_print(f"Failed to test test__compute_binary_relevance", color='red')
       print(e, end='\n'*2)
    color_print(f"Текущий скор: {round(total_score, 3):,}", color='magenta')
    print()
    print(f"Testing ap_at_k...")
    try:
      total_score += test_ap_at_k()
    except Exception as e:
       color_print(f"Failed to test test_ap_at_k", color='red')
       print(e, end='\n'*2)
    color_print(f"Текущий скор: {round(total_score, 3):,}", color='magenta')
    print()
    print(f"Testing map_at_k...")
    try:
      total_score += test_map_at_k()
    except Exception as e:
       color_print(f"Failed to test test_map_at_k", color='red')
       print(e, end='\n'*2)
    color_print(f"Текущий скор: {round(total_score, 3):,}", color='magenta')
    print()
    color_print(f"Общий скор: {round(total_score, 3):,}", color='magenta')