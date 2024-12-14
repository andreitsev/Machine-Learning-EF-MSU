from functools import partial
from pprint import pprint

import numpy as np
try:
    from fabulous import color as fb_color
    color_print = lambda x, color='green': print(getattr(fb_color, color)(x)) if 'fb_color' in globals() else print(x)
except:
    color_print = lambda x, color='green': print(x)

from public_tests import (
   _compute_binary_relevance_test_cases,
   ap_at_k_test_cases,
   map_at_k_test_cases,
   jaccard_sim_test_cases,
   user2user_similarity_output_length_test_cases,
   user2user_similarity_test_cases,
   user2user_get_items_scores_test_cases,
  #  als_initialise_embeddings_test_cases,
   _als_user_step_test_cases,
   _als_item_step_test_cases,
   fitted_als_test_cases
)

from utils.metrics import (
   _compute_binary_relevance,
   ap_at_k,
   map_at_k
)
from utils.distances import jaccard_sim

from utils.models import (
  User2User, 
  EmbeddingType, 
  ALS, 
  _als_user_step, 
  _als_item_step
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
      add_score_flag = False
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
    try:
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
    except Exception as e:
      add_score_flag = False
      color_print(f"Failed to test test_map_at_k for test {i}!", color='red')
      print(e, end='\n'*2)
  if add_score_flag:
    score += add_score_for_this_test
  return score


def test_jaccard_sim(add_score_for_this_test: float=1.0) -> float:
  score = 0
  add_score_flag = True
  test_cases = jaccard_sim_test_cases
  for i, test_case in enumerate(test_cases, start=1):
    try:
      print(f"Test {i}:")
      computed_output = jaccard_sim(**test_case['args'])
      decision = (
          'passed ✓' if all(np.abs(computed_output - test_case['expected_output']) < 1e-2) else 'failed x'
      )
      color_print(decision, color='green' if decision == 'passed ✓' else 'red')
      if decision == 'failed x':
        add_score_flag = False
        print(test_case)
        print('got output:')
        print(computed_output)
        print()
    except Exception as e:
      add_score_flag = False
      color_print(f"Failed to test test_jaccard_sim for test {i}!", color='red')
      print(e, end='\n'*2)
  if add_score_flag:
    score += add_score_for_this_test
  return score


def test_user2user_similarity_output_length(add_score_for_this_test: float=1.0) -> float:
  score = 0
  add_score_flag = True
  test_cases = user2user_similarity_output_length_test_cases
  for i, test_case in enumerate(test_cases, start=1):
    try:
      print(f"Test {i}:")
      user2user_model = User2User(**test_case['init_args'])
      computed_output = len(user2user_model.similarity(**test_case['similarity_args']))
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
    except Exception as e:
      add_score_flag = False
      color_print(f"Failed to test test_user2user_similarity_output_length for test {i}!", color='red')
      print(e, end='\n'*2)
  if add_score_flag:
    score += add_score_for_this_test
  return score

def test_user2user_similarity(add_score_for_this_test: float=1.0) -> float:
  score = 0
  add_score_flag = True
  test_cases = user2user_similarity_test_cases
  for i, test_case in enumerate(test_cases, start=1):
    try:
      print(f"Test {i}:")
      user2user_model = User2User(**test_case['init_args'])
      computed_output = user2user_model.similarity(**test_case['similarity_args'])
      decision = (
          'passed ✓' if np.allclose(test_case['expected_output'], computed_output, atol=1e-2) else 'failed x'
      )
      color_print(decision, color='green' if decision == 'passed ✓' else 'red')
      if decision == 'failed x':
        add_score_flag = False
        print(test_case)
        print('got output:')
        print(computed_output)
        print()
    except Exception as e:
      add_score_flag = False
      color_print(f"Failed to test test_user2user_similarity for test {i}!", color='red')
      print(e, end='\n'*2)
  if add_score_flag:
    score += add_score_for_this_test
  return score

def test_user2user_get_items_scores(add_score_for_this_test: float=1.0) -> float:
  score = 0
  add_score_flag = True
  test_cases = user2user_get_items_scores_test_cases
  for i, test_case in enumerate(test_cases, start=1):
    try:
      print(f"Test {i}:")
      user2user_model = User2User(**test_case['init_args'])
      computed_output = user2user_model.get_items_scores(**test_case['get_items_scores_args'])
      decision = (
          'passed ✓' if np.allclose(test_case['expected_output'], computed_output, atol=1e-2) else 'failed x'
      )
      color_print(decision, color='green' if decision == 'passed ✓' else 'red')
      if decision == 'failed x':
        add_score_flag = False
        print('test_case:')
        print({k:v for k, v in test_case.items() if k != 'expected_output'})
        print('expected_output:')
        print(test_case.get('expected_output', 'No expected output in test_case dict!'))
        print('got output:')
        print(computed_output)
        print()
    except Exception as e:
      add_score_flag = False
      color_print(f"Failed to test test_user2user_get_items_scores for test {i}!", color='red')
      print(e, end='\n'*2)
  if add_score_flag:
    score += add_score_for_this_test
  return score

# def test_initialise_embeddings(add_score_for_this_test: float=1.0) -> float:
#     score = 0
#     add_score_flag = True
      
#     test_cases = als_initialise_embeddings_test_cases
#     for i, test_case in enumerate(test_cases, start=1):
#       try:
#         print(f"Test {i}:")
#         model = ALS(**test_case['init_args'])
#         model._initialise_embeddings(**test_case['_initialise_embeddings_args'])
#         decision = (
#             'passed ✓' 
#             if (
#               model.users_embeddings.shape == test_case['expected_output']['users_embeddings_shape']
#               and model.items_embeddings.shape == test_case['expected_output']['items_embeddings_shape']
#             ) 
#             else 'failed x'
#         )
#         color_print(decision, color='green' if decision == 'passed ✓' else 'red')
#         if decision == 'failed x':
#           add_score_flag = False
#           pprint(test_case)
#           print('got output:')
#           print(f'\tmodel.users_embeddings.shape: {model.users_embeddings.shape}')
#           print(f'\tmodel.items_embeddings.shape: {model.items_embeddings.shape}')
#           print()
#       except Exception as e:
#         add_score_flag = False
#         color_print(f"Failed to test test_initialise_embeddings for test {i}!", color='red')
#         print(e, end='\n'*2)
#     if add_score_flag:
#       score += add_score_for_this_test
#     return score

def test__als_user_step(add_score_for_this_test: float = 1.0) -> float:
    score = 0
    add_score_flag = True
    test_cases = _als_user_step_test_cases
    for i, test_case in enumerate(test_cases, start=1):
        try:
            print(f"Test {i}:")
            computed_output = _als_user_step(**test_case['args'])
            # Compare with expected output
            decision = (
                "passed ✓" if np.allclose(computed_output, test_case["expected_output"], atol=1e-3) else "failed x"
            )
            color_print(decision, color="green" if decision == "passed ✓" else "red")
            if decision == "failed x":
                add_score_flag = False
                print("Test Case:", test_case)
                print("Computed Output:", computed_output)
                print("Expected Output:", test_case["expected_output"])
                print()
        except Exception as e:
            add_score_flag = False
            color_print(f"Failed to test _als_user_step for test {i}!", color="red")
            print(e, end="\n" * 2)

    if add_score_flag:
        score += add_score_for_this_test
    return score
  
def test__als_item_step(add_score_for_this_test: float = 1.0) -> float:
    score = 0
    add_score_flag = True
    test_cases = _als_item_step_test_cases
    for i, test_case in enumerate(test_cases, start=1):
        try:
            print(f"Test {i}:")
            computed_output = _als_item_step(**test_case['args'])
            # Compare with expected output
            decision = (
                "passed ✓" if np.allclose(computed_output, test_case["expected_output"], atol=1e-3) else "failed x"
            )
            color_print(decision, color="green" if decision == "passed ✓" else "red")
            if decision == "failed x":
                add_score_flag = False
                print("Test Case:", test_case)
                print("Computed Output:", computed_output)
                print("Expected Output:", test_case["expected_output"])
                print()
        except Exception as e:
            add_score_flag = False
            color_print(f"Failed to test _als_user_step for test {i}!", color="red")
            print(e, end="\n" * 2)

    if add_score_flag:
        score += add_score_for_this_test
    return score
  
def check_embeddings_equal(
  embeddings1: EmbeddingType, 
  embeddings2: EmbeddingType, 
  tol: float=1e-3
) -> bool: 

  if len(set(embeddings1).symmetric_difference(set(embeddings2))) > 0:
    return False

  embs_equal_flag = True
  for key in embeddings1:
    emb1, emb2 = embeddings1[key], embeddings2[key]
    if not np.allclose(emb1, emb2, atol=tol):
      embs_equal_flag = False
      break
  return embs_equal_flag
  
  
def test__als_fit(add_score_for_this_test: float = 1.0) -> float:
    score = 0
    add_score_flag = True
    test_cases = fitted_als_test_cases
    for i, test_case in enumerate(test_cases, start=1):
        try:
            print(f"Test {i}:")
            als = ALS(**test_case['init_args'])
            # create new dict, so that als.<...>_embeddings and test_case['init_<...>_embeddings'] don't point
            # at the same object (and thus test_case['init_<...>_embeddings'] might get changed after als.fit(...))
            als.users_embeddings = {user_id: emb for user_id, emb in test_case['init_users_embeddings'].items()}
            als.items_embeddings = {item_id: emb for item_id, emb in test_case['init_items_embeddings'].items()}
            als.fit(**test_case['fit_args'])
            # Compare with expected output
            decision = (
                "passed ✓" 
                if (
                  check_embeddings_equal(
                    embeddings1=als.users_embeddings, 
                    embeddings2=test_case['expected_output']['users_embeddings'], 
                    tol=1e-3,
                  )
                  and check_embeddings_equal(
                    embeddings1=als.items_embeddings, 
                    embeddings2=test_case['expected_output']['items_embeddings'], 
                    tol=1e-3,
                  )
                )
                else "failed x"
            )
            color_print(decision, color="green" if decision == "passed ✓" else "red")
            if decision == "failed x":
                add_score_flag = False
                print("Test Case:")
                pprint(test_case)
                print()
                print("Computed Output:")
                print("\tusers_embeddings:")
                pprint(als.users_embeddings)
                print("\titems_embeddings:")
                pprint(als.items_embeddings)
                print("\nExpected Output:")
                print("\tusers_embeddings:")
                pprint(test_case['expected_output']['users_embeddings'])
                print("\titems_embeddings:")
                pprint(test_case['expected_output']['items_embeddings'])
                print()
        except Exception as e:
            add_score_flag = False
            color_print(f"Failed to test _als_user_step for test {i}!", color="red")
            print(e, end="\n" * 2)

    if add_score_flag:
        score += add_score_for_this_test
    return score


if __name__ == '__main__':
    total_score = 0
    for testing_function in [
      partial(test__compute_binary_relevance, add_score_for_this_test=1.0),
      partial(test_ap_at_k, add_score_for_this_test=2.0),
      partial(test_map_at_k, add_score_for_this_test=1.0),
      partial(test_jaccard_sim, add_score_for_this_test=1.0),
      partial(test_user2user_similarity_output_length, add_score_for_this_test=0.0),
      partial(test_user2user_similarity, add_score_for_this_test=1.0),
      partial(test_user2user_get_items_scores, add_score_for_this_test=3.0),
      partial(test__als_user_step, add_score_for_this_test=2.0),
      partial(test__als_item_step, add_score_for_this_test=2.0),
      partial(test__als_fit, add_score_for_this_test=8.0),
    ]:
      function_name = testing_function.func.__name__
      print(f"\n{function_name}...")
      try:
        add_score_for_this_test = testing_function()
        total_score += add_score_for_this_test
        color_print(f"+{add_score_for_this_test} балла(ов)", color='magenta' if add_score_for_this_test > 0 else 'red')
      except Exception as e:
        color_print(f"Failed to test {function_name}", color='red')
        print(e, end='\n'*2)
      color_print(f"Текущий скор: {round(total_score, 3):,}\n", color='magenta')
    color_print(f"\nОбщий скор: {round(total_score, 3):,}", color='magenta')