import os
from os.path import join as p_join
import numpy as np
from copy import deepcopy
try:
    from fabulous import color as fb_color
    color_print = lambda x, color='green': print(getattr(fb_color, color)(x)) if 'fb_color' in globals() else print(x)
except:
    color_print = lambda x, color='green': print(x)

from submission import *
# from submission_solved import *

# Для тестирования обученных деревьев
from deepdiff import DeepDiff
from sklearn.metrics import accuracy_score

from public_tests import *
if 'private_tests.py' in os.listdir(os.getcwd()):
    from private_tests import *

global_score = 0

def test_compute_criterion():
    global global_score
    print('Тестируем compute_criterion:')
    passed_all_tests = True
    # Public tests ---------------------------------------------------------
    for public_test_number, test in enumerate(compute_criterion_tests):
        test_str = ' '*4 + f'Test {public_test_number+1}: '
        try:
            true_result = test['true_result']
            del test['true_result']
            test_result = compute_criterion(**test)
            if abs(true_result - test_result) <= 1e-3:
                test_str += '✓'
                color_print(x=test_str)
            else:
                test_str += 'x'
                passed_all_tests = False
                color_print(x=test_str, color='red')
        except:
            test_str += 'Failed to test'
            passed_all_tests = False
            color_print(x=test_str, color='red')
    # Private tests ---------------------------------------------------------
    if 'compute_criterion_tests_private' in globals():
        for private_test_number, test in enumerate(compute_criterion_tests_private):
            test_str = ' '*4 + f'Test {private_test_number+public_test_number+2}: '
            try:
                true_result = test['true_result']
                del test['true_result']
                test_result = compute_criterion(**test)
                if abs(true_result - test_result) <= 1e-3:
                    test_str += '✓'
                    color_print(x=test_str)
                else:
                    test_str += 'x'
                    passed_all_tests = False
                    color_print(x=test_str, color='red')
            except:
                test_str += 'Failed to test'
                passed_all_tests = False
                color_print(x=test_str, color='red')

    if passed_all_tests:
        global_score += 2
    color_print(x=f'Current total score: {global_score}', color='magenta')


def test_find_best_split():
    global global_score
    print('Тестируем find_best_split:')
    passed_all_tests = True
    # Public tests ---------------------------------------------------------
    for public_test_number, test in enumerate(find_best_split_tests):
        test_str = ' ' * 4 + f'Test {public_test_number + 1}: '
        try:
            true_result = test['true_result']
            del test['true_result']
            test_result = find_best_split(**test)
            if np.all(np.abs(true_result[0] - test_result[0]) <= 1e-2)\
                and np.all(np.abs(true_result[1] - test_result[1]) <= 1e-2)\
                and np.all(np.abs(true_result[2] - test_result[2]) <= 1e-2)\
                and np.all(np.abs(true_result[3] - test_result[3]) <= 1e-2):
                test_str += '✓'
                color_print(x=test_str)
            else:
                test_str += 'x'
                passed_all_tests = False
                color_print(x=test_str, color='red')
        except:
            test_str += 'Failed to test'
            passed_all_tests = False
            color_print(x=test_str, color='red')
    # Private tests ---------------------------------------------------------
    if 'find_best_split_tests_private' in globals():
        for private_test_number, test in enumerate(find_best_split_tests_private):
            test_str = ' ' * 4 + f'Test {private_test_number + public_test_number + 2}: '
            try:
                true_result = test['true_result']
                del test['true_result']
                test_result = find_best_split(**test)
                if np.all(np.abs(true_result[0] - test_result[0]) <= 1e-2) \
                        and np.all(np.abs(true_result[1] - test_result[1]) <= 1e-2) \
                        and np.all(np.abs(true_result[2] - test_result[2]) <= 1e-2) \
                        and np.all(np.abs(true_result[3] - test_result[3]) <= 1e-2):
                    test_str += '✓'
                    color_print(x=test_str)
                else:
                    test_str += 'x'
                    passed_all_tests = False
                    color_print(x=test_str, color='red')
            except:
                test_str += 'Failed to test'
                passed_all_tests = False
                color_print(x=test_str, color='red')

    if passed_all_tests:
        global_score += 3
    color_print(x=f'Current total score: {global_score}', color='magenta')



def test_DecisionTree_fit():
    global global_score
    print('Тестируем DecisionTree_fit:')
    # Public tests ---------------------------------------------------------
    for public_test_number, test in enumerate(DecisionTree_fit_tests):
        test_str = ' ' * 4 + f'Test {public_test_number + 1}: '
        if 'true_result' in test:
            true_result = test.pop('true_result')
        try:
            dt = DecisionTree(**test['tree_params'])
            dt.fit(X=test["X"], y=test["y"])
            test_res = DeepDiff(t1=dt._tree, t2=true_result, ignore_numeric_type_changes=True, significant_digits=3)
            if test_res == {}:
                test_str += '✓'
                color_print(x=test_str)
                if 'ADD_SCORE_FOR_THIS_TEST' in test:
                    global_score += test['ADD_SCORE_FOR_THIS_TEST']
                    color_print('\t'+f"+{test['ADD_SCORE_FOR_THIS_TEST']} балла")
            else:
                test_str += 'x'
                passed_all_tests = False
                color_print(x=test_str, color='red')
                color_print(test_res, color='red')
        except:
            test_str += 'Failed to test'
            passed_all_tests = False
            color_print(x=test_str, color='red')
    # Private tests ---------------------------------------------------------
    if 'DecisionTree_fit_tests_private' in globals():
        for private_test_number, test in enumerate(DecisionTree_fit_tests_private):
            test_str = ' ' * 4 + f'Test {private_test_number + public_test_number + 2}: '

            if 'true_result' in test:
                true_result = test.pop('true_result')
            try:
                dt = DecisionTree(**test['tree_params'])
                dt.fit(X=test["X"], y=test["y"])
                test_res = DeepDiff(t1=dt._tree, t2=true_result, ignore_numeric_type_changes=True, significant_digits=3)
                if test_res == {}:
                    test_str += '✓'
                    color_print(x=test_str)
                    if 'ADD_SCORE_FOR_THIS_TEST' in test:
                        global_score += test['ADD_SCORE_FOR_THIS_TEST']
                        color_print('\t' + f"+{test['ADD_SCORE_FOR_THIS_TEST']} балла")
                else:
                    test_str += 'x'
                    passed_all_tests = False
                    color_print(x=test_str, color='red')
                    color_print(test_res, color='red')
            except:
                test_str += 'Failed to test'
                passed_all_tests = False
                color_print(x=test_str, color='red')

    color_print(x=f'Current total score: {round(global_score, 3)}', color='magenta')


def test_DecisionTree_predict():
    global global_score
    print('Тестируем DecisionTree_predict(proba):')
    # Public tests ---------------------------------------------------------
    for public_test_number, test in enumerate(DecisionTree_predict_proba_tests):
        test_str = ' ' * 4 + f'Test {public_test_number + 1}: '
        if 'true_result' in test:
            true_result = test.pop('true_result')
        try:
            dt = DecisionTree(**test['tree_params'])
            X_copy, y_copy = deepcopy(test['X']), deepcopy(test['y'])
            dt.fit(X=X_copy, y=y_copy)
            y_pred = dt.predict(test["X"])
            acc = accuracy_score(y_true=test["y"], y_pred=y_pred)
            if abs(acc - true_result) < 1e-2:
                test_str += '✓'
                color_print(x=test_str)
                if 'ADD_SCORE_FOR_THIS_TEST' in test:
                    global_score += test['ADD_SCORE_FOR_THIS_TEST']
                    color_print('\t'+f"+{test['ADD_SCORE_FOR_THIS_TEST']} балла")
            else:
                test_str += 'x'
                passed_all_tests = False
                color_print(x=test_str, color='red')
        except:
            test_str += 'Failed to test'
            passed_all_tests = False
            color_print(x=test_str, color='red')
    # Private tests ---------------------------------------------------------
    if 'DecisionTree_predict_proba_tests_private' in globals():
        for private_test_number, test in enumerate(DecisionTree_predict_proba_tests_private):
            test_str = ' ' * 4 + f'Test {private_test_number + public_test_number + 2}: '

            if 'true_result' in test:
                true_result = test.pop('true_result')
            try:
                dt = DecisionTree(**test['tree_params'])
                X_copy, y_copy = deepcopy(test['X']), deepcopy(test['y'])
                dt.fit(X=X_copy, y=y_copy)
                y_pred = dt.predict(test["X"])
                acc = accuracy_score(y_true=test["y"], y_pred=y_pred)
                if abs(acc - true_result) < 1e-2:
                    test_str += '✓'
                    color_print(x=test_str)
                    if 'ADD_SCORE_FOR_THIS_TEST' in test:
                        global_score += test['ADD_SCORE_FOR_THIS_TEST']
                        color_print('\t' + f"+{test['ADD_SCORE_FOR_THIS_TEST']} балла")
                else:
                    test_str += 'x'
                    passed_all_tests = False
                    color_print(x=test_str, color='red')
            except:
                test_str += 'Failed to test'
                passed_all_tests = False
                color_print(x=test_str, color='red')

    color_print(x=f'Current total score: {round(global_score, 3)}', color='magenta')



if __name__ == '__main__':
    test_compute_criterion()
    test_find_best_split()
    test_DecisionTree_fit()
    test_DecisionTree_predict()
    print()
    # Выводим Общий балл и записываем его в файл:
    color_print(x=f"Итоговый балл: {round(global_score, 3)}", color='magenta')
    try:
        with open(p_join('/prac_folder/main_task_score.txt'), mode='w', encoding='utf-8') as f:
            f.writelines(str(round(global_score, 3)) + '\n')
    except:
        print("Не получилось записать main_task_score.txt")