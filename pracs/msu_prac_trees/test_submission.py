import os
from os.path import join as p_join
import numpy as np
from fabulous import color as fb_color

# from submission import *
from submission_solved import *

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
                print(fb_color.green(test_str))
            else:
                test_str += 'x'
                passed_all_tests = False
                print(fb_color.red(test_str))
        except:
            test_str += 'Failed to test'
            passed_all_tests = False
            print(fb_color.red(test_str))
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
                    print(fb_color.green(test_str))
                else:
                    test_str += 'x'
                    passed_all_tests = False
                    print(fb_color.red(test_str))
            except:
                test_str += 'Failed to test'
                passed_all_tests = False
                print(fb_color.red(test_str))

    if passed_all_tests:
        global_score += 1
    print(fb_color.bold(fb_color.magenta(f'Current total score: {global_score}')))


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
                print(fb_color.green(test_str))
            else:
                test_str += 'x'
                passed_all_tests = False
                print(fb_color.red(test_str))
        except:
            test_str += 'Failed to test'
            passed_all_tests = False
            print(fb_color.red(test_str))
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
                    print(fb_color.green(test_str))
                else:
                    test_str += 'x'
                    passed_all_tests = False
                    print(fb_color.red(test_str))
            except:
                test_str += 'Failed to test'
                passed_all_tests = False
                print(fb_color.red(test_str))

    if passed_all_tests:
        global_score += 1
    print(fb_color.bold(fb_color.magenta(f'Current total score: {global_score}')))






if __name__ == '__main__':
    test_compute_criterion()
    test_find_best_split()

    print()
    print(fb_color.bold(fb_color.magenta(f"Итоговый балл: {global_score}")))