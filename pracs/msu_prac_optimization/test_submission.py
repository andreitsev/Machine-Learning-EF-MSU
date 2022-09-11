import os
from os.path import join as p_join
import numpy as np
from fabulous import color as fb_color
try:
    from fabulous import color as fb_color
    color_print = lambda text, color='green': print(getattr(fb_color, color)(text))
except:
    color_print = lambda text, color='green': print(text)

from submission import *
# from submission_empty import *

global_score = 0


# Testing function computations -------------------------------------------------------------------------------------
def test_f1_value():
    """
    f(x) = x^2
    """

    test_cases = [
        (0, 0),
        (-1, 1),
        (3.43, 3.43**2),
        (-15, 15**2),
        (np.cos(15), np.cos(15)**2)
    ]

    func = f1()

    global global_score
    passed = True
    print('Testing f1 __call__ method...')
    for x, f_x in test_cases:
        decision = " "*4 + f"x={x} your f(x)={func(x)}"
        if abs(func(x) - f_x) <= 1e-4:
            decision += ' ✓'
        else:
            passed = False
            decision += ' x'
        #print(fb_color.green(decision) if passed else fb_color.red(decision))
        color_print(decision, color='green' if passed else 'red')

    if passed:
        print(f'f1 __call__ method is correct!')
        global_score += 1
    else:
        print(f'f1 __call__ method failed!')
    print(f"Total score: {global_score}", end='\n'*2)

def test_f2_value():
    """
    f(x) = np.sin(3*np.sqrt(x**3)+2) + x**2
    """

    test_cases = [
        (0, 0.9092974268256817),
        (1, 0.041075725336861546),
        (2.085, 3.3478856303323523),
        (3.602, 12.47970423033661),
        (0.001, 0.9092589435785655),
        (1.512, 3.248198623755454),
        (0.734, -0.13917495438635596),
        (0.577, 0.16051102745867718),
    ]

    func = f2()

    global global_score
    passed = True
    print('Testing f2 __call__ method...')
    for x, f_x in test_cases:
        decision = " "*4 + f"x={x} your f(x)={func(x)}"
        if abs(func(x) - f_x) <= 1e-4:
            decision += ' ✓'
        else:
            passed = False
            decision += ' x'
        #print(fb_color.green(decision) if passed else fb_color.red(decision))
        color_print(decision, color='green' if passed else 'red')

    if passed:
        print(f'f2 __call__ method is correct!')
        global_score += 1
    else:
        print(f'f2 __call__ method failed!')
    print(f"Total score: {global_score}", end='\n'*2)

def test_f3_value():
    """
    f(x) = (x[0] - 3.3)**2/4 + (x[1] + 1.7)**2/15
    """

    test_cases = [
        (np.array([0, 0]), 2.9151666666666665),
        (np.array([3.303, -1.7]), 2.2500000000001707e-06),
        (np.array([-0.83, 2.2]), 5.278225),
        (np.array([-5.0, -1.98]), 17.22772666666667),
        (np.array([-3.53, -4.08]), 12.039851666666666),
        (np.array([-3.14, -1.54]), 10.370106666666665),
        (np.array([-1.03, 0.39]), 4.978431666666666),
    ]

    func = f3()

    global global_score
    passed = True
    print('Testing f3 __call__ method...')
    for x, f_x in test_cases:
        decision = " "*4 + f"x=np.array([{x[0]}, {x[1]}]) your f(x)={func(x)}"
        if abs(func(x) - f_x) <= 1e-4:
            decision += ' ✓'
        else:
            passed = False
            decision += ' x'
        #print(fb_color.green(decision) if passed else fb_color.red(decision))
        color_print(decision, color='green' if passed else 'red')

    if passed:
        print(f'f3 __call__ method is correct!')
        global_score += 1
    else:
        print(f'f3 __call__ method failed!')
    print(f"Total score: {global_score}", end='\n'*2)

def test_SquaredL2Norm_value():
    """
    f(x) = ||x||^2
    """

    test_cases = [
        (np.array([0, 0]), 0),
        (np.array([-0.83, 2.2]), 5.528900000000001),
        (np.array([-5.0, -1.98]), 28.9204),
        (np.array([-0.83, 2.2, -5.0, -1.98, -3.53]), 46.9102),
        (np.array([-4.08, -3.14, -1.54, -1.03, 0.39]), 30.090600000000002),
        (np.array([3.01, 4.68, -1.87, 1.92, 3.76, 3.95, -4.15, -4.61, -3.3, 3.78]), 131.53889999999998),
        (np.array([-4.02, -0.79, 4.58, 0.33, 1.92, -1.84, 1.87, 3.35, -4.82, 2.5]), 89.14359999999999),
    ]

    func = SquaredL2Norm()

    global global_score
    passed = True
    print('Testing SquaredL2Norm __call__ method...')
    for x, f_x in test_cases:
        decision = " "*4 + f"x=np.array({[val for val in x]}) your f(x)={func(x)}"
        if abs(func(x) - f_x) <= 1e-4:
            decision += ' ✓'
        else:
            passed = False
            decision += ' x'
        #print(fb_color.green(decision) if passed else fb_color.red(decision))
        color_print(decision, color='green' if passed else 'red')

    if passed:
        print(f'SquaredL2Norm __call__ method is correct!')
        global_score += 1
    else:
        print(f'SquaredL2Norm __call__ method failed!')
    print(f"Total score: {global_score}", end='\n'*2)

def test_Himmelblau_value():

    test_cases = [
        (np.array([-0.83, 2.2]), 74.73004321),
        (np.array([-5.0, -1.98]), 209.76033616),
        (np.array([3, 2]), 0),
        (np.array([-2.8051, 3.1313]), 1.6631016200238794e-08),
        (np.array([3.58, -1.84]), 0.0017403199999999868),
        (np.array([-3.53, -4.08]), 44.27003377000001),
        (np.array([-3.14, -1.54]), 67.53258272),
        (np.array([-1.03, 0.39]), 153.24661921999999),
    ]

    func = Himmelblau()

    global global_score
    passed = True
    print('Testing Himmelblau __call__ method...')
    for x, f_x in test_cases:
        decision = " "*4 + f"x=np.array({[val for val in x]}) your f(x)={func(x)}"
        if abs(func(x) - f_x) <= 1e-4:
            decision += ' ✓'
        else:
            passed = False
            decision += ' x'
        #print(fb_color.green(decision) if passed else fb_color.red(decision))
        color_print(decision, color='green' if passed else 'red')

    if passed:
        print(f'Himmelblau __call__ method is correct!')
        global_score += 1
    else:
        print(f'Himmelblau __call__ method failed!')
    print(f"Total score: {global_score}", end='\n'*2)

def test_Rosenbrok_value():

    test_cases = [
        (np.array([-0.83, 2.2]), 231.6912210000001),
        (np.array([-5.0, -1.98]), 72828.04),
        (np.array([-0.83, 2.2, -5.0]), 9915.691221000001),
        (np.array([-1.98, -3.53, -4.08]), 32940.384597000004),
        (np.array([-0.81, 1.85, -2.96, 3.78, -4.73]), 42895.166658),
        (np.array([1.7, -0.83, 0.59, -3.6, -3.02]), 28504.774482),
        (np.array([-0.83, 2.2, -5.0, -1.98, -3.53, -4.08, -3.14, -1.54, -1.03, 0.39, -0.81, 1.85, -2.96]),
         173398.088454),
        (np.array([3.78, -4.73, 1.7, -0.83, 0.59, -3.6, -3.02, 3.01, 4.68, -1.87, 1.92, 3.76, 3.95]),
         180295.46190500003),
    ]

    func = Rosenbrok()

    global global_score
    passed = True
    print('Testing Rosenbrok __call__ method...')
    for x, f_x in test_cases:
        decision = " "*4 + f"x=np.array({[val for val in x]}) your f(x)={func(x)}"
        if abs(func(x) - f_x) <= 1e-4:
            decision += ' ✓'
        else:
            passed = False
            decision += ' x'
        #print(fb_color.green(decision) if passed else fb_color.red(decision))
        color_print(decision, color='green' if passed else 'red')

    if passed:
        print(f'Rosenbrok __call__ method is correct!')
        global_score += 2
    else:
        print(f'Rosenbrok __call__ method failed!')
    print(f"Total score: {global_score}", end='\n'*2)

# Testing function gradients -------------------------------------------------------------------------------------
def test_f1_grad():
    """
    f(x) = x^2
    """

    test_cases = [
        (0, 0),
        (-1, -2),
        (3.43, 2*3.43),
        (-15, 2*(-15)),
        (np.cos(15), 2*np.cos(15))
    ]

    func = f1()

    global global_score
    passed = True
    print('Testing f1 grad method...')
    for x, grad_f_x in test_cases:
        grad_f = func.grad(x)
        decision = " "*4 + f"x={x} your grad f(x)={grad_f}"
        if abs(grad_f - grad_f_x) <= 1e-4:
            decision += ' ✓'
        else:
            passed = False
            decision += ' x'
        #print(fb_color.green(decision) if passed else fb_color.red(decision))
        color_print(decision, color='green' if passed else 'red')

    if passed:
        print(f'f1 grad method is correct!')
        global_score += 1
    else:
        print(f'f1 grad method failed!')
    print(f"Total score: {global_score}", end='\n'*2)

def test_f2_grad():
    """
    f(x) = sin(3*sqrt(x**3)+2) + x**2
    """

    test_cases = [
        (0.05, -0.3491871921575964),
        (1, 3.276479834584518),
        (2.085, 4.406150058472785),
        (3.602, -0.2182589131698629),
        (0.001, -0.057231008254084115),
        (1.512, 4.5338137911959295),
        (0.734, -1.3661450920857936),
    ]

    func = f2()

    global global_score
    passed = True
    print('Testing f2 grad method...')
    for x, grad_f_x in test_cases:
        grad_f = func.grad(x)
        decision = " "*4 + f"x={x} your grad f(x)={grad_f}"
        if abs(grad_f - grad_f_x) <= 1e-4:
            decision += ' ✓'
        else:
            passed = False
            decision += ' x'
        #print(fb_color.green(decision) if passed else fb_color.red(decision))
        color_print(decision, color='green' if passed else 'red')

    if passed:
        print(f'f2 grad method is correct!')
        global_score += 1
    else:
        print(f'f2 grad method failed!')
    print(f"Total score: {global_score}", end='\n'*2)

def test_f3_grad():
    """
    f(x) = (x - 3.3)^2/4 + (y + 1.7)^2/15
    """

    test_cases = [
        (np.array([-0.83, 2.2]), np.array([-2.065, 0.52])),
        (np.array([-5.0, -1.98]), np.array([-4.15, -0.037333333333333336])),
        (np.array([-3.53, -4.08]), np.array([-3.415, -0.3173333333333333])),
        (np.array([-3.14, -1.54]), np.array([-3.2199999999999998, 0.021333333333333322])),
        (np.array([-1.03, 0.39]), np.array([-2.165, 0.2786666666666666])),
        (np.array([3.3, 1.7]), np.array([0.0, 0.4533333333333333])),
        (np.array([3.3, -1.7]), np.array([0.0, 0.0])),
    ]

    func = f3()

    global global_score
    passed = True
    print('Testing f3 grad method...')
    for x, grad_f_x in test_cases:
        grad_f = func.grad(x)
        decision = " "*4 + f"x={x} your grad f(x)={grad_f}"
        if abs(np.linalg.norm(grad_f - grad_f_x)) <= 1e-4:
            decision += ' ✓'
        else:
            passed = False
            decision += ' x'
        #print(fb_color.green(decision) if passed else fb_color.red(decision))
        color_print(decision, color='green' if passed else 'red')

    if passed:
        print(f'f3 grad method is correct!')
        global_score += 1
    else:
        print(f'f3 grad method failed!')
    print(f"Total score: {global_score}", end='\n'*2)

def test_SquaredL2Norm_grad():
    """
    f(x) = ||x||^2
    """

    test_cases = [
        (np.array([-0.83]), np.array([-1.66])),
        (np.array([2.2]), np.array([4.4])),
        (np.array([-0.83, 2.2]), np.array([-1.66, 4.4])),
        (np.array([-5.0, -1.98]), np.array([-10.0, -3.96])),
        (np.array([-0.83, 2.2, -5.0, -1.98, -3.53]), np.array([-1.66, 4.4, -10.0, -3.96, -7.06])),
        (np.array([-4.08, -3.14, -1.54, -1.03, 0.39]), np.array([-8.16, -6.28, -3.08, -2.06, 0.78])),
        (np.array([-0.83, 2.2, -5.0, -1.98, -3.53, -4.08, -3.14, -1.54, -1.03, 0.39]),
         np.array([-1.66, 4.4, -10.0, -3.96, -7.06, -8.16, -6.28, -3.08, -2.06, 0.78])),
        (np.array([-0.81, 1.85, -2.96, 3.78, -4.73, 1.7, -0.83, 0.59, -3.6, -3.02]),
         np.array([-1.62, 3.7, -5.92, 7.56, -9.46, 3.4, -1.66, 1.18, -7.2, -6.04])),
    ]

    func = SquaredL2Norm()

    global global_score
    passed = True
    print('Testing SquaredL2Norm grad method...')
    for x, grad_f_x in test_cases:
        grad_f = func.grad(x)
        decision = " "*4 + f"x={x} your grad f(x)={grad_f}"
        if abs(np.linalg.norm(grad_f - grad_f_x)) <= 1e-4:
            decision += ' ✓'
        else:
            passed = False
            decision += ' x'
        #print(fb_color.green(decision) if passed else fb_color.red(decision))
        color_print(decision, color='green' if passed else 'red')

    if passed:
        print(f'SquaredL2Norm grad method is correct!')
        global_score += 1
    else:
        print(f'SquaredL2Norm grad method failed!')
    print(f"Total score: {global_score}", end='\n'*2)

def test_Himmelblau_grad():

    test_cases = [
        (np.array([3, 2]), np.array([0, 0])),
        (np.array([-0.83, 2.2]), np.array([20.948852000000002, -42.5342])),
        (np.array([-2.8051, 3.1313]), np.array([0.0011583933960094875, -0.0009833748120053679])),
        (np.array([-5.0, -1.98]), np.array([-256.5592, 88.03043199999999])),
        (np.array([-3.53, -4.08]), np.array([49.21449200000002, -105.05784800000002])),
        (np.array([-3.14, -1.54]), np.array([18.129023999999987, 42.492544])),
        (np.array([-1.03, 0.39]), np.array([23.586492, -31.387724])),
    ]

    func = Himmelblau()

    global global_score
    passed = True
    print('Testing Himmelblau grad method...')
    for x, grad_f_x in test_cases:
        grad_f = func.grad(x)
        decision = " "*4 + f"x={x} your grad f(x)={grad_f}"
        if abs(np.linalg.norm(grad_f - grad_f_x)) <= 1e-4:
            decision += ' ✓'
        else:
            passed = False
            decision += ' x'
        #print(fb_color.green(decision) if passed else fb_color.red(decision))
        color_print(decision, color='green' if passed else 'red')

    if passed:
        print(f'Himmelblau grad method is correct!')
        global_score += 1
    else:
        print(f'Himmelblau grad method failed!')
    print(f"Total score: {global_score}", end='\n'*2)

def test_Rosenbrok_grad():

    test_cases = [
        (np.array([-0.83, 2.2]), np.array([498.0252000000001, 302.2200000000001])),
        (np.array([-5.0, -1.98]), np.array([-53972.0, -5396.0])),
        (np.array([1, 1]), np.array([0, 0])),
        (np.array([-0.83, 2.2, -5.0]), np.array([498.0252000000001, 8963.82, -1968.0])),
        (np.array([-1.98, -3.53, -4.08]), np.array([-5906.676799999999, -24854.8908, -3308.1800000000003])),
        (np.array([1, 1, 1]), np.array([0, 0, 0])),
        (np.array([-0.83, 2.2, -5.0, -1.98, -3.53]),
         np.array([498.0252000000001, 8963.82, -55940.0, -11302.6768, -1490.08])),
        (np.array([1, 1, 1, 1, 1]), np.array([0, 0, 0, 0, 0])),
        (np.array([-4.08, -3.14, -1.54, -1.03, 0.39]),
         np.array([-32301.564800000004, -18283.4576, -4380.3856, -960.7908, -134.17999999999998])),
        (np.array([-0.81, 1.85, -2.96, 3.78, -4.73]), np.array([383.2036, 4963.53, -7182.6344, 27765.0608, -3803.68])),
        (np.array([-0.83, 2.2, -5.0, -1.98, -3.53, -4.08, -3.14, -1.54, -1.03, 0.39]), np.array(
            [498.0252000000001, 8963.82, -55940.0, -11302.6768, -24854.8908, -35609.7448, -18283.4576, -4380.3856,
             -960.7908, -134.17999999999998])),
        (np.array([-0.81, 1.85, -2.96, 3.78, -4.73, 1.7, -0.83, 0.59, -3.6, -3.02]), np.array(
            [383.2036, 4963.53, -7182.6344, 27765.0608, -42928.26680000001, -1603.5800000000008, -780.4947999999999,
             911.1515999999999, -23810.02, -3196.0])),
        (np.array([3.01, 4.68, -1.87, 1.92, 3.76, 3.95, -4.15, -4.61, -3.3, 3.78]), np.array(
            [5277.660399999999, 43633.2728, -5939.7412, -370.0648, 15342.390399999998, 29177.330000000005,
             -40202.750000000015, -49651.79240000001, -14304.220000000001, -1422.0])),
    ]

    func = Rosenbrok()

    global global_score
    passed = True
    print('Testing Rosenbrok grad method...')
    for x, grad_f_x in test_cases:
        grad_f = func.grad(x)
        decision = " "*4 + f"x={x} your grad f(x)={grad_f}"
        if abs(np.linalg.norm(grad_f - grad_f_x)) <= 1e-4:
            decision += ' ✓'
        else:
            passed = False
            decision += ' x'
        #print(fb_color.green(decision) if passed else fb_color.red(decision))
        color_print(decision, color='green' if passed else 'red')

    if passed:
        print(f'Rosenbrok grad method is correct!')
        global_score += 3
    else:
        print(f'Rosenbrok grad method failed!')
    print(f"Total score: {global_score}", end='\n'*2)
# Testing function hessians -------------------------------------------------------------------------------------
def test_f1_hess():
    """
    f(x) = x^2
    """

    test_cases = [
        (0, 2),
        (-1, 2),
        (3.43, 2),
        (-15, 2),
        (np.cos(15), 2)
    ]

    func = f1()

    global global_score
    passed = True
    print('Testing f1 hess method...')
    for x, hess_f_x in test_cases:
        hess_f = func.hess(x)
        decision = " "*4 + f"x={x} your hess f(x)={hess_f}"
        if abs(hess_f - hess_f_x) <= 1e-4:
            decision += ' ✓'
        else:
            passed = False
            decision += ' x'
        #print(fb_color.green(decision) if passed else fb_color.red(decision))
        color_print(decision, color='green' if passed else 'red')

    if passed:
        print(f'f1 hess method is correct!')
        global_score += 1
    else:
        print(f'f1 hess method failed!')
    print(f"Total score: {global_score}", end='\n'*2)

def test_f2_hess():
    """
    f(x) = sin(3*sqrt(x**3)+2) + x**2
    """

    test_cases = [
        (2.085, 44.24998807104605),
        (3.602, 37.05335164280684),
        (0.001, -27.633916600399534),
        (1.512, -26.956911422791325),
        (0.734, 10.145810192779237),
    ]

    func = f2()

    global global_score
    passed = True
    print('Testing f2 hess method...')
    for x, hess_f_x in test_cases:
        hess_f = func.hess(x)
        decision = " "*4 + f"x={x} your hess f(x)={hess_f}"
        if abs(hess_f - hess_f_x) <= 1e-4:
            decision += ' ✓'
        else:
            passed = False
            decision += ' x'
        #print(fb_color.green(decision) if passed else fb_color.red(decision))
        color_print(decision, color='green' if passed else 'red')

    if passed:
        print(f'f2 hess method is correct!')
        global_score += 1
    else:
        print(f'f2 hess method failed!')
    print(f"Total score: {global_score}", end='\n'*2)

def test_f3_hess():
    """
    f(x) = (x - 3.3)^2/4 + (y + 1.7)^2/15
    """

    test_cases = [
        (np.array([-0.83, 2.2]), np.array([[0.5, 0.0], [0.0, 0.13333333333333333]])),
        (np.array([-5.0, -1.98]), np.array([[0.5, 0.0], [0.0, 0.13333333333333333]])),
        (np.array([-3.53, -4.08]), np.array([[0.5, 0.0], [0.0, 0.13333333333333333]])),
        (np.array([-3.14, -1.54]), np.array([[0.5, 0.0], [0.0, 0.13333333333333333]])),
        (np.array([-1.03, 0.39]), np.array([[0.5, 0.0], [0.0, 0.13333333333333333]])),
    ]

    func = f3()

    global global_score
    passed = True
    print('Testing f3 hess method...')
    for x, hess_f_x in test_cases:
        hess_f = func.hess(x)
        decision = " "*4 + f"x={x}\nyour hess:\n"
        decision += ' '*4 + f"f(x)={hess_f.tolist()}\n"

        if abs(np.linalg.norm(hess_f - hess_f_x)) <= 1e-4:
            decision += ' ✓'
        else:
            passed = False
            decision += ' x'
        #print(fb_color.green(decision) if passed else fb_color.red(decision))
        color_print(decision, color='green' if passed else 'red')

    if passed:
        print(f'f3 hess method is correct!')
        global_score += 1
    else:
        print(f'f3 hess method failed!')
    print(f"Total score: {global_score}", end='\n'*2)

def test_SquaredL2Norm_hess():
    """
    f(x) = ||x||^2
    """

    test_cases = [
        (np.array([-0.83, 2.2]), 2*np.eye(2)),
        (np.array([-5.0, -1.98]), 2*np.eye(2)),
        (np.array([-0.83, 2.2, -5.0]), 2*np.eye(3)),
        (np.array([-1.98, -3.53, -4.08]), 2*np.eye(3)),
        (np.array([-0.83, 2.2, -5.0, -1.98, -3.53]), 2*np.eye(5)),
        (np.array([-4.08, -3.14, -1.54, -1.03, 0.39]), 2*np.eye(5)),
        (np.array([-0.83, 2.2, -5.0, -1.98, -3.53, -4.08, -3.14]), 2*np.eye(7))
    ]

    func = SquaredL2Norm()

    global global_score
    passed = True
    print('Testing SquaredL2Norm hess method...')
    for x, hess_f_x in test_cases:
        hess_f = func.hess(x)
        decision = " "*4 + f"x={x}\nyour hess:\n"
        decision += ' '*4 + f"f(x)={hess_f.tolist()}\n"

        if abs(np.linalg.norm(hess_f - hess_f_x)) <= 1e-4:
            decision += ' ✓'
        else:
            passed = False
            decision += ' x'
        #print(fb_color.green(decision) if passed else fb_color.red(decision))
        color_print(decision, color='green' if passed else 'red')

    if passed:
        print(f'SquaredL2Norm hess method is correct!')
        global_score += 1
    else:
        print(f'SquaredL2Norm hess method failed!')
    print(f"Total score: {global_score}", end='\n'*2)

def test_Himmelblau_hess():

    test_cases = [
        (np.array([-0.83, 2.2]), np.array([[-24.933200000000003, 5.48], [5.48, 28.76000000000001]])),
        (np.array([-5.0, -1.98]), np.array([[250.07999999999998, -27.92], [-27.92, 1.0448000000000022]])),
        (np.array([-3.53, -4.08]),
         np.array([[91.21079999999998, -30.439999999999998], [-30.439999999999998, 159.6368]])),
        (np.array([3, 2]), np.array([[74, 20], [20, 34]])),
        (np.array([-2.8051, 3.1313]), np.array([[64.94823212, 1.3048000000000002], [1.3048000000000002, 80.44007628]])),
        (np.array([-3.14, -1.54]), np.array([[70.15520000000001, -18.72], [-18.72, -10.1008]])),
        (np.array([-1.03, 0.39]), np.array([[-27.709199999999996, -2.56], [-2.56, -28.294800000000002]])),
    ]

    func = Himmelblau()

    global global_score
    passed = True
    print('Testing Himmelblau hess method...')
    for x, hess_f_x in test_cases:
        hess_f = func.hess(x)
        decision = " "*4 + f"x={x}\nyour hess:\n"
        decision += ' '*4 + f"f(x)={hess_f.tolist()}\n"

        if abs(np.linalg.norm(hess_f - hess_f_x)) <= 1e-4:
            decision += ' ✓'
        else:
            passed = False
            decision += ' x'
        #print(fb_color.green(decision) if passed else fb_color.red(decision))
        color_print(decision, color='green' if passed else 'red')

    if passed:
        print(f'Himmelblau hess method is correct!')
        global_score += 1
    else:
        print(f'Himmelblau hess method failed!')
    print(f"Total score: {global_score}", end='\n'*2)

def test_Rosenbrok_hess():

    test_cases = [
        (np.array([-0.83, 2.2]), np.array([[-51.32, 332.], [332., 200.]])),

        (np.array([-5.0, -1.98]), np.array([[30794., 2000.], [2000., 200.]])),

        (np.array([-0.83, 2.2, -5.0]), np.array([[-51.320, 332.0, 0.], [332., 8010., -880.], [0., -880., 200.]])),

        (np.array([-1.98, -3.53, -4.08]),
         np.array([[6118.48, 792.0, 0.0], [792.0, 16787.08, 1412.0], [0.0, 1412.0, 200.0]])),

        (np.array([-0.83, 2.2, -5.0, -1.98, -3.53]), np.array(
            [[-51.32, 332.0, 0.0, 0.0, 0.0], [332.0, 8010.0, -880.0, 0.0, 0.0],
             [0.0, -880.0, 30994.0, 2000.0, 0.0], [0.0, 0.0, 2000.0, 6318.48, 792.0],
             [0.0, 0.0, 0.0, 792.0, 200.0]]
        )),

        (np.array([-4.08, -3.14, -1.54, -1.03, 0.39]), np.array(
            [[21233.68, 1632.0, 0.0, 0.0, 0.0], [1632.0, 12649.52, 1256.0, 0.0, 0.0],
             [0.0, 1256.0, 3459.92, 616.0, 0.0], [0.0, 0.0, 616.0, 1319.079, 412.0],
             [0.0, 0.0, 0.0, 412.0, 200.0]])),

        (np.array([-0.81, 1.85, -2.96, 3.78, -4.73]), np.array(
            [[49.32, 324.0, 0.0, 0.0, 0.0], [324.0, 5493.0, -740.0, 0.0, 0.0],
             [0.0, -740.0, 9203.92, 1184.0, 0.0], [0.0, 0.0, 1184.0, 19240.079, -1512.0],
             [0.0, 0.0, 0.0, -1512.0, 200.0]])),

        (np.array([-0.83, 2.2, -5.0, -1.98, -3.53, -4.08, -3.14]), np.array(
            [[-51.32, 332.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [332.0, 8010.0, -880.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, -880.0, 30994.0, 2000.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 2000.0, 6318.48, 792.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 792.0, 16787.08, 1412.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 1412.0, 21433.68, 1632.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 1632.0, 200.0]])),
    ]

    func = Rosenbrok()

    global global_score
    passed = True
    print('Testing Rosenbrok hess method...')
    for x, hess_f_x in test_cases:
        hess_f = func.hess(x)
        decision = " "*4 + f"x={x}\nyour hess:\n"
        decision += ' '*4 + f"f(x)={hess_f.tolist()}\n"

        if abs(np.linalg.norm(hess_f - hess_f_x)) <= 1e-2:
            decision += ' ✓'
        else:
            passed = False
            decision += ' x'
        #print(fb_color.green(decision) if passed else fb_color.red(decision))
        color_print(decision, color='green' if passed else 'red')

    if passed:
        print(f'Rosenbrok hess method is correct!')
        global_score += 3
    else:
        print(f'Rosenbrok hess method failed!')
    print(f"Total score: {global_score}", end='\n'*2)


# Testing minimize function -------------------------------------------------------------------------------------
def test_minimize_function_gd():
    try:

        test_cases = [

            {"x_init": -1.25, "max_iter": 1, 'func': f1(), 'method': 'gd', "stopping_criteria": "gradient",
             "tolerance": 1e-7, 'learning_rate': lambda x: 0.1, "correct_result": -1.0}, #1

            {"x_init": -1.25, "max_iter": 4, 'func': f1(), 'method': 'gd', "stopping_criteria": "gradient",
             "tolerance": 1e-7, 'learning_rate': lambda x: 0.1, "correct_result": -0.512}, #2

            {"x_init": -1.25, "max_iter": 10000, 'func': f1(), 'method': 'gd', "stopping_criteria": "gradient",
             "tolerance": 1e-7, 'learning_rate': lambda x: 0.1, "correct_result": 0.0}, #3

            {"x_init": 1.87, "max_iter": 1, 'func': f2(), 'method': 'gd', "stopping_criteria": "gradient",
             "tolerance": 1e-7, 'learning_rate': lambda x: 0.1, "correct_result": 2.092722398320625}, #4

            {"x_init": 1.87, "max_iter": 15, 'func': f2(), 'method': 'gd', "stopping_criteria": "gradient",
             "tolerance": 1e-7, 'learning_rate': lambda x: 0.1, "correct_result": 0.8355066903851668}, #5

            {"x_init": 1.87, "max_iter": 10000, 'func': f2(), 'method': 'gd', "stopping_criteria": "gradient",
             "tolerance": 1e-7, 'learning_rate': lambda x: 0.1/((x+1)**0.1), "correct_result": 0.8361761684075012}, #6

            {"x_init": np.array([-1.25, 4.51]), "max_iter": 1, 'func': f3(), 'method': 'gd',
             "stopping_criteria": "gradient", "tolerance": 1e-7, 'learning_rate': lambda x: 0.1,
             "correct_result": np.array([-1.0225, 4.4272])}, #7

            {"x_init": np.array([-1.25, 4.51]), "max_iter": 1, 'func': f3(), 'method': 'gd',
             "stopping_criteria": "gradient", "tolerance": 1e-7, 'learning_rate': lambda x: 1,
             "correct_result": np.array([1.025, 3.682])}, #8

            {"x_init": np.array([-1.25, 4.51]), "max_iter": 1, 'func': SquaredL2Norm(), 'method': 'gd',
             "stopping_criteria": "gradient", "tolerance": 1e-7, 'learning_rate': lambda x: 0.1,
             "correct_result": np.array([-1., 3.608])}, #9

            {"x_init": np.array([-1.25, 4.51]), "max_iter": 1, 'func': SquaredL2Norm(), 'method': 'gd',
             "stopping_criteria": "gradient", "tolerance": 1e-7, 'learning_rate': lambda x: 1,
             "correct_result": np.array([1.25, -4.51])}, #10

            {"x_init": np.array([-3.85, 1.09, -3.67, -2.59, -1.73]), "max_iter": 1, 'func': SquaredL2Norm(),
             'method': 'gd', "stopping_criteria": "gradient", "tolerance": 1e-7, 'learning_rate': lambda x: 0.1,
             "correct_result": np.array([-3.08, 0.872, -2.936, -2.072, -1.384])}, #11

            {"x_init": np.array([-3.85, 1.09, -3.67, -2.59, -1.73]), "max_iter": 10000, 'func': SquaredL2Norm(),
             'method': 'gd', "stopping_criteria": "gradient", "tolerance": 1e-7, 'learning_rate': lambda x: 0.1,
             "correct_result": np.array([0, 0, 0, 0, 0])}, #12

            {"x_init": np.array([-3.85, 1.09]), "max_iter": 1, 'func': Himmelblau(),
             'method': 'gd', "stopping_criteria": "gradient", "tolerance": 1e-7, 'learning_rate': lambda x: 0.01,
             "correct_result": np.array([-2.900237, 1.41300884])}, #13

            {"x_init": np.array([-3.85, 1.09]), "max_iter": 10000, 'func': Himmelblau(),
             'method': 'gd', "stopping_criteria": "gradient", "tolerance": 1e-7, 'learning_rate': lambda x: 0.01,
             "correct_result": np.array([-2.80511807, 3.13131252])}, #14

        ]

        global global_score
        passed = True
        print("Testing minimize function (GD)...")
        for i, test_case_dict in enumerate(test_cases):
            minimize_result, *_ = minimize(
                func=test_case_dict['func'],
                x_init=test_case_dict['x_init'],
                learning_rate=test_case_dict['learning_rate'],
                method=test_case_dict['method'],
                max_iter=test_case_dict['max_iter'],
                stopping_criteria=test_case_dict['stopping_criteria'],
                tolerance=test_case_dict["tolerance"],
            )
            correct_result = test_case_dict['correct_result']
            decision = f'test case {i+1}:'

            test_condition = abs(minimize_result - correct_result) if \
                (isinstance(minimize_result, float) or isinstance(minimize_result, int)) else \
                np.linalg.norm(minimize_result - correct_result)
            if test_condition < 1e-3:
                decision += ' ✓'
            else:
                passed = False
                decision += ' x'
            #print(fb_color.green(decision) if passed else fb_color.red(decision))
            color_print(decision, color='green' if passed else 'red')

        if passed:
            global_score += 5

        print(f"Total score: {global_score}", end='\n' * 2)
    except:
        #print(fb_color.red('Failed test_minimize_function_gd'))
        color_print('Failed test_minimize_function_gd', color='red')
        #print(fb_color.red("Total score:", global_score))
        color_print(f"Total score: {global_score}", color='red')

def test_minimize_function_newtone():
    try:

        test_cases = [

            {"x_init": -3.85, "max_iter": 1, 'func': f1(), 'method': 'newtone', "stopping_criteria": "function",
             "tolerance": 1e-7, 'learning_rate': lambda x: 0.1, "correct_result": -3.465}, #1

            {"x_init": 4, "max_iter": 4, 'func': f1(), 'method': 'newtone', "stopping_criteria": "gradient",
             "tolerance": 1e-7, 'learning_rate': lambda x: 0.01, "correct_result": 3.8423840400000002}, #2

            {"x_init": -1.25, "max_iter": 1, 'func': f1(), 'method': 'newtone', "stopping_criteria": "gradient",
             "tolerance": 1e-7, 'learning_rate': lambda x: 1, "correct_result": 0.0}, #3

            {"x_init": 4.5, "max_iter": 1, 'func': f2(), 'method': 'newtone', "stopping_criteria": "gradient",
             "tolerance": 1e-7, 'learning_rate': lambda x: 1, "correct_result": 4.26319881725494}, #4

            {"x_init": 1.87, "max_iter": 15, 'func': f2(), 'method': 'newtone', "stopping_criteria": "gradient",
             "tolerance": 1e-7, 'learning_rate': lambda x: 0.1, "correct_result": 1.9603193926468359}, #5

            {"x_init": 1.87, "max_iter": 2, 'func': f2(), 'method': 'newtone', "stopping_criteria": "gradient",
             "tolerance": 1e-7, 'learning_rate': lambda x: 0.1 / ((x + 1) ** 0.1), "correct_result": 1.9051245616689205}, #6

            {"x_init": np.array([-1.25, 4.51]), "max_iter": 1, 'func': f3(), 'method': 'newtone',
             "stopping_criteria": "gradient", "tolerance": 1e-7, 'learning_rate': lambda x: 0.1,
             "correct_result": np.array([-0.795, 3.889])}, #7

            {"x_init": np.array([-1.25, 4.51]), "max_iter": 1, 'func': f3(), 'method': 'newtone',
             "stopping_criteria": "gradient", "tolerance": 1e-7, 'learning_rate': lambda x: 1,
             "correct_result": np.array([3.3, -1.7])}, #8

            {"x_init": np.array([-1.25, 4.51]), "max_iter": 1, 'func': SquaredL2Norm(), 'method': 'newtone',
             "stopping_criteria": "function", "tolerance": 1e-7, 'learning_rate': lambda x: 0.1,
             "correct_result": np.array([-1.125, 4.059])}, #9

            {"x_init": np.array([-1.25, 4.51]), "max_iter": 1, 'func': SquaredL2Norm(), 'method': 'newtone',
             "stopping_criteria": "function", "tolerance": 1e-7, 'learning_rate': lambda x: 1,
             "correct_result": np.array([0, 0])}, #10

            {"x_init": np.array([-3.85, 1.09, -3.67, -2.59, -1.73]), "max_iter": 1, 'func': SquaredL2Norm(),
             'method': 'newtone', "stopping_criteria": "function", "tolerance": 1e-7, 'learning_rate': lambda x: 0.1,
             "correct_result": np.array([-3.465, 0.981, -3.303, -2.331, -1.557])}, #11

            {"x_init": np.array([-3.85, 1.09, -3.67, -2.59, -1.73]), "max_iter": 10000, 'func': SquaredL2Norm(),
             'method': 'newtone', "stopping_criteria": "function", "tolerance": 1e-7, 'learning_rate': lambda x: 0.1,
             "correct_result": np.array([-0.00036208, 0.00010251, -0.00034515, -0.00024358, -0.0001627])}, #12

            {"x_init": np.array([-3.85, 1.09]), "max_iter": 1, 'func': Himmelblau(),
             'method': 'newtone', "stopping_criteria": "function", "tolerance": 1e-7, 'learning_rate': lambda x: 0.01,
             "correct_result": np.array([-3.84434507, 1.07579958])}, #13

            {"x_init": np.array([-3.85, 1.09]), "max_iter": 10000, 'func': Himmelblau(),
             'method': 'newtone', "stopping_criteria": "function", "tolerance": 1e-7, 'learning_rate': lambda x: 0.01,
             "correct_result": np.array([-3.07345161, -0.08089318])}, #14

            # {"x_init": np.array([-3.85, 1.09, 11, 3, -1]), "max_iter": 10000, 'func': Rosenbrok(),
            #  'method': 'newtone', "stopping_criteria": "gradient", "tolerance": 1e-7, 'learning_rate': lambda x: 0.01,
            #  "correct_result": np.array([-1.17032208, 1.3789261, 1.90615193, 3.6368349, 13.22656809])},  # 15

        ]

        global global_score
        passed = True
        print("Testing minimize function (Newtone)...")
        for i, test_case_dict in enumerate(test_cases):
            minimize_result, *_ = minimize(
                func=test_case_dict['func'],
                x_init=test_case_dict['x_init'],
                learning_rate=test_case_dict['learning_rate'],
                method=test_case_dict['method'],
                max_iter=test_case_dict['max_iter'],
                stopping_criteria=test_case_dict['stopping_criteria'],
                tolerance=test_case_dict["tolerance"],
            )
            correct_result = test_case_dict['correct_result']
            decision = f'test case {i+1}:'

            test_condition = abs(minimize_result - correct_result) if \
                (isinstance(minimize_result, float) or isinstance(minimize_result, int)) else \
                np.linalg.norm(minimize_result - correct_result)
            if test_condition < 1e-3:
                decision += ' ✓'
            else:
                passed = False
                decision += ' x'
            #print(fb_color.green(decision) if passed else fb_color.red(decision))
            color_print(decision, color='green' if passed else 'red')


        if passed:
            global_score += 5

        print(f"Total score: {global_score}", end='\n' * 2)
    except:
        #print(fb_color.red('Failed test_minimize_function_newtone'))
        #print(fb_color.red("Total score:", global_score))
        color_print('Failed test_minimize_function_newtone', color='red')
        color_print(f"Total score: {global_score}", color='red')


if __name__ == '__main__':
    # functions computations
    try:
        test_f1_value()
    except:
        print("Total score:", global_score)
    try:
        test_f2_value()
    except:
        print("Total score:", global_score)
    try:
        test_f3_value()
    except:
        print("Total score:", global_score)
    try:
        test_SquaredL2Norm_value()
    except:
        print("Total score:", global_score)
    try:
        test_Himmelblau_value()
    except:
        print("Total score:", global_score)
    try:
        test_Rosenbrok_value()
    except:
        print("Total score:", global_score)
    # functions gradients
    try:
        test_f1_grad()
    except:
        print("Total score:", global_score)
    try:
        test_f2_grad()
    except:
        print("Total score:", global_score)
    try:
        test_f3_grad()
    except:
        print("Total score:", global_score)
    try:
        test_SquaredL2Norm_grad()
    except:
        print("Total score:", global_score)
    try:
        test_Himmelblau_grad()
    except:
        print("Total score:", global_score)
    try:
        test_Rosenbrok_grad()
    except:
        print("Total score:", global_score)
    # functions hessians
    try:
        test_f1_hess()
    except:
        print("Total score:", global_score)
    try:
        test_f2_hess()
    except:
        print("Total score:", global_score)
    try:
        test_f3_hess()
    except:
        print("Total score:", global_score)
    try:
        test_SquaredL2Norm_hess()
    except:
        print("Total score:", global_score)
    try:
        test_Himmelblau_hess()
    except:
        print("Total score:", global_score)
    try:
        test_Rosenbrok_hess()
    except:
        print("Total score:", global_score)
    # test minimize function 1 step
    test_minimize_function_gd()
    test_minimize_function_newtone()
    # Выводим Общий балл и записываем его в файл:
    try:
        print(fb_color.bold(fb_color.magenta(f"Суммарный балл: {global_score}")))
    except:
        print(f"Суммарный балл: {global_score}")
    try:
        with open(p_join('/prac_folder/main_task_score.txt'), mode='w', encoding='utf-8') as f:
            f.writelines(str(global_score) + '\n')
    except:
        print("Не получилось записать main_task_score.txt")