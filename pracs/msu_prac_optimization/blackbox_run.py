import os
from os.path import join as pjoin
import numpy as np
from blackboxfunction import black_box_function
from blackbox_optimize import blackbox_optimize
from fabulous import color as fb_color
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--regime", help="Режим оптимизации: manual или auto", type=str, default='auto')
args = parser.parse_args()
regime = args.regime
print('regime:', regime)

def show_best_trial(trials_history: np.ndarray, func_vals_history: np.ndarray) -> float:

    """
    Печатает результат лучшей попытки и возвращает лучшее значение blackbox function
    """

    best_iteration = np.argsort(func_vals_history)[0]
    print(fb_color.green(f'\nBest trial on iteration: {best_iteration + 1}'))
    print(fb_color.green('Best trial x:'))
    print(trials_history[best_iteration].tolist())
    print(fb_color.green("Best blackbox function value:"))
    print(func_vals_history[best_iteration])
    return func_vals_history[best_iteration]


max_iter = 1000
trials_history = []
func_vals_history = []
trial = None
for iteration in tqdm(range(1, max_iter+1)):
    if regime == 'manual':
        print('-' * 50)
        print(f'Iteration: {iteration}')
        inpt = input().strip()
        if inpt == 'stop':
            print('stop')
            best_bbfunc_value = show_best_trial(trials_history=np.array(trials_history),
                                                func_vals_history=np.array(func_vals_history))
            break
        trial = np.array([float(val.strip()) for val in inpt.split(',')])
    else:
        trial = blackbox_optimize(
            args_history=np.array(trials_history),
            func_vals_history=np.array(func_vals_history)
        )
        if isinstance(trial, str):
            if trial == 'stop':
                print('stop')
                best_bbfunc_value = show_best_trial(trials_history=np.array(trials_history),
                                                    func_vals_history=np.array(func_vals_history))
                break
    trials_history.append(trial)
    func_vals_history.append(black_box_function(trial))
    if regime == 'manual':
        print('Next trial:')
        print(trial.tolist())
        print(f'black_box_function value:')
        print(func_vals_history[-1])
        print()

best_bbfunc_value = show_best_trial(trials_history=np.array(trials_history),
                                    func_vals_history=np.array(func_vals_history))

with open(pjoin('/prac_folder/blackbox_function_score.txt'), mode='w', encoding='utf-8') as f:
    f.writelines(str(best_bbfunc_value) + '\n' if regime == 'auto' else 'В ручном режиме оценка не ставится!\n')
