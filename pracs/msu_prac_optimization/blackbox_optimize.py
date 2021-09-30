import numpy as np
from typing import Tuple, Union

def blackbox_optimize(
        args_history: np.ndarray,
        func_vals_history: np.ndarray
) -> Union[np.ndarray, str]:

    """
    Функция, которая по истории проверенных точек и значений blackbox функции в них возращает точку, которую следует
    проверить следующей или же строку "stop". Учтите случай, что должна выдавать функция, когда истории нет
    (args_history и func_vals_history это пустые arrays)

    Args:
        args_history: история аргументов (args_history.shape = (n, 10))
        func_vals_history: история значений функции в соответствующих аргументах
    Returns:
        Следующая точка (np.ndarray размера 10)
    """

    # Пример такой функции:
    if len(args_history) == 0:
        return np.array([1]*10)
    else:
        return args_history[-1] + np.ones_like(args_history[-1])
