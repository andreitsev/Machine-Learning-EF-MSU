import numpy as np
from typing import Callable, Tuple, Union, List

class f1:
    def __call__(self, x: float):
        """
        Args:
            x: float
        Returns:
            float
        """
        pass

    def grad(self, x: float):
        """
        Args:
            x: float
        Returns:
            float
        """
        pass

    def hess(self, x: float):
        """
        Args:
            x: float
        Returns:
            float
        """
        pass


class f2:
    def __call__(self, x: float):
        """
        Args:
            x: float
        Returns:
            float
        """
        pass

    def grad(self, x: float):
        """
        Args:
            x: float
        Returns:
            float
        """
        pass

    def hess(self, x: float):
        """
        Args:
            x: float
        Returns:
            float
        """
        pass


class f3:
    def __call__(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (2,)
        Returns:
            float
        """
        pass

    def grad(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (2,)
        Returns:
            np.ndarray of shape (2,)
        """
        pass

    def hess(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (2,)
        Returns:
            np.ndarray of shape (2, 2)
        """
        pass


class SquaredL2Norm:
    def __call__(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (n,)
        Returns:
            float
        """
        pass

    def grad(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (n,)
        Returns:
            np.ndarray of shape (n,)
        """
        pass

    def hess(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (n,)
        Returns:
            np.ndarray of shape (n, n)
        """
        pass


class Himmelblau:
    def __call__(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (2,)
        Returns:
            float
        """
        pass

    def grad(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (2,)
        Returns:
            numpy array of shape (2,)
        """
        pass

    def hess(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (2,)
        Returns:
            numpy array of shape (2, 2)
        """
        pass


class Rosenbrok:
    def __call__(self, x: np.ndarray):

        """
        Args:
            x: numpy array of shape (n,) (n >= 2)
        Returns:
            float
        """

        assert x.shape[0] >= 2, "x.shape[0] должен быть >= 2"

        pass

    def grad(self, x: np.ndarray):

        """
        Args:
            x: numpy array of shape (n,) (n >= 2)
        Returns:
            numpy array of shape (n,)
        """

        len_x = x.shape[0]

        assert len_x >= 2, "x.shape[0] должен быть >= 2"

        pass

    def hess(self, x: np.ndarray):

        """
        Args:
            x: numpy array of shape (n,) (n >= 2)
        Returns:
            numpy array of shape (n, n)
        """

        len_x = x.shape[0]

        assert len_x >= 2, "x.shape[0] должен быть >= 2"

        pass


def minimize(
        func: Callable,
        x_init: np.ndarray,
        learning_rate: Callable = lambda x: 0.1,
        method: str = 'gd',
        max_iter: int = 10_000,
        stopping_criteria: str = 'function',
        tolerance: float = 1e-2,
) -> Tuple:
    """
    Args:
        func: функция, у которой необходимо найти минимум (объект класса, который только что написали)
            (у него должны быть методы: __call__, grad, hess)
        x_init: начальная точка
        learning_rate: коэффициент перед направлением спуска
        method:
            "gd" - Градиентный спуск
            "newtone" - Метод Ньютона
        max_iter: максимально возможное число итераций для алгоритма
        stopping_criteria: когда останавливать алгоритм
            'points' - остановка по норме разности точек на соседних итерациях
            'function' - остановка по норме разности значений функции на соседних итерациях
            'gradient' - остановка по норме градиента функции
        tolerance: c какой точностью искать решение (участвует в критерии остановки)
    Returns:
        x_opt: найденная точка локального минимума
        points_history_list: (list) список с историей точек
        functions_history_list: (list) список с историей значений функции
        grad_history_list: (list) список с исторей значений градиентов функции
    """

    assert max_iter > 0, 'max_iter должен быть > 0'
    assert method in ['gd', 'newtone'], 'method can be "gd" or "newtone"!'
    assert stopping_criteria in ['points', 'function', 'gradient'], \
        'stopping_criteria can be "points", "function" or "gradient"!'

    pass


