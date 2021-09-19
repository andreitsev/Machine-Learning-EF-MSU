import numpy as np
from collections import Counter
from typing import Callable, Tuple, Union, List


def compute_criterion(target_vector: np.ndarray, feature_vector: np.ndarray, threshold: float, criterion: str = 'gini') -> float:
    """
    Вычисляет критерий расщепления листа на два
        Q = H(R) - |R_l|/|R| * H(R_l) - |R_r|/|R| * H(R_r)

    Если в feature_vector только 1 уникальное значение, функция должна вернуть 0

    Args:
        target_vector: вектор таргетов (бинарный)
        feature_vector: вектор с конкретной фичёй объектов (вещественный)
        threshold: погор для разбиение на левое и правое поддеревья
        criterion: какой критерий считать ("gini" или "entropy")
    Returns:
        Q: критерий расщепления (максимизируя который мы выбираем оптимальное разбиение листа)
    """

    assert criterion in {'gini', 'entropy'}, "Критерий может быть только 'gini' или 'entropy'!"

    pass


def find_best_split(feature_vector: np.ndarray, target_vector: np.ndarray, criterion: str = 'gini'):
    """
    Функция, находящая оптимальное рабиение с точки зрения критерия gini или entropy

    Args:
        feature_vector: вещественнозначный вектор значений признака
        target_vector: вектор классов объектов (многоклассовый),  len(feature_vector) == len(target_vector)
    Returns:
        thresholds: (np.ndarray) отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
                     разделить на две различные подвыборки, или поддерева
        criterion_vals: (np.ndarray) вектор со значениями критерия Джини/энтропийного критерия для каждого из порогов
                в thresholds. len(criterion_vals) == len(thresholds)
        threshold_best: (float) оптимальный порог
        criterion_best: (float) оптимальное значение критерия

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    """

    pass