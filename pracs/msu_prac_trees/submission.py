import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from collections import Counter
from category_encoders import TargetEncoder

from typing import List, Union, Tuple


def compute_criterion(target_vector: np.array, feature_vector: np.array, threshold: float, criterion: str = 'gini') -> float:
    """
    Вычисляет критерий расщепления листа на два
        Q = H(R) - |R_l|/|R| * H(R_l) - |R_r|/|R| * H(R_r)

    Предикат для расщепления: [feature_vector < threshold]

    Если в feature_vector только 1 уникальное значение, функция должна вернуть 0

    Args:
        target_vector: вектор таргетов (многоклассовый)
        feature_vector: вектор с конкретной фичёй объектов (вещественный)
        threshold: погор для разбиение на левое и правое поддеревья
        criterion: какой критерий считать ("gini" или "entropy")
    Returns:
        Q: критерий расщепления (максимизируя который мы выбираем оптимальное разбиение листа)
    """

    assert criterion in ['gini', 'entropy'], "Критерий может быть только 'gini' или 'entropy'!"

    pass
    # your code here:


    return Q


def find_best_split(feature_vector: np.ndarray, target_vector: np.ndarray, criterion: str = 'gini') -> Tuple:
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

    unq_vals = np.sort(np.unique(feature_vector))

    if len(unq_vals) == 1:
        return None, None, None, 0

    pass
    # your code here:

    return


class DecisionTree(BaseEstimator):

    def __init__(
            self,
            feature_types: list,
            criterion: str = 'gini',
            max_depth: int = None,
            min_samples_split: int = None,
    ):
        """
        Args:
            feature_types: список типов фичей (может состоять из 'real' и "categorical")
            criterion: может быть 'gini' или "entropy"
            max_depth: максимальная глубина дерева
            min_samples_split: минимальное число объектов в листе, чтобы можно было расщиплять этот лист
        """

        self._feature_types = feature_types
        self._tree = {}
        # Сюда будут сохраняться обученные таргет энкодеры категориальных фичей
        self.target_encodings = {} # Dict[int<номер категориальной фичи>, category_encoders.target_encoder.TargetEncoder]
        self._criterion = criterion
        self.max_depth = max_depth
        self._min_samples_split = min_samples_split

    def _fit_node(self, sub_X: np.ndarray, sub_y: np.ndarray, node: dict):
        """
        Ищет оптимальное расщепление для листа, состоящего из объектов sub_X и таргетов sub_y. Если для данного листа
        выполненые критерии останова - то завершает работу и обозначает тип листа терминальным (type="terminal")
        Args:
            sub_X: array размера (n, len(self._feature_types)), матрица объект-признак для объектов, попавших в текущих
                лист
            sub_y: array размера (n,) - вектор таргетов для объектов, попавших в текущих лист
            node: словарь, содержащий дерево, обученное к текущему моменту
        Returns:
                None

        ***
        В случае если фича типа "categorical" - нужно применить к ней таргет энкодинг и записать обученный энкодинг в
            self.target_encodings
        ***


        По сути этот метод нужен для рекурсивного вызова в методе self.fit(). Его цель - заполнение словаря node
            (он же self._tree)

        в node (self._tree) в результате обучения должны быть следующие ключи:
            "type" - может быть "terminal" или "nonterminal" (тип текущей вершины: лист ("terminal")
                или внутренняя вершина ("nonterminal"))

                Для листьев (вершин типа "terminal") должны быть следующие ключи:
                    "classes_distribution": список или np.ndarray с распределением классов в данном листе
                        (число объектов каждого класса)

                Для внутренних вершин (типа "nonterminal") должны быть следующие ключи:
                    "feature_type" - (str) тип переменной, по которой эта вершина дальше разделяется ("real" или "categorical")
                    "feature_number" - (int) номер переменной (начиная с нуля), по которой проиходит дальнейшее разделение
                        этой вершины
                    "threshold" - (float) порог рабиения

                    (Иными словами, дальнейшее разбиение этой вершины происходит по формуле:
                        [sub_X[:, feature_number] < threshold])

        Примеры обученных деревьев (self._tree):

            {
                'type': 'nonterminal',
                 'feature_type': 'real',
                 'feature_number': 1,
                 'threshold': 0.535,
                 'left_child': {
                          'type': 'nonterminal',
                          'feature_type': 'real',
                          'feature_number': 0,
                          'threshold': -0.408,
                          'left_child': {'type': 'terminal', 'classes_distribution': [84, 5]},
                          'right_child': {'type': 'terminal', 'classes_distribution': [99, 466]}
                      },
                 'right_child': {
                             'type': 'nonterminal',
                              'feature_type': 'categorical',
                              'feature_number': 3,
                              'threshold': 1.443,
                              'left_child': {'type': 'terminal', 'classes_distribution': [315, 13]},
                              'right_child': {'type': 'terminal', 'classes_distribution': [2, 16]}
                    }
            }

            Обратите внимание, что порядок в classes_distribution должен совпадать с порядком в таргете
                (то есть на нулевом месте - число объектов нулевого класса, на первом - первого и тд.)

        """
        # your code here:
        pass

    def _predict_proba_object(self, x: np.array, node: dict) -> Union[List, np.ndarray]:
        """
        Должен либо вернуть распределение классов для объекта х (Это будет нормированный classes_distribution
        из терминального листа), либо рекурсивно просеить его в левое или правое поддерево.
        Args:
            x: объект размера (len(self._feature_types),)
            node: обученное дерево, которое нужно применить для предсказания
        """
        # your code here:
        pass

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: матрица объект-признак размеров (n, len(self._feature_types))
            y: вектор таргетов (состоящий из int) размера (n,)
        """
        assert len(set(y)) > 1, 'Таргет должен содержать более одного класса!'

        # prepare category encoding (your code here):
        
        self._fit_node(sub_X=X, sub_y=y, node=self._tree)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Применяет self._predict_proba_node для каждой строки из X
        Args:
            X: множество объектов, для которых сделать предсказание, матрица размеров (m, len(self._feature_types))
        Returns:
            np.ndarray размера (len(X), len(set(y)) (где y - вектор таргетов, участвовавший в обучении (self.fit))
        """
        assert self._tree != {}, "Cначала обучите модель!"
        predicted = []
        for x in X:
            predicted.append(self._predict_proba_object(x, self._tree))
        return np.array(predicted)


    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X=X), axis=1).ravel()






















