import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from collections import Counter
from category_encoders import TargetEncoder

from typing import List, Union, Tuple


def compute_criterion(target_vector: np.array, feature_vector: int, threshold: float, criterion: str = 'gini') -> float:
    """
    Args:
        target_vector: вектор таргетов (бинарный)
        feature_vector: вектор с конкретной фичёй объектов (вещественный)
        threshold: погор для разбиение на левое и правое поддеревья
        criterion: какой критерий считать ("gini" или "entropy")
    Returns:
        Q: критерий расщепления (максимизируя который мы выбираем оптимальное разбиение листа)
    """

    assert criterion in ['gini', 'entropy'], "Критерий может быть только 'gini' или 'entropy'!"

    root_y = Counter(target_vector)
    N_root = len(target_vector)
    root_probs_dict = dict([(key, val / N_root) for key, val in root_y.items()])

    left_y = target_vector[feature_vector < threshold]
    N_left = len(left_y)
    left_probs_dict = dict([(key, val / N_left) for key, val in Counter(left_y).items()])

    right_y = target_vector[feature_vector >= threshold]
    N_right = len(right_y)
    right_probs_dict = dict([(key, val / N_right) for key, val in Counter(right_y).items()])

    if criterion == 'gini':
        H_root = sum([prob * (1 - prob) for prob in root_probs_dict.values()])
        H_left = sum([prob * (1 - prob) for prob in left_probs_dict.values()])
        H_right = sum([prob * (1 - prob) for prob in right_probs_dict.values()])
    elif criterion == 'entropy':
        H_root = sum([-prob * np.log2(prob) for prob in root_probs_dict.values() if prob > 0])
        H_left = sum([-prob * np.log2(prob) for prob in left_probs_dict.values() if prob > 0])
        H_right = sum([-prob * np.log2(prob) for prob in right_probs_dict.values() if prob > 0])

    Q = H_root - (N_left / N_root) * H_left - (N_right / N_root) * H_right
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

    thresholds = np.array([(i + j) / 2 for (i, j) in zip(unq_vals, unq_vals[1:]) if
                           target_vector[feature_vector < (i + j) / 2].shape[0] > 0 and
                           target_vector[feature_vector >= (i + j) / 2].shape[0] > 0
                           ])

    criterion_vals_list = []
    threshold_best, criterion_best = None, None
    for threshold in thresholds:
        curr_crit = compute_criterion(
            target_vector=target_vector,
            feature_vector=feature_vector,
            threshold=threshold,
            criterion=criterion
        )
        criterion_vals_list.append(curr_crit)
        if criterion_best is None:
            criterion_best = curr_crit
            threshold_best = threshold
        if curr_crit > criterion_best:
            criterion_best = curr_crit
            threshold_best = threshold

    return thresholds, np.array(criterion_vals_list), threshold_best, criterion_best


class DecisionTree(BaseEstimator):

    def __init__(self, feature_types: list, criterion: str = 'gini',
                 max_depth: int = None, min_samples_split: int = None, min_samples_leaf: int = None):

        """
        Args:
            feature_types: список типов фичей (может состоять из 'real' и "categorical")
            criterion: может быть 'gini' или "entropy"
            max_depth: максимальная глубина дерева
            min_samples_split: минимальное число объектов в листе, чтобы можно было расщиплять этот лист
            min_samples_leaf: минимальное число объектов в полученных листьях
        """

        self._feature_types = feature_types
        self._tree = {}
        self.target_encodings = {}
        self._criterion = criterion
        self.max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X: np.array, sub_y: np.array, node: dict):

        if np.all(sub_y == sub_y[0]):
            node['type'] = 'terminal'
            node['class'] = sub_y[0]
            return

        best_criterion, best_feature_number, best_threshold = None, None, None

        for feature_number in range(sub_X.shape[1]):

            if self._feature_types[feature_number] == 'real':
                feature_vector = sub_X[:, feature_number]
            elif self._feature_types[feature_number] == 'categorical':
                target_enc = TargetEncoder().fit(sub_X[:, feature_number], sub_y)
                feature_vector = target_enc.transform(sub_X[:, feature_number]).values.ravel()
                self.target_encodings[feature_number] = target_enc
            else:
                raise ValueError('feature_type может быть "real" или "categorical"')

            _, _, curr_threshold, curr_crit = find_best_split(
                feature_vector=feature_vector,
                target_vector=sub_y,
                criterion=self._criterion
            )
            if best_criterion is None or curr_crit > best_criterion:
                best_criterion = curr_crit
                best_threshold = curr_threshold
                best_feature_number = feature_number

                split = feature_vector < best_threshold if best_threshold is not None else None

        if best_threshold is None:
            node['type'] = 'terminal'
            node['class'] = Counter(sub_y).most_common(1)[0][0]
            return

        node['type'] = 'nonterminal'
        node['feature_type'] = self._feature_types[best_feature_number]
        node['feature_number'] = best_feature_number
        node['threshold'] = best_threshold

        node['left_child'], node['right_child'] = {}, {}
        self._fit_node(sub_X=sub_X[split], sub_y=sub_y[split], node=node['left_child'])
        self._fit_node(sub_X=sub_X[np.logical_not(split)], sub_y=sub_y[np.logical_not(split)],
                       node=node['right_child'])

    def _predict_node(self, x: np.array, node: dict):
        """
        Должен либо вернуть класс для объекта х, либо рекурсивно просеить его в левое или правое поддерево.
        """

        if node['type'] == 'terminal':
            return node['class']

        if node['feature_type'] == 'real':
            if x[node['feature_number']] < node['threshold']:
                return self._predict_node(x=x, node=node['left_child'])
            else:
                return self._predict_node(x=x, node=node['right_child'])

        elif node['feature_type'] == 'categorical':
            mapped_value = \
            self.target_encodings[node['feature_number']].transform([x[node['feature_number']]]).values.ravel()[0]
            if mapped_value < node['threshold']:
                return self._predict_node(x=x, node=node['left_child'])
            else:
                return self._predict_node(x=x, node=node['right_child'])

    def fit(self, X: np.array, y: np.array):
        self._fit_node(sub_X=X, sub_y=y, node=self._tree)

    def predict(self, X: np.array):
        assert self._tree != {}, "Cначала обучите модель!"
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

    def cv_result(self, X: np.array, y: np.array, n_folds: int = 10, scorer: callable = accuracy_score):

        """
        Вспомогательный метод для задания в практической. Этот метод получает оценку scorer функции для модели по n_folds фолдам.
        """

        # Создаём индексы объектов по фолдам
        idxs_by_folds = np.array_split(
            np.random.permutation(len(X)), n_folds
        )

        train_val_idxs_dict = {
            fold_number: {
                'train_idxs': np.array(list(set(np.arange(len(X))) - set(idxs_by_folds[fold_number]))),
                'val_idxs': idxs_by_folds[fold_number]
            } for fold_number in range(n_folds)
        }

        cv_results = []
        for fold_number in range(n_folds):
            self._tree = {}
            self.fit(X=X[train_val_idxs_dict[fold_number]['train_idxs']],
                     y=y[train_val_idxs_dict[fold_number]['train_idxs']])
            y_pred = self.predict(X=X[train_val_idxs_dict[fold_number]['val_idxs']])
            y_true = y[train_val_idxs_dict[fold_number]['val_idxs']]

            cv_results.append(
                scorer(y_true=y_true, y_pred=y_pred)
            )
        return cv_results
