import numpy as np
from scipy.special import expit
from sklearn.metrics import log_loss
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--regime", help="Режим оптимизации: ручной или авто", type=str, default='авто')
args = parser.parse_args()
regime = args.regime

def black_box_function(x: np.ndarray) -> float:
    """
    Неизвестная функция, которую нужно минимизировать
    Args:
        x: np.ndarray (x.shape = (10,))
    Returns:
        значение функции в точке x (float)
    """

    np.random.seed(42)
    N = 1000
    X = np.random.uniform(-15, 15, size=(N, 10))
    w = np.array([2.63007154, -5.46182011, 1.61492748, 6.92120653,
                  -13.72995162, -14.66402084, -1.25799183, -11.04490147,
                  4.93032874, -8.57842631])
    error_term = np.random.normal(size=N)
    y = expit(X @ w + error_term).round()

    logloss = log_loss(
        y_true=y,
        y_pred=expit(X @ x)
    ) * 100
    return logloss