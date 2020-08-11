import typing as tp
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt

class LogReg(BaseEstimator):
    
    def __init__(self):
        self.params = {}
        self.weights = None
        self.epochs_results = None
        self.fitted_weight = None
        self.last_epoch = None
        
    def fit(
        self, 
        X, 
        y, 
        fit_intercept: bool=True, 
        reg_coef: float=1.0, 
        tol: float=1e-2,
        init_weights: np.array=None,
        track_epochs: bool=True,
        max_iter: int=5000,
        learning_rate: float=1.0,
        verbose: bool=True,
        criterion: tp.List[str]='weight',
    ):
        """
        Finds optimal weights for logistic regression
        
        Args:
            criterion: could be one of ["loss", "weight", "grad"]
                "loss": check's if precentage loss change is lower than tol
                "weight": check's if weights absolute norm is lower than tol
                "grad": check's if gradient norm is lower than tol
        """
        
        self.fit_intercept = fit_intercept
        self.reg_coef = reg_coef
        assert criterion in ["loss", "weight", "grad"], 'criterion should be one of ["loss", "weight", "grad"]'
        
        epoch = 1
        
        X_copy = deepcopy(X)
        y_copy = transform_target(y)
        
        if fit_intercept:
            X_copy = self._add_intercept(X_copy)
        
        if init_weights is None:
            prev_weights = np.random.normal(size=X_copy.shape[1])
        else:
            prev_weights = init_weights
            
        criterion_history = [np.inf]
        
        if track_epochs:
            self.epochs_results = {'loss': {}, 'weights': {}, 'grad': {}}
            prev_loss = self.compute_loss(X=X_copy, y=y_copy, weights=prev_weights, 
                                          reg_coef=reg_coef)
            prev_grad = self.compute_grad(X=X_copy, y=y_copy, weights=prev_weights, 
                                          reg_coef=reg_coef)
            self.epochs_results['loss'][epoch] = prev_loss
            self.epochs_results['weights'][epoch] = prev_weights
            self.epochs_results['grad'][epoch] = np.linalg.norm(prev_grad)
            if verbose:
                print(f'epoch: {epoch}, loss: {round(self.epochs_results["loss"][epoch], 3)}, grad norm: {round(np.linalg.norm(prev_grad), 3)}')
        
        success_flag = True
        while criterion_history[-1] > tol:
            epoch += 1
            curr_grad = self.compute_grad(X=X_copy, y=y_copy, weights=prev_weights, reg_coef=reg_coef)
            curr_loss = self.compute_loss(X=X_copy, y=y_copy, weights=prev_weights, reg_coef=reg_coef)
            curr_weights = prev_weights - learning_rate * curr_grad
            
            if criterion == 'weight':
                criterion_history.append(np.linalg.norm(prev_weights - curr_weights)/np.linalg.norm(prev_weights))
            elif criterion == 'loss':
                criterion_history.append(abs(prev_loss - curr_loss)/abs(prev_loss))
            else:
                criterion_history.append(np.linalg.norm(curr_grad))
            
            prev_weights = curr_weights
            prev_loss = curr_loss
            prev_grad = curr_grad
                      
            if track_epochs:
                curr_loss = self.compute_loss(X=X_copy, y=y_copy, weights=curr_weights, reg_coef=reg_coef)
                curr_grad = self.compute_grad(X=X_copy, y=y_copy, weights=curr_weights, reg_coef=reg_coef)
                self.epochs_results['loss'][epoch] = curr_loss
                self.epochs_results['weights'][epoch] = curr_weights
                self.epochs_results['grad'][epoch] = np.linalg.norm(curr_grad)
            if verbose:
                print(f"epoch: {epoch}, loss: {round(curr_loss, 3)}, grad norm: {round(np.linalg.norm(curr_grad), 3)}")
                
            if max_iter is not None:
                if epoch > max_iter:
                    print('Число эпох достигло max_iter!')
                    success_flag = False
                    self.fitted_weight = curr_weights
                    self.last_epoch = epoch
                    break 
        if success_flag:
            print('Достигнута требуемая точность')
        self.fitted_weight = curr_weights
        self.last_epoch = epoch
                      
    def predict_proba(self, X, weights: np.array=None):
                      
        if weights is None:
            weights = self.fitted_weight
        if self.fit_intercept:
            X_copy = self._add_intercept(X)
        else:
            X_copy = X
                      
        return sigmoid(X_copy @ weights)
                      
    def predict(self, X, threshold: float=0.5, weights: np.array=None):
        if weights is None:
            weights = self.fitted_weight
        return (self.predict_proba(X, weights=weights) > threshold).astype(int)
            
    def compute_grad(self, X, y, weights, reg_coef: float=None):
        
        y = transform_target(y)
        
        margin = y * (X @ weights)
        
        grad = (- y.reshape(-1, 1) * X * sigmoid(-margin).reshape(-1, 1)).sum(axis=0)
        
        if reg_coef is not None:
            grad += reg_coef * weights
        
        return grad
        
        
    def compute_loss(self, X, y, weights: np.array, reg_coef: float=None):
        
        y = transform_target(y)
        
        margin = y * (X @ weights)
        
        Q = -np.log(sigmoid(margin)).sum()
        
        if reg_coef is not None:
            Q += (reg_coef/2) * np.linalg.norm(weights)**2
            
        return Q
            
    def _add_intercept(self, X):
        
        X_copy = deepcopy(X)
        
        if isinstance(X_copy, pd.core.frame.DataFrame):
            X_copy = X_copy.values
        # if first columns is not of ones
        if (X_copy[:, 0] == np.ones(X_copy.shape[0])).sum() != X.shape[0]:
            X_copy = np.hstack([np.ones(X_copy.shape[0]).reshape(-1, 1), X_copy])
            
        return X_copy
                      
    def plot_learning_progress(self, y_axis: str='loss', epoch_range: tuple=None):
                      
        epoch, val = [], []
        [(epoch.append(i), val.append(j)) for i, j in self.epochs_results[y_axis].items()];
        
        if epoch_range is not None:
            epoch = [ep for ep in epoch if ep <= epoch_range[1] and ep >= epoch_range[0]]
            val = [val for val, ep in zip(val, epoch) if ep <= epoch_range[1] and ep >= epoch_range[0]]
        
        plt.figure(figsize=(16, 8));
        plt.plot(epoch, val);
        plt.scatter(epoch, val, s=15);
        plt.xlabel('epoch', fontsize=15);
        plt.ylabel(y_axis, fontsize=15);
                      
    def plot2d_level_curves(self, X, y, points_color: str='black', figsize=(8, 8), n_levels: int=50, 
                            track_every_k_points: int=None, points_size: int=12, text_size: int=12):
                      
        assert self.epochs_results is not None, 'параметр track_epochs должен быть True'
        assert len(self.epochs_results['weights'][1]) == 2, 'данная функция работает только для модели с двумя весами'
    
        min_loss, max_loss = min(list(self.epochs_results['loss'].values()))-300, max(list(self.epochs_results['loss'].values()))+300

        weights_by_history = []
        [weights_by_history.append(self.epochs_results['weights'][epoch]) 
                 for epoch in range(1, len(self.epochs_results['weights']) + 1)]
        weights_by_history = np.array(weights_by_history)


        weight0_lim = (weights_by_history[:, 0].min()-5, weights_by_history[:, 0].max()+5)
        weight1_lim = (weights_by_history[:, 1].min()-5, weights_by_history[:, 1].max()+5)

        levels = np.linspace(min_loss, max_loss, n_levels)
        W0, W1 = np.meshgrid(np.linspace(*weight0_lim, 50), np.linspace(*weight1_lim, 50))
        Z = np.zeros_like(W0)
        for i in range(W0.shape[0]):
            for j in range(W0.shape[1]):
                Z[i, j] = self.compute_loss(X, y, weights=np.array([W0[i, j], W1[i, j]]), reg_coef=self.reg_coef)

        plt.figure(figsize=figsize)
        plt.title(f'reg_coef: {round(self.reg_coef, 3)}', fontsize=15);
        cs = plt.contour(W0, W1, Z, levels=levels);
        plt.clabel(cs);
        plt.colorbar();
        if track_every_k_points is not None:
            for i in range(0, len(weights_by_history), track_every_k_points):
                plt.text(weights_by_history[i, 0], weights_by_history[i, 1], i+1, size=text_size);
                plt.scatter(weights_by_history[i, 0], weights_by_history[i, 1], s=points_size, color=points_color);
                plt.scatter(weights_by_history[-1, 0], weights_by_history[-1, 1], s=50, color='yellow', marker='*', 
                    label=f'последняя точка: {(round(weights_by_history[-1, 0], 2), round(weights_by_history[-1, 1], 2))}');
        else:
            plt.scatter(weights_by_history[:, 0], weights_by_history[:, 1], s=points_size, color=points_color);
            plt.scatter(weights_by_history[-1, 0], weights_by_history[-1, 1], s=50, color='yellow', marker='*', 
                    label=f'последняя точка: {(round(weights_by_history[-1, 0], 2), round(weights_by_history[-1, 1], 2))}');
        plt.legend(fontsize=15);
        plt.xlim(weight0_lim[0], weight0_lim[1])
        plt.ylim(weight1_lim[0], weight1_lim[1]);
        plt.xlabel(r'$w_{0}$', fontsize=15);
        plt.ylabel(r'$w_{1}$', fontsize=15);
        plt.show();
                      
    
def transform_target(y):
                      
    """
    Преобразует метки 0, 1 в метки -1, 1
    """
    
    y_new = deepcopy(y)
    
    if 0 in set(y_new):
        y_new[y_new == 0] = -1

    return y_new

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
                      
def plot_decision_function(X, y, decision_function: callable, title: str=None):
                      
    """
    Работает только для двумерной выборки
    """
                      
    plt.figure(figsize=(8, 8))
    plt.title(title, fontsize=15);
    plt.scatter(X[:, 0], X[:, 1], color=np.array(['blue', 'red'])[y]);
    
    x_dom = np.linspace(X[:, 0].min(), X[:, 0].max(), 1000)
    granz = np.array([decision_function(x) for x in x_dom])
    
    plt.plot(x_dom, granz, color='black', linestyle='--', linewidth=2);
    plt.xlim(X[:, 0].min(), X[:, 0].max());
    plt.ylim(X[:, 1].min(), X[:, 1].max());
