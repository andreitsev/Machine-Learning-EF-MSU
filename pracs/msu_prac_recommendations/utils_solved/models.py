from typing import Callable, List

import numpy as np
from numpy.typing import NDArray
import pandas as pd

# from utils.distances import jaccard_sim
from utils_solved.distances import jaccard_sim


similarityFuncType = Callable[[NDArray[float], NDArray[float]], NDArray[float]]

class BaseModel:
    def __init__(self, ratings: pd.DataFrame):
        self.ratings = ratings
        self.n_users = len(np.unique(self.ratings['userId']))
        self.n_items = len(np.unique(self.ratings['trackId']))

        self.R = np.zeros((self.n_users, self.n_items))
        self.R[self.ratings['userId'], self.ratings['trackId']] = 1.
        
    def recommend(self, uid: int):
        """
        param uid: int - user's id
        return: [n_items] - vector of recommended items sorted by their scores in descending order
        """
        raise NotImplementedError

    def remove_train_items(self, preds: List[List[int]], k: int):
        """
        param preds: [n_users, n_items] - recommended items for each user
        param k: int
        return: np.array [n_users, k] - recommended items without training examples
        """
        new_preds = np.zeros((len(preds), k), dtype=int)
        for user_id, user_data in self.ratings.groupby('userId'):
            user_preds = preds[user_id]
            new_preds[user_id] = user_preds[~np.in1d(user_preds, user_data['trackId'])][:k]

        return new_preds

    def get_test_recommendations(self, test_idxs: List[int], k: int) -> NDArray[int]:
        # your code here
        pass
        

class RandomRecommender(BaseModel):
    def __init__(self, ratings):
        super().__init__(ratings)

    def recommend(self, uid: int):
        unique_items = self.ratings['trackId'].unique()
        predictions_u = np.random.permutation(unique_items)
        return predictions_u
    

class User2User(BaseModel):
    def __init__(self, ratings, similarity_func: similarityFuncType=jaccard_sim, alpha: float=0.02):
        super().__init__(ratings)

        self.similarity_func = similarity_func
        self.alpha = alpha

    def similarity(self, user_vector: NDArray[int]) -> NDArray[float]:
        """Computes similarities between user_vector and all vectors in self.R
        Args:
            user_vector: vector of ratings, user has given to all tracks
        Returns:
            vector of simillarities between this user and all users in self.R
        """
        # your code here:
        pass
    
    def get_items_scores(self, uid: int) -> NDArray[float]:
        """Computes scores \hat{r}_{ui} for all items in rating matrix for
        a particular user uid

        Args:
            uid (int): index of user from rating matrix
        Returns:
            scores_u (NDArray[float]): array of scores for all items
        """
        # your code here:
        pass

    def recommend(self, uid: int):
        scores_u = self.get_items_scores(uid=uid)
        predictions_u = np.array([idx for idx in np.argsort(scores_u)[::-1]])
        return predictions_u