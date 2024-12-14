import numpy as np

def jaccard_sim(ratings: np.array, user_vector: np.array) -> np.array:
    """
    Computes jaccard similarities between user (represented by user_vector) and users in ratings matrix
    
    Args:
        ratings: matrix of ratings of shape (# users we want to find distance to) x n_items
        user_vector: user ratings vector
    Returns:
        jaccard_sim_arr: jaccard distances from this user to others 
            (this vector length should equals ratings.shape[0])
    """
    # your code here:
    ...
    