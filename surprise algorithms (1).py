import numpy as np 
import pandas as pd
import random
from surprise import Dataset
from surprise import KNNBasic, KNNBaseline
from surprise import SVDpp
from surprise import SlopeOne, CoClustering
from surprise import BaselineOnly
from surprise.model_selection import KFold



## function to generate results for all algorithms:
# returns df with userid, movieid, rating and MAE for all algorithms
def generate_ml_df_results(algos, folds):
    p_dict = {}
    kf = KFold(n_splits=folds)
    c = 0
    for trainset, testset in kf.split(data):
        c = c + 1
        for klass in algos:
            klass.fit(trainset)        
            predictions = klass.test(testset)
            for p in predictions:
                key = (p.uid, p.iid, c)
                algo_name = klass.__class__.__name__
                if key not in p_dict:
                    p_dict[key] = { 'userid': p.uid, 'movieid': p.iid, 'rating': p.r_ui, 'fold': c }
                delta = np.abs(p.est - p.r_ui )
                p_dict[key][str(algo_name)+'_est'] = p.est
                p_dict[key][str(algo_name)+'_delta'] = delta
                          
    return pd.DataFrame(p_dict.values())


#set RNG
np.random.seed(0)
random.seed(0)
 
#using ratings dataset downloaded from grouplens website instead of inbuilt movielens dataset as cannot view Dataset in surpirse
#reader sets rating scale  
reader = Reader(rating_scale = (0,5))
data = Dataset.load_from_df(ratings, reader)

#algorithms to be fed into function. Using all algorithms available in surprise
algos = (BaselineOnly(), SlopeOne(), CoClustering(), KNNBasic(), KNNBaseline(), SVDpp())

#call function with 5-fold cross validation
surprise_res_mae_2 = generate_ml_df_results(algos, 5)
