import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
from matplotlib import pyplot as plt
from surprise import Dataset
from surprise import SVDpp, KNNBasic
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from surprise.model_selection import train_test_split as tts
from surprise import dump
from sklearn import metrics



## Data cleaning
users_copy = pd.read_csv(r'C:\Users\niall\OneDrive\Documents\Laura - College\MovieLens\Datasets\users.csv')
movies = pd.read_csv(r'C:\Users\niall\OneDrive\Documents\Laura - College\MovieLens\Datasets\movies.csv', encoding = 'latin')
ratings = pd.read_csv(r'C:\Users\niall\OneDrive\Documents\Laura - College\MovieLens\Datasets\ratings.csv')
zip_income = pd.read_csv(r'C:\Users\niall\OneDrive\Documents\Laura - College\MovieLens\Datasets\Median.csv')

ratings.drop(['timestamp'], axis=1, inplace= True)


users_copy['zipCode']= np.where(users_copy['zipCode'].str.isnumeric(), users_copy['zipCode'], 0)
users_copy['zipCode']= users_copy['zipCode'].astype(str).astype(int)
users_copy = pd.merge(users_copy, zip_income[['zipCode', 'Median_Income']], on= 'zipCode', how='left')
users_copy.drop(['zipCode'], axis=1, inplace= True)
mu_age = round(np.mean(users_copy['Median_Income']), 0)
users_copy['Median_Income']= users_copy['Median_Income'].fillna(mu_age)

rate_user = ratings.join(users_copy.set_index('userid') , on = 'userid')
#rate_user.drop(['userid'], axis=1, inplace= True)
rate_user = pd.concat([rate_user ,pd.get_dummies(rate_user['gender'], prefix='Gender' , drop_first=True)],axis=1)
rate_user = pd.concat([rate_user ,pd.get_dummies(rate_user['occupation'], prefix='Occ' , drop_first=True )],axis=1)
rate_user.drop(['gender'], axis=1, inplace= True)
rate_user.drop(['occupation'], axis=1, inplace= True)


movies= movies.rename(columns = {'movieid ' :'movieid'})
movies.drop([' movie title '], axis =1, inplace = True)
movies[' release date '] = movies[' release date '].str[-2:]
movies[' release date ']= movies[' release date '].fillna(0)
movies[' release date ']= movies[' release date '].astype(str).astype(int)
movies.drop([' video release date '], axis =1, inplace = True)
movies.drop([' IMDb URL '], axis =1, inplace = True)


rate_user_genre = rate_user.join(movies.set_index('movieid') , on = 'movieid')

userstats = rate_user_genre.groupby('userid', as_index=False)['rating'].agg({'usermean':np.mean,'userstd':np.std,'usermin':np.min, 'usermax':np.max,'usermedian':np.median,'usercount' : 'count' })
moviestats = rate_user_genre.groupby('movieid', as_index=False)['rating'].agg({'moviemean':np.mean,'moviestd':lambda x : np.std(x, ddof=0),'moviemin':np.min, 'moviemax':np.max, 'moviemedian':np.median,'moviecount' : 'count'})


rate_user_genre = rate_user_genre.join(userstats.set_index('userid') , on = 'userid')
rate_user_genre = rate_user_genre.join(moviestats.set_index('movieid') , on = 'movieid')


## Scaling
rate_user_genre_colnames = rate_user_genre.columns
scaler = StandardScaler()
rate_user_genre_scaled = pd.DataFrame(scaler.fit_transform(rate_user_genre), columns = rate_user_genre_colnames) 

# =============================================================================
#  LGBM data 
# =============================================================================
X =rate_user_genre_scaled.loc[:, rate_user_genre.columns != 'rating']
y = ratings['rating']


X_train = X.loc[:89999, :]
y_train = y.loc[:89999]
X_test = X.loc[90000:, :]
y_test =  y.loc[90000:]


LGBM = lgb.LGBMRegressor()
LGBM.fit(X_train, y_train)
LGBM_pred = LGBM.predict(X_test)
LGBM_test = rate_user_genre.loc[90000:,['userid', 'movieid', 'rating']]
LGBM_test['LGBM_predict'] = LGBM_pred
LGBM_test = LGBM_test.reset_index()
LGBM_test.drop(['index'], axis = 1 , inplace = True)


# =============================================================================
# SVD++
# =============================================================================
reader = Reader(rating_scale = (0,5))
data = Dataset.load_from_df(ratings, reader)

np.random.seed(1)
random.seed(1)
raw_ratings = data.raw_ratings
threshold = int(.9 * len(raw_ratings))
A_raw_ratings = raw_ratings[:threshold]
B_raw_ratings = raw_ratings[threshold:]

data.raw_ratings = A_raw_ratings  # data is now the set A
trainset = data.build_full_trainset()
algo = SVDpp()
algo.fit(trainset)         
testset = data.construct_testset(B_raw_ratings)                    
predictions = algo.test(testset)
dump.dump('./dump_file', predictions, algo)
predictions, algo = dump.load('./dump_file')

SVDpp_test = pd.DataFrame(predictions, columns=['userid', 'movieid', 'rating', 'SVDpp_predict', 'details'])    
SVDpp_test.drop(['details'], axis =1 , inplace = True)

# =============================================================================
# Joined predictions 
# =============================================================================
two_alg = SVDpp_test.join(LGBM_test.set_index(['userid', 'movieid', 'rating']) , on = ['userid', 'movieid', 'rating'])
two_alg['SVDpp_error'] = np.abs(np.subtract(two_alg['rating'], two_alg['SVDpp_predict']))
two_alg['LGBM_error'] = np.abs(np.subtract(two_alg['rating'], two_alg['LGBM_predict']))

two_alg.iloc[:, 5:7].mean()



## KNN to get cluster splitting dataa
## dataset withiut rating and with cluster 
dataset_no_rating = rate_user_genre_scaled.drop(['rating'], axis = 1, inplace = False)
dataset_no_rating['Cluster'] = kmeans_userslabels 

Xc =dataset_no_rating.loc[:, dataset_no_rating.columns != 'Cluster']
yc = dataset_no_rating['Cluster']
Xc_train = Xc.loc[:89999, :]
yc_train = yc.loc[:89999]
Xc_test = Xc.loc[90000:, :]
yc_test =  yc.loc[90000:]


knn = KNeighborsClassifier(n_neighbors=7, metric='euclidean')
knn.fit(Xc_train, yc_train)
yc_pred = knn.predict(Xc_test)
knn_test = rate_user_genre.loc[90000:,['userid', 'movieid', 'rating']]
knn_test['knn_predict'] = yc_pred


print("Accuracy:",metrics.accuracy_score(yc_test, yc_pred))
confusion_matrix(yc_test, yc_pred)

value = np.arange(5, 9, 1)
for i in value:
    knn = KNeighborsClassifier(n_neighbors=i, metric='euclidean')
    model = knn.fit(Xc_train, yc_train)
    yc_pred = knn.predict(Xc_test)
    w = metrics.accuracy_score(yc_test, yc_pred)
    print(i)
    print(w)



# =============================================================================
# 
    # LGBM  - 0, 1, 4, 6, 7
    # SVD++ - 2, 3, 5, 8, 9
# =============================================================================
two_alg['knn_predict'] = yc_pred
for in :
    if two_alg['knn_predict'] == 0 or two_alg['knn_predict'] ==



