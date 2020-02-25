import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
from matplotlib import pyplot as plt
import itertools


## Merging all rsults : suprise algorithm results, LGBM results and Linear Regression results
surprise_res_mae[['userid', 'movieid', 'rating']] = surprise_res_mae[['userid', 'movieid', 'rating']].astype(str).astype(float)
results_mae = LGBM_res_mae.join(surprise_res_mae.set_index(['userid', 'movieid', 'rating']) , on = ['userid', 'movieid', 'rating'])
results_mae = LR_res_mae.join(results_mae.set_index(['userid', 'movieid', 'rating']) , on = ['userid', 'movieid', 'rating'])

#writing results to csv for future use
#results_mae.to_csv(r'C:\Users\niall\OneDrive\Documents\Laura - College\MovieLens\MAE\all instances results.csv')

#results_mae = pd.read_csv(r'C:\Users\Me\Documents\College\Final Year Project\Clusters\4\All alg.csv') 
#results_mae.drop(['Unnamed: 0'], axis=1, inplace= True)

#### Perfect metalearner
df = pd.DataFrame(results_mae[[ 'KNNBasic_delta','SVDpp_delta', 'CoClustering_delta']])
min_error = df.min(axis=1)
min_error.mean()

alg = ['BaselineOnly_delta', 'KNNBasic_delta', 'KNNBaseline_delta','SVDpp_delta', 'SlopeOne_delta', 'CoClustering_delta', 'LGBM_delta', 'Reg_delta']

def compute_n_algos(algs, x):
    combinations = set(itertools.combinations(algs, x))
    smallest_val = 9999999999999999
    smallest_combination = ''
    for c in combinations:  
        df = pd.DataFrame(results_mae[list(c)])
        min_error = df.min(axis=1)
        res = min_error.mean()
        if res < smallest_val:
            smallest_val = res
            smallest_combination = c
    return {'num_algos': x, 'combination': smallest_combination, 'mae': smallest_val }
    #print('For {0} combinations - {1} has the smallest error of {2}'.format(x,smallest_combination,smallest_val))

def compute_df_of_lowest_error_for_each_combination_size(algs):
    df = pd.DataFrame()
    for i in range(len(algs)):
        df = df.append(compute_n_algos(algs,i+1), ignore_index=True)
    return df

mae_combinations = compute_df_of_lowest_error_for_each_combination_size(alg)
print (mae_combinations)



## aadding columns to df with most and least accurate algorithm
results_mae['Most_Acc'] = results_mae.iloc[:, 3:11].idxmin(axis=1)
results_mae['Least_Acc'] = results_mae.iloc[:, 3:11].idxmax(axis=1)

## Overall results: MAE of each algorithm, count of number and % of times each algorithm is best and worst
results_mae_acc = pd.DataFrame()
results_mae_acc['Count_most_acc'] = results_mae.groupby(['Most_Acc'], sort = True).count().iloc[:,1]
results_mae_acc['percent_most_acc'] = (results_mae_acc['Count_most_acc']/results_mae_acc['Count_most_acc'].sum())*100
results_mae_acc['Count_least_acc'] = results_mae.groupby(['Least_Acc'], sort = True).count().iloc[:,1]
results_mae_acc['percent_least_acc'] = (results_mae_acc['Count_least_acc']/results_mae_acc['Count_least_acc'].sum())*100
mae = results_mae.iloc[:, 3:11].mean()
results_mae_acc['mae'] = mae
results_mae_acc.to_csv(r'C:\Users\niall\OneDrive\Documents\Laura - College\MovieLens\MAE\Count and MAE.csv')



## PCA for kmeans using all features including added user and item features
pca = PCA(n_components = 3) 
pca = pca.fit_transform(rate_user_genre_scaled)
pca = pd.DataFrame(pca) 
pca.columns = ['P1', 'P2', 'P3'] 

##Kmeans loop for elbow plot
SS_distances = []
K = range(1,25)
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans = kmeans.fit(pca) 
    SS_distances.append(kmeans.inertia_)

plt.plot(K, SS_distances)
plt.xlabel('k')
plt.ylabel('SS_distances')
plt.title('Elbow plot for k')
plt.show()

#kmeans with selected number of clusters
kmeans = KMeans(n_clusters=10)
kmeans.fit(pca)
kmeans_userslabels = pd.DataFrame(kmeans.predict(pca))

#adding kmeans labels to result dataset
results_mae['Cluster'] = kmeans_userslabels
results_mae.to_csv(r'C:\Users\niall\OneDrive\Documents\Laura - College\MovieLens\all with clusters alg.csv')


#### MAE by cluser
means_clusters = results_mae.groupby(['Cluster'], sort = True).mean().iloc[:,3:11]
#mae_clusters = (means_clusters)
#mae_clusters.to_csv(r'C:\Users\niall\OneDrive\Documents\Laura - College\MovieLens\MAE\10.csv')
means_clusters.to_csv(r'C:\Users\niall\OneDrive\Documents\Laura - College\MovieLens\MAE\cluster rmse.csv')

## Count of number of times each algorithm is best and worst witin each cluster
countma_clusters = results_mae.groupby(['Cluster', 'Most_Acc'], sort = True).count().iloc[:,1:11]
countma_clusters.to_csv(r'C:\Users\niall\OneDrive\Documents\Laura - College\MovieLens\MAE\countma_clusters.csv')
