# FYP Per-Instance-Algorithm-Selection-Method

Laura Tierney
Final Year Mathematics Project 
2019-2020

Per-Instance-Algorithm-Selection-Method
Classifying new instances into a cluster using KNN and using the best performing algorithm for that cluster to predict the rating.

Abstract
Meta-learning in the area of Recommender systems is primarily used to select the overall
best-performing algorithm to be used in the system on a global level. This project details a
method to use meta-learning at a per-instance level for recommender systems using K-means
clustering and K-Nearest Neighbors.

In the project we show the difference in algorithm performance for the 14 algorithms
investigated in the project; Normal predictor has the largest MAE if 1.2228, however, it is the
best algorithm available for 16.3% of instances; LGBM_EF has the smallest MAE of 0.7206,
however, it is best for just 3.1% of instances.

We aim to exploit this difference in algorithm performance in our Per-Instance Algorithm
Selection Method, MetaClust. We developed 3 different versions of this method;

MetaClust2/10 with 2 algorithms and 10 clusters - this made an improvement of 1% on the
overall best-performing algorithm (MetaClust2/10 : MAE – 0.7142; SD – 0.5561, p-value –
0.1942);

MetaClust5/25 with 5 algorithms and 25 clusters - this made an improvement of 1.19% on the
overall best-performing algorithm (MetaClust5/25 : MAE – 0.7128; SD – 0.5579; p-value –
0.1216);

MetaClust2/25 with 2 algorithms and 25 clusters- this made an improvement of 1.48% on the
overall best-performing algorithm (MetaClust2/25 : MAE – 0.7107; SD – 0.5559; p-value –
0.0532).


