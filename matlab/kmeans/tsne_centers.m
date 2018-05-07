X = csvread('kmeans_100_c_cluster_centers.csv');
rng default % for reproducibility
Y = tsne(X);
