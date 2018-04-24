close all

par_inds = 10000:10200;

num_vec = csvread('num_vec.csv');
subplot(2,1,1); 
idx = csvread('kmeans_labels.csv');
idx = idx + 1;
F = ind2vec(idx')';
visited_clusters = unique(sort(idx(par_inds)));
imagesc(F(par_inds,visited_clusters))
set(gca,'XTick',1:length(visited_clusters))
set(gca,'XTickLabel',visited_clusters)
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'fontsize',6)



subplot(2,1,2);
idx_c = csvread('kmeans_c_labels.csv');
idx_c = idx_c + 1;
F_c = ind2vec(idx_c')';
visited_clusters_c = unique(sort(idx_c(par_inds)));
imagesc(F_c(par_inds,visited_clusters_c))
set(gca,'XTick',1:length(visited_clusters_c))
set(gca,'XTickLabel',visited_clusters_c)
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'fontsize',6)

figure(2);
subplot(2,1,1); hist(idx); ylim([0 5e5])
subplot(2,1,2); hist(idx_c); ylim([0 5e5])
