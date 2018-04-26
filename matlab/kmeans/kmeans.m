close all

num_vec = csvread('num_vec.csv');
book_num = 755 - 1;
num_books = 1;
par_inds = sum(num_vec(1:book_num))+1:sum(num_vec(1:book_num+num_books));

%subplot(3,1,1); 
% idx = csvread('kmeans_labels.csv');
% idx = idx + 1;
% F = ind2vec(idx')';
% visited_clusters = unique(sort(idx(par_inds)));
% imagesc(F(par_inds,visited_clusters))
% set(gca,'XTick',1:length(visited_clusters))
% set(gca,'XTickLabel',visited_clusters)
% a = get(gca,'XTickLabel');
% set(gca,'XTickLabel',a,'fontsize',5)

% hold on
% plot(-1000:6667, (232)*ones(7668,1), 'k')
% plot(-1000:6667, (232+237)*ones(7668,1), 'k')


% subplot(3,1,2);
idx_c = csvread('kmeans_100_c_labels.csv');
idx_c = idx_c + 1;
F_c = ind2vec(idx_c')';
visited_clusters_c = unique(sort(idx_c(par_inds)));
imagesc(F_c(par_inds,visited_clusters_c))
set(gca,'XTick',1:length(visited_clusters_c))
set(gca,'XTickLabel',visited_clusters_c)
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'fontsize',6)
title('centered')
% 
% subplot(3,1,3);
% idx_d = csvread('kmeans_d_labels.csv');
% idx_d = idx_d + 1;
% F_d = ind2vec(idx_d')';
% visited_clusters_d = unique(sort(idx_d(par_inds)));
% imagesc(F_d(par_inds,visited_clusters_d))
% set(gca,'XTick',1:length(visited_clusters_d))
% set(gca,'XTickLabel',visited_clusters_d)
% a = get(gca,'XTickLabel');
% set(gca,'XTickLabel',a,'fontsize',6)
% title('differences')

% figure(2);
% subplot(2,1,1); hist(idx); ylim([0 5e5])
% subplot(2,1,2); hist(idx_c); ylim([0 5e5])
