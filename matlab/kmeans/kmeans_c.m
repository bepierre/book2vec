close all

num_vec = csvread('num_vec.csv');
book_num = 1452 - 1;
num_books = 1;
par_inds = sum(num_vec(1:book_num))+1:sum(num_vec(1:book_num+num_books));
%subplot(3,1,1);
idx_c = csvread('kmeans_100_c_labels.csv');
idx_c = idx_c + 1;
F_c = ind2vec(idx_c')';
imagesc(F_c(par_inds,:))

% book_num = 728 - 1;
% num_books = 1;
% par_inds2 = sum(num_vec(1:book_num))+1:sum(num_vec(1:book_num+num_books));
% subplot(3,1,2);
% idx_c = csvread('kmeans_100_c_labels.csv');
% idx_c = idx_c + 1;
% F_c = ind2vec(idx_c')';
% imagesc(F_c(par_inds2,:))
% title('HG 2')
% 
% book_num = 805 - 1;
% num_books = 1;
% par_inds3 = sum(num_vec(1:book_num))+1:sum(num_vec(1:book_num+num_books));
% subplot(3,1,3);
% idx_c = csvread('kmeans_100_c_labels.csv');
% idx_c = idx_c + 1;
% F_c = ind2vec(idx_c')';
% imagesc(F_c(par_inds3,:))
% title('random')

% subplot(2,1,2);
% idx_mbc = csvread('mb_kmeans_100_c_labels.csv');
% idx_mbc = idx_mbc + 1;
% F_c = ind2vec(idx_mbc')';
% visited_clusters_c = unique(sort(idx_mbc(par_inds)));
% imagesc(F_c(par_inds,visited_clusters_c))
% set(gca,'XTick',1:length(visited_clusters_c))
% set(gca,'XTickLabel',visited_clusters_c)
% a = get(gca,'XTickLabel');
% set(gca,'XTickLabel',a,'fontsize',6)
% title('centered')
