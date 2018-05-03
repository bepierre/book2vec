close all

num_vec = csvread('num_vec.csv');
book_num = 1090 - 1;
num_books = 1;
par_inds = sum(num_vec(1:book_num))+1:sum(num_vec(1:book_num+num_books));
idx = csvread('kmeans_100_c_labels.csv');
idx = idx(par_inds);
idx = idx + 1;
m = 100;
n = numel(idx);
y = zeros(m,1);
p = zeros(m,m);
for k=1:n-1
    y(idx(k)) = y(idx(k)) + 1;
    p(idx(k),idx(k+1)) = p(idx(k),idx(k+1)) + 1;
end
p = bsxfun(@rdivide,p,y); p(isnan(p)) = 0;
mc = dtmc(p);
graphplot(mc,'ColorEdges',true);