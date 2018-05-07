close all

num_vec = csvread('num_vec.csv');
book_num = 1452 - 1;
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
G = digraph(p);

X = csvread('kmeans_100_c_cluster_centers.csv');
rng default % for reproducibility
Y = tsne(X);
x = Y(:,1);
y = Y(:,2);

h = plot(G,'XData',x,'YData',y); 
colormap jet
h.EdgeCData = G.Edges.Weight;
h.MarkerSize = 5;
h.NodeColor = 'red';
cb = colorbar
caxis([0 1])
% hide axes:
set(gca,'XTickLabel',{' '})
set(gca,'YTickLabel',{' '})
set(gca,'YTick',[])
set(gca,'XTick',[])

colorTitleHandle = get(cb,'Title');
titleString = 'Transition Probabilities';
set(colorTitleHandle ,'String',titleString);