M = csvread('Hobbit_4c_w.csv');
%h = heatmap(M);

figure(1); hold on
M = M-mean(M);
s = svd(M);
plot(s)

for i = 1:1
    R = sqrt(var(M)).*randn(size(M));
    sr = svd(R);
    plot(sr)
end
    
legend('hobbit', 'random')