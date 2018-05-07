b = 346;
top = 5;

figure(1)
idx = books{b}.target + 1;
F = ind2vec(double(idx),100);
%imagesc(F')

% figure(2)
probs = books{b}.probs;
% probsc = probs
% imagesc(probs)

%figure(3)
sprobs = sort(probs);
probs(probs < min(maxk(probs,top,2),[],2)) = 0;
imagesc(probs)
hold on
for i = 1:length(idx)
    plot(idx(i), i, 'rx')
end

% figure(4)
% imagesc(full(F')-probs*10)