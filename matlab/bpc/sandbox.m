b = 23;

figure(1)
idx = books{b}.target + 1;
F = ind2vec(double(idx),100);
imagesc(F')

figure(2)
probs = books{b}.probs;
probsc = probs
imagesc(probs)

figure(3)
probs(probs < 0.035) = 0;
imagesc(probs)

figure(4)
imagesc(full(F')-probs*10)