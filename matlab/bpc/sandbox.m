idx = books{1}.target + 1;
F = ind2vec(double(idx),100);
imagesc(F')

probs = books{1}.probs;