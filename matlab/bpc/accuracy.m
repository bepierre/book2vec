b = 1;
top = 1;

acc = [];
for b = 1:500
    p = books{b}.probs;
    t = books{b}.target;
    b_acc = 0;
    for i = 1:size(p,1)
        [~, guess] = maxk(p(i,:), top);
        target = double(t(i)+1);
        b_acc = b_acc + sum(ismember(target,guess));
    end
    acc = [acc, b_acc / size(p,1)];
end
mean(acc)