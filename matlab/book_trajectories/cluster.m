close all

M1 = csvread('Hobbit_4c_w.csv');
M2 = csvread('Silmarillon_4c_w.csv');
M3 = csvread('Magyk_4c_w.csv');
M4 = csvread('Mistborn-1_4c_w.csv');
M5 = csvread('Mistborn-2_4c_w.csv');
M6 = csvread('Mistborn-3_4c_w.csv');
M7 = csvread('Warbreaker_4c_w.csv');
M8 = csvread('Discworld_4c_w.csv');
M9 = csvread('Hunger_Games_4c_w.csv');
M10 = csvread('Da_Vinci_Code_4c_w.csv');
M11 = csvread('James_Bond_4c_w.csv');
M12 = csvread('Marriage_Trap_4c_w.csv');
M13 = csvread('Thoughtless-1_4c_w.csv');
M14 = csvread('Thoughtless-2_4c_w.csv');
M15 = csvread('Thoughtless-3_4c_w.csv');
M16 = csvread('Vampireville_4c_w.csv');
M17 = csvread('Vampire_Brat_4c_w.csv');
M18 = csvread('Vampalicious_4c_w.csv');


num_c = 18

M = [M1; M2; M3; M4; M5; M6; M7; M8; M9; M10; M11; M12; M13; M14; M15; ...
     M16; M17; M18];
[idx,C] = kmeans(M, num_c);
F = ind2vec(idx')';
subplot(2,1,1); imagesc(F)
title('original')

M_c = [M1-mean(M1); M2-mean(M2); M3-mean(M3); M4-mean(M4); M5-mean(M5); ...
       M6-mean(M6); M7-mean(M7); M8-mean(M8); M9-mean(M9); M10-mean(M10); ...
       M11-mean(M11); M12-mean(M12); M13-mean(M13); M14-mean(M14); ...
       M15-mean(M15); M16-mean(M16); M17-mean(M17); M18-mean(M18)];
[idx_c,C_c] = kmeans(M_c, num_c);
F_c = ind2vec(idx_c')';
subplot(2,1,2); imagesc(F_c)
title('centered')

% M1 = csvread('Da_Vinci_Code_4c_w.csv');
% M2 = csvread('Hobbit_4c_w.csv');
% M3 = csvread('James_Bond_4c_w.csv');
% M4 = csvread('Marriage_Trap_4c_w.csv');
% M5 = csvread('Vampireville_4c_w.csv');
%
% M = [M1; M2; M3; M4; M5];
% [idx,C] = kmeans(M, num_c);
% F = ind2vec(idx')';
% subplot(2,2,1); imagesc(F)
% title('original')
% 
% M_c = [M1-mean(M1); M2-mean(M2); M3-mean(M3); M4-mean(M4); M5-mean(M5)];
% [idx_c,C_c] = kmeans(M_c, num_c);
% F_c = ind2vec(idx_c')';
% subplot(2,2,2); imagesc(F_c)
% title('centered')
% 
% M_n = [(M1-mean(M1))/var(M1); (M2-mean(M2))/var(M2); ...
%     (M3-mean(M3))/var(M3); (M4-mean(M4))/var(M4); (M5-mean(M5))/var(M5)];
% [idx_n,C_n] = kmeans(M_n, num_c);
% F_n = ind2vec(idx_n')';
% subplot(2,2,3); imagesc(F_n)
% title('normalized')