M1 = csvread('Da_Vinci_Code_4c_w.csv');
M2 = csvread('Hobbit_4c_w.csv');
M3 = csvread('James_Bond_4c_w.csv');
M4 = csvread('Marriage_Trap_4c_w.csv');
M5 = csvread('Vampireville_4c_w.csv');

D = pdist(M3);
Z = squareform(D);
heatmap(Z, 'ColorLimits', [25, 50], 'Colormap', hot)
