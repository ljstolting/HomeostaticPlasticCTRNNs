n = 20;
pardim = (n^2) + (2*n);

MinSearchValue = -1;
MaxSeachValue = 1; %both from the C code

p = haltonset(pardim,'Skip',1e3,'Leap',1e2);
p = scramble(p,'RR2')
X0 = net(p,50000);
X0 = (X0 - .5)*2;
writematrix(X0,'quasidim20.txt');
