p = haltonset(2,'Skip',1e3,'Leap',1e2);
p = scramble(p,'RR2')
X0 = net(p,50000);
