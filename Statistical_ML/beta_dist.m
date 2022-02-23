X=0:0.01:1
as1 = 0.1; bs1 = 0.1;
ps1 = betapdf(X,as1,bs1);
as2 = 1; bs2 = 1;
ps2 = betapdf(X,as2,bs2)
as3 = 2; bs3 = 3;
ps3 = betapdf(X,as3,bs3)
as4 = 8; bs4 = 4;
ps4 = betapdf(X,as4,bs4)
figure;
plot(X, ps1,'Color','b','LineStyle','--', 'LineWidth', 3);
hold on;
plot(X, ps2,'Color','r', 'LineWidth', 3);
plot(X, ps3,'Color','g', 'LineWidth', 3);
plot(X, ps4,'Color','b', 'LineWidth', 3);
legend({'a = b = 0.1','a = b =1','a =2, b = 3','a=8,b=4'},'Location','NorthWest');
title('Beta distributions')
axis([-0.02,1.02,-0.1,3.1])
