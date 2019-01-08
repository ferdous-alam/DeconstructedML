%taking radius of the circle as 1.0
r=1;
S=5000;
xs = unifrnd(-r,r,S,1);
ys = unifrnd(-r,r,S,1);
rs = xs.^2 + ys.^2;
inside = (rs <= r^2);
samples = 4*(r^2)*inside;
Ihat = mean(samples)
piHat = Ihat/(r^2)

figure(1);clf
outside = ~inside;
plot(xs(inside), ys(inside), 'bo');
hold on
plot(xs(outside), ys(outside), 'rx');
axis square
axis([-1.03 1.03 -1.03 1.03])
title ('Estimating pi using Monte carlo integration')
