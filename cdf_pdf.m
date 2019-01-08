%Figure 1
%cdf and pdf 
subplot(1,2,1);
q=@(x)normcdf(x, 0, 1);
x = -4:0.1:4;
y = q(x);
plot(x,y,'b','LineWidth',3.5);
title('Gaussian CDF');
subplot(1,2,2);
p = @(x)normpdf(x, 0, 1);
x = -4:0.1:4;
y = p(x);
plot(x,y,'b','LineWidth',3.5);
axis([-4,4,0,0.45]);
title('Gaussian PDF');