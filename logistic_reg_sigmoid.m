x = -5:0.1:5;
y = 1 ./ (1+exp(-x));
plot(x,y,'LineWidth',3);
hold on
p = line([0 0],[0 1]);
q = line([-5 5],[0.5 0.5]);
xlabel('x');
ylabel('sigmoid function');
title('Sigmoid function for logistic regression')
