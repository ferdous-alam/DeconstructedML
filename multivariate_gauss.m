mu = [0 0]; 
Sigma = [.7 .5; .5 1]; 
x1 = -5:.2:5; x2 = -5:.2:5; 
[X1,X2] = meshgrid(x1,x2);
F = mvnpdf([X1(:) X2(:)],mu,Sigma);
F = reshape(F,length(x2),length(x1));
subplot(2,1,1)
contour(x1,x2,F);
title('Contour plot of MVN')
subplot(2,1,2)
surf(x1,x2,F);
title('Surface plot of MVN')
caxis([min(F(:))-.5*range(F(:)),max(F(:))]);
axis([-5 5 -5 5 0 0.3])
xlabel('x1'); ylabel('x2'); zlabel('Probability Density');