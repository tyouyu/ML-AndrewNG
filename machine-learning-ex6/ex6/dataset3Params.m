function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

vector = [0.01 0.03 0.1 0.3 1 3 10 30];
e=10;
for i = 1:length(vector)
    for j = 1:length(vector)
        model = svmTrain(X, y, vector(i), @(x1, x2) gaussianKernel(x1, x2, vector(j)));
        predictions = svmPredict(model, Xval);
        error = mean(double(predictions~=yval));
        if (error<e)
            e=error;
            C=vector(i);
            sigma=vector(j);
            fprintf('[C, sigma] = [%f %f]\n', C, sigma);
        end
    end
end


%{
error_min = inf;
values = [0.01 0.03 0.1 0.3 1 3 10 30];

for C1 = values
  for sigma1 = values
    fprintf('Train and evaluate (on cross validation set) for\n[_C, _sigma] = [%f %f]\n', C1, sigma1);
    model = svmTrain(X, y, C1, @(x1, x2) gaussianKernel(x1, x2, sigma1));
    e = mean(double(svmPredict(model, Xval) ~= yval));
    fprintf('prediction error: %f\n', e);
    if( e <= error_min )
      fprintf('error_min updated!\n');
      C = C1;
      sigma = sigma1;
      error_min = e;
      fprintf('[C, sigma] = [%f %f]\n', C, sigma);
    end
    fprintf('--------\n');
  end
end
%}

% =========================================================================

end
