function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
X = [ones(m) X]
a1_val = sigmoid(X * Theta1');
for i = 1:size(a1_val,1)
  for j = 1:size(a1_val,2)
    if a1_val(i,j) >= 0.5
      a1_val(i,j) = 1;
    else 
      a1_val(i,j) = 0;
    end
  end
end

a1_val = [ones(size(a1_val,1)) a1_val];

a2_val = sigmoid(a2_val * Theta2);

for i = 1:size(a2_val,1)
  for j = 1:size(a2_val,2)
    if a2_val(i,j) >= 0.5
      a2_val(i,j) = 1;
    else 
      a2_val(i,j) = 0;
    end
  end
end

[C,p] = max(a2_val);







% =========================================================================


end
