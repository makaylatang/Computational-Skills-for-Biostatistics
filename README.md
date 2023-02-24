# Computational Skills for Biostatistics
#### UW BIOST561: Computational Skills fro Biostatistics (Spring 2022)
instructor: Eardi Lila

> Contents

🌟 Stochastic gradient descent 

🌟 S3 method shallow neural network and C implementation

## Stochastic gradient descent 

In class, we have introduced one of the most popular optimization techniques: gradient descent. However, for problems with very large n, gradient descent can be inefficient and stochastic gradient descent (a.k.a. batched gradient descent) is instead typically used. The main difference is that stochastic gradient descent, at each iteration, uses only random subsets of the n training sample to compute the gradient and update the parameter of interest (β in our problem).

In this problem, you will modify the gradient descent algorithm for linear regression introduced in class (you can also work on the logistic regression problem – slightly more challenging) to perform stochastic gradient descent optimization.

### Pseudocode for gradient descent

Let `x` be a n × (p + 1) data-matrix and let `y` be the associated vector of outcomes of length n. We assume a linear relationship between input and output. Let `beta = beta_init` be the parameter to be optimized and let `niter` be the desired number of iterations.

`for it = 1, 2, 3, ..., niter:`
- Compute the loss function (using the entire training set)
- Compute the gradient of the loss function w.r.t. beta (using the entire training set)
- Update parameters: `beta = beta – learning_rate*gradient`

### Pseudocode for stochastic gradient descent

Let `x` be a n × (p + 1) data-matrix and let `y` be the associated vector of outcomes of length n. We assume a linear relationship between input and output. Let `beta = beta_init` be the parameter to be optimized and let niter be the desired number of iterations.

`for it = 1, 2, 3, ..., niter:`

- Split the n samples in the training set (same split for both x and y) into B groups that we call
mini-batches (Similarly to what you do with k-fold cross-validation).
- `for mini_batch = 1, 2, ..., B:`
    - Compute the loss function (using the entire training set)
    - Compute gradient of the loss function w.r.t. beta (using only the observations in the current mini_batch)
    - Update parameters: `beta = beta – learning_rate*gradient`

> Tasks:
1. Implement and test a function lm_sgd that performs stochastic gradient descent as described above. Describe (in words and small code sections) the main changes you have made to the lm_gd code, introduced in class, to implement lm_sgd. Put the entire code in an appendix.

2. Display the values of the loss function at every iteration in a scatter plot # iteration vs loss function for both lm_gd and lm_sgd. In light of this plot, why do you think the technique is called stochastic gradient descent?

3. Generate a list of 20 random vectors beta_init. For every element in the list run stochastic gradient descent with that initialization value. Use purrr:map for both the generation of the random vectors and the application of lm_sgd (see Lecture 4). Display the 20 estimation errors ∥β − β0∥2, where β0 is the true beta used to generate the data and β is the estimated one from lm_sgd.

## S3 method shallow neural network and C implementation
