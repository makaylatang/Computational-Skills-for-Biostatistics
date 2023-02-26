# Computational Skills for Biostatistics
#### UW BIOST561: Computational Skills fro Biostatistics (Spring 2022)
instructor: Eardi Lila

> Contents

🌟 Stochastic Gradient Descent 

🌟 S3 Method Bootstrap

🌟 S3 Method Shallow Neural Network & C++ imputation

## Stochastic Gradient Descent 

In class, we have introduced one of the most popular optimization techniques: gradient descent. However, for problems with very large n, gradient descent can be inefficient and stochastic gradient descent (a.k.a. batched gradient descent) is instead typically used. The main difference is that stochastic gradient descent, at each iteration, uses only random subsets of the n training sample to compute the gradient and update the parameter of interest (β in our problem).

In this problem, you will modify the gradient descent algorithm for linear regression introduced in class (you can also work on the logistic regression problem – slightly more challenging) to perform stochastic gradient descent optimization.

#### Pseudocode for gradient descent

Let `x` be a n × (p + 1) data-matrix and let `y` be the associated vector of outcomes of length n. We assume a linear relationship between input and output. Let `beta = beta_init` be the parameter to be optimized and let `niter` be the desired number of iterations.

`for it = 1, 2, 3, ..., niter:`
- Compute the loss function (using the entire training set)
- Compute the gradient of the loss function w.r.t. beta (using the entire training set)
- Update parameters: `beta = beta – learning_rate*gradient`

#### Pseudocode for stochastic gradient descent

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

## S3 Method Bootstrap

In this exercise, you will construct an S3 method bootstrap, for both the class `numeric` and `stratified`, with the following interface.

```
bootstrap.my_class <- function(object, nboot, stat){... your code here ...}
```
The function `bootstrap.my_class` will return the evaluation of the statistics encoded in the function `stat` on each one of the bootstrapped vectors. 

- Illustrate the use of your bootstrap generic function on objects of the class `numeric` and `stratified` using the mean, the median, and the standard deviation as the statistics of interest (e.g. make a histogram).

- Generalize the methods bootstrap defined above to the case of an argument `stat` that is a function that can take additional arguments, e.g. a function the computes the kth moment. Test it.

```
moment <- function(x, k) {
  (1/length(x))*sum((x-mean(x))ˆk)
}
```

## S3 Method Shallow Neural Network & C++ imputation

#### S3 Method Shallow Neural Network

In this exercise, you will refactor the shallow neural network code for binary classification provided in class and use the S3 object-oriented system.

To this end, you will define a new S3 class named shallow_net that contains the parameters of your model. More in detail, define the following:

- A constructor that given `p` and `q` returns an object shallow_net with randomly initialized parameters `theta` and `beta`.
- An S3 method `predict` that for a given object of the type `shallow_net` and a *n2 × p* data matrix `X` returns a *n2*-vector with the predicted probabilities.
- An S3 method `train`, that for a given a *n × p* data matrix `X`, a *n*-vector `y` of categorical outputs, the learning rate and number of iterations, uses gradient descent to learn the parameters of the neural network.

Re-run Examples 1 and 2 provided in class by using the redesigned code.

#### C++ imputation

- Profile the S3 method `train`. Comment on the results.
- Re-implement in C++ the computation of the gradient of the parameter `theta` of the neural network.
- Define a new S3 method `train_fast` that is a variation of `train` that replaces the code for the computation of the gradient of `theta` with the Cpp function implemented by calling

```
dL_dtheta <- compute_gradient_theta(X_aug, f_hat, y, beta, A)
```

- Check that train and train_fast give the same results by re-running the Example 1 and 2 with train_fast. Compare their performance.



