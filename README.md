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


### Pseudocode for stochastic gradient descent


## S3 method shallow neural network and C implementation
