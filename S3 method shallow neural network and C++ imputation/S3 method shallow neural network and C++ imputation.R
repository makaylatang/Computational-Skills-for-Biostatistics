## S3 Shallow Neural Network

## Create a constructor function for the "shallow_net" class
## given p & q returns initialized parameters theta & beta
shallow_net <- function(p,q) {
        if (p<1) stop("'p' needs to be >= 1" )
        if (q<1) stop("'q' needs to be >= 1" )
        
        parameter <- list(theta = replicate(q, 0.1*runif(p+1, -.5, .5)), 
                          beta = 0.1*runif(q+1, -.5, .5))
        attr(parameter, "class") <- "shallow_net"
        parameter
}

# sigmoid function
sigmoid =  function(x) {
        1/(1+exp(-x))
}

# generic predict function
predict <- function(object, ...) UseMethod("predict")

# method for shallow_net
predict.shallow_net <- function(object, X){  
        if ( !is( object, "shallow_net") ) 
                stop( "predict.shallow_net requires an object of class 'shallow_net'" )
        if ( !is( X, "matrix") ) 
                stop( "X requires an object of class 'matrix'" )
        
        sigmoid =  function(x){ 1/(1+exp(-x))}
        
        theta = object$theta
        beta = object$beta
        
        n = nrow(X)
        X_aug = cbind(rep(1,n),X)
        A = sigmoid(X_aug %*% theta)
        A_aug = cbind(rep(1,n),A)
        f_pred = sigmoid(A_aug %*% beta)
        
        return(f_pred)
}

# Test
n = 100
set.seed(101)
X = as.matrix(runif(n, -2, 2))
y_prob = sigmoid(2 - 3*X^2)
y = rbinom(n,1,y_prob)

s <- shallow_net(1,4)
predict.shallow_net(s, X)

# train function for shallow_net
train <- function(X, y, learn_rate = 0.001, n_iter = 200, object)
{
        if ( !is( object, "shallow_net") ) 
                stop( "predict.shallow_net requires an object of class 'shallow_net'" )
        if ( !is( X, "matrix") ) 
                stop( "X requires an object of class 'matrix'" )
        
        beta = object$beta
        theta = object$theta
        
        q = ncol(theta)
        n = nrow(X)
        p = ncol(X)
        
        for (it in 1:n_iter)
        {
                if(it %% 1000 == 0) cat("Iter: ", it, "\n")
                
                # Forward pass
                X_aug = cbind(rep(1,n),X)
                A = sigmoid(X_aug %*% theta)
                A_aug = cbind(rep(1,n),A)
                f_hat = sigmoid(A_aug %*% beta)
                
                # Backward pass
                dloss_beta = (1/n)*t(A_aug)%*%(f_hat - y)
                dloss_theta = matrix(rep(NA, (p+1)*q), ncol = q)
                
                sum_theta = matrix(rep(0, (p+1)*q), ncol = q)
                for(i in 1:n)
                {
                        sum_theta = sum_theta + 
                                X_aug[i,]%*%t((f_hat[i] - y[i])*(A[i,]*(1-A[i,]))*beta[-1])   
                }
                
                dloss_theta = sum_theta/n
                
                beta  = beta - learn_rate*dloss_beta
                theta = theta - learn_rate*dloss_theta
                
                # Check objective function value
                # print(loss_func(y,f_hat_func(X,theta,beta)))
                
        }
        
        out = list(theta = theta, beta = beta)
        class(out) <- 'shallow_net'
        
        return(out)
}

##------------------------------------------------------------------------------
## Example 1 

n = 100
p = 1
q = 4

set.seed(101)
X = as.matrix(runif(n, -2, 2))
y_prob = sigmoid(2 - 3*X^2)
y = rbinom(n,1,y_prob)

object.shallow_net = shallow_net(p, q)

## Forward pass

f_hat = predict(object.shallow_net, X)

## Plot prediction
X_grid = as.matrix(seq(-2,2,length.out = 100))
y_prob_grid = sigmoid(2 - 3*X_grid^2)
f_hat_grid = predict(object.shallow_net, X_grid)

plot(X,y, pch = 20)
lines(X_grid,y_prob_grid, col = 'red')
lines(X_grid,f_hat_grid, col = 'blue')


## Animation

for(s_it in 1:60)
{
        out_nn = train(X, y, learn_rate = .3, n_iter = 100, object.shallow_net)
        
        ## Update the shallow_net parameters
        object.shallow_net = out_nn
        
        X_grid = as.matrix(seq(-2,2,length.out = 100))
        y_prob_grid = sigmoid(2 - 3*X_grid^2)
        f_hat_grid = predict(out_nn, X_grid)
        
        #plot(X,y, pch = 20)
        #lines(X_grid,y_prob_grid, col = 'red')
        #lines(X_grid,f_hat_grid, col = 'blue')
        #Sys.sleep(.8)
}

##------------------------------------------------------------------------------
## Example 2 

n = 200
p = 1
q = 8

set.seed(101)
X = as.matrix(runif(n, -2, 2))
y_prob = sigmoid(3 + X - 3*X^2 + 3*cos(4*X))
y = rbinom(n,1,y_prob)

object.shallow_net = shallow_net(p, q)

## Forward pass

f_hat = predict(object.shallow_net, X)

## Plot prediction
X_grid = as.matrix(seq(-2,2,length.out = 100))
y_prob_grid = sigmoid(3 + X_grid - 3*X_grid^2 + 3*cos(4*X_grid))
f_hat_grid = predict(object.shallow_net, X_grid)

plot(X,y, pch = 20)
lines(X_grid,y_prob_grid, col = 'red')
lines(X_grid,f_hat_grid, col = 'blue')

## Animation

for(s_it in 1:1000)
{
        out_nn = train(X, y, learn_rate = .5, n_iter = 500, object.shallow_net)
        
        ## Update the shallow_net parameters
        object.shallow_net = out_nn
        
        X_grid = as.matrix(seq(-2,2,length.out = 100))
        y_prob_grid = sigmoid(3 + X_grid - 3*X_grid^2 + 3*cos(4*X_grid))
        f_hat_grid = predict(object.shallow_net, X_grid)
        
        #plot(X,y, pch = 20)
        #lines(X_grid,y_prob_grid, col = 'red')
        #lines(X_grid,f_hat_grid, col = 'blue')
        #Sys.sleep(.8)
}

## Profile the S3 method train
library(microbenchmark)
time_comp <- microbenchmark(
        Rcode = train(X, y, learn_rate = .5, n_iter = 500, object.shallow_net),
        times = 100L
)
summary(time_comp)

##------------------------------------------------------------------------------
##  C++ Imputation 
# The matrix X passed has an intercept column (i.e. this is X_aug)
# The matrix A passed does NOT have an intercept column
Rcpp::cppFunction(
        " NumericVector compute_gradient_theta(NumericMatrix X,
        NumericVector f_hat,
        NumericVector y,
        NumericVector beta,
        NumericMatrix A) {
   int q = beta.size() - 1, p = X.ncol(), n = X.nrow(); // Compute q,p, and n
   
   NumericMatrix dL_dtheta(p, q); // Matrix with gradient of theta
   double sum_theta;
    for(int l = 0; l < q; l++){
      for(int j = 0; j < p; j++){
        sum_theta = 0;
        for(int i = 0; i < n; i++){
         sum_theta = sum_theta + (f_hat(i) - y(i))*A(i,l)*(1-A(i,l))*beta(l+1)*X(i,j);
        }
        dL_dtheta(j,l) = sum_theta/n;
     }
   }
   return dL_dtheta;
  }
"
)

# train_fast function with C++ code
train_fast <- function(X, y, learn_rate = 0.001, n_iter = 200, object)
{
        if ( !is( object, "shallow_net") ) 
                stop( "predict.shallow_net requires an object of class 'shallow_net'" )
        if ( !is( X, "matrix") ) 
                stop( "X requires an object of class 'matrix'" )
        
        beta = object$beta
        theta = object$theta
        
        q = ncol(theta)
        n = nrow(X)
        p = ncol(X)
        
        for (it in 1:n_iter)
        {
                if(it %% 1000 == 0) cat("Iter: ", it, "\n")
                
                # Forward pass
                X_aug = cbind(rep(1,n),X)
                A = sigmoid(X_aug %*% theta)
                A_aug = cbind(rep(1,n),A)
                f_hat = sigmoid(A_aug %*% beta)
                
                # Backward pass
                dL_dbeta = (1/n)*t(A_aug)%*%(f_hat - y)
                dL_dtheta = matrix(rep(NA, (p+1)*q), ncol = q)
                
                dL_dtheta <- compute_gradient_theta(X_aug, f_hat, y, beta, A)
                
                beta  = beta - learn_rate*dL_dbeta
                theta = theta - learn_rate*dL_dtheta
                
                # Check objective function value
                # print(loss_func(y,f_hat_func(X,theta,beta)))
                
        }
        
        out = list(theta = theta, beta = beta)
        class(out) <- 'shallow_net'
        
        return(out)
}

##------------------------------------------------------------------------------
# Check that `train` and `train_fast` give the same results by re-running the Example 1 and 2
# Compare performance with system time

## Example 1 

n = 100
p = 1
q = 4

set.seed(101)
X = as.matrix(runif(n, -2, 2))
y_prob = sigmoid(2 - 3*X^2)
y = rbinom(n,1,y_prob)

object.shallow_net = shallow_net(p, q)

# Forward pass

f_hat = predict(object.shallow_net, X)

# Plot prediction
X_grid = as.matrix(seq(-2,2,length.out = 100))
y_prob_grid = sigmoid(2 - 3*X_grid^2)
f_hat_grid = predict(object.shallow_net, X_grid)

plot(X,y, pch = 20)
lines(X_grid,y_prob_grid, col = 'red')
lines(X_grid,f_hat_grid, col = 'blue')

# Animation

for(s_it in 1:60)
{
        out_nn = train_fast(X, y, learn_rate = .3, n_iter = 100, object.shallow_net)
        
        ## Update the shallow_net parameters
        object.shallow_net = out_nn
        
        X_grid = as.matrix(seq(-2,2,length.out = 100))
        y_prob_grid = sigmoid(2 - 3*X_grid^2)
        f_hat_grid = predict(out_nn, X_grid)
        
        #plot(X,y, pch = 20)
        #lines(X_grid,y_prob_grid, col = 'red')
        #lines(X_grid,f_hat_grid, col = 'blue')
        #Sys.sleep(.8)
}

##------------------------------------------------------------------------------
## Example 2 

n = 200
p = 1
q = 8

set.seed(101)
X = as.matrix(runif(n, -2, 2))
y_prob = sigmoid(3 + X - 3*X^2 + 3*cos(4*X))
y = rbinom(n,1,y_prob)

object.shallow_net = shallow_net(p, q)

## Forward pass

f_hat = predict(object.shallow_net, X)

## Plot prediction
X_grid = as.matrix(seq(-2,2,length.out = 100))
y_prob_grid = sigmoid(3 + X_grid - 3*X_grid^2 + 3*cos(4*X_grid))
f_hat_grid = predict(object.shallow_net, X_grid)

plot(X,y, pch = 20)
lines(X_grid,y_prob_grid, col = 'red')
lines(X_grid,f_hat_grid, col = 'blue')

## Animation

for(s_it in 1:1000)
{
        out_nn = train_fast(X, y, learn_rate = .5, n_iter = 500, object.shallow_net)
        
        ## Update the shallow_net parameters
        object.shallow_net = out_nn
        
        X_grid = as.matrix(seq(-2,2,length.out = 100))
        y_prob_grid = sigmoid(3 + X_grid - 3*X_grid^2 + 3*cos(4*X_grid))
        f_hat_grid = predict(object.shallow_net, X_grid)
        
        #plot(X,y, pch = 20)
        #lines(X_grid,y_prob_grid, col = 'red')
        #lines(X_grid,f_hat_grid, col = 'blue')
        #Sys.sleep(.8)
}

time_com <- microbenchmark(
        Rcode = train(X, y, learn_rate = .5, n_iter = 500, object.shallow_net),
        Rcppcode = train_fast(X, y, learn_rate = .5, n_iter = 500, object.shallow_net),
        times = 100L
)
summary(time_com)