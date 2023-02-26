## S3 method shallow neural network 

rm(list=ls())
# shallow_net
# returns initialized parameters theta, beta
shallow_net <- function(p,q) {
        if (p<1) stop("'p' needs to be >= 1" )
        if (q<1) stop("'q' needs to be >= 1" )
        
        parameter <- list(theta = replicate(q, 0.1*runif(p+1, -.5, .5)), 
                          beta = 0.1*runif(q+1, -.5, .5))
        attr(parameter, "class") <- "shallow_net"
        parameter
}
# sigmoid function
sigmoid =  function(x){
        1/(1+exp(-x))
}
# generic predict function
predict <- function(object, ...) UseMethod("predict")
#  shallow_net
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

# train function for shallow_net
train <- function(X, y, learn_rate = 0.001, n_iter = 200, object){
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
                
        }
        
        out = list(theta = theta, beta = beta)
        class(out) <- 'shallow_net'
        
        return(out)
}

## -----------------------------------------------------------------------------
## Example 1
n = 100
p = 1
q = 4
set.seed(1)
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
        out_nn = train(X, y, learn_rate = .3, n_iter = 100, object.shallow_net)
        
        ## Update the shallow_net parameters
        object.shallow_net = out_nn
        
        X_grid = as.matrix(seq(-2,2,length.out = 100))
        y_prob_grid = sigmoid(2 - 3*X_grid^2)
        f_hat_grid = predict(out_nn, X_grid)
        
        plot(X,y, pch = 20)
        lines(X_grid,y_prob_grid, col = 'red')
        lines(X_grid,f_hat_grid, col = 'blue')
        Sys.sleep(.8)
}

## -----------------------------------------------------------------------------
## Example 2 
n = 200
p = 1
q = 8
set.seed(1)
X = as.matrix(runif(n, -2, 2))
y_prob = sigmoid(3 + X - 3*X^2 + 3*cos(4*X))
y = rbinom(n,1,y_prob)
object.shallow_net = shallow_net(p, q)
# Forward pass
f_hat = predict(object.shallow_net, X)

# Plot prediction
X_grid = as.matrix(seq(-2,2,length.out = 100))
y_prob_grid = sigmoid(3 + X_grid - 3*X_grid^2 + 3*cos(4*X_grid))
f_hat_grid = predict(object.shallow_net, X_grid)
plot(X,y, pch = 20)
lines(X_grid,y_prob_grid, col = 'red')
lines(X_grid,f_hat_grid, col = 'blue')
# Animation
for(s_it in 1:1000)
{
        out_nn = train(X, y, learn_rate = .5, n_iter = 500, object.shallow_net)
        
        ## Update the shallow_net parameters
        object.shallow_net = out_nn
        
        X_grid = as.matrix(seq(-2,2,length.out = 100))
        y_prob_grid = sigmoid(3 + X_grid - 3*X_grid^2 + 3*cos(4*X_grid))
        f_hat_grid = predict(object.shallow_net, X_grid)
        
        plot(X,y, pch = 20)
        lines(X_grid,y_prob_grid, col = 'red')
        lines(X_grid,f_hat_grid, col = 'blue')
        Sys.sleep(.8)
}
