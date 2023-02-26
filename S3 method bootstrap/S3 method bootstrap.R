## S3 method bootstrap

bootstrap <- function(object, ...) UseMethod("bootstrap")

bootstrap.numeric <- function(object, nboot, stat, ...){ 
        if (!is( object, "numeric"))
                stop( "bootstrap.numeric must be 'numeric'" ) 
        if (nboot < 1 | is.infinite(nboot))
                stop( "'nboot' should be a positive integer" ) 
        if (!is( stat, "function"))
                stop( "bootstrap.numeric requires 'stat' of class 'function'" )
        n <- length(object)
        purrr::map_dbl(seq(nboot), function(x) stat(sample(object, size=n, replace=TRUE), ...)) 
}

# Define function moment
moment <- function(x, k) {
        (1/length(x))*sum((x-mean(x))^k)
}

x <- rnorm(100,5,1)
par(mfrow=c(1,4))
hist(bootstrap(x, 1000, mean), main="Mean", xlab="")
hist(bootstrap(x, 100, median), main="Median", xlab="")
hist(bootstrap(x, 100, sd), main="SD", xlab="")
hist(bootstrap(x, 100, moment, k = 2), main="Second moment", xlab="")

# Construct function for stratified objects
stratified <- function(y, strata) {
        if (!is.numeric(y)) stop("'y' must be numeric")
        if (!is.factor(strata)) stop("'strata' must be a factor")
        if (length(y) != length(strata)) stop("'y' and 'strata' must have equal length")
        structure(list(y=y, strata=strata), class = "stratified")
}

# bootstrap method for stratified objects
bootstrap.stratified <- function(object, nboot, stat, ...){ 
        if ( !is( object, "stratified") )
                stop( "bootstrap.stratified requires an object of class 'stratified'" ) 
        if ( nboot < 1 | is.infinite(nboot) )
                stop( "'nboot' should be a positive integer" ) 
        if ( !is( stat, "function") )
                stop( "bootstrap.numeric requires 'stat' of class 'function'" )
        
        stat_with_args = function(x) stat(x, ...)
        tapply(object$y, object$strata, bootstrap.numeric, nboot, stat_with_args)
}

# Visualization 
my_str_samp <- stratified(y = c(rgamma(50,3), rnorm(100, 30)),
                          strata = factor(c(rep("a",50),rep("b",100))))
bootstrap(my_str_samp, 10, mean)
bootstrap(my_str_samp, 10, median)
bootstrap(my_str_samp, 10, sd)
bootstrap(my_str_samp, 10, moment, 2)
