# Javascript library for statistics and multivariate data analysis

A simple library with a set of most common methods for descriptive and inferential statistics as well as matrix operations and projection based methods for multivariate data analysis. The library is currently under development, breaking changes may occur in the coming versions.

## Last release (0.6.1)

* `diag()` method, creating a diagonal matrix from vector was renamed to `diagm()`
* `getdiag()` method, returning a vector with diagonal elements was renamed to `diag()`
* improvements to `inv()` method
* small bug fixes

## Descriptive and inferential statistics

Functions can be imported from `'mdatools/stat'`.

If nothing specific is written, then in all functions variables named `x` and `y` are expected to be vectors with numbers (`number[]`).

### Quick example

```javascript
export {rnorm, mean, sd} from "mdatools/stat";

// generate vector of n = 10 random numbers from N(µ = 5, σ = 2)
const x = rnorm(10, 5, 2);

// compute mean and standard deviation
const m = mean(x);
const s = sd(x);

// show both statistics
console.log([m, s]);
```

### Computing statistics

Following functions compute single statistic for one or two vectors.

* `min(x)` — smallest value in a vector.
* `max(x)` — largest value in a vector.
* `sum(x)` — sum of all values in a vector.
* `prod(x)` — product of all values in a vector.
* `mean(x)` — mean (average) value.
* `sd(x, biased = false)` — standard deviation.
* `skewness(x)` — skewness (measure of symmetry)
* `kurtosis(x)` — kurtosis (measure of tailedness)
* `cov(x, y)` — covariance between `x` and `y`.
* `cor(x, y, type = 'pearson')` — correlation between `x` and `y`.

Following functions compute and return either single statistic or a vector with statistics.

* `quantile(x, p)` — computes p-th quantile, `0 > p > 1`.
* `range(x)` — returns a vector with smallest (min) and largest (max) values.
* `mrange(x, margin)` — similar to `range()` but with margins on both sides of the interval.
* `split(x, n)` — splits a range of values from `x` into `n` equal intervals.
* `count(x, bins`) — counts how many values from `x` fall into bins defined by `bins`.
* `mids(x)` — returns a vector with middle points between the adjacent values from `x`.
* `diff(x)` — returns a vector with differences between the adjacent values from `x`.
* `getOutliers(x, Q1, Q3)` — finds outliers in `x` based on 1.5 IQR rule (like in boxplot).
* `ppoints(n)` — computes probability points for QQ plot.
* `rank(x)` — returns a vector with ranks of values from `x`.
* `cumsum(x)` — computes cumulative sum of a vector of values.
* `scale(x, center, scale)` — center and scale values from `x` (e.g. standardize).

### Manipulations with values

* `seq(a, b, n)` — creates a sequence of `n` values equally distanced in interval [a, b].
* `rep(x, n)` — replicates values in a vector `x` `n` times (`n` can be a vector).
* `sort(x, decreasing = false)` — sorts values in `x`.
* `subset(x, ind, method)` — returns a subset of `x`, by selecting values specified vector of indices `ind` (if `method="select"` or not specified) or by removing the values (if `method="remove"`).
* `shuffle(x)` — shuffles values in `x` using Fisher–Yates algorithm.
* `round(x, n)` — rounds `x` to `n` decimals.
* `expandGrid(...args)` — generates all combinations of vector values (full factorial design).

### Theoretical distributions

The package has support for several known theoretical distributions. Every distribution is represented by four functions `d*` for computing density, `p*` for computing probability, `q*` for computing quantiles and `r*` for generating random numbers. The following distributions are supported:

#### Uniform distribution
* `dunif(x, a = 0, b = 1)` — `x` is a vector of values, `a` and `b` — distribution parameters.
* `punif(x, a = 0, b = 1)` — same parameters as for `dunif`
* `runif(n, a = 0, b = 1)` — `n` is a number of values to generate.

#### Normal distribution
* `dnorm(x, mean = 0, sd = 1)`
* `pnorm(x, mean = 0, sd = 1)`
* `qnorm(p, mean = 0, sd = 1)`
* `rnorm(n, mean = 0, sd = 1)`

#### Student's t-distribution
* `dt(t, dof)`
* `pt(t, dof)`
* `qt(p, dof)`

#### F-distribution
* `df(F, d1, d2)`
* `pf(F, d1, d2)`

### Hypothesis testing

#### Student t-test for means

The tests compute the observed effect, standard error, t- and p-values as well
as confidence interval and return everything as JSON. The two-sample t-test assumes
that population variances are equal. Confidence intervals always computed around
the observed effect and for both tails.

* `tTest1(x, mu = 0, alpha = 0.05, tail = "both")` — one sample t-test
* `tTest2(x, y, alpha = 0.05, tail = "both")` — two sample t-test


### Additional functions (helpers)

* `integrate(f, a, b)` — numeric integration of function `f` with limits `(a, b)`.
* `gamma(z)` — Lanczos approximation of the Gamma function.
* `beta(x, y)` — Beta function computed via approximation of Gamma function.
* `ibeta(x, a, b)` — Standardized incomplete Beta function (a.k.a. I<sub>x</sub>(a, b)).


## Matrix operations

Functions can be imported from `'mdatools/matrix'`.

The library operates with vectors (1D Array, e.g. `x = [1, 2, 3]`) and matrices (2D Arrays or Arrays of Arrays of equal length, e.g. `X = [[1, 2, 3], [4, 5, 6]]`). Any vector is considered as a column vector, so `x` from the example above will have a dimension of 1 column and 3 rows. You can think of a matrix as a vector of vectors, e.g. `X` from the example above will have a dimension of 2 columns and 3 rows.

### Quick example

```javascript
export {crossprod, tomatrix} from "mdatools/matrix";

// create a matrix with 2 columns and 3 rows manually
const X = [[1, 2, 3], [4, 5, 6]];

// create matrix with 2 columns and 3 rows from a vector (sequence)
const Y = tomatrix(seq(1, 6), 3, 2);

// compute X'Y
const Z = crossprod(X, Y);

// show the result
// must be [ [ 14, 32 ], [ 32, 77 ] ]
console.log(Z);
```

### Generation of matrices

* `matrix(n, m, a)` — creates a matrix with `n` rows and `m` columns and fills it with a value `a`.
* `zeros(n, m)` — creates a matrix with `n` rows and `m` columns and fills it with zeros.
* `diagm(x)` — creates a squared matrix and fills diagonal elements with values from vector `x`.
* `eye(n)` — creates an identity matrix of size `n`.
* `tomatrix(x, n, m)` — creates a matrix with `n` rows and `m` columns from a vector `x`.
* `diag(x)` — returns a vector with elements from main diagonal of matrix `x`.

The `tomatrix(x, n, m)` works as follows. If `x` has the same number of elements as number of rows, it replicates the vector column wise, so every column of the matrix will have values from the vector. If `x` has the same number of elements as number of columns — it does replication row wise. Otherwise it expects `x` to have length `n * m` and simply reshape the values into a matrix (column wise).

### Checking or computing properties of vectors and matrices
* `nrow(X)` — returns number of elements in a vector or number of rows in a matrix.
* `ncol(X)` — returns 1 for a vector or number of columns in a matrix.
* `vnorm2(x)` — computes 2-norm (Euclidean norm) of a vector.
* `ismatrix(X)` — returns true if `X` is a matrix (2D Array).
* `isvector(X)` — returns true if `X` is a vector (1D Array).
* `isarray(X)` — returns true if `X` is an Array (shortcut for `Array.isArray()`).
* `issquaredmat(X)` — returns true if `X` is a squared matrix.
* `islowertrianmat(X)` — returns true if `X` is a lower triangular matrix.
* `isuppertrianmat(X)` — returns true if `X` is a lower triangular matrix.

### Manipulations with matrices and vectors
* `cbind(X, Y)` — binds (merge) `X` and `Y` column wise.
* `rbind(X, Y)` — binds (merge) `X` and `Y` row wise.
* `msubset(X, rowInd, colInd, method)` — subset matrix `X`by selecting or removing rows and columns.
* `mreplace(X, Y, rowInd, colInd)` — replaces values in matrix `X` specified by row and column indices by values from matrix `Y`.
* `vreplace(x, y, ind)` — replaces values in vector `x` specified by indices by values from vector `y`.

### Simple operations with vectors
* `vadd(x, y)` — element wise addition of two vectors, or a vector and a scalar.
* `vsubtract(x, y)` — element wise subtraction of two vectors, or a vector and a scalar.
* `vmult(x, y)` — element wise multiplication of two vectors, or a vector and a scalar.
* `vdiv(x, y)` — element wise division of two vectors, or a vector and a scalar.
* `vapply(x, fun)` — applies function `fun` to every element of the vector (shortcut for `Array.map()`).
* `vdot(x, y)` — computes a dot product of two vectors.

### Simple operations with matrices
* `transpose(X)` — transposition of a matrix or a vector.
* `madd(X, Y)` — element wise sum of two matrices, a matrix and a scalar or a matrix and a vector.
* `msubtract(X, Y)` — element wise difference of two matrices, a matrix and a scalar or a matrix and a vector.
* `mmult(X, Y)` — element wise product of two matrices, a matrix and a scalar or a matrix and a vector.
* `mdivide(X, Y)` — element wise division of two matrices, a matrix and a scalar or a matrix and a vector.
* `mdot(X, Y)` — inner product of two matrices (as dot products of rows of `X` and columns of `Y`).
* `crossprod(X, Y)` — computes inner product of X'Y.
* `tcrossprod(X, Y)` — computes inner product of XY'.


For element wise operations of a matrix and a vector, functions check how many elements the vector has. If vector has the same number of elements as number of rows in the matrix the operation will
me carried out column wise. If vector has the same number of elements as number of columns in the matrix, the operation will be applied to every row of the matrix.


## Decompositions

Functions can be imported from `'mdatools/decomp'`.

Methods for computing decomposition of matrices and related methods (e.g. inverse).

* `qr(X)` — create QR decomposition of `X` using Householder reflections.
* `inv(X)` — computes inverse of squared matrix `X` using QR decomposition.

More methods are coming.

## Modelling

Functions can be imported from `'mdatools/models'`.

Methods for fitting various models. Every method returns a JSON with fitted model parameters, their inference (where it can be done), as well as performance statistics for a trained data.

### Regression
* `lmfit(X, y)` — linear regression model (simple or multiple).
* `lmpredict(m, X)` — computes predictions using MLR model from `lmfit()` and vector or matrix with predictors.
* `polyfit(x, y, d)` — polynomial regression model (univariate).
* `polypredict(m, x)` — computes predictions using polynomial model from `polyfit()`.
