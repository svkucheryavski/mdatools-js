# Javascript library for statistics and multivariate data analysis

A simple library with implementation of most common methods for descriptive and inferential statistics as well as matrix operations, decompositions and projection based methods for multivariate data analysis. The library is currently under development, breaking changes may occur in the coming versions.

## What is new

<span style="color:crimson">Version **1.0.1** introduces many breaking changes as the library was almost fully re-written from the scratch. If you used a pre-release version (0.6.1), do not upgrade.</span>

The documentation below has been also re-written to match the new version.

## Matrices, vectors and indices

The values are represented using instances of the following classes:

* `Vector` is a class for representing sequence of values. The values are stored inside the class instances as `Float64Array`.
* `Matrix` is a class for representing numerical matrices (2D Arrays). The values are stored inside the class instances as `Float64Array`.
* `Index` is a class for representing vectors of indices — integer numbers specifying positions of values in vectors and matrices. The values are stored inside the class instances as `Int32Array`.
* `Factor` is a class for representing categorical variable, which is turned to array of labels for the categories (as strings) and array of indices (as `Uint8Array`).

These classes and their methods can be imported from `'mdatools/arrays'` module.

### Vectors of indices

The simplest way to create an instance of `Index` class is to use method `index(x)` where `x` is a conventional JavaScript array, e.g. `index([1, 2, 3])`. Alternatively index values can be generated as a sequence using static method `seq(start, end, by)`:

```javascript
import { Index } from 'mdatools/arrays';

const ind1 = Index.seq(1, 5); // returns index([1, 2, 3, 4, 5])
const ind2 = Index.seq(2, 10, 2); // returns index([2, 4, 6, 8, 10])
```

There are also two additional static methods for creating index vectors:

* `Index.fill(v, n)` — creates an index vector of size `n` filled with integer value `v`.
* `Index.ones(n)` — creates an index vector of size `n` filled with ones.

Values of any index vector can be replicated using class methods (here `ind` is an instance of class `Index`):

* `ind.rep(n)` — replicates the index vector `n` times.
* `ind.repeach(n)` — replicates each value from the index vector `n` times.
* `ind.sort([decreasing=false])` — sorts indices in `ind`.
* `ind.shuffle()` — shuffles indices in `ind` using Fisher–Yates algorithm.

All methods above return instance of `Index` as a result. It must be noted that when indices are used for subsetting values in vectors and matrices, it is expected that they start from 1, not 0.

Two (or more) vectors with indices, e.g. `i1` and `i2` can be concatenated into one by using method `i = c(i1, i2)`.

### Class Vector

The simplest way to create an instance of `Vector` class is to use method `vector(x)` where `x` is a conventional JavaScript array, e.g. `vector([1.1, 2.2, 3.3])`. The values can also be generated using the following static methods:

* `Vector.zeros(n)` — creates a vector of size `n` filled with zeros.
* `Vector.fill(v, n)` — creates a vector of size `n` filled with value `v`.
* `Vector.ones(n)` — creates a vector of size `n` filled with ones.
* `Vector.rand(n, [a=0], [b=1])` — creates a vector of size `n` filled with random values uniformly distributed between `a` and `b`.
* `Vector.randn(n, [mu=0], [sigma=1])` — creates a vector of size `n` filled with normally distributed random values.
* `Vector.seq(start, end, by)` — creates a sequence of values, similar to the `seq` method for `Index` class.
* `Vector.c(a, b, ...)` — concatenates any amount of numbers, arrays or/and vectors into a vector.

The `Vector` object object also has class methods `rep(n)` and `repeach(n)` which work similar to the methods for `Index` objects.

Similar to indices, several vectors can be concatenated into a single vector by using method `c`:

```javascript
import {vector, c} from 'mdatools/arrays';

const v1 = vector([1, 2, 3]);
const v2 = vector([10, 11, 12]);
const v3 = vector([20, 21, 22]);

const x = c(v1, v2, v3);
```

Other class methods are listed below.

#### Method for subsetting of vectors

The indices used for subsetting of vectors and matrices, must start with one, not zero. All subsetting methods create a copy of values (not a reference to original values).

* `x.subset(ind)` — creates a subset of vector `x` by taking values, whose position is specified by vector with indices `ind`.
* `x.slice(start, end)` — creates a subset of vector `x` by taking a sequence of values between the two positions (both are included to the subset).
* `x.copy()` — creates a deep copy of the vector.

#### Methods for manipulation with vector values

Following methods do something with vector values, e.g. arithmetic operations, transformations, etc.

* `x.apply(fun)` — applies function `fun` to every value of `x` and creates a new vector with values returned by the function.
* `x.add(y)` — adds values from another vector or a number.
* `x.subtract(y)` — subtracts values from another vector or a number.
* `x.mult(y)` — multiplies to values from another vector or to a number.
* `x.divide(y)` — divides by values from another vector or by a number.
* `x.dot(y)` — takes a dot (inner) product of two vectors (results in a number).
* `x.sort([decreasing=false])` — sorts values in `x`.
* `x.shuffle()` — shuffles values in `x` using Fisher–Yates algorithm.

### Class Factor

Factors are vectors representing categorical variables. To create an instance of `Factor` use method `factor()` and provide it an array of categories as strings:

```javascript
import {factor} from 'mdatools/arrays';

const f = factor(["red", "green", "red", "green", "red", "green", "blue]);
```

Factor has following class methods:

* `f.which(label)` — returns indices (as instance of `Index` class) of all entries of given label inside the factor.



### Methods for class Matrix

The simplest way to create an instance of `Matrix`  class is to use method `matrix(x, nrows, ncols)` where `x` is a conventional JavaScript array, e.g. `matrix([1, 2, 3, 4], 2, 2)`. The values can also be generated using the following static methods:

* `Matrix.zeros(nrows, ncols)` — creates a matrix filled with zeros.
* `Matrix.fill(v, nrows, ncols)` — creates a matrix filled with value `v`.
* `Matrix.ones(nrows, ncols)` — creates a matrix filled with ones.
* `Matrix.rand(nrows, ncols, [a=0], [b=1])` — creates a matrix filled with random values uniformly distributed between `a` and `b`.
* `Matrix.randn(nrows, ncols, [mu=0], [sigma=1])` — creates a matrix filled with normally distributed random values.
* `Matrix.eye(nrows, ncols)` — creates an identity matrix (can be rectangular).
* `Matrix.diagm(x)` — creates a squared matrix with main diagonal filled with values from vector `x`.
* `Matrix.outer(x, y, fun)` — creates a matrix by applying function `fun` to all possible pairs of values from vectors `x` and `y`.

One can also create a matrix by reshaping a vector as shown below.


```javascript
import { vector, reshape } from 'mdatools/arrays';

const x = vector([1, 3, 5, 7, 9, 11, 13, 15, 17]);
const Y = reshape(x, 3, 3);
```
Matrix can be created by merging/binding several vectors as rows (method `rbind()`) or as columns (method `cbind()`) of the matrix:

```javascript
import {vector, cbind, rbind} from 'mdatools/arrays';

const v1 = vector([1, 2, 3]);
const v2 = vector([10, 11, 12]);
const v3 = vector([20, 21, 22]);

const X1 = cbind(v1, v2, v3);
const X2 = rbind(v1, v2, v3);
```
Both methods can also bind matrices as well as vectors and matrices.

#### Method for subsetting of matrices

The indices in the methods below can be single numbers, conventional Javascript arrays or instances of Index class. They must start with one, not zero. All subsetting methods create a copy of values (not reference to original values).

* `X.subset(rind, cind)` — creates a subset of matrix `X` by taking values, whose position is specified by vector with row and column indices. If all rows or all columns must be taken just provide an empty array, `[]` as a value for index argument. Indices must start from 1.
* `X.copy()` — creates a deep copy of the matrix object `X`.
* `X.getcolumn(ic)` — returns values of `ic`-th column of  `X` as a vector (index must start from 1).
* `X.getrow(ir)` — returns values of `ir`-th  row of  `X` as a vector (index must start from 1).
* `X.diag()` — returns values from main diagonal of the matrix as a vector.

One can also replace some of the values in a matrix by providing another matrix with new values as well as row and column indices for the values to replace. In the example below we replace values for all odd rows and columns.

```javascript
import { matrix } from 'mdatools/arrays';

const X = matrix([1, 3, 5, 7, 9, 11, 13, 15, 17], 3, 3);
const Y = matrix([2, 4, 6, 8], 2, 2);

X.replace(Y, [1, 3], [1, 3]);
```

#### Methods for manipulation with matrix values

Following methods do something with matrix values, e.g. arithmetic operations, transformations, etc.

* `X.apply(fun, [dims=2])` — applies function `fun` to every row (if `dims=1`), every column (if `dims=2`) or every single value (if `dims=0`) of matrix  `X` and returns either a vector or a new matrix with values returned by the function.
* `X.t()` — transposes matrix `X` (swaps columns and rows).
* `X.inv()` — computes inverse of squared matrix `X` (if exists).

The following methods implement element-wise arithmetic operations with matrices, a matrix and a vector and a matrix and a number.

* `X.add(Y, [dims=2])` — adds values from another matrix, a vector or a number.
* `X.subtract(Y, [dims=2])` — subtracts values from another matrix, a vector or a number.
* `X.mult(Y, [dims=2])` — multiplies to values from another matrix, a vector or to a number.
* `X.divide(Y, [dims=2])` — divides by values from another matrix, a vector or from by a number.
* `X.dot(Y, [dims=2])` — takes a dot product of two matrices or a matrix and a vector.

In case if the argument is a vector its length must match one of the matrix dimension (number of rows or columns) and you need to specify this dimension explicitly by providing second argument, `dims`. In the example below we subtract mean from every column of matrix `X` (the statistical functions are introduced later in this document).

```javascript
import { matrix } from 'mdatools/arrays';
import { mean } from 'mdatools/stat';

// create matrix X
const X = matrix([1, 3, 5, 7, 9, 11, 13, 15, 17], 3, 3);

// compute mean of every column of X
const m = X.apply(mean, 2);

// mean center columns of X
const Y = X.subtract(m, 2);
```

Following methods are implemented as a standalone methods (not class methods):

* `crossprod(X, Y)` — computes a cross-product of two matrices as $\mathbf{X}^\textrm{T}\mathbf{Y}$.
* `tcrossprod(X, Y)` — computes a cross-product of two matrices as $\mathbf{XY}^\textrm{T}$.

#### Additional methods for matrices

There is also a set of methods which check property of a matrix:

* `X.islowertriangular()` — returns `true` if `X` is lower triangular matrix.
* `X.isuppertriangular()` — returns `true` if `X` is upper triangular matrix.

You can also convert a matrix to a string which looks nice if you want to show/check the
matrix values by e.g. `console.log()`:

* `X.toString(ndec)` — here `ndec` is number of decimals to show in the output.

Finally there are two methods which can be helpful when it is necessary to read a matrix values from a CSV file or save the values as CSV.

Class method `X.toCSV([sep=","], [dec="."], [header=[]], [labels=[]])` — generates a string which can be saved as CSV file. The parameters are: `sep`: symbol to use for separating values, `dec`: symbol to use for separating decimals, `header`: a conventional Javascript array with column names (variable names), `labels`: a conventional Javascript array with row names (observation labels).

Static method `Matrix.parseCSV(str, sep, hasHeader, hasLabels)` — parses a string, which is a result of reading a CSV file, into a JSON. The parameters are: `str`: string with CSV file content with rows delimited by `\n` or `\r\n`, `sep`: symbol to use for separating values, `hasHeader`: logical, does the data have header or not, `hasLabels`: logical, does the data have labels or not. The method returns a JSON with matrix values, symbols for values and decimal separators, arrays with header and labels.

## Descriptive statistics

The following methods can be used to compute most common statistics (or a vector with statistics). The methods can work with both conventional Javascript arrays and instances of class `Vector` although it is recommended to provide the latter in any case.

Methods can be imported from `'mdatools/stat'` module.

### Methods for computing single statistics

* `min(x)` — returns the smallest value in vector `x`.
* `minind(x)` — returns index (position) of the smallest value in vector `x`.
* `max(x)` — returns the largest value in vector `x`.
* `maxind(x)` — returns index (position) of the largest value in vector `x`. 
* `prod(x)` — returns a product of all values from vector `x`.
* `sum(x)` — returns a sum of all values from vector `x`.
* `mean(x)` — returns mean (average) of values from vector `x`.
* `variance(x, [biased=false])` — returns variance of values from vector `x`.
* `sd(x, [biased=false])` — returns standard deviation of values from vector `x`.
* `ssq(x)` — returns a sum of squared values from vector `x`.
* `norm2(x)` — returns a Eucledian norm (length) of vector `x`.
* `skewness(x)` — returns skewness of values from vector `x`.
* `kurtosis(x)` — returns kurtosis of values from vector `x`.
* `cor(x, y, method='pearson')` — returns a correlation ("pearson" or "spearman") of values from the two vectors.
* `cov(x, y)` — returns a covariance ("pearson" or "spearman") of values from the two vectors.


### Methods for computing vectors of statistics

* `range(x)` — returns an array with two values, smallest and largest.
* `mrange(x, m)` — computes range with margin, `m` is how much range must be stretched (e.g. `0.1` for 10%).
* `rank(x)` — returns rank (position) for every value from vector `x`.
* `cumsum(x)` — returns a cumulative sum of the values from vector `x`.
* `ppoints(n)` — computes `n` probabilities
* `diff(x)` — returns difference between the adjacent values from vector `x`.
* `mids(x)` — returns middle points for each pair of values from vector `x`.
* `split(x, n)` — splits the range of values in `x` into equal intervals (bins).
* `count(x, s)` — counts how many values from vector `x`, falls into each interval from `s`.
* `quantile(x, p)` — returns `p`-th quantiles computed based on values from vector `x`.

## Distributions

The following methods implement Probability Density Function (PDF), Cumulative Distribution Function (CDF) and Inverse Cumulative Distribution Function (ICDF) a.k.a. quantile function for several known distributions. For uniform and normal distributions there are also functions which generate random numbers.

Methods can be imported from `'mdatools/distributions'` module.


* `dnorm(x, mu, sigma)`, `pnorm(x, mu, sigma)`, `qnorm(p, mu, sigma)`, `rnorm(x, mu, sigma)` — PDF, CDF, ICDF, and random numbers generator for normal distribution. Default values for the paramaters are `mu=0`, `sigma=1`.
* `dunif(x, a, b)`, `punif(x, a, b)`, `runif(x, a, b)` — PDF, CDF, and random numbers generator for uniform distribution. Default values for the paramaters are `a=0`, `b=1`.
* `dt(t, dof)`, `pt(x, mu, sigma)`, `qt(p, mu, sigma)` — PDF, CDF, abd ICDF for Student's t-distribution.
* `df(F, dof1, dof2)`, `pf(F, dof1, dof2)` — PDF and CDF for F-distribution.
* `pchisq(x, dof)`, `qchisq(p, dof)` — CDF and ICDF function for chi-square distribution.

There are also a set of helper functions used to compute the values for the distributions, which can be useful:

* `erf(x)` — error function for normal distribution.
* `beta(x, y)` — Beta function (approximation via numerical integration).
* `gamma(z)` — Gamma function (approximation).
* `ibeta(x, a, b)` — incomplete Beta function (approximation via numerical integration).

Because the distribution values are computed using approximated functions they are not very accurate. The accuracy achieved in tests is about 10<sup>-5</sup> comparing to implementation of the corresponding methods in e.g. R.

## Additional methods useful in statistics

Methods can be imported from `'mdatools/misc'` module.

* `closestind(x, a)` — finds index of a value from vector `x` which is the closest to value `a`.
* `integrate(f, a, b)` — computes an integral of function `f` with limits `a`and `b`.
* `getoutliers(x)` — returns values from `x`, lying beyond the 1.5IQR distance from the first and the third quartiles.
* `expandgrid(...args)` — generates all possible combinations of values from two or more vectors provided as arguments to the function. Values must be unique.

## Hypothesis testing

Methods can be imported from `'mdatools/tests'` module.

### Student's t-test for means

The tests compute the observed effect, standard error, t- and p-values as well
as confidence interval and return everything as JSON. The two-sample t-test assumes
that population variances are equal. Confidence intervals always computed around
the observed effect and for both tails.

* `ttest(x, mu=0, alpha=0.05, tail='both')` — one sample t-test
* `ttest2(x, y, alpha=0.05, tail='both')` — two sample t-test

The `x` and `y` arguments must be instances of `Vector` class.


## Decompositions

Functions can be imported from `'mdatools/decomp'` module.

Methods for computing decomposition of matrices and related methods (e.g. inverse).

* `qr(X)` — computes QR decomposition of `X` using Householder reflections.
* `lu(X)` — computes LU decomposition of `X` using Givens rotations.
* `svd(X)` — computes SVD decomposition of `X` using Golub-Reinsch bidiagonalization.
* `rsvd(X)` — randomized version of SVD which is must faster than the original method.

## Preprocessing

Functions can be imported from `'mdatools/prep'`.

Methods for preprocessing of data values. Every method takes a matrix as a main argument and returns a matrix as a result (plus potentially other outcomes).

* `scale(X, [center=true], [scale=false])` — center and scale each column of matrix `X`. If values for `center` and `scale` arguments are logical, the method will do standardization (using mean values for centring and standard deviation for scaling). Alternatively the values for these two arguments can be specified by user as a vector of values. Number of values must be equal to the number of columns in `X`.
* `unscale(X, centerValues, scaleValues)` — does the inverse to `scale()` operation using vectors with values used to center and scale the data.

More methods are coming in future versions.

## Modelling

Functions can be imported from `'mdatools/models'`.

Methods for fitting various models. Every method returns a JSON with fitted model parameters, their inference (where it can be done). The prediction methods contain the predicted values as well as performance statistics (when it is possible to compute).

### Simple and multiple linear regression
* `lmfit(X, y)` — fits linear regression model (simple or multiple).
* `lmpredict(m, X)` — computes predictions using model from `lmfit()` and vector or matrix with predictors.


### Polynomial regression
* `polyfit(x, y, d)` — polynomial regression model (univariate).
* `polypredict(m, x)` — computes predictions using polynomial model from `polyfit()`.

### Principal component anaylsis

* `pcafit(X, ncomp)` — fits PCA model with given number of components.
* `pcapredict(m, X)` — projects data from `X` to the PCA model and computes main outcomes (scores, distances, variance, etc.).

## DD-SIMCA classification

DD-SIMCA classification is done based on PCA model with additional parameters.

* `getsimcaparams(className, alpha, limType)` — creates an object with SIMCA parameters, necessary for classification.
* `simcapredict(m, params, X, [cRef])` — classifies rows of data matrix `X` using PCA model `m` and DD-SIMCA parameters `params`. If factor with reference classes, `cRef` is provided, the method also computes classification statistics and adds it to he outcome.

### Principal components regression

* `pcrfit(X, Y, ncomp)` — fits PCR model with given number of components.
* `pcrpredict(m, X, [Y])` — projects data from `X` to the PCR model and computes main outcomes (predicted y-values for each component, scores, distances, variances and other performance statistics).
*
### Partial least squares regression

* `plsfit(X, Y, ncomp)` — fits PLS1 model with given number of components.
* `plspredict(m, X, [Y])` — projects data from `X` to the PLS model and computes main outcomes (predicted y-values for each component, scores, distances, variances and other performance statistics).



