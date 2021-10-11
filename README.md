# Javascript package for statistic analysis and data manipulations

A simple library for statistic analysis and manipulations with single vector values. The function syntax is similar to corresponding functions in R.

## List of functions

In all functions variables named `x` and `y` are expected to be a vector with numbers (`number[]`).

### Computing statistics

Following functions compute different statistics for one or two vectors and result in a single value.

* `min(x)` — smallest value in a vector.
* `max(x)` — largest value in a vector.
* `sum(x)` — sum of all values in a vector.
* `mean(x)` — mean (average) value.
* `sd(x, biased = false)` — standard deviation.
* `quantile(x, p)` — p-th quantile (0 > p > 1).
* `skewness(x)` — skewness (measure of symmetry)
* `kurtosis(x)` — kurtosis (measure of tailedness)

Following functions compute and return a vector with statistics.

* `range(x)` — a vector with smallest and largest values.
* `mrange(x, margin)` — similar to `range()` but with margins on both sides of the interval.
* `split(x, n)` — split a range of values from `x` into `n` equal intervals.
* `count(x, bins`) — counts how many values from `x` falls into bins defined by `bins`.
* `mids(x)` — return a vector with middle points between the adjacent values from `x`.
* `diff(x)` — return a vector with difference between adjacent values (e.g. `x[1] - x[0]`, ...).
* `getOutliers(x, Q1, Q3)` — finds outliers in `x` based on 1.5 IQR rule (like in boxplots).
* `seq()` !!!
* `ppoints()` !!!

Not implemented yet
* `rank(x)` — return vector with ranks of values from `x`.
* `cov(x, y)` — compute covariance between `x` and `y`.
* `cor(x, y, type = 'pearson')` — compute correlation between `x` and `y`.

## Manipulations with values in a vector

* `sort(x, decreasing = false)` — sort values in `x`.

Not implemented yet
* `subset(x, ind`)` — returns a subset of x, defined by indices or by vector with logical values.
* `gt(x, value)`

### Theoretical distributions

The package has support for several known theoretical distributions. Every distribution is represented by four functions `d*` for computing density, `p*` for computing probability, `q*` for computing quantiles and `r*` for generating random numbers. The following distributions are supported:

#### Uniform distribution
* `dunif(x, a = 0, b = 1)` — `x` is a vector of values, `a` and `b` — distribution parameters.
* `punif(x, a = 0, b = 1)` — same as for `dunif`
!!!* `qunif(p, a = 0, b = 1)` — `p` is a probability for the needed quantile, can be a vector.
* `runif(n, a = 0, b = 1)` — `n` is a number of values to generate.

#### Normal distribution
* `dnorm(x, mean = 0, sd = 1)`
* `pnorm(x, mean = 0, sd = 1)`
!!!* `qnorm(p, mean = 0, sd = 1)`
* `rnorm(n, mean = 0, sd = 1)`

