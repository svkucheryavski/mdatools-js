# Javascript library for statistic analysis and data manipulations

A simple library for statistic analysis and manipulations with single vector values. Most of the functions work similar to statistical functions in R. The function names use camel case convention, e.g. R function `expand.grid()` has name `expandGrid()` in this library.

The library is currently under development, breaking changes may occur in the coming versions.

## List of functions

If nothing specific is written, then in all functions variables named `x` and `y` are expected to be vectors with numbers (`number[]`).

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

Following functions compute and return either single or a vector with statistics.

* `quantile(x, p)` — p-th quantile, `0 > p > 1`, if `p` is a vector, it returns a vector with quantiles.
* `range(x)` — a vector with smallest (min) and largest (max) values.
* `mrange(x, margin)` — similar to `range()` but with margins on both sides of the interval.
* `split(x, n)` — split a range of values from `x` into `n` equal intervals.
* `count(x, bins`) — counts how many values from `x` falls into bins defined by `bins`.
* `mids(x)` — return a vector with middle points between the adjacent values from `x`.
* `diff(x)` — return a vector with differences between adjacent values (e.g. `x[1] - x[0]`, ...).
* `getOutliers(x, Q1, Q3)` — finds outliers in `x` based on 1.5 IQR rule (like in boxplots).
* `seq(a, b, n)` — creates a sequence of `n` values equally distanced in interval [a, b].
* `ppoints(n)` — generate probability points for QQ plot.

**Not implemented yet**

* `rank(x)` — return vector with ranks of values from `x`.
* `cov(x, y)` — compute covariance between `x` and `y`.
* `cor(x, y, type = 'pearson')` — compute correlation between `x` and `y`.

## Manipulations with values

* `sort(x, decreasing = false)` — sort values in `x`.
* `subset(x, ind)` — returns a subset of `x`, defined by vector of indices `ind`. Indices start with 1.
* `rep(x, n)` — replicates vector `x` `n` times. If `n` is a vector of the same length as `x` then it is assumed that it contains separate number of replicates for each value in `x`.
* `expandGrid(...args)` — generates all combinations of all values of the provided vectors.

### Theoretical distributions

The package has support for several known theoretical distributions. Every distribution is represented by four functions `d*` for computing density, `p*` for computing probability, and `r*` for generating random numbers. The following distributions are supported:

#### Uniform distribution
* `dunif(x, a = 0, b = 1)` — `x` is a vector of values, `a` and `b` — distribution parameters.
* `punif(x, a = 0, b = 1)` — same as for `dunif`
* `runif(n, a = 0, b = 1)` — `n` is a number of values to generate.

#### Normal distribution
* `dnorm(x, mean = 0, sd = 1)`
* `pnorm(x, mean = 0, sd = 1)`
* `rnorm(n, mean = 0, sd = 1)`

