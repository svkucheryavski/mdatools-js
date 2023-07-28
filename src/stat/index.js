/*************************************************/
/*    Methods for computing statistics           */
/*************************************************/

import { isnumber, isindex, isvector, vector, Vector, _sort } from '../arrays/index.js';


/**
 * Compute median of vector values.
 *
 * @param {Array|Vector} x - vector with values.
 *
 * @returns {number} median of x.
 *
 */
export function median(x) {
   return quantile(x, 0.5)
}

/**
 * Compute inter-quartile range for vector of values.
 *
 * @param {Array|Vector} x - vector with values.
 *
 * @returns {number} IQR of x (Q3 - Q1).
 *
 */
export function iqr(x) {
   return quantile(x, 0.75) - quantile(x, 0.25);
}


/**
 * Computes a p-th quantile/quantiles for a numeric vector.
 *
 * @param {Vector} x - vector with values.
 * @param {number|Array|Vector} p - probability (one value or a vector).
 *
 * @returns {number|Vector} quantile value or a vector with quantiles.
 */
export function quantile(x, p) {

   if (isvector(x)) {
      return quantile(x.v, p);
   }

   if (isvector(p)) {
      return quantile(x, p.v);
   }

   const n = x.length;

   if (!Array.isArray(p)) p = [p];

   if (!isnumber(p[0]) || min(p) < 0 || max(p) > 1) {
      throw new Error("Parameter 'p' must be between 0 and 1 (both included).");
   }

   function q(x, p) {
      const h = (n - 1) * p + 1;
      const n1 = Math.floor(h);
      const n2 = Math.ceil(h);
      return x[n1 - 1] + (x[n2 - 1] - x[n1 - 1]) * (h - Math.floor(h));
   }

   const xs = _sort(x);
   const out =  p.map(v => q(xs, v));
   return p.length == 1 ? out[0] : vector(out);
}


/**
 * Counts how many values from a vector falls into provided intervals (bins).
 *
 * @param {Array|Vector} x - vector with values.
 * @param {Array|Vector} bins - vector with bins boundaries.
 *
 * @returns {Vector} vector with counts for each bean.
 *
 */
export function count(x, bins) {

   if (isvector(x)) {
      return count(x.v, isvector(bins) ? bins.v : bins);
   }

   const n = bins.length;

   // add a bit extra to right side of the last bin
   bins[n - 1] = bins[n - 1] * 1.0001

   // count
   let counts = new Vector.valuesConstructor(n - 1);
   for (let i = 0; i < x.length; i++) {
      for (let j = 0; j < n - 1; j++) {
         if (x[i] >= bins[j] && x[i] < bins[j + 1]) counts[j] += 1;
      }
   }

   return new Vector(counts);
}


/**
 * Computes middle points between values of a vector.
 *
 * @param {Array|Vector} x - vector with values.
 *
 * @returns {Vector} vector with middle points.
 *
 */
export function mids(x) {

   if (isvector(x)) {
      return mids(x.v);
   }

   const out = new Vector.valuesConstructor(x.length - 1);
   for (let i = 0; i < out.length; i++) {
      out[i] = 0.5 * (x[i] + x[i + 1]);
   }

   return new Vector(out);
}


/**
 * Splits range of vector values into equal intervals.
 *
 * @param {Array|Vector} x - vector with values.
 * @param {number} n - number of intervals.
 *
 * @returns {Vector} vector with boundaries of the intervals.
 *
 */
export function split(x, n) {

   if (isvector(x)) {
      split(x.v, n);
   }

   const rn = range(x);

   if (rn[0] === rn[1]) {
      throw new Error('split: values in a vector "x" should vary.');
   }

   const by = (rn[1] - rn[0]) / n;
   return Vector.seq(rn[0], rn[1], by);
}


/**
 * Computes difference between all adjacent values in a vector.
 *
 * @param {Array|Vector} x - vector with values.
 *
 * @returns {Vector} vector with the differences.
 *
 */
export function diff(x) {

   if (isvector(x)) {
      return diff(x.v);
   }

   const out = new Vector.valuesConstructor(x.length - 1);
   for (let i = 0; i < x.length - 1; i++) {
      out[i] = x[i + 1] - x[i];
   }

   return new Vector(out);
}


/**
 * Generate probability points for QQ plot.
 *
 * @param {number} n - number of points.
 *
 * @returns {Vector} a sequence of probabilities between 0 and 1.
 *
 */
export function ppoints(n) {

   const a = n < 10 ? 3.0/8.0 : 0.5;
   const out = new Vector.valuesConstructor(n);

   for (let i = 0; i < n; i++) {
      out[i] = (i + 1 - a) / (n + (1 - a) - a);
   }

   return new Vector(out);
}


/**
 * Computes cumulative sums for the vector values.
 *
 * @param {Array|Vector} x - vector with values.
 *
 * @returns {Vectors} vector with cumulative sums.
 *
 */
export function cumsum(x) {

   if (isvector(x)) {
      return cumsum(x.v);
   }

   let s = 0;
   let out = new Vector.valuesConstructor(x.length);
   for (let i = 0; i < x.length; i++) {
      s += x[i];
      out[i] = s
   }

   return new Vector(out);
}


/**
 * Computes kurtosis of values.
 *
 * @param {Array|Vector} x - vector with values.
 *
 * @returns {number} kurtosis of x.
 *
 */
export function kurtosis(x) {

   if (isvector(x)) {
      return kurtosis(x.v);
   }

   let n = x.length;
   let m = mean(x);

   let m4 = 0.0;
   let m2 = 0.0;
   for (let i = 0; i < n; i++) {
      m2 = m2 + Math.pow((x[i] - m), 2);
      m4 = m4 + Math.pow((x[i] - m), 4);
   }

   return (m4/n) / Math.pow((m2/n), 2);
}


/**
 * Computes skewness of values.
 *
 * @param {Array|Vector} x - vector with values.
 *
 * @returns {number} skewness of x.
 *
 */
export function skewness(x) {

   if (isvector(x)) {
      return skewness(x.v);
   }

   let n = x.length;
   let m = mean(x);

   let m3 = 0.0;
   let m2 = 0.0;
   for (let i = 0; i < n; i++) {
      m2 = m2 + Math.pow((x[i] - m), 2);
      m3 = m3 + Math.pow((x[i] - m), 3);
   }

   return (m3/n) / Math.pow((m2/n), 1.5);
}


/**
 * Computes correlation between two vectors.
 *
 * @param {Array|Vector} x - vector with values.
 * @param {Array|Vector} y - vector with values.
 * @param {string} method - which method to use ("pearson" or "spearman").
 *
 * @returns {number} correlation between x and y.
 *
 */
export function cor(x, y, method = "pearson") {

   if (isvector(x)) {
      return cor(x.v, y.v, method);
   }

   if (method === "spearman") {
      return cor(rank(x), rank(y));
   }

   return cov(x, y) / (sd(x) * sd(y));
}


/**
 * Computes covariance between two vectors.
 *
 * @param {Array|Vector} x - vector with values.
 * @param {Array|Vector} y - vector with values.
 * @param {boolean} [biased=false] - compute a biased version with n degrees of freedom or not (with n - 1).
 * @param {number} [mx=undefined] - mean of x values (if already known).
 * @param {number} [my=undefined] - mean of y values (if already known).
 *
 * @returns {number} covariance between x and y.
 *
 */
export function cov(x, y, biased = false, mx = undefined, my = undefined) {

   if (isvector(x)) {
      return cov(x.v, y.v, biased, mx, my);
   }

   const n = x.length;

   if (y.length !== n) {
      throw Error("Vectors 'x' and 'y' must have the same length.");
   }

   if (n < 2) {
      throw Error("Vectors 'x' and 'y' must have at least two values.");
   }

   if (mx === undefined) mx = mean(x);
   if (my === undefined) my = mean(y);

   let s = 0;
   for (let i = 0; i < n; i++) {
      s = s + (x[i] - mx) * (y[i] - my);
   }

   return s / (biased ? n : n - 1);
}


/**
 * Returns ranks of values in a vector (ranks start from 1, not 0).
 *
 * @param {Array|Vector} x - vector with values.
 *
 * @returns {Vector} vector with ranks.
 *
 */
export function rank(x) {

   if (isvector(x)) {
      return rank(x.v);
   }

   const y = [...x].sort((a, b) => a - b);

   return new Vector(x.map(v => y.indexOf(v) + 1));
}


/**
 * Compute marginal range of values as [min, max]
 *
 * @param {Array|Vector} x - vector with values.
 * @param {number} m - margin as per cent of range (value between 0 and 1).
 *
 * @return {Array} array with two values, min and max.
 *
 */
export function mrange(x, m) {

   if (isvector(x)) {
      return mrange(x.v, m);
   }

   if (m === undefined) m = 0.10;
   const r = range(x);
   const d = (r[1] - r[0]) * m;

   return [r[0] - d, r[1] + d];
}


/**
 * Compute range of values as [min, max].
 *
 * @param {Array|Vector} x - vector with values.
 *
 * @return {Array} array with two values, min and max.
 *
 */
export function range(x) {

   if (isvector(x)) {
      return range(x.v);
   }

   let min = x[0];
   let max = x[0];

   for (let i = 1; i < x.length; i++) {
      if (x[i] < min) min = x[i];
      if (x[i] > max) max = x[i];
   }

   return [min, max];
}


/**
 * Compute norm2 of a vector (Euclidean distance).
 *
 * @param {Array|Vector} x - vector with values.
 *
 * @returns {number} norm value.
 *
 */
export function norm2(x) {

   if (isvector(x)) {
      return norm2(x.v);
   }

   return Math.sqrt(ssq(x));
}


/**
 * Compute sum of squared vector values
 *
 * @param {Array|Vector} x - vector with values.
 *
 * @returns {number} sum of squared values of x.
 *
 */
export function ssq(x) {

   if (isvector(x)) {
      return ssq(x.v);
   }

   let ssqv = 0;
   const n = x.length;
   for (let i = 0; i < n; i++) {
      const v = x[i]
      ssqv += v * v;
   }

   return ssqv;
}


/**
 * Compute standard deviation of vector values.
 *
 * @param {Array|Vector} x - vector with values.
 * @param {boolean} [biased=false] - compute a biased value (n degrees of freedom) or unbiased (n - 1 degrees of freedom)
 *
 * @returns {number} standard deviation of x.
 *
 */
export function sd(x, biased) {

   if (isvector(x)) {
      return sd(x.v, biased);
   }

   return Math.sqrt(variance(x, biased));
}


/**
 * Compute variance of vector values.
 *
 * @param {Array|Vector} x - vector with values.
 * @param {boolean} [biased=false] - compute a biased variance (n degrees of freedom) or unbiased (n - 1 degrees of freedom)
 *
 * @returns {number} variance of x.
 *
 */
export function variance(x, biased) {

   if (biased === undefined) {
      biased = false;
   }

   if (isvector(x)) {
      return variance(x.v);
   }

   const m = mean(x);
   let s = 0;
   for (let i = 0; i < x.length; i++) {
      const d = (x[i] - m);
      s = s + d * d ;
   }

   return s / (biased ? x.length : (x.length - 1));
}


/**
 * Compute average of vector values.
 *
 * @param {Array|Vector} x - vector with values.
 *
 * @returns {number} mean of x.
 *
 */
export function mean(x) {

   if (isvector(x)) {
      return mean(x.v);
   }

   return sum(x) / x.length;
}


/**
 * Compute sum of all values in a vector.
 *
 * @param {Array|Vector} x - vector with values.
 *
 * @returns {number} sum of x.
 *
 */
export function sum(x) {

   if (isvector(x)) {
      return sum(x.v);
   }

   let s = 0;
   for (let i = 0; i < x.length; i++) {
      s = s + x[i];
   }

   return s;
}


/**
 * Compute product of all values in a vector.
 *
 * @param {Array|Vector} x - vector with values.
 *
 * @returns {number} product of x.
 *
 */
export function prod(x) {

   if (isvector(x)) {
      return prod(x.v);
   }

   let p = 1;
   for (let i = 0; i < x.length; i++) {
      p = p * x[i];
   }

   return p;
}


/**
 * Find the smallest element in a vector.
 *
 * @param {Array|Vector|Index} x - vector or index with values.
 *
 * @returns {number} the smallest value.
 *
 */
export function min(x) {

   if (isvector(x) || isindex(x)) {
      return x.v[minind(x.v) - 1];
   }

   return x[minind(x) - 1];
}


/**
 * Find index of the smallest element in a vector.
 *
 * @param {Array|Vector|Index} x - vector or index with values.
 *
 * @returns {number} index of the smallest value (starting from 1).
 *
 */
export function minind(x) {

   if (isvector(x) || isindex(x)) {
      return minind(x.v);
   }

   let outind = 1;
   let out = x[0]
   for (let i = 2; i <= x.length; i++) {
      if (x[i - 1] < out) {
         outind = i;
         out = x[i - 1];
      }
   }

   return outind;
}


/**
 * Find the largest element in a vector.
 *
 * @param {Array|Vector|Index} x - vector or index with values.
 *
 * @returns {number} the largest value.
 *
 */
export function max(x) {

   if (isvector(x) || isindex(x)) {
      return x.v[maxind(x.v) - 1];
   }

   return x[maxind(x) - 1];
}


/**
 * Find index of the largest element in a vector.
 *
 * @param {Array|Vector|Index} x - vector or index with values.
 *
 * @returns {number} index of the largest value (starting from 1).
 *
 */
export function maxind(x) {

   if (isvector(x) || isindex(x)) {
      return maxind(x.v);
   }

   let outind = 1;
   let out = x[0]
   for (let i = 2; i <= x.length; i++) {
      if (x[i - 1] > out) {
         outind = i;
         out = x[i - 1];
      }
   }

   return outind;
}


