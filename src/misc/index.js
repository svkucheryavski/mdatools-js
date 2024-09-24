/*************************************************/
/*    Misc extra methods for statistics          */
/*************************************************/

import { isvector, vector, Index, Vector } from "../arrays/index.js";
import { prod, quantile } from "../stat/index.js";


/**
 * Finds index of value in x which is closest to the value a.
 *
 * @param {Array|Vector} x - a vector with values.
 * @param {number} a - a value
 *
 * @returns {number} the index value (starts from 1).
 *
 */
export function closestind(x, a) {

   if (isvector(x)) {
      return closestind(x.v, a);
   }

   const c = x.reduce((prev, curr) => Math.abs(curr - a) < Math.abs(prev - a) ? curr : prev);
   return x.indexOf(c) + 1;
}

/**
 * Finds index of value in x which is closest to the value a from the left.
 *
 * @param {Array|Vector} x - a vector with values.
 * @param {number} a - a value
 *
 * @returns {number} the index value (starts from 1).
 *
 */
export function closestindleft(x, a) {

   if (isvector(x)) {
      return closestindleft(x.v, a);
   }

   if (x[0] > a) return 1;
   for (let i = 1; i < x.length; i++) {
      if (x[i] > a) return i;
   }
   return x.length;
}

/**
 * Finds index of value in x which is closest to the value a from the right.
 *
 * @param {Array|Vector} x - a vector with values.
 * @param {number} a - a value
 *
 * @returns {number} the index value (starts from 1).
 *
 */
export function closestindright(x, a) {

   if (isvector(x)) {
      return closestindright(x.v, a);
   }

   if (x[x.length - 1] < a) return x.length;
   for (let i = (x.length - 1); i >= 0; i--) {
      if (x[i] < a) return i + 2;
   }
   return 1;
}


/**
 * Computes numeric integral for function "f" with limits (a, b).
 *
 * @param {function} f - a reference to a function.
 * @param {number} a - lower limit for integration.
 * @param {number} b - upper limit for integration.
 * @param {number} [acc=0.000001] - absolute accuracy.
 * @param {number} [eps=0.00001] - relative accuracy.
 * @param {number[]} oldfs - vector of values needed for recursion.
 *
 * @returns {number} result of integration.
 *
 */
export function integrate(f, a, b, acc, eps, oldfs) {

   if (acc === undefined) {
      acc = 0.000001;
   }

   if (eps === undefined) {
      eps = 0.00001
   }

   if (typeof(a) !== "number" || typeof(b) !== "number") {
      throw Error("Parameters 'a' and 'b' must be numbers.");
   }

   if (b < a) {
      throw Error("Parameter 'b' must be larger 'a'.");
   }

   // special case when left limit is minus infinity
   if (a === -Infinity && b !== Infinity) {
      return integrate((t) => f(b - (1 - t) / t) / (t ** 2), 0, 1);
   }

   // special case when right limit is plus infinity
   if (a !== -Infinity && b === Infinity) {
      return integrate((t) => f(a + (1 - t) / t) / (t ** 2), 0, 1);
   }

   // special case when both limits are infinite
   if (a === -Infinity && b === Infinity) {
      return integrate((t) => (f((1 - t) / t) + f((t - 1) / t)) / t ** 2, 0, 1);
   }

   // constants for splitting the integration interval
   const x = [1/6, 2/6, 4/6, 5/6];
   const w = [2/6, 1/6, 1/6, 2/6];
   const v = [1/4, 1/4, 1/4, 1/4];
   const p = [1, 0, 0, 1];

   let n = x.length, h = b - a;
   let fs;

   if (oldfs === undefined) {
      fs = x.map(v => f(a + v * h));
   } else {
      fs = new Array(n);
      for (let k = 0, i = 0; i < n; i++) {
         fs[i] = p[i] === 1 ? f(a + x[i] * h) : oldfs[k++];
      }
   }

   let q4 = 0, q2 = 0;
   for (let i = 0; i < n; i++) {
      q4 += w[i] * fs[i] * h;
      q2 += v[i] * fs[i] * h;
   }

   if (isNaN(q2) || isNaN(q4)) {
      throw Error("Numerical integration ended up with NaN number.")
   }

   let tol = acc + eps * Math.abs(q4);
   let err = Math.abs((q4 - q2)/3);

   if (err < tol) return q4;

   acc = acc / Math.sqrt(2.);
   const mid = (a + b) / 2;
   const left = fs.filter((v, i) => i < n/2);
   const right = fs.filter((v, i) => i >= n/2);

   const ql = integrate(f, a, mid, eps, acc, left);
   const qr = integrate(f, mid, b, eps, acc, right);

   return ql + qr;
}


/**
 * Finds outliers in a vector based on inter-quartile range distance.
 *
 * @param {Array|Vector} x - vector with values.
 * @param {number} q1 - first quartile (optional parameter).
 * @param {number} q3 - third quartile (optional parameter).
 *
 * @returns {Vector} vector with outliers or empty vector if none were found.
 *
 */
export function getoutliers(x, q1 = undefined, q3 = undefined) {

   if (q1 === undefined) q1 = quantile(x, 0.25);
   if (q3 === undefined) q3 = quantile(x, 0.75);

   const iqr15 = (q3 - q1) * 1.5;
   const bl = q1 - iqr15
   const bu = q3 + iqr15

   return new Vector(x.v.filter(v => v < bl || v > bu));
}


/**
 * Rounds number (or vector of numbers) to given amount of decimals.
 *
 * @param {numbr|Array|Vector} x - a number or a vector with values.
 *
 * @return {number|Array|Vector} a number or a vector with rounded values.
 *
 */
export function round(x, n = 0) {

   if (isvector(x)) {
      return vector(x.v.map(v => round(v, n)));
   }

   if (Array.isArray(x)) {
      return x.map(v => round(v, n));
   }

   return Number.parseFloat(x.toFixed(n));
}


/**
 * Generate combination of all levels of vectors.
 *
 * @param {...} args - a sequence of vectors.
 *
 * @returns {Array} array of arrays with values for each vector.
 *
 */
export function expandgrid(...args) {

   const nargs = args.length;
   const d = args.map(v => v.length);
   let orep = prod(d);

   let grid = Array(nargs);
   let repFac = 1;

   for (let i = 0; i < nargs; i++) {
      const x = isvector(args[i]) ? args[i] : vector(args[i]);
      const nx = x.length;
      orep = orep / nx;
      grid[i] = x.subset(Index.seq(1, nx).repeach(repFac).rep(orep));
      repFac = repFac * nx;
   }

   return grid;
}
