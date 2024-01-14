/* =================================================================== */
/*    Methods for statistical distributions                            */
/* =================================================================== */

import { range } from '../stat/index.js';
import { isvector, vector, Vector } from '../arrays/index.js';
import { closestind, integrate } from '../misc/index.js';

// threshold to switch between two approximation for chi-square distribution
const CHISQ_DOF_THRESHOLD = 30;
// tolerance for chi-square distribution search
const CHISQ_TOL = 0.0000001;

/**
 * Generates 'n' random numbers from a uniform distribution.
 *
 * @param {number} n - amount of numbers to generate.
 * @param {number} [a=0] - smallest value (min) of the population.
 * @param {number} [b=1] - largest value (max) of the population.
 *
 * @returns {Vector} vector with generated random numbers.
 *
 */
export function runif(n, a = 0, b = 1) {
   let out = Vector.zeros(n);
   for (let i = 0; i < n; i++) {
      out.v[i] = (a + Math.random() * (b - a));
   }
   return out;
}


/**
 * Cumulative distribution function for uniform distribution.
 *
 * @param {number|Vector} x - vector of values or a single value.
 * @param {number} [a=0] - smallest value (min) of the population.
 * @param {number} [b=1] - largest value (max) of the population.
 *
 * @returns {number|Vector} computed probabilities.
 *
 */
export function punif(x, a = 0, b = 1) {

   if (isvector(x)) {
      return x.apply(v => punif(v, a, b));
   }

   return x < a ? 0 : (x > b ? 1 : (x - a) / (b - a));
}


/**
 * Probability density function for uniform distribution.
 *
 * @param {number|Vector} x - vector of values or a single value.
 * @param {number} [a=0] - smallest value (min) of the population.
 * @param {number} [b=1] - largest value (max) of the population.
 *
 * @returns {number|Vector} computed densities.
 *
 */
export function dunif(x, a = 0, b = 1) {

   if (isvector(x)) {
      return x.apply(v => dunif(v, a, b));
   }

   return x < a || x > b ? 0 : 1 / (b - a);
}


 /**
 * Generates 'n' random numbers from a normal distribution.
 *
 * @param {number} n - amount of numbers to generate.
 * @param {number} [mu=0] - average value of the population.
 * @param {number} [sigma=1] - standard deviation of the population.
 *
 * @returns {Vector} vector with generated random numbers.
 */
export function rnorm(n, mu = 0, sigma = 1) {

   let out = Vector.zeros(n);
   for (let i = 0; i < n; i ++) {
      const a = Math.sqrt(-2 * Math.log(Math.random()));
      const b = 2 * Math.PI * Math.random();
      out.v[i] = (a * Math.sin(b) * sigma + mu);
   }

   return out;
}


/**
 * Probability density function for normal distribution.
 *
 * @param {number|Vector} x - vector of values of a single value.
 * @param {number} [mu=0] - average value of the population.
 * @param {number} [sigma=1] - standard deviation of the population.
 *
 * @returns {number|Vector} computed densities.
 *
 */
export function dnorm(x, mu = 0, sigma = 1) {

   if (isvector(x)) {
      return x.apply(v => dnorm(v, mu, sigma));
   }

   const z = (x - mu) / sigma;
   return Math.exp(-0.5 * z * z) / (Math.sqrt(2 * Math.PI) * sigma);
}


/**
 * Cumulative distribution function for normal distribution.
 *
 * @param {number|Vector} x - vector of values or a single value.
 * @param {number} [mu=0] - average value of the population.
 * @param {number} [sigma=1] - standard deviation of the population.
 *
 * @returns {number|Vector} computed probabilities.
 *
 */
export function pnorm(x, mu = 0, sigma = 1) {

   if (isvector(x)) {
      return x.apply(v => pnorm(v, mu, sigma));
   }

   const frac = 1 / (Math.sqrt(2) * sigma);
   return  0.5 * (1 + erf((x - mu) * frac));
}


/**
 * Inverse cumulative distribution function for normal distribution.
 *
 * @param {number|Vector} p - vector of probabilities or a single probability value.
 * @param {number} [mu=0] - average value of the population.
 * @param {number} [sigma=1] - standard deviation of the population.
 *
 * @returns {number|Vector} computed quantiles.
 *
 */
export function qnorm(p, mu = 0, sigma = 1) {

   if (isvector(p)) {
      return p.apply(v => qnorm(v, mu, sigma));
   }

   if (mu !== 0 || sigma !== 1) {
      return qnorm(p) * sigma + mu;
   }

   if (p < 0 || p > 1) {
      throw Error('Parameter "p" must be between 0 and 1.');
   }

   if (p < 0.0000000001) return -Infinity;
   if (p > 0.9999999999) return +Infinity;

   const SP1 = 0.425;
   const SP2 = 5.0;
   const C1 = 0.180625;
   const C2 = 1.6;

   const a0 = 3.3871327179;
   const a1 = 5.0434271938 * 10;
   const a2 = 1.5929113202 * 100;
   const a3 = 5.9109374720 * 10;
   const b1 = 1.7895169469 * 10;
   const b2 = 7.8757757664 * 10;
   const b3 = 6.7187563600 * 10;

   const c0 = 1.4234372777;
   const c1 = 2.7568153900;
   const c2 = 1.3067284816;
   const c3 = 1.7023821103 * 0.1;
   const d1 = 7.3700164250 * 0.1;
   const d2 = 1.2021132975 * 0.1;

   const e0 = 6.6579051150;
   const e1 = 3.0812263860;
   const e2 = 4.2868294337 * 0.1;
   const e3 = 1.7337203997 * 0.01;
   const f1 = 2.4197894225 * 0.1;
   const f2 = 1.2258202635 * 0.01;

   const q = p - 0.5;
   let r;

   if (Math.abs(q) <= SP1) {
      r = C1 - q * q;
      return q * (((a3 * r + a2) * r + a1) *r + a0) / (((b3 * r + b2) * r + b1) * r + 1.0);
   }

   r = q < 0 ? p : 1 - p;
   r = Math.sqrt(-Math.log(r));
   let res;

   if (r <= SP2) {
      r = r - C2;
      res = (((c3 * r + c2) * r + c1) * r + c0) / ((d2 * r + d1) * r + 1.0);
   } else {
      r = r - SP2;
      res = (((e3 * r + e2) * r + e1) + e0) / ((f2 * r + f1) * r + 1.0);
   }

   return q < 0 ? -res : res;
}


/**
 * Probability density function for Student's t-distribution.
 *
 * @param {number|Vector} t - t-value or a vector of t-values.
 * @param {number} dof - degrees of freedom.
 *
 * @returns {number|Vector} computed densities.
 *
 */
export function dt(t, dof) {

   if (dof < 0) {
      throw new Error('Parameter "dof" should be a positive number.');
   }

   if (isvector(t)) {
      return t.apply(v => dt(v, dof));
   }

   const pow = -0.5 * (dof + 1);
   const A = 1 / (Math.sqrt(dof) * beta(0.5, dof/2));

   return (A * Math.pow((1 + t * t / dof), pow));
}


/**
 * Cumulative distribution function for Student's t-distribution.
 *
 * @param {number|Vector} t - t-value or a vector of t-values.
 * @param {number} dof - degrees of freedom.
 *
 * @returns {number|Vector} computed probabilities.
 *
 */
export function pt(t, dof) {

   if (dof === undefined || dof === null || dof < 1) {
      throw Error('Parameter "dof" (degrees of freedom) must be an integer number >= 1.');
   }

   if (isvector(t)) {
      return t.apply(v => pt(v, dof));
   }

   // since distribution in symmetric we can use only left tail
   if (t === 0) return 0.5;
   if (t === -Infinity) return 0;
   if (t === Infinity) return 1;
   if (t > 0) return (1 - pt(-t, dof));

   return integrate((x) => dt(x, dof), -Infinity, t);
}


/**
 * Inverse cumulative distribution (quantile) function for Student's t-distribution.
 *
 * @param {number|Vector} p - probability or vector with probabilities.
 * @param {number} dof - degrees of freedom.
 *
 * @returns {number|Vector} computed quantiles.
 *
 */
export function qt(p, dof) {

   if (dof === undefined || dof === null || dof < 1) {
      throw Error('Parameter "dof" (degrees of freedom) must be an integer number >= 1.');
   }

   if (p < 0 || p > 1) {
      throw Error('Parameter "p" must be between 0 and 1.');
   }

   if (isvector(p)) {
      return p.apply(v => qt(v, dof));
   }

   if (p < 0.0000000001) return -Infinity;
   if (p > 0.9999999999) return +Infinity;


   // simple cases — exact solutions
   if (dof === 1) {
      return Math.tan(Math.PI * (p - 0.5));
   }

   if (dof === 2) {
      return 2 * (p - 0.5) * Math.sqrt(2 / (4 * p * (1 - p)));
   }

   // approximation

   let sign = -1;
   if (p >= 0.5){
      sign = +1 ;
      p = 2 * (1 - p);
   } else {
      sign = -1;
      p = 2 * p;
   }

   const a = 1.0 / (dof - 0.5);
   const b = 48.0 / (a ** 2);
   let c = ((20700 * a / b - 98) * a - 16) * a + 96.36;
   const d = ((94.5 / (b + c) - 3.0)/b + 1.0) * Math.sqrt(a * Math.PI / 2) * dof;

   let x = d * p;
   let y = x ** (2.0/dof);

   if (y > 0.05 + a) {

      // asymptotic inverse expansion about normal
      x = qnorm(p * 0.5);
      y = x * x;

      if (dof < 5) {
         c = c + 0.3 * (dof - 4.5) * (x + 0.6);
      }

      c = (((0.05 * d * x - 5.0) * x - 7.0) * x - 2.0) * x + b + c;
      y = (((((0.4 * y + 6.3) * y + 36.0) * y + 94.5) / c - y - 3.0)/b + 1.0) * x;
      y = a * (y ** 2);
      y = y > 0.002 ? Math.exp(y) - 1.0 : 0.5 * (y ** 2) + y;
   } else {
      y = ((1.0 / (((dof + 6.0)/(dof * y) - 0.089 * d - 0.822) * (dof + 2.0) * 3.0) + 0.5/(dof + 4.0)) * y - 1.0) *
         (dof + 1.0)/(dof + 2.0) + 1.0/y;
   }

   return sign * Math.sqrt(dof * y);
}


/**
 * Probability density function for F-distribution.
 *
 * @param {number|Vector} F - F-value or a vector of t-values
 * @param {number} d1 - degrees of freedom.
 * @param {number} d2 - degrees of freedom.
 *
 * @returns {number|Vector} computed densities.
 *
 */
export function df(F, d1, d2) {

   if (F < 0 || d1 < 0 || d2 < 0) {
      throw new Error('All 3 parameters must be positive.');
   }

   if (d2 <= d1) {
      throw new Error('Parameter "d1" must be larger "d2".');
   }

   if (isvector(F)) {
      return F.apply(v => df(v, d1, d2));
   }

   return Math.sqrt( ( Math.pow(d1 * F, d1) * Math.pow(d2, d2)) / Math.pow(d1 * F + d2, d1 + d2)) / (F * beta(d1/2, d2/2))
}


/**
 * Cumulative distribution function for F-distribution.
 *
 * @param {number|Vector} F - F-value or a vector of t-values.
 * @param {number} d1 - degrees of freedom.
 * @param {number} d2 - degrees of freedom.
 *
 * @returns {number|Vector} computed probabilities.
 *
 */
export function pf(F, d1, d2) {

   if (F < 0 || d1 < 0 || d2 < 0) {
      throw new Error('All 3 parameters must be positive.');
   }

   if (isvector(F)) {
      return F.apply(v => pf(v, d1, d2));
   }

   return ibeta(d1 * F / (d1 * F + d2), d1/2, d2/2)
}


/**
 * Cumulative distribution function for chi-square distribution.
 *
 * @param {number|Vector|Array} x - chi-square value.
 * @param {number} dof - degrees of freedom.
 *
 * @description if DoF is relatively small (< 30) the method computes probabilities via
 * approximation of lower incomplete gamma function, `igamma()`, and gamma function, `gamma()`.
 * If the DoF is larger than 30 it uses the a modified power series approximation
 * (doi:10.1016/j.csda.2004.04.001) and normal distribution.
 *
 * @returns {number|Vector} computed probabilities.
 *
 */
export function pchisq(x, dof) {

   if (Array.isArray(x)) {
      return pchisq(vector(x), dof);
   }

   const mu = 5/6 - 1 / (9 * dof) - 7 / (648 * dof * dof) + 25 / (2187 * dof * dof * dof);
   const sigma2 = 1 / (18 * dof) + 1 / (162 * dof * dof) - 37 / (1164 * dof * dof * dof);
   const sigma = Math.sqrt(sigma2);

   // works well for large dof
   function F2(x) {
      if (x === 0) return 0;
      if (dof === 0) return 1;
      if (dof > 100 && x > dof * 2) return 1;
      if (dof > 50 && x > dof * 3) return 1;
      if (dof > 10 && x > dof * 5) return 1;
      if (dof > 5 && x > dof * 8) return 1;
      if (x > dof * 30) return 1;
      const l0 = x / dof;
      const l = Math.pow(l0, 1/6) - 0.5 * Math.pow(l0, 1/3) + (1/3) * Math.pow(l0, 1/2);
      return pnorm(l, mu, sigma);
   }

   // works well for small dof
   function F1(x) {
      if (x === 0) return 0;
      if (dof === 0) return 1;
      if (dof > 100 && x > dof * 2) return 1;
      if (dof > 50 && x > dof * 3) return 1;
      if (dof > 10 && x > dof * 5) return 1;
      if (dof > 5 && x > dof * 8) return 1;
      if (x > dof * 30) return 1;
      return igamma(x/2, dof/2) / gamma(dof/2)
   }

   const F = dof > CHISQ_DOF_THRESHOLD ? F2 : F1;
   return isvector(x) ? x.apply(F) : F(x)
}


/**
 * Inverse cumulative distribution (quantile) function for chi-square distribution.
 *
 * @param {number|Vector|Array} p - probability or vector/array with probabilities.
 * @param {number} dof - degrees of freedom.
 *
 * @description if DoF is relatively small (< 30) the method computes quantiles by sequential
 * improving, splitting possible quantile range into intervals, computing probabilities for each
 * element of the interval using `pchisq()`, then find the quantile which has the closest
 * probability value, make a new split around it and so on until the probability of the
 * currently selected quintile is close to the desired one. For large DoF (>30) the it uses
 * the a modified power series approximation (doi:10.1016/j.csda.2004.04.001) and
 * normal distribution.
 *
 * @returns {number|Vector} computed quantiles.
 *
 */
export function qchisq(p, dof) {

   if (Array.isArray(p)) {
      return qchisq(vector(p), dof);
   }

   // compute parameters of normal distribution for power approximation
   const mu = 5/6 - 1 / (9 * dof) - 7 / (648 * dof * dof) + 25 / (2187 * dof * dof * dof);
   const sigma2 = 1 / (18 * dof) + 1 / (162 * dof * dof) - 37 / (1164 * dof * dof * dof);
   const sigma = Math.sqrt(sigma2);

   // compute quantile for one probability value using power approximations
   // and inverse solution
   function F2(x) {
      if (x === 0) return 0;
      if (x === 1) return Inf;
      if (dof === 0) return 0;
      const l = qnorm(x, mu, sigma);
      const o = Math.pow(Math.sqrt(36 * l * l - 30 * l + 13)/4 + (3 * l - 3/2)/2 + 1/8, 1/3);
      return Math.pow(o - 3 / (4 * o) + 0.5, 6) * dof;
   }

   // compute quantile using sequential splits of quantile range
   function F1(x) {
      if (x === 0) return 0;
      if (x === 1) return Inf;
      if (dof === 0) return 0;

      // initial split (start, end, and delta)
      let qs = 0;
      let qe = CHISQ_DOF_THRESHOLD * 10;
      let qd = qe / 30;

      // loop for sequential splits
      for (let i = 0; i < 30; i++) {

         // compute vector of quantiles and corresponding probabilities
         const q = Vector.seq(qs, qe, qd);
         const p = pchisq(q, dof);

         // find index of quantile which has probability closest to x
         const ind = closestind(p, x) - 1;

         // if the probability is close enough to the desired one return the quantile
         if (Math.abs(p.v[ind] - x) < CHISQ_TOL) return q.v[ind];

         // if not make a new split by taking left and right neighbors quantiles
         qs = ind == 0 ? q.v[0] : q.v[ind - 1];
         qe = ind == q.v.length - 1? q.v[ind] : q.v[ind + 1];

         // if the boundaries are the same stop
         if (qs == qe) return qs;

         // compute delta for new boundaries
         qd = (qe - qs) / 30;
      }

      // if the loop did not converge we simply return the middle quantile
      return (qe + qs) / 2;
   }

   const F = dof > CHISQ_DOF_THRESHOLD ? F2 : F1;
   return isvector(p) ? p.apply(F) : F(p)
}


/**
 * Incomplete Betta function (approximation via numeric integration).
 *
 * @param {number} x - first argument.
 * @param {number} a - second argument.
 * @param {number} b - third argument.
 *
 * @returns {number} value of the function.
 *
 */
export function ibeta(x, a, b) {

   if (x === 0) return 0;
   if (x === 1) return 1;
   if (b === 1) return x ** a;
   if (a === 1) return (1 - (1 - x)**b);

   return integrate((t) => t ** (a - 1) * (1 - t) ** (b - 1), 0, x) / beta(a, b);
}


/**
 * Gamma function (approximation).
 *
 * @param {number|Vector} z - argument (one value or a vector).
 *
 * @returns {number} value of the Gamma function.
 *
 */
export function gamma(z) {

   if (isvector(z)) {
      return z.apply(gamma);
   }

   if (z <= 0) {
      throw new Error('gamma: the function only works with arguments > 0.');
   }

   // coefficients
   const p = [
        676.5203681218851,
      -1259.1392167224028,
        771.32342877765313,
       -176.61502916214059,
         12.507343278686905,
         -0.13857109526572012,
          9.9843695780195716e-6,
          1.5056327351493116e-7
    ];

   if (z < 0.5) {
      return Math.PI / (Math.sin(Math.PI * z) + gamma(1 - z));
   }

   z = z - 1;
   let x = 0.99999999999980993;

   for (let i = 0; i < p.length; i++) {
      x = x + p[i] / (z + i + 1);
   }

   const t = z + p.length - 0.5;
   return Math.sqrt(2 * Math.PI) * Math.pow(t, z + 0.5) * Math.exp(-t) * x;
}



/**
 * Lower incomplete gamma function (approximation)
 *
 * @param {number} x - first argument (one value).
 * @param {number} a - second argument (one value).
 *
 * @returns {number} value of the lower incomplete gamma function.
 *
 */
export function igamma(x, a) {

   const epsilon = 1e-18; // Desired accuracy
   const maxIterations = 10000; // Maximum number of iterations

   let sum = 0;

   for (let k = 0; k < maxIterations; k++) {
      const s = Math.pow(x, k) / gamma(a + k + 1);
      if (isNaN(s) || s < epsilon) break;
      sum += s
   }

   return Math.pow(x, a) * gamma(a) * Math.exp(-x) * sum;
}



/**
 * Beta function (approximation).
 *
 * @param {number} x - first argument (one value).
 * @param {number} y - second argument (one value).
 *
 * @returns {number} value of the Beta function.
 *
 */
export function beta(x, y) {

   if (y > 100) {
      // for large y we use slower integrate version
      return integrate((u) => Math.pow(u, x - 1) / Math.pow(1 + u, x + y), 0, Infinity)
   }

   return gamma(x) * gamma(y) / gamma(x + y);
}


/**
 * Error function for normal distribution.
 *
 * @param {number} x - a number.
 *
 * @returns {number} value for erf.
 *
 */
export function erf(x) {

  const sign = (x >= 0) ? 1 : -1;
  x = Math.abs(x);

  // constants
  const a1 =  0.254829592;
  const a2 = -0.284496736;
  const a3 =  1.421413741;
  const a4 = -1.453152027;
  const a5 =  1.061405429;
  const p  =  0.3275911;

  // approximation
  const t = 1.0 / (1.0 + p * x);
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

  return sign * y;
}


