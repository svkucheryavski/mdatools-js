/* =================================================================== */
/*    Methods for statistical distributions                            */
/* ------------------------------------------------------------------- */
/*    Some of the methods and helper functions are based               */
/*    on C translation of ACM TOMS 708                                 */
/* =================================================================== */

import { isvector, vector, Vector } from "../arrays/index.js";

// threshold to switch between two approximation for chi-square distribution
const CHISQ_DOF_THRESHOLD = 30;

// tolerance for chi-square and f- distribution search
const PTOL = 0.0000001;
const QTOL = 0.0000001;
const QMIN = 0.0000000000001;
const QMAX = 1000;
const QNITER = 100;

/**
 * Search quantile value using CDF function and splits.
 *
 * @param {Number} p - probability for quantile (left tail).
 * @param {Function} pfun - CDF function
 * @param {Number} min - smallest possible quantile.
 * @param {Number} max - largest possible quantile.
 * @param {Number} niter - maximum number of iterations.
 * @returns quantile value for given "p".
 */
export function quantsearch(p, pfun, min, max, niter) {
   if (max === undefined) max = QMAX;
   if (min === undefined) min = QMIN;
   if (niter === undefined) niter = QNITER;

   // initial split (start, end, and delta)
   let qs = min;
   let qe = max;
   let qd = (qe - qs) / niter;

   // loop for sequential splits
   for (let i = 0; i < niter; i++) {


      // find q1/q2 so F(q1) ≤ p ≤ F(q2)
      let q1, q2;
      for (let j = 0; j < niter; j++) {
         q1 = qs + j * qd;
         q2 = qs + (j + 1) * qd;
         const p1 = pfun(q1);
         const p2 = pfun(q2);
         if (p1 <= p && p2 >= p) break;
      }

      // compute new quantile value as mean of q1 and q2
      const q0 = (q1 + q2) / 2;
      const p0 = pfun(q0);

      // check convergence
      if (Math.abs(p0 - p) < PTOL) return q0;

      // prepare interval and delta for next iteration
      qs = q1;
      qe = q2;
      qd = (qe - qs) / niter;

      // check if delta is small enough
      if (qd / qe < QTOL) return q0;
   }

   // if did not converge return anyway
   return (qs + qe) / 2;
}


/******************************************************/
/* Uniform distribution                               */
/******************************************************/

/**
 * Generates 'n' random numbers from a uniform distribution.
 *
 * @param {number} n - amount of numbers to generate.
 * @param {number} [a=0] - smallest value (min) of the population.
 * @param {number} [b=1] - largest value (max) of the population.
 *
 * @returns {Vector} vector with generated random numbers.
 */
export function runif(n, a = 0, b = 1) {
  let out = Vector.zeros(n);
  for (let i = 0; i < n; i++) {
    out.v[i] = a + Math.random() * (b - a);
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
 */
export function punif(x, a = 0, b = 1) {
  if (isvector(x)) {
    return x.apply((v) => punif(v, a, b));
  }

  return x < a ? 0 : x > b ? 1 : (x - a) / (b - a);
}

/**
 * Probability density function for uniform distribution.
 *
 * @param {number|Vector} x - vector of values or a single value.
 * @param {number} [a=0] - smallest value (min) of the population.
 * @param {number} [b=1] - largest value (max) of the population.
 *
 * @returns {number|Vector} computed densities.
 */
export function dunif(x, a = 0, b = 1) {
  if (isvector(x)) {
    return x.apply((v) => dunif(v, a, b));
  }

  return x < a || x > b ? 0 : 1 / (b - a);
}


/******************************************************/
/* Normal distribution                                */
/******************************************************/

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
  for (let i = 0; i < n; i++) {
    const a = Math.sqrt(-2 * Math.log(Math.random()));
    const b = 2 * Math.PI * Math.random();
    out.v[i] = a * Math.sin(b) * sigma + mu;
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
    return x.apply((v) => dnorm(v, mu, sigma));
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
    return x.apply((v) => pnorm(v, mu, sigma));
  }

  const frac = 1 / (Math.sqrt(2) * sigma);
  return 0.5 * (1 + erf((x - mu) * frac));
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
    return p.apply((v) => qnorm(v, mu, sigma));
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
  const a3 = 5.910937472 * 10;
  const b1 = 1.7895169469 * 10;
  const b2 = 7.8757757664 * 10;
  const b3 = 6.71875636 * 10;

  const c0 = 1.4234372777;
  const c1 = 2.75681539;
  const c2 = 1.3067284816;
  const c3 = 1.7023821103 * 0.1;
  const d1 = 7.370016425 * 0.1;
  const d2 = 1.2021132975 * 0.1;

  const e0 = 6.657905115;
  const e1 = 3.081226386;
  const e2 = 4.2868294337 * 0.1;
  const e3 = 1.7337203997 * 0.01;
  const f1 = 2.4197894225 * 0.1;
  const f2 = 1.2258202635 * 0.01;

  const q = p - 0.5;
  let r;

  if (Math.abs(q) <= SP1) {
    r = C1 - q * q;
    return (
      (q * (((a3 * r + a2) * r + a1) * r + a0)) /
      (((b3 * r + b2) * r + b1) * r + 1.0)
    );
  }

  r = q < 0 ? p : 1 - p;
  r = Math.sqrt(-Math.log(r));
  let res;

  if (r <= SP2) {
    r = r - C2;
    res = (((c3 * r + c2) * r + c1) * r + c0) / ((d2 * r + d1) * r + 1.0);
  } else {
    r = r - SP2;
    res = ((e3 * r + e2) * r + e1 + e0) / ((f2 * r + f1) * r + 1.0);
  }

  return q < 0 ? -res : res;
}


/******************************************************/
/* Student's t-distribution                           */
/******************************************************/

/**
 * Probability density function for Student's t-distribution.
 *
 * @param {number|Vector} t - t-value or a vector of t-values.
 * @param {number} df - degrees of freedom.
 *
 * @returns {number|Vector} computed densities.
 *
 */
export function dt(t, df) {
   if (df < 0) {
      throw new Error('Parameter "df" should be a positive number.');
   }

   if (isvector(t)) {
      return t.apply((v) => dt(v, df));
   }

   const pow = -0.5 * (df + 1);
   const A = 1 / (Math.sqrt(df) * beta(0.5, df / 2));

   return A * Math.pow(1 + (t * t) / df, pow);
}

/**
 * Cumulative distribution function for Student's t-distribution.
 *
 * @param {number|Vector} t - t-value or a vector of t-values.
 * @param {number} df - degrees of freedom.
 *
 * @returns {number|Vector} computed probabilities.
 *
 */
export function pt(t, df) {
   if (df === undefined || df === null || df < 1) {
      throw Error('Parameter "df" (degrees of freedom) must be an integer number >= 1.',);
   }

   if (isvector(t)) {
      return t.apply((v) => pt(v, df));
   }

   // since distribution in symmetric we can use only left tail
   if (t === 0) return 0.5;
   if (t === -Infinity) return 0;
   if (t === Infinity) return 1;
   if (t < 0) return 1 - pt(-t, df);

   if (df === 1) return 0.5 + Math.atan(t) / Math.PI;
   if (df === 2) return 0.5 + t / (2 * Math.sqrt(2 + t * t));
   if (df === 3) return (0.5 + (t / Math.sqrt(3) / (1 + (t * t) / 3) + Math.atan(t / Math.sqrt(3))) / Math.PI);
   if (df === 4) return (0.5 + ((t / Math.sqrt(1 + t * t * 0.25)) * (1 - (t * t) / (12 * (1 + t * t * 0.25))) * 3) / 8);

   if (!Number.isFinite(df) || df > 1e5) return pnorm(x, 0.0, 1.0);

   const dft = 1 + (t / df) * t;
   const val = (df > t * t) ? (1 - pbeta(t * t / (df + t * t), 0.5, df / 2.)) : pbeta(1. / dft, df / 2., 0.5);
   return 1 - val / 2.;
}


/**
 * Inverse cumulative distribution (quantile) function for Student's t-distribution.
 *
 * @param {number|Vector} p - probability or vector with probabilities.
 * @param {number} df - degrees of freedom.
 *
 * @returns {number|Vector} computed quantiles.
 *
 */
export function qt(p, df) {
   if (df === undefined || df === null || df < 1) {
      throw Error('Parameter "df" (degrees of freedom) must be an integer number >= 1.',);
   }

   if (isvector(p)) {
      return p.apply((v) => qt(v, df));
   }

   if (p < 0 || p > 1) {
      throw Error('Parameter "p" must be between 0 and 1.');
   }

   if (p < 0.0000000001) return -Infinity;
   if (p > 0.9999999999) return +Infinity;

   // simple cases — exact solutions
   if (df === 1) {
      return Math.tan(Math.PI * (p - 0.5));
   }

   if (df === 2) {
      return 2 * (p - 0.5) * Math.sqrt(2 / (4 * p * (1 - p)));
   }

  // approximation

   let sign = -1;
   if (p >= 0.5) {
      sign = +1;
      p = 2 * (1 - p);
   } else {
      sign = -1;
      p = 2 * p;
   }

   const a = 1.0 / (df - 0.5);
   const b = 48.0 / a ** 2;
   let c = (((20700 * a) / b - 98) * a - 16) * a + 96.36;
   const d = ((94.5 / (b + c) - 3.0) / b + 1.0) * Math.sqrt((a * Math.PI) / 2) * df;

   let x = d * p;
   let y = x ** (2.0 / df);

   if (y > 0.05 + a) {
      // asymptotic inverse expansion about normal
      x = qnorm(p * 0.5);
      y = x * x;

      if (df < 5) {
         c = c + 0.3 * (df - 4.5) * (x + 0.6);
      }

      c = (((0.05 * d * x - 5.0) * x - 7.0) * x - 2.0) * x + b + c;
      y = (((((0.4 * y + 6.3) * y + 36.0) * y + 94.5) / c - y - 3.0) / b + 1.0) * x;
      y = a * y ** 2;
      y = y > 0.002 ? Math.exp(y) - 1.0 : 0.5 * y ** 2 + y;
   } else {
      y = (((1.0 / (((df + 6.0) / (df * y) - 0.089 * d - 0.822) * (df + 2.0) * 3.0) +
         0.5 / (df + 4.0)) * y - 1.0) * (df + 1.0)) / (df + 2.0) + 1.0 / y;
   }

   return sign * Math.sqrt(df * y);
}


/******************************************************/
/* F-distribution                                     */
/******************************************************/

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
    throw new Error("All 3 parameters must be positive.");
  }

  if (d2 <= d1) {
    throw new Error('Parameter "d1" must be larger "d2".');
  }

  if (isvector(F)) {
    return F.apply((v) => df(v, d1, d2));
  }

  return (
    Math.sqrt(
      (Math.pow(d1 * F, d1) * Math.pow(d2, d2)) /
        Math.pow(d1 * F + d2, d1 + d2),
    ) /
    (F * beta(d1 / 2, d2 / 2))
  );
}

/**
 * Cumulative distribution function for F-distribution.
 *
 * @param {number|Vector} F - F-value or a vector of t-values.
 * @param {number} df1 - degrees of freedom.
 * @param {number} df2 - degrees of freedom.
 *
 * @returns {number|Vector} computed probabilities.
 *
 */
export function pf(F, df1, df2) {

   if (df1 <= 0 || df2 <= 0) {
      throw new Error('pf: parameters "df1" and "df2" must be positive numbers.')
   }
   if (Array.isArray(F)) {
      return pf(vector(F), df1, df2);
   }

   if (isvector(F)) {
      return F.apply((v) => pf(v, df1, df2));
   }

   if (F <= 0) return 0;

   if (df1 <= 0. || df2 <= 0.) ML_ERR_return_NAN;

   if (df2 == Number.POSITIVE_INFINITY) {

   	if (df1 == Number.POSITIVE_INFINITY) {
	      if(F <  1.) return 0.0;
	      if(F == 1.) return 0.5;
	      if(F >  1.) return 1.0;
	   }

	   return pchisq(F * df1, df1);
   }

   if (df1 == Number.POSITIVE_INFINITY) return 1 - pchisq(df2 / F , df2);

   if (df1 * F > df2)
   	return 1 - pbeta(df2 / (df2 + df1 * F), df2 / 2., df1 / 2.);
   else
	   return pbeta(df1 * F / (df2 + df1 * F), df1 / 2., df2 / 2.);

}


/**
 * Inverse cumulative distribution (quantile) function for F-distribution.
 *
 * @param {number|Vector|Array} p - probability or vector/array with probabilities.
 * @param {number} d1 - degrees of freedom.
 * @param {number} d2 - degrees of freedom.
 *
 * @returns {number|Vector} computed probabilities.
 *
 * @description the method computes quantiles by sequential improving, splitting possible quantile
 * range into intervals, computing probabilities for each element of the interval using `pf()`,
 * then find the quantile which has the closest probability value, make a new split around it
 * and so on until the probability of the currently selected quintile is close to the desired
 * one.
 *
 * @returns {number|Vector} computed quantiles.
 *
 */
export function qf(p, df1, df2) {

   if (p === 0) return 0;
   if (p === 1) return Inf;
   if (df1 <= 0) return NaN;
   if (df2 <= 0) return NaN;

   let MAXQ = QMAX;
   let MINQ = QMIN;
   if (p > 0.9997 && df2 === 1) { MAXQ = 10000000; MINQ = 600000; p = 0.9997}
   else if (p > 0.9992 && df2 === 1) { MAXQ = 10000000; MINQ = 600000; }
   else if (p > 0.997 && df2 === 1) { MAXQ = 1000000; MINQ = 45000; }
   else if (p > 0.992 && df2 === 1) { MAXQ = 100000; MINQ = 5000; }
   else if (p > 0.97 && df2 === 1) { MAXQ = 10000; MINQ = 400; }

   if (Array.isArray(p)) {
     return qf(vector(p), df1, df2);
   }

   const pfun = (v) => pf(v, df1, df2);

  // compute quantile using sequential splits of quantile range
   function F(x, df1, df2) {
     if (x === 0) return 0;
     if (x === 1) return Inf;
     if (df1 === 0) return NaN;
     if (df2 === 0) return NaN;

     return quantsearch(x, pfun, MINQ, MAXQ);
   }

   return isvector(p) ? p.apply(F, df1, df2) : F(p, df1, df2);
}



/******************************************************/
/* Chi-square distribution                            */
/******************************************************/


/**
 * Cumulative distribution function for chi-square distribution.
 *
 * @param {number|Vector|Array} x - chi-square value.
 * @param {number} df - degrees of freedom.
 *
 * @description if DoF is relatively small (< 30) the method computes probabilities via
 * approximation of lower incomplete gamma function, `igamma()`, and gamma function, `gamma()`.
 * If the DoF is larger than 30 it uses the a modified power series approximation
 * (doi:10.1016/j.csda.2004.04.001) and normal distribution.
 *
 * @returns {number|Vector} computed probabilities.
 *
 */
export function pchisq(x, df) {

   if (Array.isArray(x)) {
      return pchisq(vector(x), df);
   }

   const mu = 5 / 6 - 1 / (9 * df) - 7 / (648 * df * df) + 25 / (2187 * df * df * df);
   const sigma2 = 1 / (18 * df) + 1 / (162 * df * df) - 37 / (1164 * df * df * df);
   const sigma = Math.sqrt(sigma2);

  // works well for large df
   function F2(x) {
      if (x === 0) return 0;
      if (df === 0) return 1;
      if (df > 100 && x > df * 2) return 1;
      if (df > 50 && x > df * 3) return 1;
      if (df > 10 && x > df * 5) return 1;
      if (df > 5 && x > df * 8) return 1;
      if (x > df * 30) return 1;
      const l0 = x / df;
      const l = Math.pow(l0, 1 / 6) - 0.5 * Math.pow(l0, 1 / 3) + (1 / 3) * Math.pow(l0, 1 / 2);
      return pnorm(l, mu, sigma);
   }

  // works well for small df
  function F1(x) {
    if (x === 0) return 0;
    if (df === 0) return 1;
    if (df > 100 && x > df * 2) return 1;
    if (df > 50 && x > df * 3) return 1;
    if (df > 10 && x > df * 5) return 1;
    if (df > 5 && x > df * 8) return 1;
    if (x > df * 30) return 1;
    return igamma(x / 2, df / 2) / gamma(df / 2);
  }

   const F = df > CHISQ_DOF_THRESHOLD ? F2 : F1;
   return isvector(x) ? x.apply(F) : F(x);
}


/**
 * Inverse cumulative distribution (quantile) function for chi-square distribution.
 *
 * @param {number|Vector|Array} p - probability or vector/array with probabilities.
 * @param {number} df - degrees of freedom.
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
export function qchisq(p, df) {
   if (Array.isArray(p)) {
      return qchisq(vector(p), df);
   }

   // compute parameters of normal distribution for power approximation
   const mu = 5 / 6 - 1 / (9 * df) - 7 / (648 * df * df) + 25 / (2187 * df * df * df);
   const sigma2 = 1 / (18 * df) + 1 / (162 * df * df) - 37 / (1164 * df * df * df);
   const sigma = Math.sqrt(sigma2);

   // compute quantile for one probability value using power approximations
   // and inverse solution
   function F2(x) {
      if (x === 0) return 0;
      if (x === 1) return Inf;
      if (df === 0) return 0;
      const l = qnorm(x, mu, sigma);
      const o = Math.pow(Math.sqrt(36 * l * l - 30 * l + 13) / 4 + (3 * l - 3 / 2) / 2 + 1 / 8, 1 / 3);
      return Math.pow(o - 3 / (4 * o) + 0.5, 6) * df;
   }

   // compute quantile using sequential splits of quantile range
   function F1(x) {
      if (x === 0) return 0;
      if (x === 1) return Inf;
      if (df === 0) return 0;

      const pfun = (v) => pchisq(v, df);
      return quantsearch(x, pfun, 0, CHISQ_DOF_THRESHOLD * 10);
   }

   const F = df > CHISQ_DOF_THRESHOLD ? F2 : F1;
   return isvector(p) ? p.apply(F) : F(p);
}



/******************************************************/
/* Beta distribution                            */
/******************************************************/


/**
 * Cumulative distribution function for beta distribution
 * (a.k.a regularized incomplete beta function I_x(a, b)).
 *
 * @param {number|Vector|Array} x - argument value or vector/array.
 * @param {number} a - degrees of freedom.
 * @param {number} b - degrees of freedom.
 *
 * @description
 *
 * @returns {number|Vector} computed probabilities.
 *
 */
function pbeta(x, a, b) {

   if (Array.isArray(x)) {
      return pbeta(vector(x), a, b);
   }

   if (isvector(x)) {
      return x.apply(v => pbeta(v, a, b));
   }

   // check corner cases for x
   if (x <= 0) return 0;
   if (x >= 1) return 1;

   // check corner cases for a and b
   if (a == 0 || b == 0 || !Number.isFinite(a) || !Number.isFinite(b)) {
      if (a == 0 && b == 0) return 0.5;
      if (a == 0 || a / b == 0) return 1.0;
      if (b == 0 || b / a == 0) return 0.0;
      // remaining corner case when both a and b are infinite
      return x < 0.5 ? 0.0 : 1.0;
   }

   const x1 = 0.5 - x + 0.5;
   return bratio(a, b, x, x1);
}

/******************************************************/
/* Helper functions                                   */
/******************************************************/

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
      throw new Error("gamma: the function only works with arguments > 0.");
   }

   // coefficients
   const p = [
    676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012,
    9.9843695780195716e-6, 1.5056327351493116e-7,
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
      sum += s;
   }

   return Math.pow(x, a) * gamma(a) * Math.exp(-x) * sum;
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
   const sign = x >= 0 ? 1 : -1;
   x = Math.abs(x);

   // constants
   const a1 = 0.254829592;
   const a2 = -0.284496736;
   const a3 = 1.421413741;
   const a4 = -1.453152027;
   const a5 = 1.061405429;
   const p = 0.3275911;

   // approximation
   const t = 1.0 / (1.0 + p * x);
   const y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

   return sign * y;
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
  return Math.exp(lgamma(x) + lgamma(y) - lgamma(x + y));
}


/**
 * Evaluation of complementary error function
 * @param {Number} ind a flag (0/1)
 * @param {Number} x an argument
 * @returns {Number} erfx(x) if ind = 0, exp(x * x) * erfc(x) otherwise
 */
function erfc1(ind, x) {
    /* Initialized data */

   const c = .564189583547756;
   const a = [7.7105849500132e-5,-.00133733772997339,
	    .0323076579225834,.0479137145607681,.128379167095513];
   const b = [.00301048631703895,.0538971687740286,
	    .375795757275549];
   const p = [-1.36864857382717e-7,.564195517478974,
	    7.21175825088309,43.1622272220567,152.98928504694,
	    339.320816734344,451.918953711873,300.459261020162];
   const q = [1.,12.7827273196294,77.0001529352295,
	    277.585444743988,638.980264465631,931.35409485061,
	    790.950925327898,300.459260956983];
   const r = [2.10144126479064,26.2370141675169,
	    21.3688200555087,4.6580782871847,.282094791773523];
   const s = [94.153775055546,187.11481179959,
	    99.0191814623914,18.0124575948747];

   let ret_val;
   let e, t, w, bot, top;
   let ax = Math.abs(x);
   if (ax <= 0.5) {
	   t = x * x;
	   top = (((a[0] * t + a[1]) * t + a[2]) * t + a[3]) * t + a[4] + 1.;
	   bot = ((b[0] * t + b[1]) * t + b[2]) * t + 1.;
	   ret_val = 0.5 - x * (top / bot) + 0.5;
   	if (ind != 0) {
         ret_val = Math.exp(t) * ret_val;
	   }
	   return ret_val;
   }

   if (ax <= 4.) {
   	top = ((((((p[0] * ax + p[1]) * ax + p[2]) * ax + p[3]) * ax + p[4]) * ax + p[5]) * ax + p[6]) * ax + p[7];
	   bot = ((((((q[0] * ax + q[1]) * ax + q[2]) * ax + q[3]) * ax + q[4]) * ax + q[5]) * ax + q[6]) * ax + q[7];
	   ret_val = top / bot;
   } else {
   	if (x <= -5.6) {
   	    ret_val = 2.;
	       if (ind != 0) {
		      ret_val = Math.exp(x * x) * 2.;
	      }
	      return ret_val;
	   }

      if (ind == 0 && (x > 100. || x * x > -exparg(1))) {
         return 0.;
	   }

      t = 1. / (x * x);
      top = (((r[0] * t + r[1]) * t + r[2]) * t + r[3]) * t + r[4];
      bot = (((s[0] * t + s[1]) * t + s[2]) * t + s[3]) * t + 1.;
      ret_val = (c - t * top / bot) / ax;
   }

   if (ind != 0) {
   	if (x < 0.) {
         ret_val = Math.exp(x * x) * 2. - ret_val;
      }
   } else {
      w = x * x;
      t = w;
      e = w - t;
      ret_val = (0.5 - e + 0.5) * Math.exp(-t) * ret_val;
      if (x < 0.) {
         ret_val = 2. - ret_val;
      }
   }
   return ret_val;

}


/**
 * log-gamma function
 * @param {Number} z
 * @returns function value (single)
 */
function lgamma(z) {
  const g = 7,
    p = [
      0.99999999999980993, 676.5203681218851, -1259.1392167224028,
      771.32342877765313, -176.61502916214059, 12.507343278686905,
      -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
    ];

  if (z < 0.5) {
    return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * z)) - lgamma(1 - z);
  } else {
    z -= 1;
    let x = p[0];
    for (let i = 1; i < g + 2; i++) {
      x += p[i] / (z + i);
    }
    var t = z + g + 0.5;
    return (
      Math.log(2 * Math.PI) / 2 + (z + 0.5) * Math.log(t) - t + Math.log(x)
    );
  }
}


/**
 * Evaluation of the function ln(gamma(a + b))
 * @param {Number} a
 * @param {Number} b
 * @returns function value.
 */
function gsumln(a, b) {

   let x = a + b - 2.;

   if (x <= 0.25) return gamln1(x + 1.);
   if (x <= 1.25) return gamln1(x) + alnrel(x);
   return gamln1(x - 1.) + Math.log(x * (x + 1.));
}


/**
 * Evaluation of del(a0) + del(b0) - del(a0 + b0)
 * @param {Number} a0
 * @param {Number} b0
 * @returns function value
 */
function bcorr(a0, b0) {
   const c0 = .0833333333333333;
   const c1 = -.00277777777760991;
   const c2 = 7.9365066682539e-4;
   const c3 = -5.9520293135187e-4;
   const c4 = 8.37308034031215e-4;
   const c5 = -.00165322962780713;

   let ret_val, r1;
   let a, b, c, h, t, w, x, s3, s5, x2, s7, s9, s11;

   a = Math.min(a0, b0);
   b = Math.max(a0, b0);

   h = a / b;
   c = h / (h + 1.);
   x = 1. / (h + 1.);
   x2 = x * x;

   s3 = x + x2 + 1.;
   s5 = x + x2 * s3 + 1.;
   s7 = x + x2 * s5 + 1.;
   s9 = x + x2 * s7 + 1.;
   s11 = x + x2 * s9 + 1.;

   r1 = 1. / b;
   t = r1 * r1;
   w = ((((c5 * s11 * t + c4 * s9) * t + c3 * s7) * t + c2 * s5) * t + c1 *
      s3) * t + c0;
   w *= c / b;

   r1 = 1. / a;
   t = r1 * r1;
   ret_val = (((((c5 * t + c4) * t + c3) * t + c2) * t + c1) * t + c0) / a +
      w;

   return ret_val;
}

/**
 * Computation of 1 / gamma(1 + a)
 * @param {Number} a
 * @returns function value.
 */
function gam1(a) {
   let d, t, w, bot, top;

   t = a;
   d = a - 0.5;
   if (d > 0.) t = d - 0.5;
   if (t < 0.) { /* L30: */
	   const r = [-.422784335098468,-.771330383816272,
		     -.244757765222226,.118378989872749,9.30357293360349e-4,
		     -.0118290993445146,.00223047661158249,2.66505979058923e-4,
		     -1.32674909766242e-4];
	   const s1 = .273076135303957;
	   const s2 = .0559398236957378;

   	top = (((((((r[8] * t + r[7]) * t + r[6]) * t + r[5]) * t + r[4]) * t + r[3]) * t + r[2]) * t + r[1]) * t + r[0];
	   bot = (s2 * t + s1) * t + 1.;
	   w = top / bot;
      return d > 0. ? t * w / a : a * (w + 0.5 + 0.5);
   } else if (t == 0) {
	   return 0.;
   } else {
	   const p = [.577215664901533,-.409078193005776,
		     -.230975380857675,.0597275330452234,.0076696818164949,
		     -.00514889771323592,5.89597428611429e-4];
	   const q = [1.,.427569613095214,.158451672430138,
		     .0261132021441447,.00423244297896961];

	   top = (((((p[6] * t + p[5]) * t + p[4]) * t + p[3]) * t + p[2]) * t + p[1]) * t + p[0];
   	bot = (((q[4] * t + q[3]) * t + q[2]) * t + q[1]) * t + 1.;
	   w = top / bot;
	   return d > 0. ? t / a * (w - 0.5 - 0.5) : a * w;
   }
}

/**
 *
 * @param {*} x
 * @param {*} a
 * @param {*} n
 * @returns
 */
function chebyshev_eval(x, a, n) {
   let b0, b1, b2, twox;

   if (n > 1000) return Number.NaN;
   if (x < -1.1 || x > 1.1) return Number.NaN;

   twox = x * 2;
   b2 = b1 = 0;
   b0 = 0;
   for (let i = 1; i <= n; i++) {
	   b2 = b1;
	   b1 = b0;
	   b0 = twox * b1 - b2 + a[n - i];
   }
   return (b0 - b2) * 0.5;
}

/**
 * More precise evaluation of log(1 + x) for small x.
 * @param {Number} x
 * @returns function value.
 */
function log1p(x) {
   const alnrcs = [
	+.10378693562743769800686267719098e+1,
	-.13364301504908918098766041553133e+0,
	+.19408249135520563357926199374750e-1,
	-.30107551127535777690376537776592e-2,
	+.48694614797154850090456366509137e-3,
	-.81054881893175356066809943008622e-4,
	+.13778847799559524782938251496059e-4,
	-.23802210894358970251369992914935e-5,
	+.41640416213865183476391859901989e-6,
	-.73595828378075994984266837031998e-7,
	+.13117611876241674949152294345011e-7,
	-.23546709317742425136696092330175e-8,
	+.42522773276034997775638052962567e-9,
	-.77190894134840796826108107493300e-10,
	+.14075746481359069909215356472191e-10,
	-.25769072058024680627537078627584e-11,
	+.47342406666294421849154395005938e-12,
	-.87249012674742641745301263292675e-13,
	+.16124614902740551465739833119115e-13,
	-.29875652015665773006710792416815e-14,
	+.55480701209082887983041321697279e-15,
	-.10324619158271569595141333961932e-15,
	+.19250239203049851177878503244868e-16,
	-.35955073465265150011189707844266e-17,
	+.67264542537876857892194574226773e-18,
	-.12602624168735219252082425637546e-18,
	+.23644884408606210044916158955519e-19,
	-.44419377050807936898878389179733e-20,
	+.83546594464034259016241293994666e-21,
	-.15731559416479562574899253521066e-21,
	+.29653128740247422686154369706666e-22,
	-.55949583481815947292156013226666e-23,
	+.10566354268835681048187284138666e-23,
	-.19972483680670204548314999466666e-24,
	+.37782977818839361421049855999999e-25,
	-.71531586889081740345038165333333e-26,
	+.13552488463674213646502024533333e-26,
	-.25694673048487567430079829333333e-27,
	+.48747756066216949076459519999999e-28,
	-.92542112530849715321132373333333e-29,
	+.17578597841760239233269760000000e-29,
	-.33410026677731010351377066666666e-30,
	+.63533936180236187354180266666666e-31,
   ];

   const nlnrel = 22;
   const xmin = -0.999999985;

   if (x == 0.) return 0.;/* speed */
   if (x == -1) return(Number.NEGATIVE_INFINITY);
   if (x  < -1) return(Number.NaN);

   if (Math.abs(x) <= .375) {
   	if(Math.abs(x) < .5 * Number.EPSILON) return x;
	   if( (0 < x && x < 1e-8) || (-1e-9 < x && x < 0)) return x * (1 - .5 * x);
   	return x * (1 - x * chebyshev_eval(x / .375, alnrcs, nlnrel));
   }

   if (x < xmin) {
      throw new Error('log1p: error evaluating the functio value.')
   }

   return Math.log(1 + x);
}

/**
 * Computation of ln(gamma(b) / gamma(a + b)) when b ≥ 8
 * @param {Number} a
 * @param {Number} b
 * @returns function value
 */
function algdiv(a, b) {
   const c0 = .0833333333333333;
   const c1 = -.00277777777760991;
   const c2 = 7.9365066682539e-4;
   const c3 = -5.9520293135187e-4;
   const c4 = 8.37308034031215e-4;
   const c5 = -.00165322962780713;

   let c, d, h, t, u, v, w, x, s3, s5, x2, s7, s9, s11;

   if (a > b) {
      h = b / a;
      c = 1. / (h + 1.);
      x = h / (h + 1.);
      d = a + (b - 0.5);
   } else {
      h = a / b;
      c = h / (h + 1.);
      x = 1. / (h + 1.);
      d = b + (a - 0.5);
   }

   x2 = x * x;
   s3 = x + x2 + 1.;
   s5 = x + x2 * s3 + 1.;
   s7 = x + x2 * s5 + 1.;
   s9 = x + x2 * s7 + 1.;
   s11 = x + x2 * s9 + 1.;

   t = 1./ (b * b);
   w = ((((c5 * s11 * t + c4 * s9) * t + c3 * s7) * t + c2 * s5) * t + c1 * s3) * t + c0;
   w *= c / b;

   u = d * alnrel(a / b);
   v = a * (Math.log(b) - 1.);

   return u > v ? w - v - u : w - u - v;
}

/**
 * Evaluation of ln(gamma(1 + a)) for –0.2 ≤ a ≤ 1.25
 * @param {Number} a
 * @returns function value.
 */
function gamln1(a) {
   if (a < 0.6) {
	   const p0 = .577215664901533;
	   const p1 = .844203922187225;
	   const p2 = -.168860593646662;
	   const p3 = -.780427615533591;
	   const p4 = -.402055799310489;
	   const p5 = -.0673562214325671;
	   const p6 = -.00271935708322958;
	   const q1 = 2.88743195473681;
	   const q2 = 3.12755088914843;
	   const q3 = 1.56875193295039;
	   const q4 = .361951990101499;
	   const q5 = .0325038868253937;
	   const q6 = 6.67465618796164e-4;
	   const w = ((((((p6 * a + p5)* a + p4)* a + p3)* a + p2)* a + p1)* a + p0) /
	      ((((((q6 * a + q5)* a + q4)* a + q3)* a + q2)* a + q1)* a + 1.);
	   return -(a) * w;
   } else {
	   const r0 = .422784335098467;
	   const r1 = .848044614534529;
	   const r2 = .565221050691933;
	   const r3 = .156513060486551;
	   const r4 = .017050248402265;
	   const r5 = 4.97958207639485e-4;
	   const s1 = 1.24313399877507;
	   const s2 = .548042109832463;
	   const s3 = .10155218743983;
	   const s4 = .00713309612391;
	   const s5 = 1.16165475989616e-4;
	   const x = a - 0.5 - 0.5;
	   const w = (((((r5 * x + r4) * x + r3) * x + r2) * x + r1) * x + r0) /
	       (((((s5 * x + s4) * x + s3) * x + s2) * x + s1) * x + 1.);
	   return x * w;
   }
}

/**
 * Evaluation of  ln(gamma(a))  for positive a
 * @param {Number} a
 * @returns function value
 */
function gamln(a) {
   const d  = .418938533204673;
   const c0 = .0833333333333333;
   const c1 = -.00277777777760991;
   const c2 = 7.9365066682539e-4;
   const c3 = -5.9520293135187e-4;
   const c4 = 8.37308034031215e-4;
   const c5 = -.00165322962780713;

   if (a <= 0.8) return gamln1(a) - Math.log(a);
   if (a <= 2.25) return gamln1(a - 0.5 - 0.5);

   if (a < 10.) {
   	let n = Math.round(a - 1.25);
	   let t = a;
	   let w = 1.;
	   for (let i = 1; i <= n; ++i) {
	      t += -1.;
	      w *= t;
	   }
	   return gamln1(t - 1.) + Math.log(w);
   }

	let t = 1. / (a * a);
	let w = (((((c5 * t + c4) * t + c3) * t + c2) * t + c1) * t + c0) / a;
	return d + w + (a - 0.5) * (Math.log(a) - 1.);
}

/**
 * Evaluation of the logarithm of the beta function  ln(beta(a0,b0))
 * @param {Number} a0
 * @param {Number} b0
 * @returns function value
 */
function betaln(a0, b0) {

   const e = .918938533204673;
	let a = Math.min(a0, b0);
	let b = Math.max(a0, b0);

   function l140(w, a, b) {
	   let n = Math.round(b - 1);
      let z = 1.;
	   for (let i = 1; i <= n; ++i) {
		   b += -1.;
		   z *= b / (a + b);
      }
      return w + Math.log(z) + (gamln(a) + (gamln(b) - gsumln(a, b)));
   }

   if (a < 8.) {

	   if (a < 1.) {
	      if (b < 8.) {
		      return gamln(a) + (gamln(b) - gamln(a+b));
         } else {
   		   return gamln(a) + algdiv(a, b);
         }
	   }

   	let w;
	   if (a < 2.) {
	      if (b <= 2.) {
		      return gamln(a) + gamln(b) - gsumln(a, b);
	      }

   	    if (b < 8.) {
	      	w = 0.;
		      return l140(w, a, b);
	      }

	      return gamln(a) + algdiv(a, b);
      }

      if (b <= 1e3) {
         let n = Math.round(a - 1);
         w = 1.;
         for (let i = 1; i <= n; ++i) {
            a += -1.;
            let h = a / b;
            w *= h / (h + 1.);
         }
         w = Math.log(w);

         if (b >= 8.) {
            return w + gamln(a) + algdiv(a, b);
         }

         return l140(w, a, b);

      } else {

         let n = Math.round(a - 1);
         w = 1.;
         for (let i = 1; i <= n; ++i) {
            a += -1.;
            w *= a / (a / b + 1.);
         }
         return log(w) - n * log(b) + (gamln(a) + algdiv(a, b));
      }

    } else {
	   let w = bcorr(a, b);
	   let h = a / b;
	   let u = -(a - 0.5) * Math.log(h / (h + 1.));
	   let v = b * alnrel(h);
   	return (u > v) ? Math.log(b) * -0.5 + e + w - v - u : Math.log(b) * -0.5 + e + w - u - v;
   }
}

/**
 * If l = 0 then returns largest positive W for which exp(W) can be computed.
 * @param {Number} l
 * @returns
 */
function exparg(l) {
   const lnb = .69314718055995;
   let m = (l == 0) ? 1024 : -300; // TODO: constants are chosen manually
   return m * lnb * .99999;
}

/**
 * Evaluation of the function  x - ln(1 + x)
 * @param {Number} x
 * @returns
 */
function rlog1(x) {
   const a = .0566749439387324;
   const b = .0456512608815524;
   const p0 = .333333333333333;
   const p1 = -.224696413112536;
   const p2 = .00620886815375787;
   const q1 = -1.27408923933623;
   const q2 = .354508718369557;

   let h, r, t, w, w1;
   if (x < -0.39 || x > 0.57) {
	   w = x + 0.5 + 0.5;
	   return x - Math.log(w);
   }

   if (x < -0.18) {
      h = x + .3;
      h /= .7;
      w1 = a - h * .3;
   } else if (x > 0.18) {
      h = x * .75 - .25;
      w1 = b + h / 3.;
   } else {
      h = x;
      w1 = 0.;
   }

   r = h / (h + 2.);
   t = r * r;
   w = ((p2 * t + p1) * t + p0) / ((q2 * t + q1) * t + 1.);
   return t * 2. * (1. / (1. - r) - r * w) + w1;
}

/**
 * Evaluation of the function ln(1 + a)
 * @param {Number} a
 * @returns function value
 */
function alnrel(a) {

   if (Math.abs(a) > 0.375) return Math.log(1. + a);

	const p1 = -1.29418923021993;
	const p2 = .405303492862024;
	const p3 = -.0178874546012214;
	const q1 = -1.62752256355323;
	const q2 = .747811014037616;
	const q3 = -.0845104217945565;

	let t = a / (a + 2.);
	let t2 = t * t;
	let w = (((p3 * t2 + p2) * t2 + p1) * t2 + 1.) / (((q3 * t2 + q2) * t2 + q1) * t2 + 1.);

   return t * 2. * w;
}

/**
 * Evaluation of the real error function.
 * @param {Number} x
 */
function erf__(x) {
   const c = .564189583547756;
   const a = [7.7105849500132e-5, -.00133733772997339, .0323076579225834, .0479137145607681, .128379167095513];
   const b = [.00301048631703895, .0538971687740286, .375795757275549];
   const p = [-1.36864857382717e-7,.564195517478974,
	    7.21175825088309,43.1622272220567,152.98928504694,
	    339.320816734344,451.918953711873,300.459261020162];
   const q = [1., 12.7827273196294, 77.0001529352295,
	    277.585444743988,638.980264465631,931.35409485061,
	    790.950925327898,300.459260956983];
   const r = [2.10144126479064, 26.2370141675169,
	    21.3688200555087,4.6580782871847,.282094791773523];
   const s = [94.153775055546,187.11481179959,
	    99.0191814623914,18.0124575948747];

   let t, x2, ax, bot, top;

   ax = Math.abs(x);
   if (ax <= 0.5) {
      t = x * x;
      top = (((a[0] * t + a[1]) * t + a[2]) * t + a[3]) * t + a[4] + 1.;
      bot = ((b[0] * t + b[1]) * t + b[2]) * t + 1.;
   	return x * (top / bot);
   }

   if (ax <= 4.) {
   	top = ((((((p[0] * ax + p[1]) * ax + p[2]) * ax + p[3]) * ax + p[4]) * ax
	   	+ p[5]) * ax + p[6]) * ax + p[7];
	   bot = ((((((q[0] * ax + q[1]) * ax + q[2]) * ax + q[3]) * ax + q[4]) * ax
		   + q[5]) * ax + q[6]) * ax + q[7];
	   let R = 0.5 - Math.exp(-x * x) * top / bot + 0.5;
	   return (x < 0) ? -R : R;
   }

   if (ax >= 5.8) {
   	return x > 0 ? 1 : -1;
   }

   x2 = x * x;
   t = 1. / x2;
   top = (((r[0] * t + r[1]) * t + r[2]) * t + r[3]) * t + r[4];
   bot = (((s[0] * t + s[1]) * t + s[2]) * t + s[3]) * t + 1.;
   t = (c - top / (x2 * bot)) / ax;
   let R = 0.5 - Math.exp(-x2) * t + 0.5;
   return (x < 0) ? -R : R;
}


/**
 * Evaluation of the function exp(x) - 1
 * @param {Number} x
 * @returns function value
 */
function rexpm1(x) {
   const p1 = 9.14041914819518e-10;
   const p2 = .0238082361044469;
   const q1 = -.499999999085958;
   const q2 = .107141568980644;
   const q3 = -.0119041179760821;
   const q4 = 5.95130811860248e-4;

   if (Math.abs(x) <= 0.15) {
      return x * (((p2 * x + p1) * x + 1.) / ((((q4 * x + q3) * x + q2) * x + q1) * x + 1.));
   } else {
   	let w = Math.exp(x);
	   return x > 0. ? w * (0.5 - 1. / w + 0.5) : w - 0.5 - 0.5;
   }
}

function esum(mu, x) {

   let w;
   if (x > 0.) {
	   if (mu > 0) return Math.exp(mu) * Math.exp(x);
	   w = mu + x;
	   if (w < 0.) return Math.exp(mu) * Math.exp(x);
   } else {
	   if (mu < 0) return Math.exp(mu) * Math.exp(x);
	   w = mu + x;
	   if (w > 0.) return Math.exp(mu) * Math.exp(x);
   }

   return Math.exp(w);
}

function psi(x) {

   const piov4 = .785398163397448;
   const dx0 = 1.461632144968362341262659542325721325;

   const p1 = [.0089538502298197,4.77762828042627,
	    142.441585084029,1186.45200713425,3633.51846806499,
	    4138.10161269013,1305.60269827897];
   const q1 = [44.8452573429826, 520.752771467162,
	    2210.0079924783, 3641.27349079381, 1908.310765963,
	    6.91091682714533e-6];
   const p2 = [-2.12940445131011,-7.01677227766759,
	    -4.48616543918019,-.648157123766197 ];
   const q2 = [32.2703493791143, 89.2920700481861,
	    54.6117738103215, 7.77788548522962];

   let i, m, n, nq;
   let d2;
   let w, z;
   let den, aug, sgn, xmx0, xmax1, upper, xsmall;

   xmax1 = Number.MAX_SAFE_INTEGER;
   d2 = 0.5 / (0.5 * Number.EPSILON);
   if(xmax1 > d2) xmax1 = d2;
   xsmall = 1e-9;

   aug = 0.;
   if (x < 0.5) {
   	if (fabs(x) <= xsmall) {
	      if (x == 0.) return 0;
         aug = -1. / x;
	   } else {
         w = -x;
         sgn = piov4;

         if (w <= 0.) {
		      w = -w;
		      sgn = -sgn;
	      }

         if (w >= xmax1) return 0;

	      nq = Math.round(w);
	      w -= nq;
	      nq = Math.round(w * 4.);
	      w = (w - nq * 0.25) * 4.;

   	   n = nq / 2;
	      if (n + n != nq) {
		      w = 1. - w;
	      }

         z = piov4 * w;
	      m = n / 2;
	      if (m + m != n) {
		      sgn = -sgn;
	      }

         n = (nq + 1) / 2;
	      m = n / 2;
	      m += m;
	      if (m == n) {
      		if (z == 0.) return 0;
   		   aug = sgn * (cos(z) / sin(z) * 4.);
	      } else {
		      aug = sgn * (sin(z) / cos(z) * 4.);
	      }
	   }

	   x = 1. - x;
   }


   if (x <= 3.) {
	   den = x;
	   upper = p1[0] * x;

	   for (let i = 1; i <= 5; ++i) {
	      den = (den + q1[i - 1]) * x;
	      upper = (upper + p1[i]) * x;
	   }

	   den = (upper + p1[6]) / (den + q1[5]);
	   xmx0 = x - dx0;
	   return den * xmx0 + aug;
   }

   if (x < xmax1) {
	   w = 1. / (x * x);
	   den = w;
	   upper = p2[0] * w;

	   for (let i = 1; i <= 3; ++i) {
	      den = (den + q2[i - 1]) * w;
	      upper = (upper + p2[i]) * w;
	   }

	   aug = upper / (den + q2[3]) - 0.5 / x + aug;
   }

   return aug + log(x);
}

function fsper(a, b, x, eps) {

   let ans, c, s, t, an, tol;

   if (a > eps * 0.001) {
	   t = a * Math.log(x);
	   if (t < Math.exparg(1)) {
	      return 0.;
	   }
	   ans = Math.exp(t);
   } else {
	   ans = 1.;
   }

	ans *= b / a;
   tol = eps / a;
   an = a + 1.;
   t = x;
   s = t / an;


   // first iteration for do while
   an += 1.;
   t = x * t;
   c = t / an;
   s += c;

   while (Math.abs(x) > tol) {
      an += 1.;
      t = x * t;
      c = t / an;
      s += c;
   }

	ans *= a * s + 1.;
   return ans;
}

function apser(a, b, x, eps) {

   const g = .577215664901533;

   let tol, c, j, s, t, aj;
   let bx = b * x;

   t = x - bx;
   c = b * eps <= 0.02 ? Math.log(x) + psi(b) + g + t : Math.log(bx) + g + t;
   tol = eps * 5. * Math.abs(c);
   j = 1.;
   s = 0.;

   // first iteration of do while
   j += 1.;
   t *= x - bx / j;
   aj = t / j;
   s += aj;
   while (Math.abs(aj) > tol) {
   	j += 1.;
	   t *= x - bx / j;
	   aj = t / j;
	   s += aj;
   }

   return -a * (c + s);
}

function bpser(a, b, x, eps) {
   let m;
   let ans, c, t, u, z, a0, b0, apb;

   if (x == 0.) return 0.0;

   a0 = Math.min(a, b);
   if (a0 >= 1.) {
	   z = a * Math.log(x) - betaln(a, b);
	   ans = Math.exp(z) / a;
   } else {
	   b0 = Math.max(a, b);

   	if (b0 < 8.) {

	      if (b0 <= 1.) {

            ans = Math.pow(x, a);
            if (ans == 0.) return ans;


            apb = a + b;
            if (apb > 1.) {
               u = a + b - 1.;
               z = (gam1(u) + 1.) / apb;
            } else {
               z = gam1(apb) + 1.;
            }

            c = (gam1(a) + 1.) * (gam1(b) + 1.) / z;
            ans *=  c * (b / apb);

         } else {
            u = gamln1(a0);
            m = Math.round(b0 - 1.);
            if (m >= 1) {
               c = 1.;
               for (let i = 1; i <= m; ++i) {
                  b0 += -1.;
                  c *= b0 / (a0 + b0);
               }
               u += Math.log(c);
            }

            z = a * Math.log(x) - u;
            b0 += -1.;
            apb = a0 + b0;
            if (apb > 1.) {
               u = a0 + b0 - 1.;
               t = (gam1(u) + 1.) / apb;
            } else {
               t = gam1(apb) + 1.;
            }

            ans = Math.exp(z) * (a0 / a) * (gam1(b0) + 1.) / t;
         }
      } else {
	      u = gamln1(a0) + algdiv(a0, b0);
	      z = a * Math.log(x) - u;
		   ans = a0 / a * Math.exp(z);
	   }
   }

   if (ans == 0 || a <= eps * 0.1) return ans;

   let tol = eps / a;
   let n = 0.;
   let sum = 0.
   let w;
   c = 1.;

   // first iteration of do while
   n += 1.;
   c *= (0.5 - b / n + 0.5) * x;
   w = c / (a + n);
   sum += w;

   while (n < 1e7 && Math.abs(w) > tol) {
	   n += 1.;
	   c *= (0.5 - b / n + 0.5) * x;
	   w = c / (a + n);
	   sum += w;
   }
   if (a * sum > -1.) {
	   ans *= (a * sum + 1.);
   } else {
	   ans = 0.;
   }

   return ans;
}

function bup(a, b, x, y, n, eps) {

   let ret_val;
   let k, mu;
   let d, l;

   let apb = a + b,
	ap1 = a + 1.;

   if (n > 1 && a >= 1. && apb >= ap1 * 1.1) {
	   mu = Math.round(Math.abs(exparg(1)));
	   k = Math.round(exparg(0));
	   if (mu > k) mu = k;
 	   d = Math.exp(-mu);
   }  else {
	   mu = 0;
	   d = 1.;
   }

    ret_val = brcmp1(mu, a, b, x, y) / a;
    if (n == 1 || ret_val == 0.0) return ret_val;

   let nm1 = n - 1;
   let w = d;

   k = 0;
   if (b > 1.) {
	   if (y > 1e-4) {
	      let r = (b - 1.) * x / y - a;
	      if (r >= 1.) k = (r < nm1) ? Math.round(r) : nm1;
      } else {
	      k = nm1;
      }

      for (let i = 0; i < k; ++i) {
	      l = i;
	      d *= (apb + l) / (ap1 + l) * x;
	      w += d;
	   }
   }


   for (let i = k; i < nm1; ++i) {
	   l = i;
	   d *= (apb + l) / (ap1 + l) * x;
	   w += d;
	   if (d <= eps * w) break;
   }

   ret_val *= w;
   return ret_val;
}

function bfrac(a, b, x, y, lambda, eps) {
   let c, e, n, p, r, s, t, w, c0, c1, r0, an, bn, yp1, anp1, bnp1, beta, alpha, brc;
   if (!Number.isFinite(lambda)) return Number.NaN;

   brc = brcomp(a, b, x, y);
   if (Number.isNaN(brc)) return Number.NaN;
   if (brc == 0.) return 0.;

   c = lambda + 1.;
   c0 = b / a;
   c1 = 1. / a + 1.;
   yp1 = y + 1.;

   n = 0.;
   p = 1.;
   s = a + 1.;
   an = 0.;
   bn = 1.;
   anp1 = 1.;
   bnp1 = c / c1;
   r = c1 / c;

   // first iteration
   n += 1.;
   t = n / a;
   w = n * (b - n) * x;
   e = a / s;
   alpha = p * (p + c0) * e * e * (w * x);
   e = (t + 1.) / (c1 + t + t);
   beta = n + w / s + e * (c + n * yp1);
   p = t + 1.;
   s += 2.;

   t = alpha * an + beta * anp1;	an = anp1;	anp1 = t;
   t = alpha * bn + beta * bnp1;	bn = bnp1;	bnp1 = t;

   r0 = r;
   r = anp1 / bnp1;

   an /= bnp1;
   bn /= bnp1;
   anp1 = r;
   bnp1 = 1.;

   while (n < 10000) {
      n += 1.;
      t = n / a;
      w = n * (b - n) * x;
      e = a / s;
      alpha = p * (p + c0) * e * e * (w * x);
      e = (t + 1.) / (c1 + t + t);
      beta = n + w / s + e * (c + n * yp1);
      p = t + 1.;
      s += 2.;

      t = alpha * an + beta * anp1;	an = anp1;	anp1 = t;
      t = alpha * bn + beta * bnp1;	bn = bnp1;	bnp1 = t;

	   r0 = r;
	   r = anp1 / bnp1;
   	if (Math.abs(r - r0) <= eps * r) break;

	   an /= bnp1;
	   bn /= bnp1;
	   anp1 = r;
	   bnp1 = 1.;
   }

   return brc * r;
}

function brcomp(a, b, x, y) {

   const const__ = .398942280401433;
   let n, c, e, u, v, z, a0, b0, apb;

   if (x == 0. || y == 0.) return 0.0;

   a0 = Math.min(a, b);

   if (a0 < 8.) {

      let lnx, lny;
	   if (x <= .375) {
	      lnx = Math.log(x);
	      lny = alnrel(-x);
	   } else {
	      if (y > .375) {
		      lnx = Math.log(x);
		      lny = Math.log(y);
	      } else {
		      lnx = alnrel(-y);
		      lny = Math.log(y);
	      }
	   }

   	z = a * lnx + b * lny;
	   if (a0 >= 1.) {
	      z -= betaln(a, b);
	      return Math.exp(z);
	   }

	   b0 = Math.max(a, b);
	   if (b0 >= 8.) {
	      u = gamln1(a0) + algdiv(a0, b0);
	      return a0 * Math.exp(z - u);
	   }

	   if (b0 <= 1.) {

	      let e_z = Math.exp(z);
	      if (e_z == 0.) return 0.;

	      apb = a + b;
	      if (apb > 1.) {
		      u = a + b - 1.;
		      z = (gam1(u) + 1.) / apb;
	      } else {
		      z = gam1(apb) + 1.;
	      }

	      c = (gam1(a) + 1.) * (gam1(b) + 1.) / z;
	      return e_z * (a0 * c) / (a0 / b0 + 1.);
   	}

	   u = gamln1(a0);
	   n = Math.round(b0 - 1.);
	   if (n >= 1) {
	      c = 1.;
	      for (let i = 1; i <= n; ++i) {
		      b0 += -1.;
		      c *= b0 / (a0 + b0);
	      }
	      u = Math.log(c) + u;
	   }

	   z -= u;
	   b0 += -1.;
	   apb = a0 + b0;
	   let t;
	   if (apb > 1.) {
	      u = a0 + b0 - 1.;
	      t = (gam1(u) + 1.) / apb;
	   } else {
	      t = gam1(apb) + 1.;
	   }

   	return a0 * Math.exp(z) * (gam1(b0) + 1.) / t;
   } else {

      let h, x0, y0, lambda;
	   if (a <= b) {
         h = a / b;
         x0 = h / (h + 1.);
         y0 = 1. / (h + 1.);
         lambda = a - (a + b) * x;
	   } else {
         h = b / a;
         x0 = 1. / (h + 1.);
         y0 = h / (h + 1.);
         lambda = (a + b) * y - b;
	   }

	   e = -lambda / a;
	   u = Math.abs(e) > .6 ? e - Math.log(x / x0) : rlog1(e);
   	e = lambda / b;
      v = Math.abs(e) <= .6 ? rlog1(e) : e - Math.log(y / y0);
	   z = Math.exp(-(a * u + b * v));

	   return const__ * Math.sqrt(b * x0) * z * Math.exp(-bcorr(a, b));
   }
}

function brcmp1(mu, a, b, x, y) {
   const const__ = .398942280401433; /* == 1/sqrt(2*pi); */
   let c, t, u, v, z, a0, b0, apb;

   a0 = Math.min(a, b);
   if (a0 < 8.) {
      let lnx, lny;
      if (x <= .375) {
         lnx = Math.log(x);
         lny = alnrel(-x);
      } else if (y > .375) {
         lnx = Math.log(x);
         lny = Math.log(y);
      } else {
         lnx = alnrel(-y);
         lny = Math.log(y);
      }

   	z = a * lnx + b * lny;
	   if (a0 >= 1.) {
	       z -= betaln(a, b);
	      return esum(mu, z);
	   }

   	b0 = Math.max(a, b);
	   if (b0 >= 8.) {
	      u = gamln1(a0) + algdiv(a0, b0);
	      return a0  * esum(mu, z - u);
	   } else if (b0 <= 1.) {
	      let ans = esum(mu, z);
	      if (ans == 0.) return ans;
	      apb = a + b;
	      if (apb > 1.) {
   		   u = a + b - 1.;
	   	   z = (gam1(u) + 1.) / apb;
         } else {
		      z = gam1(apb) + 1.;
	      }
	      c = (gam1(a) + 1.) * (gam1(b) + 1.) / z;
   	    return ans * (a0 * c) / (a0 / b0 + 1.);
	   }

      u = gamln1(a0);
	   let n = Math.round(b0 - 1.);
	   if (n >= 1) {
	      c = 1.;
	      for (let i = 1; i <= n; ++i) {
		      b0 += -1.;
		      c *= b0 / (a0 + b0);
         }
	       u += Math.log(c);
   	}

   	z -= u;
	   b0 += -1.;
	   apb = a0 + b0;
	   t = apb > 1. ? (gam1(apb - 1.) + 1.) / apb : gam1(apb) + 1.;
   	return a0 * esum(mu, z) * (gam1(b0) + 1.) / t;

   } else {
	   let h, x0, y0, lambda;
	   if (a > b) {
         h = b / a;
         x0 = 1. / (h + 1.);
         y0 = h / (h + 1.);
         lambda = (a + b) * y - b;
	   } else {
         h = a / b;
         x0 = h / (h + 1.);
         y0 = 1. / (h + 1.);
         lambda = a - (a + b) * x;
	   }
	   let lx0 = -log1p(b/a);
	   let e = -lambda / a;
	   u = Math.abs(e) > 0.6 ? e - Math.log(x / x0) : rlog1(e);
	   e = lambda / b;
	   v = Math.abs(e) > 0.6 ? e - Math.log(y / y0) : rlog1(e);

   	z = esum(mu, -(a * u + b * v));
	   return const__ * Math.sqrt(b * x0)      * z * Math.exp(-bcorr(a, b));
   }
}


/**
 * Asymptotic expansion for regularized incomplete beta function when a is larger than b.
 * It is assumed a >= 15 and b <= 1
 * @param {Number} a
 * @param {Number} b
 * @param {Number} x
 * @param {Number} y
 * @param {Number} w
 * @param {Number} eps
 * @returns
 */
function bgrat(a, b, x, y, w, eps) {
   const  n_terms_bgrat = 30;
   let c = Array(n_terms_bgrat);
   let d = Array(n_terms_bgrat);
   let bm1 = b - 0.5 - 0.5;
	let nu = a + bm1 * 0.5;
	let lnx = (y > 0.375) ? Math.log(x) : alnrel(-y);
	let z = -nu * lnx;

   if (b * z == 0.) {
      throw new Error('bgrat: expansion can not be computed.');
   }

   let log_r = Math.log(b) + log1p(gam1(b)) + b * Math.log(z) + nu * lnx;
	let log_u = log_r - (algdiv(b, a) + b * Math.log(nu))
	let u = Math.exp(log_u);
   if (log_u == Number.NEGATIVE_INFINITY) {
      throw new Error('bgrat: expansion can not be computed.');
   }

   let u_0 = (u == 0.);
   let l = (w == 0.0 ? 0. : Math.exp(Math.log(w) - log_u));
   let q_r = grat_r(b, z, log_r, eps);
	let v = 0.25 / (nu * nu);
	let t2 = lnx * 0.25 * lnx;
	let j = q_r;
	let sum = j;
	let t = 1., cn = 1., n2 = 0.;
   for (let n = 1; n <= n_terms_bgrat; ++n) {
	   let bp2n = b + n2;
	   j = (bp2n * (bp2n + 1.) * j + (z + bp2n + 1.) * t) * v;
	   n2 += 2.;
	   t *= t2;
	   cn /= n2 * (n2 + 1.);
	   let nm1 = n - 1;
	   c[nm1] = cn;

      let s = 0.;
	   if (n > 1) {
	      let coef = b - n;
	      for (let i = 1; i <= nm1; ++i) {
		      s += coef * c[i - 1] * d[nm1 - i];
		      coef += b;
	      }
	   }

   	d[nm1] = bm1 * cn + s / n;
	   let dj = d[nm1] * j;
	   sum += dj;
   	if (sum <= 0.) {
         throw new Error('bgrat: expansion can not be computed.');
      }

      if (Math.abs(dj) <= eps * (sum + l)) {
	      break;
	   }
   }

	w += u_0 ? Math.exp(log_u + Math.log(sum)) : u * sum;
   return w;
}

/**
 * Scaled complement of incomplete gamma ratio function
 * @param {*} a
 * @param {*} x
 * @param {*} log_r
 * @param {*} eps
 * @returns function value.
 */
function grat_r(a, x, log_r, eps) {
   const M_SQRT_PI = 1.772453850905516;
   if (a * x == 0.) {
   	if (x <= a) {
	      return Math.exp(-log_r);
	   } else {
	      return 0.;
	   }
   } else if (a == 0.5) {
   	if (x < 0.25) {
	      let p = erf__(Math.sqrt(x));
	      return (0.5 - p + 0.5) * Math.exp(-log_r);
      } else {
	      let sx = Math.sqrt(x)
		   return erfc1(1, sx)/sx * M_SQRT_PI;
	   }
   } else if (x < 1.1) {
	   let an = 3.;
	   let c = x;
	   let sum = x / (a + 3.);
	   let tol = eps * 0.1 / (a + 1.), t;

      // first iteration
      an += 1.;
      c *= -(x / an);
      t = c / (a + an);
      sum += t;
      while (Math.abs(t) > tol) {
         an += 1.;
         c *= -(x / an);
         t = c / (a + an);
         sum += t;
      }

	   let j = a * x * ((sum/6. - 0.5/(a + 2.)) * x + 1./(a + 1.));
	   let z = a * Math.log(x);
	   let h = gam1(a);
	   let g = h + 1.;

   	if ((x >= 0.25 && (a < x / 2.59)) || (z > -0.13394)) {
	      let l = rexpm1(z);
		   let q = ((l + 0.5 + 0.5) * j - l) * g - h;
         return q <= 0.0 ? 0.0 : q * Math.exp(-log_r);
	   } else {
   	   let p = Math.exp(z) * g * (0.5 - j + 0.5);
	      return (0.5 - p + 0.5) * Math.exp(-log_r);
	   }
   } else {

	   let a2n_1 = 1.;
	   let a2n = 1.;
	   let b2n_1 = x;
	   let b2n = x + (1. - a);
	   let c = 1., am0, an0;

      // this is needed because in JS there is no do-while, only while-do
      a2n_1 = x * a2n + c * a2n_1;
      b2n_1 = x * b2n + c * b2n_1;
      am0 = a2n_1 / b2n_1;
      c += 1.;
      let c_a = c - a;
      a2n = a2n_1 + c_a * a2n;
      b2n = b2n_1 + c_a * b2n;
      an0 = a2n / b2n;

      while (Math.abs(an0 - am0) >= eps * an0) {
         a2n_1 = x * a2n + c * a2n_1;
         b2n_1 = x * b2n + c * b2n_1;
         am0 = a2n_1 / b2n_1;
         c += 1.;
         c_a = c - a;
         a2n = a2n_1 + c_a * a2n;
         b2n = b2n_1 + c_a * b2n;
         an0 = a2n / b2n;
   	}

	   return  an0;
   }
}

/**
 * Asymptotic expansion for regularized incomplete beta function when a and b are large.
 * @param {*} a
 * @param {*} b
 * @param {*} lambda
 * @param {*} eps
 * @returns function value.
 */
function basym(a, b, lambda, eps) {
   const num_it = 20;

   const e0 = 1.12837916709551;
   const e1 = .353553390593274;
   const ln_e0 = 0.120782237635245;

   const a0 = Array(num_it + 1);
   const b0 = Array(num_it + 1);
   const c = Array(num_it + 1);
   const d = Array(num_it + 1);

   let f = a * rlog1(-lambda/a) + b * rlog1(lambda/b), t;
	t = Math.exp(-f);
	if (t == 0.) return 0;

   let z0 = Math.sqrt(f);
	let z = z0 / e1 * 0.5;
	let z2 = f + f;
	let h, r0, r1, w0;

   if (a < b) {
	   h = a / b;
	   r0 = 1. / (h + 1.);
	   r1 = (b - a) / b;
	   w0 = 1. / Math.sqrt(a * (h + 1.));
   } else {
      h = b / a;
      r0 = 1. / (h + 1.);
      r1 = (b - a) / a;
      w0 = 1. / Math.sqrt(b * (h + 1.));
   }

   a0[0] = r1 * .66666666666666663;
   c[0] = a0[0] * -0.5;
   d[0] = -c[0];
   let j0 = 0.5 / e0 * erfc1(1, z0);
	let j1 = e1;
	let sum = j0 + d[0] * w0 * j1;

   let s = 1.0;
	let h2 = h * h;
	let hn = 1.;
	let w = w0;
	let znm1 = z;
	let zn = z2;

   for (let n = 2; n <= num_it; n += 2) {
	   hn *= h2;
	   a0[n - 1] = r0 * 2. * (h * hn + 1.) / (n + 2.);
	   let np1 = n + 1;
	   s += hn;
	   a0[np1 - 1] = r1 * 2. * s / (n + 3.);

      for (let i = n; i <= np1; ++i) {
         let r = (i + 1.) * -0.5;
         b0[0] = r * a0[0];
         for (let m = 2; m <= i; ++m) {
            let bsum = 0.;
            for (let j = 1; j <= m-1; ++j) {
               let mmj = m - j;
               bsum += (j * r - mmj) * a0[j - 1] * b0[mmj - 1];
            }
		      b0[m - 1] = r * a0[m - 1] + bsum / m;
	      }
	      c[i - 1] = b0[i - 1] / (i + 1.);

	      let dsum = 0.;
	      for (let j = 1; j <= i-1; ++j) {
		      dsum += d[i - j - 1] * c[j - 1];
	      }
	      d[i - 1] = -(dsum + c[i - 1]);
	   }

      j0 = e1 * znm1 + (n - 1.) * j0;
      j1 = e1 * zn + n * j1;
      znm1 = z2 * znm1;
      zn = z2 * zn;
      w *= w0;
	   let t0 = d[n - 1] * w * j0;
	   w *= w0;
	   let t1 = d[np1 - 1] * w * j1;
	   sum += t0 + t1;
	   if (Math.abs(t0) + Math.abs(t1) <= eps * sum) {
	      break;
	   }
   }

	let u = Math.exp(-bcorr(a, b));
	return e0 * t * u * sum;
}

/**
 * Evaluation of the Incomplete Beta function I_x(a,b) for non-negative a and b and x ≤ 1.
 * @param {Number} a
 * @param {Number} b
 * @param {Number} x
 * @param {Number} y - y = 1 - x
 * @returns function value.
 */
function bratio(a, b, x, y) {

   const eps = 1e-15;
   let w = 0, w1 = 0;

   // helper functions
   function lwbsper(a0, b0, x0, do_swap) {
      const w = bpser(a0, b0, x0, eps);
      return do_swap ? 0.5 - w + 0.5 : w;
   }

   function lw1bsper(a0, b0, y0, do_swap) {
      const w1 = bpser(b0, a0, y0, eps);
      return do_swap ? w1 : 0.5 - w1 + 0.5;
   }

   function lbfrac(a0, b0, x0, y0, lambda, do_swap) {
      const w = bfrac(a0, b0, x0, y0, lambda, eps * 15);
      return do_swap ? 0.5 - w + 0.5 : w;
   }

   function l140(a0, b0, x0, y0, do_swap) {
      let n = Math.round(b0);
      b0 = b0 - n;
      if (b0 === 0) {
         n = n - 1;
         b0 = 1.0;
      }

      let w = bup(b0, a0, y0, x0, n, eps);
      if (x0 <= 0.7) {
         w = w + bpser(a0, b0, x0, eps);
         return do_swap ? 0.5 - w + 0.5 : w;
      }

      if (a0 <= 15.0) {
         n = 20;
         w = w + bup(a0, b0, x0, y0, n, eps);
         a0 = a0 + n;
      }

      w = bgrat(a0, b0, x0, y0, w, 15 * eps);
      return do_swap ? 0.5 - w + 0.5 : w;
   }

   if (a < eps * .001 && b < eps * .001) {
      // if a and b are too small — use simple ratios
      return b / (a + b);
   }

   let a0, b0, x0, y0, do_swap;
   if (a <= 1 || b <= 1) {
      if (x > 0.5) {
         do_swap = true;
         a0 = b; x0 = y; b0 = a; y0 = x;
      } else {
         do_swap = false;
         a0 = a; x0 = x; b0 = b; y0 = y;
      }

      if (b0 < Math.min(eps, eps * a0)) {
         w = fsper(a0, b0, x0, eps);
         return do_swap ? 0.5 - w + 0.5 : w;
      }

      if (a0 < Math.min(eps, eps * b0) && b0 * x0 <= 1.) {
         w1 = apser(a0, b0, x0, eps);
         return do_swap ? w1 : 0.5 - w1 + 0.5;
      }

      if (Math.max(a0, b0) > 1.) {
         if (b0 <= 1.) return lwbsper(a0, b0, x0, do_swap);
         if (x0 >= 0.29) return lw1bsper(a0, b0, y0, do_swap);
         if (x0 < 0.1 && Math.pow(x0 * b0, a0) <= 0.7) return lwbsper(a0, b0, x0, do_swap);
      } else {
         if (a0 >= Math.min(0.2, b0)) return lwbsper(a0, b0, x0, do_swap);
         if (Math.pow(x0, a0) <= 0.9) return lwbsper(a0, b0, x0, do_swap);
         if (x0 >= 0.3)	return lw1bsper(a0, b0, y0, do_swap);
      }

      let did_bup = false;
      if (Math.max(a0, b0) > 1. && b0 > 15.0) {
         w1 = 0.0;
      } else {
         const n = 20;
         did_bup = true;
         w1 = bup(b0, a0, y0, x0, n, eps, false);
         b0 += n;
      }

      w1 = bgrat(b0, a0, y0, x0, w1, 15 * eps, false);
      if (w1 == 0 || (0 < w1 && w1 < Number.MIN_VALUE)) {
         if (did_bup) {
            w1 = bup(b0 - n, a0, y0, x0, n, eps, true);
         } else {
            w1 = Number.NEGATIVE_INFINITY;
         }
         w1 = bgrat(b0, a0, y0, x0, w1, 15 * eps, true)
         w  = -expm1(w1);
         return do_swap ? 0.5 - w + 0.5 : w;
      }

      w = 0.5 - w1 + 0.5;
      return do_swap ? 0.5 - w + 0.5 : w;
   } else {
      // a > 1 and b > 1 (as well as a0 and b0)
      let lambda = Number.isFinite(a + b) ? (a > b ? (a + b) * y - b : a - (a + b) * x) : a * y - b * x;
      const do_swap = lambda < 0;
      let a0, b0, x0, y0;
      if (do_swap) {
         lambda = -lambda;
         a0 = b; x0 = y; b0 = a; y0 = x;
      } else {
         a0 = a; x0 = x; b0 = b; y0 = y;
      }
      if (b0 < 40.0) {
         if (b0 * x0 <= 0.7) return lwbsper(a0, b0, x0, do_swap);
         else return l140(a0, b0, x0, y0, do_swap);
      } else if (a0 > b0) {
         if (b0 <= 100.0 || lambda > b0 * 0.03) return lbfrac(a0, b0, x0, y0, lambda, do_swap);
      } else if (a0 <= 100.0) {
         return lbfrac(a0, b0, x0, y0, lambda, do_swap)
      } else if (lambda > a0 * 0.03) {
         return lbfrac(a0, b0, x0, y0, lambda, do_swap)
      }

      w = basym(a0, b0, lambda, eps * 100);
      return do_swap ? 0.5 - w + 0.5 : w;
   }
}


