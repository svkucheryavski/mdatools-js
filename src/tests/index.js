/*************************************************/
/*    Methods for hypothesis testing             */
/*************************************************/

import { min, mean, variance, sd } from '../stat/index.js';
import { pt, qt } from '../distributions/index.js';

/**
 * Returns a p-value for any test.
 *
 * @param {function} pfun - reference to a CDF function (e.g. pnorm).
 * @param {number} crit - critical value for the test (e.g. z-score or t-score).
 * @param {string} tail - which tail to use ("left", "right", or "both").
 * @param {Array} params - additional parameters to CDF function.
 *
 * @returns {number} a p-value for the test.
 *
 */
export function getpvalue(pfun, crit, tail, params = []) {

   if (tail === "left") {
      return(pfun(crit, ...params));
   }

   if (tail === "right") {
      return(1 - pfun(crit, ...params));
   }

   if (tail === "both") {
      let p = pfun(crit, ...params);
      return min([p, 1 - p]) * 2;
   }
}


/**
 * Makes two-sample t-test for a difference of means assuming population variances equal.
 *
 * @param {Vector} x - vector with sample 1 values.
 * @param {Vector} y - vector with sample 2 values.
 * @param {number} alpha - significance level (used to compute confidence interval).
 * @param {string} tail - which tail to use ("left", "right", or "both").
 *
 * @returns {Object} a JSON with test results.
 *
 */
export function ttest2(x, y, alpha = 0.05, tail = "both") {

   const nx = x.length;
   const mx = mean(x);
   const my = mean(y);
   const ny = y.length;

   const effectExpected = 0;
   const effectObserved = mx - my;
   const se = Math.sqrt( (variance(x) / nx) + (variance(y) / ny));
   const tValue = (effectObserved - effectExpected) / se;
   const DoF = (nx - 1) + (ny - 1);
   const errMargin = qt(1 - alpha/2, DoF) * se;

   return {
      test: "Two sample t-test",
      effectExpected: effectExpected,
      effectObserved: effectObserved,
      se: se,
      tValue: tValue,
      alpha: alpha,
      tail: tail,
      DoF: DoF,
      pValue: getpvalue(pt, tValue, tail, [DoF]),
      ci: [effectObserved - errMargin, effectObserved + errMargin]
   };
}


/**
 * Makes one-sample t-test for a mean.
 *
 * @param {Vector} x - vector with sample values.
 * @param {number} mu - assumed mean value for population (H0).
 * @param {number} alpha - significance level (used to compute confidence interval).
 * @param {string} tail - which tail to use ("left", "right", or "both").
 *
 * @returns {Object} a JSON with test results.
 *
 */
export function ttest(x, mu = 0, alpha = 0.05, tail = "both") {

   if (typeof(mu) !== "number") {
      throw Error("Parameter 'mu' should be a number.");
   }

   const nx = x.length;

   const effectExpected = mu;
   const effectObserved = mean(x);
   const se = sd(x) / Math.sqrt(nx);
   const tValue = (effectObserved - effectExpected) / se;
   const DoF = nx - 1
   const errMargin = qt(1 - alpha/2, DoF) * se;

   return {
      test: "One sample t-test",
      effectExpected: mu,
      effectObserved: effectObserved,
      se: se,
      tValue: tValue,
      alpha: alpha,
      tail: tail,
      DoF: DoF,
      pValue: getpvalue(pt, tValue, tail, [DoF]),
      ci: [effectObserved - errMargin, effectObserved + errMargin]
   };
}

