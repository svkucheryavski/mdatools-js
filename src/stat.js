/**********************************************
 * Functions for statistical tests            *
 **********************************************/

/**
 * Returns a p-value for any test
 * @param {function} pfun - reference to a CDF function (e.g. pnorm)
 * @param {number} crit - critical value for the test (e.g. z-score or t-score)
 * @param {string} tail - which tail to use ("left", "right", or "both")
 * @param {Array} params - additional parameters to CDF function
 * @returns {number} - a p-value for the test
 */
export function getPValue(pfun, crit, tail, params = []) {

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
 * Makes one-sample t-test for a mean
 * @param {number[]} x - vector with sample values
 * @param {number} mu - assumed mean value for population (H0)
 * @param {number} alpha - significance level (used to compute confidence interval)
 * @param {string} tail - which tail to use ("left", "right", or "both")
 * @returns {Object} - a JSON with test results
 */
export function tTest1(x, mu = 0, alpha = 0.05, tail = "both") {

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
      pValue: getPValue(pt, tValue, tail, [DoF]),
      ci: [effectObserved - errMargin, effectObserved + errMargin]
   };
}

/**
 * Makes two-sample t-test for a difference of means assuming population variances equal
 * @param {number[]} x - vector with sample 1 values
 * @param {number[]} y - vector with sample 2 values
 * @param {number} alpha - significance level (used to compute confidence interval)
 * @param {string} tail - which tail to use ("left", "right", or "both")
 * @returns {Object} - a JSON with test results
 */
export function tTest2(x, y, alpha = 0.05, tail = "both") {
   const nx = x.length;
   const mx = mean(x);
   const my = mean(y);
   const ny = y.length;

   const effectExpected = 0;
   const effectObserved = mx - my;
   const se = Math.sqrt( (sd(x)**2 / nx) + (sd(y)**2 / ny));
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
      pValue: getPValue(pt, tValue, tail, [DoF]),
      ci: [effectObserved - errMargin, effectObserved + errMargin]
   };
}

/**********************************************
 *
 * Functions for computing single statistics  *
 **********************************************/

/**
 * Computes kurtosis of values
 * @param {number[]} x - vector with values
 * @returns {number}
 */
export function kurtosis(x) {
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
 * Computes skewness of values
 * @param {number[]} x - vector with values
 * @returns {number}
 */
export function skewness(x) {
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
 * Finds smallest value in a vector
 * @param {number[]} x - vector with values
 * @returns {number}
 */
export function min(x) {
   let n = x.length;
   let min = Number.POSITIVE_INFINITY

   while (n--) min = x[n] < min ? x[n] : min;
   return min;
}


/**
 * Finds largest value in a vector
 * @param {number[]} x - vector with values
 * @returns {number}
 */
export function max(x) {
   let n = x.length;
   let max = Number.NEGATIVE_INFINITY

   while (n--) max = x[n] > max ? x[n] : max;
   return max;
}


/**
 * Computes sum of all values in a vector
 * @param {number[]} x - vector with values
 * @returns {number}
 */
export function sum(x) {
   let s = 0;
   for (let i = 0; i < x.length; i++) {
      s = s + x[i];
   }

   return s;
}


/**
 * Computes product of all value in a vector
 * @param {number[]} x - vector with values
 * @returns {number}
 */
export function prod(x) {
   let p = 1;
   for (let i = 0; i < x.length; i++) {
      p = p * x[i];
   }

   return p;
}


/**
 * Computes mean (average) value for a vector
 * @param {number[]} x - vector with values
 * @returns {number}
 */
export function mean(x) {
   return sum(x) / x.length;
}


/**
 * Computes covariance between two vectors
 * @param {number[]} x - vector with values
 * @param {number[]} y - vector with values
 * @param {boolean} biased - compute a biased version with n degrees of freedom or not (with n-1).
 * @param {number} mx - mean of x values (if already known)
 * @param {number} my - mean of y values (if already known)
 * @returns {number}
 */
export function cov(x, y, biased = false, mx = undefined, my = undefined) {

   const n = x.length;

   if (y.length !== n) {
      throw Error("Vectors 'x' and 'y' must have the same length.");
   }

   if (n < 2) {
      throw Error("Vectors 'x' and 'y' must have at least two values.");
   }

   if (mx === undefined) mx = mean(x);
   if (my === undefined) my = mean(y);

   let cov = 0;
   for (let i = 0; i < n; i++) {
      cov = cov + (x[i] - mx) * (y[i] - my);
   }

   return cov / (biased ? n : n - 1);
}


/**
 * Computes variance for a vector
 * @param {number[]} x - vector with values
 * @param {boolean} biased - compute a biased version with n degrees of freedom or not (with n-1).
 * @param {number} m - mean value (e.g. if already computed).
 * @returns {number}
 */
export function variance(x, biased = false, m = undefined) {
   return cov(x, x, biased, m, m);
}


/**
 * Computes standard deviation for a vector
 * @param {number[]} x - vector with values
 * @param {boolean} biased - compute a biased version with n degrees of freedom or not (with n-1).
 * @param {number} m - mean value (e.g. if already computed).
 * @returns {number}
 */
export function sd(x, biased = false, m = undefined) {
   return Math.sqrt(variance(x, biased, m));
}


/**
 * Computes correlation between two vectors
 * @param {number[]} x - vector with values
 * @param {number[]} y - vector with values
 * @param {string} method - which method to use ("pearson" or "spearman")
 * @returns {number}
 */
export function cor(x, y, method = "pearson") {

   if (method === "spearman") {
      return cor(rank(x), rank(y));
   }

   return cov(x, y) / (sd(x) * sd(y));
}



/***************************************************
 * Functions for computing vectors of statistics   *
 ***************************************************/


/**
 * Computes a p-th quantile/quantiles for a numeric vector
 * @param {number[]} x - vector with values
 * @param {number|number[]} p - probability (one value or a vector)
 * @returns {number}
 */
export function quantile(x, p) {

   x = sort(x);
   const n = x.length;

   if (!Array.isArray(p)) p = [p];
   if (typeof(p[0]) !== "number" || min(p) < 0 || max(p) > 1) {
      throw new Error("Parameter 'p' must be between 0 and 1 (both included).");
   }

   function q(x, p) {
      const h = (n - 1) * p + 1;
      const n1 = Math.floor(h);
      const n2 = Math.ceil(h);
      return x[n1 - 1] + (x[n2 - 1] - x[n1 - 1]) * (h - Math.floor(h));
   }

   const out =  p.map(v => q(x, v));
   return p.length == 1 ? out[0] : out;
}


/**
 * Generate a sequence of n numbers between min and max.
 * @param {number} min - first value in the sequence
 * @param {number} max - last value in the sequence
 * @param {number} n - number of values in the sequence
 * @returns {number[]} array with the sequence values
 */
export function seq(min, max, n) {

   if (n < 2) {
      throw new Error("Parameter 'n' should be ≥ 2.");
   }

   if (n === undefined && Number.isInteger(min) && Number.isInteger(max)) {
      n = max - min + 1;
   }

   const step = (max - min + 0.0) / (n - 1 + 0.0)
   let out = [...Array(n)].map((x, i) => min + i * step);

   // if step is smaller than 1 round values to remove small decimals accidentally added by JS
   if (Math.abs(step) < 1) {
      const r = Math.pow(10, Math.round(-Math.log10(step)) + 1);
      out = out.map(v => Math.round((v + Number.EPSILON) * r) / r)
   }

   return(out)
}


/**
 * Finds a range of values in a vector (min and max)
 * @param {number[]} x - vector with values
 * @returns {number[]} array with min and max values
 */
export function range(x) {
   return [min(x), max(x)];
}


/**
 * Computes a range of values in a vector with a margin
 * @param {number[]} x - vector with values
 * @param {number} margin - margin in parts of one (e.g. 0.1 for 10% or 2 for 200%)
 * @returns{number[]} array with marginal range boundaries
 */
export function mrange(x, margin = 0.05) {
   const mn = min(x);
   const mx = max(x);
   const d = mx - mn;

   return [mn - d * margin, max(x) + d * margin];
}


/**
 * Splits range of vector values into equal intervals
 * @param {number[]} x - vector with values
 * @param {number} n - number of intervals
 * @returns {number[]} vector with boundaries of the intervals
 */
export function split(x, n) {
   if (x === undefined || !Array.isArray(x) || x.length < 2) {
      throw new Error("split: 'x' must bet a vector with numbers.");
   }

   if (n === undefined || n < 2) {
      throw new Erro("split: 'n' must be a positive integer number.");
   }

   const mn = min(x);
   const mx = max(x);

   if (mn === mx) {
      throw new Error("split: values in a vector 'x' should vary.");
   }

   const step = (mx - mn) / n;
   return Array.from({length: n + 1}, (v, i) => mn + i * step + 0.0);
}


/**
 * Counts how many values from a vector falls into provided intervals (bins)
 * @param {number[]} x - vector with values
 * @param {number[]} bins - vector with bins boundaries
 * @returns {number[]} vector with counts for each bean
 */
export function count(x, bins) {

   if (x === undefined || !Array.isArray(x) ||x.length < 2) {
      throw new Error("count: 'x' must be a vector with numbers.")
   }

   if (bins === undefined || !Array.isArray(bins) || bins.length < 2) {
      throw new Error("count: 'bins' must be a vector with numbers.")
   }

   const n = bins.length;

   // add a bit extra to right side of the last bin
   bins[n - 1] = bins[n - 1] * 1.0001

   // count
   let counts = Array(n - 1).fill(0);
   for (let i = 0; i < x.length; i++) {
      for (let j = 0; j < n - 1; j++) {
         if (x[i] >= bins[j] && x[i] < bins[j + 1]) counts[j] = counts[j] + 1;
      }
   }
   return counts;
}


/**
 * Computes middle points between values of a vector
 * @param {number[]} x - vector with values
 * @returns {number[]} vector with middle points
 */
export function mids(x) {
   return x.slice(1).map((v, i) => (0.5 * (v + x[i])));
}


/**
 * Computes difference between all adjacent values in a vector
 * @param {number[]} x - vector with values
 * @returns {number[]} vector with the differences
 */
export function diff(x) {
   return x.slice(1).map( (y, i) => (y - x[i]));
}


/**
 * Finds outliers in a vector based on inter-quartile range distance
 * @param {Array} x - vector with values
 * @param {number} Q1 - first quartile (optional parameter)
 * @param {Array} Q3 - third quartile (optional parameter)
 * @returns {Array} vector with outliers or empty vector if none were found.
 */
export function getOutliers(x, Q1 = undefined, Q3 = undefined) {

   if (Q1 === undefined) Q1 = quantile(x, 0.25);
   if (Q3 === undefined) Q3 = quantile(x, 0.75);

   const IQR = Q3 - Q1;
   const bl = Q1 - 1.5 * IQR
   const bu = Q3 + 1.5 * IQR
   return(x.filter(v => v < bl || v > bu));
}


/**
 * Returns ranks of values in a vector (ranks start from 1, not 0)
 * @param {number[]} x - vector with values
 * @returns {number[]} vector with ranks
 */
export function rank(x) {
   const y = [...x].sort((a, b) => a - b);
   return(x.map(v => y.indexOf(v) + 1));
}


/**
 * Generate probability points for QQ plot
 * @param {number} n - number of points
 * @returns {Array} a sequence of probabilities between 0 and 1
 */
export function ppoints(n) {
   const a = n < 10 ? 3.0/8.0 : 0.5;
   return Array.from({length: n}, (v, i) => (i + 1 - a) / (n + (1 - a) - a));
}


/**
 * Computes cumulative sums for the vector values
 * @param {number[]} x - vector with values
 * @returns {number[]}
 */
export function cumsum(x) {
   let s = 0;
   return x.map(v => s += v);
}



/***********************************************
 * Functions for theoretical distribution      *
 ***********************************************/


/**
 * Generates 'n' random numbers from a uniform distribution
 * @param {number} n - amount of numbers to generate
 * @param {number} a - smallest value (min) of the population
 * @param {number} b - largest value (max) of the population
 * @returns {number[]} vector with generated numbers
 */
export function runif(n, a = 0, b = 1) {
   let out = Array(n);
   for (let i = 0; i < n; i++) out[i] = (a + Math.random() * (b - a));
   return out;
}


/**
 * Probability density function for uniform distribution
 * @param {Array} x - vector of values
 * @param {number} a - smallest value (min) of the population
 * @param {number} b - largest value (max) of the population
 * @returns {Array} vector with densities
 */
export function dunif(x, a = 0, b = 1) {

   if (!Array.isArray(x)) x = [x];

   const n = x.length;
   const A = 1 / (b - a);
   let d = Array(n);

   for (let i = 0; i < n; i++) {
      d[i] = (x[i] < a || x[i] > b ) ? 0 : A;
   }

   return d;
}


/**
 * Cumulative distribution function for uniform distribution
 * @param {Array} x - vector of values
 * @param {number} a - smallest value (min) of the population
 * @param {number} b - largest value (max) of the population
 * @returns {Array} vector with probabilities
 */
export function punif(x, a = 0, b = 1) {

   if (!Array.isArray(x)) x = [x];

   const n = x.length;
   let p = Array(n);
   for (let i = 0; i < n; i++) {
      if (x[i] < a) {
         p[i] = 0;
      } else if (x[i] > b) {
         p[i] = 1;
      } else {
         p[i] = (x[i] - a) / (b - a)
      }
   }

   return p;
}


 /**
 * Generates 'n' random numbers from a normal distribution
 * @param {number} n - amount of numbers to generate
 * @param {number} mu - average value of the population
 * @param {number} sigma - standard deviation of the population
 * @returns {Array} vector with generated numbers
 */
export function rnorm(n, mu = 0, sigma = 1) {

   let out = Array(n);
   for (let i = 0; i < n; i ++) {
      const a = Math.sqrt(-2 * Math.log(Math.random()));
      const b = 2 * Math.PI * Math.random();
      out[i] = (a * Math.sin(b) * sigma + mu);
   }

   return out;
}


/**
 * Probability density function for normal distribution
 * @param {Array} x - vector of values
 * @param {number} mu - average value of the population
 * @param {number} sigma - standard deviation of the population
 * @returns {Array} vector with densities
 */
export function dnorm(x, mu = 0, sigma = 1) {

   if (!Array.isArray(x)) x = [x];

   const n = x.length;
   const A = 1 / (Math.sqrt(2 * Math.PI) * sigma);
   const frac = -0.5 / sigma ** 2;

   let d = Array(n);
   for (let i = 0; i < n; i++) {
      const df = x[i] - mu;
      d[i] = A * Math.exp(frac * df * df);
   }

   return x.length === 1 ? d[0] : d;
}


/**
 * Cumulative distribution function for normal distribution
 * @param {Array} x - vector of values
 * @param {number} mu - average value of the population
 * @param {number} sigma - standard deviation of the population
 * @returns {Array} vector with probabilities
 */
export function pnorm(x, mu = 0, sigma = 1) {

   if (!Array.isArray(x)) x = [x];

   const n = x.length;
   const frac = 1 / (Math.sqrt(2) * sigma);

   let p = Array(n);
   for (let i = 0; i < n; i++) {
      p[i] = 0.5 * (1 + erf((x[i] - mu) * frac))
   }

   return p.length === 1 ? p[0] : p;
}

/**
 * Inverse cumulative distribution function for normal distribution
 * @param {number|number[]} p - vector of probabilities or a single probability value
 * @param {number} mu - average value of the population
 * @param {number} sigma - standard deviation of the population
 * @returns {number|number[]} vector with quantiles or single quantile value
 */
export function qnorm(p, mu = 0, sigma = 1) {

   if (Array.isArray(p)) {
      return p.map(v => qnorm(v, mu, sigma));
   }

   if (mu !== 0 || sigma !== 1) {
      return qnorm(p) * sigma + mu;
   }

   if (p < 0 || p > 1) {
      throw Error("Parameter 'p' must be between 0 and 1.");
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
 * Probability density function for Student's t-distribution
 * @param {number|number[]} t - t-value or a vector of t-values
 * @param {number} dof - degrees of freedom
 */
export function dt(t, dof) {

   if (dof < 0) {
      throw new Error("Parameter 'dof' should be a positive number.");
   }

   if (Array.isArray(t)) {
      return t.map(v => dt(v, dof));
   }

   const pow = -0.5 * (dof + 1);
   const A = 1 / (Math.sqrt(dof) * beta(0.5, dof/2));
   return (A * Math.pow((1 + t * t / dof), pow));
}


/**
 * Cumulative distribution function for Student's t-distribution
 * @param {number|number[]} t - t-value or a vector of t-values
 * @param {number} dof - degrees of freedom
 */
export function pt(t, dof) {

   if (dof === undefined || dof === null || dof < 1) {
      throw Error("Parameter 'dof' (degrees of freedom) must be an integer number >= 1.");
   }

   if (Array.isArray(t)) {
      return t.map(v => pt(v, dof));
   }

   // since distribution in symmetric we can use only left tail
   if (t === 0) return 0.5;
   if (t === -Infinity) return 0;
   if (t === Infinity) return 1;
   if (t > 0) return (1 - pt(-t, dof));

   return integrate((x) => dt(x, dof), -Infinity, t);
}


/**
 * Inverse cumulative distribution function for Student's t-distribution
 * @param {number|number[]} p - probability or vector with probabilities
 * @param {number} dof - degrees of freedom
 */
export function qt(p, dof) {

   if (dof === undefined || dof === null || dof < 1) {
      throw Error("Parameter 'dof' (degrees of freedom) must be an integer number >= 1.");
   }

   if (p < 0 || p > 1) {
      throw Error("Parameter 'p' must be between 0 and 1.");
   }

   if (Array.isArray(p)) {
      return p.map(v => qt(v, dof));
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
      y = x ** 2;

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
 * Probability density function for F-distribution
 * @param {number|number[]} F - F-value or a vector of t-values
 * @param {number} d1 - degrees of freedom
 * @param {number} d2 - degrees of freedom
 */
export function df(F, d1, d2) {

   if (F < 0 || d1 < 0 || d2 < 0) {
      throw new Error("All 3 parameters must be positive.");
   }

   if (d2 <= d1) {
      throw new Error("Parameter 'd1' must be larger 'd2'.");
   }

   if (Array.isArray(F)) {
      return F.map(v => df(v, d1, d2));
   }

   return Math.sqrt( ( (d1 * F)**d1 * d2**d2) / ((d1 * F + d2)**(d1 + d2))) / (F * beta(d1/2, d2/2))
}


/**
 * Cumulative distribution function for F-distribution
 * @param {number|number[]} F - F-value or a vector of t-values
 * @param {number} d1 - degrees of freedom
 * @param {number} d2 - degrees of freedom
 */
export function pf(F, d1, d2) {

   if (F < 0 || d1 < 0 || d2 < 0) {
      throw new Error("All 3 parameters must be positive.");
   }

   if (d2 <= d1) {
      throw new Error("Parameter 'd1' must be larger 'd2'.");
   }

   if (Array.isArray(F)) {
      return F.map(v => pf(v, d1, d2));
   }

   return ibeta(d1 * F / (d1 * F + d2), d1/2, d2/2)
}



/***********************************************
 * Functions for manipulations with values     *
 ***********************************************/


/**
 * Sorts values in a vector
 * @param {Array} x - vector with values
 * @returns {Array} vector with sorted values
 */
export function sort(x, decreasing = false) {
   return decreasing ? [...x].sort((a, b) => b - a) : [...x].sort((a, b) => a - b);
}


/**
 * Replicates values in x n times
 * @param {any} x - single value or a vector with values
 * @param {number} n - how many times to replicate
 */
export function rep(x, n) {

   if (Array.isArray(n)) {
      if (x.length != n.length) {
         throw new Error("Parameter 'n' should be a single value or a vector of the same length as x.");
      }

      let out = [];
      for (let i = 0; i < n.length; i++) {
         out.push(...rep([x[i]], n[i]));
      }

      return out;
   }

   if (!Array.isArray(x)) x = [x];
   if (n <= 1) return x;

   const nx = x.length;
   x.length = nx * n
   for (let i = 0; i < n - 1; i ++) {
      for (let j = 0; j < nx; j++) {
         x[nx * (i + 1) + j] = x[j];
      }
   }

   return x;
}


/**
 * Create a subset of vectors based on a vector of indices
 * @param {number[]} x - a vector with values
 * @param {number[]} indices - a vector with element indices (first index is 1 not 0!)
 */
export function subset(x, indices) {

   if (!Array.isArray(x)) x = [x];
   if (!Array.isArray(indices)) indices = [indices];

   if (max(indices) > x.length || min(indices) < 1) {
      throw new Error("Parameter 'indices' must have values between 1 and 'x.length'.");
   }

   const n = indices.length;
   let out = Array(n);
   for (let i = 0; i < n; i++) {
      out[i] = x[indices[i] - 1];
   }

   return out;
}


/**
 * Generate combination of all levels of vectors
 * @param {...} args - a sequence of vectors
 */
export function expandGrid(...args) {

   const nargs = args.length;
   const d = args.map(v => v.length);
   let orep = prod(d);

   let grid = Array(nargs);
   let repFac = 1;

   for (let i = 0; i < nargs; i++) {
      const x = args[i];
      const nx = x.length;
      orep = orep/nx;
      grid[i] = subset(x, rep(rep(seq(1, nx, nx), rep(repFac, nx)), orep));
      repFac = repFac * nx;
   }

   return grid;
}


/**
 * Shuffles values in vector x using Fisher–Yates algorithm
 * @param {Array} x - a vector with values
 */
export function shuffle(x) {
  let y = [...x];
  let n = y.length;
  let t, i;

  while (n) {
    i = Math.floor(Math.random() * n--);
    t = y[n];
    y[n] = y[i];
    y[i] = t;
  }

  return y;
}


/**
 * Finds index of value in x which is closest to the value a
 * @param {number[]} x - a vector with values
 * @param {number}  a - a value
 */
export function closestIndex(x, a) {
   const c = x.reduce((prev, curr) => Math.abs(curr - a) < Math.abs(prev - a) ? curr : prev);
   return x.indexOf(c);
}


/**
 * Rounds number (or vector of numbers) to given amount of decimals
 * @param {number | number[]} x - a vector with values
 * @return {number | number[]}
 */
export function round(x, n = 0) {
   if (Array.isArray(x)) {
      return x.map(v => round(v, n));
   }
   return Number.parseFloat(x.toFixed(n));
}

/**
 * Standardize (mean center and sd scale) values from a vector
 * @param {number | number[]} x - a vector with values
 * @return {number} center - value for centering the values (if undefined, will use mean(x))
 * @return {number} scale - value for scaling the values (if undefined, will use sd(x))
 *
 */
export function scale(x, center = undefined, scale = undefined) {
   if (center === undefined) center = mean(x);
   if (scale === undefined) scale = sd(x);

   const n = x.length;
   let y = Array(n);

   for (let i = 0; i < n; i++) {
      y[i] = (x[i] - center) / scale;
   }

   return y;
}



/***************************************************************
 * Mathematical functions and methods needed for computations  *
 ***************************************************************/

/**
 * Computes numeric integral for function "f" with limits (a, b)
 * @param {function} f - a reference to a function
 * @param {number} a - lower limit for integration
 * @param {number} b - upper limit for integration
 * @param {number} acc - absolute accuracy
 * @param {number} eps - relative accuracy
 * @param {number[]} oldfs - vector of values needed for recursion
 * @returns {number} result of integration
 */
export function integrate(f, a, b, acc = 0.000001, eps = 0.00001, oldfs = undefined) {

   if (typeof(a) !== "number" || typeof(b) !== "number") {
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

   let tol = acc + eps * Math.abs(q4);
   let err = Math.abs((q4 - q2)/3);

   if (err < tol) return q4;

   acc = acc / Math.sqrt(2.);
   let mid = (a + b) / 2;
   let left = fs.filter((v, i) => i < n/2);
   let right = fs.filter((v, i) => i >= n/2);

   let ql = integrate(f, a, mid, eps, acc, left);
   let qr = integrate(f, mid, b, eps, acc, right);
   return (ql + qr);
}


/**
 * Error function for normal distribution
 * @param {number} x - a number
 * @returns {number} value for erf
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


/**
 * Gamma function (approximation)
 * @param {number|number[]} z - argument (one value or a vector)
 * @returns {number} value of the Gamma function
 */
export function gamma(z) {

   if (Array.isArray(z)) {
      return z.map(v => gamma(v));
   }

   if (z <= 0) {
      throw new Error("Gamma function only works with arguments > 0.");
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
 * Betta function (approximation)
 * @param {number} x - first argument (one value)
 * @param {number} y - second argument (one value)
 * @returns {number} value of the Beta function
 */
export function beta(x, y) {
   return gamma(x) * gamma(y) / gamma(x + y);
}


/**
 * Incomplete Betta function (approximation via numeric integration)
 * @param {number} x - first argument (one value)
 * @param {number} a - second argument (one value)
 * @param {number} b - third argument (one value)
 * @returns {number} value of the function
 */
export function ibeta(x, a, b) {
   if (x === 0) return 0;
   if (x === 1) return 1;
   if (b === 1) return x ** a;
   if (a === 1) return (1 - (1 - x)**b);
   return integrate((t) => t ** (a - 1) * (1 - t) ** (b - 1), 0, x) / beta(a, b);
}
