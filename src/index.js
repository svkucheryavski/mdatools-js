/**********************************************
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
 * Computes sum of all value in a vector
 * @param {number[]} x - vector with values
 * @returns {number}
 */
export function sum(x) {
   return x.reduce((t, v) => t + v);
}


/**
 * Computes product of all value in a vector
 * @param {number[]} x - vector with values
 * @returns {number}
 */
export function prod(x) {
   return x.reduce((t, v) => t * v);
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
 * Computes standard deviation for a vector
 * @param {number[]} x - vector with values
 * @param {boolean} biased - compute a biased version with n degrees of freedom or not (with n-1).
 * @param {number} m - mean value (e.g. if already computed).
 * @returns {number}
 */
export function sd(x, biased = false, m = undefined) {
   if (m === undefined) m = mean(x)
   return Math.sqrt(sum(x.map(v => (v - m) ** 2 )) / (x.length - (biased ? 0 : 1)));
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
export function mrange(x, margin) {
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

/*******************************************
 * Functions for theoretical distributions *
 *******************************************/

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

   return d;
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

   return p;
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
 * Generate probability points for QQ plot
 * @param {number} n - number of points
 * @returns {Array} a sequence of probabilities between 0 and 1
 */
export function ppoints(n) {
   const a = n < 10 ? 3.0/8.0 : 0.5;
   return Array.from({length: n}, (v, i) => (i + 1 - a) / (n + (1 - a) - a));
}


/*******************************************
 * Other functions                         *
 *******************************************/


/**
 * Sorts values in a vector
 * @param {Array} x - vector with values
 * @returns {Array} vector with sorted values
 */
export function sort(x, decreasing = false) {
   return decreasing ? x.sort((a, b) => b - a) : x.sort((a, b) => a - b);
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
 * @param {number[]} indices - a vector with element indices
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
 * Generate combination of all levels of two vectors
 * @param {number[]} a - a vector with unique values
 * @param {number[]} b - a vector with unique values
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