import {pf, pt, qt, rep, variance} from '../stat/index.js';
import {nrow, mmult, ncol, mdot, vdot, vapply, transpose, crossprod, getdiag, vdiv,
   msubset, isvector, isarray, ismatrix, vsubtract} from '../matrix/index.js';
import {inv} from '../decomp/index.js';

/**
 * Fitting a linear model (SLR or MLR)
 * @param {number[]} X - vector or matrix with predictors
 * @param {number[]} y - vector with responses
 * @return JSON with model parameters and performance statistics
 */
export function lmfit(X, y) {

   if (!isarray(X)) {
      throw Error("Argument 'X' must be a matrix or a vector.");
   }

   if (!isvector(y)) {
      throw Error("Argument 'y' must be a vector.");
   }

   const n = nrow(X);
   if (nrow(y) !== n) {
      throw Error("Arguments 'X' and 'y' must have the same number of measurements.");
   }

   if (n <= ncol(X)) {
      throw Error("Number of measurements must be larger than number of predictors.");
   }

   if (!ismatrix(X)) {
      X = [X];
   }

   // add column of ones for estimation of intercept
   const Xr = msubset(X, [], []);
   Xr.unshift(rep(1, n));

   // compute inverse of variance-covariance matrix
   const R = inv(crossprod(Xr));

   // estimate regression coefficients
   const estimate = mdot(mdot(R, transpose(Xr)), y)[0];

   // compute predicted y-values and performance statistics
   const fitted = mdot(Xr, estimate)[0];
   const stat = regstat(y, fitted, ncol(X));

   // compute standard error and t-values for regression coefficients, H0: beta = 0
   const coeffse = vapply(getdiag(mmult(R, stat.se**2)), Math.sqrt);
   const tstat = vdiv(estimate, coeffse);

   // compute critical t-value for confidence intervals
   const tCrit = qt(0.975, stat.DoF);

   // return JSON with all results
   return {
      class: "lm",
      data: {X: X, y: y},
      coeffs: {
         estimate: estimate,
         se: coeffse,
         tstat: tstat,
         p: tstat.map(t => t > 0 ? 2 * pt(-t, stat.DoF) : 2 * pt(t, stat.DoF)),
         lower: estimate.map((b, i) => b - tCrit * coeffse[i]),
         upper: estimate.map((b, i) => b + tCrit * coeffse[i])
      },
      fitted: fitted,
      stat: stat
   }
}

/**
 * Predicts response values based on the fitted model and predictors
 * @param {JSON} m - regression model (object returned by 'limfit()')
 * @param {number[]} X - vector or matrix with predictors
 * @return vector with predicted response values
 */
export function lmpredict(m, X) {

   if (!isarray(X)) {
      throw Error("Argument 'X' must be a matrix or a vector.");
   }

   if (m.class !== "lm" || !m.coeffs.estimate || m.coeffs.estimate.length < 1) {
      throw Error("Argument 'm' must be object with MLR model returned by method 'lmfit()'.");
   }

   if (ncol(X) !== (m.coeffs.estimate.length - 1)) {
      throw Error("Number of columns in 'X' do not match number of coefficients in model 'm'.");
   }


   if (!ismatrix(X)) {
      X = [X];
   }

   const Xr = msubset(X, [], []);
   Xr.unshift(rep(1, nrow(X)));
   return mdot(Xr, m.coeffs.estimate)[0];
}

/**
 * Computes performance statistics for predicted and reference response values
 * @param {number[]} y — vector with reference response values
 * @param {number[]} yp — vector with predicted response values
 * @return JSON with statistics (adjusted R2, R2, s(e), F-value, p-value)
 */
export function regstat(y, yp, p) {

   const n = nrow(y);
   if (!p) p = 1;

   const e = vsubtract(y, yp);
   const SSe = vdot(e, e);
   const SSy = variance(y) * (n - 1);
   const R2 = (1 - SSe / SSy);
   const DoF = n - p - 1
   const F = ((SSy - SSe)/p) / (SSe/DoF);

   return {
      R2: R2,
      R2adj: 1 - (1 - R2) * (n - 1) / DoF,
      Fstat: F,
      p: 1 - pf(F, p, DoF),
      DoF: DoF,
      se: Math.sqrt(SSe/DoF)
   };
}