import { rsvd } from '../decomp/index.js';
import { pf, pt, qt } from '../distributions/index.js';
import { norm2, variance, mean, sd, ssq } from '../stat/index.js';
import { scale as prep_scale, unscale as prep_unscale } from '../prep/index.js';
import { _dot, cbind, tcrossprod, crossprod, reshape, ismatrix, Index, Matrix, vector, isvector, Vector, MDAData } from '../arrays/index.js';


/**
 * Fit Partial Least Squares Regression model.
 *
 * @param {Matrix} X - matrix with predictors.
 * @param {Matrix} Y - matirx with responses (must have one column).
 * @param {number} ncomp - number of components.
 * @param {boolean} [center=true] - logical, mean center X and Y or not.
 * @param {boolean} [scale=false] - logical, standardize X and Y or not.
 *
 * @returns {JSON} object with model parameters.
 *
 */
export function plsfit(X, Y, ncomp, center, scale) {

   if (!ismatrix(X) || X.ncols < 2 || X.nrows < 2) {
      throw Error('plsfit: parameter "X" must be a matrix with at least two rows and two columns.');
   }

   if (!ismatrix(Y) || Y.ncols !== 1) {
      throw Error('plsfit: parameter "Y" must be a matrix with one column.');
   }

   if (Y.nrows !== X.nrows) {
      throw Error('plsfit: number of rows in "X" and "Y" do not match.');
   }

   if (center === undefined) {
      center = true;
   }

   if (scale === undefined) {
      scale = false;
   }

   if (!ncomp) {
      ncomp = Math.min(ncols, nrows - 1);
   }

   // center and scale the training set
   const [Xp, mX, sX] = prep_scale(X, center, scale, true);
   const [Yp, mY, sY] = prep_scale(Y, center, scale, true);

   // compute loadings and eigenvalues for X using randomized SVD
   const m = simpls(Xp, Yp, ncomp);

   // compute distances and variance
   const pcares = pcagetmainres(Xp, m.T, m.P, m.xeigenvals);

   // compute parameters for distances
   const [h0, Nh] = getDistParams(pcares.H);
   const [q0, Nq] = getDistParams(pcares.Q);

   // compute Y-loadings

   // return the model object
   return {
      class: ['plsmodel', 'regmodel', 'pcamodel'],
      ncomp: ncomp,
      center: center,
      scale: scale,

      // X part
      mX: mX,
      sX: sX,
      P: m.P,
      R: m.R,
      C: m.C,
      xeigenvals: m.xeigenvals,

      // Y part
      mY: mY,
      sY: sY,
      C: m.C,
      yeigenvals: m.yeigenvals,

      // distances
      q0: q0,
      h0: h0,
      Nq: Nq,
      Nh: Nh,
   };
}


/**
 * Make predictions for PCR model and new dataset.
 *
 * @param {JSON} m - PCR model created by 'pcrfit()'.
 * @param {Matrix} X - matrix with predictors.
 * @param {Matrix} Y - matirx with responses (must have one column, use empty array [], if no response available).
 * @param {string} name - text label for the results with objects.
 *
 * @returns {JSON} object with main results.
 *
 */
export function plspredict(m, X, Y, name) {

   const Xp = prep_scale(X, m.mX, m.sX);

   // compute scores
   const T = Xp.dot(m.R);

   // compute distances and variance
   const pcares = pcagetmainres(Xp, T, m.P, m.xeigenvals);

   // compute Y-scores if reference values provided
   let U = null;
   if (Y !== undefined && Y !== null) {

      if (!ismatrix(Y)) {
         throw Error('plcapredict: parameter "Y" must be a matrix with one column or null/undefined.');
      }

      const Yp = prep_scale(Y, m.mY, m.sY);
      U = Yp.dot(m.C);

      // oprthogonalize y-scores
      for (let a = 2; a <= m.ncomp; a++) {
         const ua = U.subset([], a);
         const Ta = T.subset([], Index.seq(1, a - 1));
         const uao = ua.subtract(Ta.dot(crossprod(Ta, ua)));
         U.v.set(uao.v, (a - 1) * U.nrows);
      }
   }

   const regres = reggetmainres(T, m.C, m.mY, m.sY, Y);

   return {
      class: ['pcares', 'regres', 'plsres'],
      name: name,
      T: T,
      U: U,
      ...pcares,
      ...regres
   };
}


/**
 * Fit Principal Component Regression model.
 *
 * @param {Matrix} X - matrix with predictors.
 * @param {Matrix} Y - matirx with responses (must have one column).
 * @param {number} ncomp - number of components.
 * @param {boolean} [center=true] - logical, mean center X and Y or not.
 * @param {boolean} [scale=false] - logical, standardize X and Y or not.
 *
 * @returns {JSON} object with model parameters.
 */
export function pcrfit(X, Y, ncomp, center, scale) {

   if (!ismatrix(X) || X.ncols < 2 || X.nrows < 2) {
      throw Error('pcrfit: parameter X must be a matrix with at least two rows and two columns.');
   }

   if (!ismatrix(Y) || Y.ncols !== 1) {
      throw Error('pcrfit: parameter Y must be a matrix with one column.');
   }

   if (Y.nrows !== X.nrows) {
      throw Error('pcrfit: number of rows in X and Y do not match.');
   }


   if (center === undefined) {
      center = true;
   }

   if (scale === undefined) {
      scale = false;
   }

   if (!ncomp) {
      ncomp = Math.min(ncols, nrows - 1);
   }

   // center and scale the training set
   const [Xp, mX, sX] = prep_scale(X, center, scale, true);
   const [Yp, mY, sY] = prep_scale(Y, center, scale, true);

   // compute loadings and eigenvalues for X using randomized SVD
   const m = rsvd(Xp, ncomp);
   const eigenvals = m.s.apply(x => x * x / (X.nrows - 1));

   // compute main PCA results for calibration set
   const calres = pcagetmainres(Xp, m.U.dot(Matrix.diagm(m.s)), m.V, eigenvals);

   // compute parameters for distances
   const [h0, Nh] = getDistParams(calres.H);
   const [q0, Nq] = getDistParams(calres.Q);

   // compute Y-loadings
   const C = Matrix.zeros(1, ncomp);
   for (let a = 1; a <= ncomp; a++) {
      const sigmaa = m.s.v[a - 1];
      const ta = calres.T.getcolref(a);
      C.v[a - 1] = _dot(Yp.v, ta, 1, X.nrows, X.nrows, 1) / (sigmaa * sigmaa);
   }

   // return the model object
   return {
      class: ['pcrmodel', 'regmodel', 'pcamodel'],

      // PCA part
      P: m.V,
      eigenvals: eigenvals,
      center: center,
      scale: scale,
      mX: mX,
      sX: sX,
      q0: q0,
      h0: h0,
      Nq: Nq,
      Nh: Nh,

      // PCR part
      ncomp: ncomp,
      C: C,
      mY: mY,
      sY: sY,
   }
}


/**
 * Make predictions for PCR model and new dataset.
 *
 * @param {JSON} m - PCR model created by 'pcrfit()'.
 * @param {Matrix} X - matrix with predictors.
 * @param {Matrix} Y - matirx with responses (must have one column, use empty array [], if no response available).
 * @param {string} name - text label for the results with objects.
 *
 * @returns {JSON} object with main results.
 */
export function pcrpredict(m, X, Y, name) {


   const Xp = prep_scale(X, m.mX, m.sX);

   // compute PCA results
   const T = Xp.dot(m.P);
   const pcares = pcagetmainres(Xp, T, m.P, m.eigenvals);

   // compute regression results
   const regres = reggetmainres(T, m.C, m.mY, m.sY, Y);

   return {
      class: ['pcares', 'regres', 'pcrres'],
      name: name,
      ...pcares,
      ...regres
   };
}


/**
 * Fit a Principal Component Analysis model.
 *
 * @param {Matrix|MDAData} data - matrix or dataset with data values.
 * @param {number} ncomp - number of components to compute.
 * @param {boolean|Vector} [center=true] - logical (mean center or not) or vector with values for centering.
 * @param {boolean|Vector} [scale=false]  - logical (standardize or not) or vector with values for scaling.
 *
 * @returns {Object} JSON with model parameters.
 *
 */
export function pcafit(data, ncomp, center, scale) {

   if (center === undefined) {
      center = true;
   }

   if (scale === undefined) {
      scale = false;
   }

   let X, varAttrs, objAttrs;
   if (data.constructor === MDAData) {
      X = data.values;
      varAttrs = data.colAttrs;
      objAttrs = data.rowAttrs;
   } else if (ismatrix(data)) {
      X = data;

      const varLabels = Array(data.ncols).fill().map((e, i) => 'X' + (i + 1));
      varAttrs = {
         labels: varLabels,
         axisLabels: varLabels,
         axisValues: Vector.seq(1, data.ncols),
         axisName: 'Variables'
      };

      const objLabels = Array(data.ncols).fill().map((e, i) => 'O' + (i + 1));
      objAttrs = {
         labels: objLabels,
         axisLabels: objLabels,
         axisValues: Vector.seq(1, data.ncols),
         axisName: 'Objects'
      };
   }

   if (!ncomp) {
      ncomp = Math.min(X.ncols, X.nrows - 1);
   }

   // center and scale the training set
   const [Xp, mX, sX] = prep_scale(X, center, scale, true);

   // compute loadings and eigenvalues
   const m = rsvd(Xp, ncomp);
   const eigenvals = m.s.apply(v => v * v / (X.nrows - 1));

   // compute main PCA results for calibration set
   const calres = pcagetmainres(Xp, m.U.dot(Matrix.diagm(m.s)), m.V, eigenvals, objAttrs);

   // compute mean values for distances
   const [h0, Nh] = getDistParams(calres.H);
   const [q0, Nq] = getDistParams(calres.Q);

   // return the model object
   return {
      class: ['pcamodel'],
      P: m.V,
      eigenvals: eigenvals,
      center: center,
      scale: scale,
      mX: mX,
      sX: sX,
      q0: q0,
      h0: h0,
      Nq: Nq,
      Nh: Nh,
      ncomp: ncomp,
      varAttrs: varAttrs,
      compAttrs: calres.compAttrs,
      results: {'cal': calres}
   }
}


/**
 * Compute parameters of distance distributions from distance matrix U.
 *
 * @param {Matrix} U - matrix with distances.
 *
 * @returns {Array} array with two vectors, u0 (scalars) and Nu (number of degrees of freedom).
 *
 */
export function getDistParams(U) {

   const ncomp = U.ncols;
   const u0 = U.apply(mean, 2);
   const su = U.apply(sd, 2);

   const Nu = Vector.zeros(ncomp);
   for (let i = 0; i < ncomp; i++) {
      const f = u0.v[i] / su.v[i];
      const N = Math.round(2 * f * f);
      Nu.v[i] = N < 1 ? 1 : N > 250 ? 250 : N;
   }

   return [u0, Nu];
}


/**
 * Compute main PCA results for already preprocessed dataset.
 *
 * @param {Matrix} Xp - matrix with preprocessed (e.g. centered) data values.
 * @param {Matrix} T - matrix with scores.
 * @param {Matrix} P - matrix with loadings.
 * @param {Vector} eigenvals - vector with eigenvalues.
 * @param {Object} objAttrs - optional JSON with object attributes (labels, axis values, etc.)
 *
 * @returns {Object} JSON with main outcomes (scores, distances, variances).
 *
 */
export function pcagetmainres(Xp, T, P, eigenvals, objAttrs) {

   const ncomp = P.ncols;
   const nrows = Xp.nrows;
   const nvars = Xp.ncols;

   // prepare objects for results
   const H = Matrix.zeros(nrows, ncomp);
   const Q = Matrix.zeros(nrows, ncomp);
   const expvar = Vector.zeros(ncomp);
   const cumexpvar = Vector.zeros(ncomp);

   // compute total sum of squares
   const ha = new Vector.valuesConstructor(nrows);
   const E = Xp.copy();
   const totssq = ssq(Xp.v);

   // prepare component based attributes
   const compAttrs = {
      labels: Array(ncomp),
      axisValues: Array(ncomp),
      axisLabels: Array(ncomp),
      axisName: "Components"
   }

   // loop for computing variances and distances
   for (let a = 1; a <= ncomp; a++) {
      const ta = T.getcolref(a);
      const pa = P.getcolref(a);
      const ts = 1 / eigenvals.v[a - 1];

      const qa = new Vector.valuesConstructor(nrows);
      let qs = 0;
      for (let c = 0; c < nvars; c++) {
         const e = E.getcolref(c + 1);
         for (let r = 0; r < nrows; r++) {
            e[r] = e[r] - ta[r] * pa[c]
            const essq = e[r] * e[r];
            qa[r] += essq;
            qs += essq;
         }
      }

      for (let r = 0; r < nrows; r++) {
         ha[r] += ta[r] * ta[r] * ts;
      }

      H.v.set(ha, (a - 1) * nrows);
      Q.v.set(qa, (a - 1) * nrows);

      cumexpvar.v[a - 1] = 100 * (1 - qs / totssq);
      expvar.v[a - 1] = a === 1 ? cumexpvar.v[a - 1] : cumexpvar.v[a - 1] - cumexpvar.v[a - 2];

      compAttrs.labels[a - 1] = "PC" + a;
      compAttrs.axisValues[a - 1] = a;
      compAttrs.axisLabels[a - 1] = "PC" + a + " (" + expvar.v[a - 1].toFixed(1) + "%)";
   }

   return {
      class: ['pcares', 'ldecomp'],
      T: T,
      H: H,
      Q: Q,
      expvar: expvar,
      cumexpvar: cumexpvar,
      objAttrs: objAttrs,
      compAttrs: compAttrs
   }
}


/**
 * Project new data to a PCA model and create object with main outcomes.
 *
 * @param {Object} m - JSON with PCA model created by 'pcafit()'.
 * @param {Matrix|MDAData} data - matrix or dataset with data values.
 * @param {string} name - name/label for the result object (e.g. "cal", "val", etc.).
 *
 * @returns {Object} - JSON wiht main outcomes (scores, distances, variance, etc.).
 *
 */
export function pcapredict(m, data, name) {

   let Xp, objAttrs;
   if (data.constructor === MDAData) {
      Xp = prep_scale(data.values, m.mX, m.sX);
      objAttrs = data.rowAttrs;
   } else if (ismatrix(data)) {
      Xp = prep_scale(data, m.mX, m.sX);

      const objLabels = Array(data.ncols).fill().map((e, i) => 'O' + (i + 1));
      objAttrs = {
         labels: objLabels,
         axisLabels: objLabels,
         axisValues: Vector.seq(1, data.ncols),
         axisName: 'Objects'
      };
   }

   let res = pcagetmainres(Xp, Xp.dot(m.P), m.P, m.eigenvals, objAttrs);
   res.name = name;

   return res;
}


/**
 * For given vector 'x' and number 'd' returns matrix with power of x values from 1 to d as columns.
 *
 * @param {Vector} x - vector with values.
 * @param {number} d - polynomial degree.
 *
 * @returns a matrix with 'd' columns.
 *
 */
export function polymat(x, d) {

   if (!isvector(x)) {
      throw Error('polymat: argument "x" must be a vector.');
   }

   const X = Matrix.zeros(x.length, d);
   for (let i = 1; i <= d; i++) {
      X.v.set(x.apply(v => v ** i).v, x.length * (i - 1));
   }

   return X;
}


/**
 * Fitting a univariate polynomial model.
 *
 * @param {Vector} x - vector with predictors.
 * @param {Vector} y - vector with responses.
 * @param {number} d - polynomial degree.
 *
 * @return JSON with model parameters and performance statistics.
 *
 */
export function polyfit(x, y, d) {

   if (d < 1 || d >= x.length) {
      throw Error('polyfit: polynomial degree "d" must a positive value smaller than number of measurements.');
   }

   let model = lmfit(polymat(x, d), y);
   model.pdegree = d;
   model.class = "pm";

   return model;
}


/**
 * Predicts response values based on the fitted model and predictors.
 *
 * @param {JSON} m - regression model (object returned by 'polyfit()').
 * @param {vector} x - vector with predictors.
 *
 * @return vector with predicted response values.
 *
 */
export function polypredict(m, x) {

   if (!isvector(x)) {
      throw Error('polypredict: Argument "x" must be a vector.');
   }

   if (m.class !== "pm") {
      throw Error('polypredict: argument "m" must be object with "pm" model.');
   }

   return lmpredict(m, polymat(x, m.pdegree));
}


/**
 * Fit a linear model (SLR or MLR).
 *
 * @param {Vector|Matrix} X - vector or matrix with predictors.
 * @param {Vector} y - vector with responses.
 *
 * @return JSON with model parameters and performance statistics.
 *
 */
export function lmfit(X, y) {

   if (isvector(X)) {
      X = reshape(X, X.length, 1);
   }

   if (!ismatrix(X)) {
      throw Error('lmfit: argument "X" must be a matrix or a vector.');
   }

   if (!isvector(y)) {
      throw Error('lmfit: argument "y" must be a vector.');
   }

   const n = X.nrows;
   if (y.length !== n) {
      throw Error('lmfit: arguments "X" and "y" must have the same number of objects.');
   }

   if (n <= X.ncols) {
      throw Error('lmfit: number of objects must be larger than number of predictors.');
   }

   // add column of ones for estimation of intercept
   const Xr = cbind(Vector.ones(n), X);

   // compute inverse of variance-covariance matrix
   const R = crossprod(Xr).inv();

   // estimate regression coefficients
   const estimate = reshape(R.dot(Xr.t()).dot(y), X.ncols + 1);

   // compute predicted y-values and performance statistics
   const fitted = reshape(Xr.dot(estimate), n);
   const stat = regstat(y, fitted, X.ncols);

   // compute standard error and t-values for regression coefficients, H0: beta = 0
   const coeffse = R.mult(stat.se * stat.se).diag().apply(Math.sqrt);
   const tstat = estimate.divide(coeffse);

   // compute critical t-value for confidence intervals
   const tCrit = qt(0.975, stat.DoF);

   // return JSON with all results
   const errMargin = coeffse.mult(tCrit);
   return {
      class: "lm",
      data: {X: X, y: y},
      coeffs: {
         estimate: estimate,
         se: coeffse,
         tstat: tstat,
         p: tstat.apply(t => t > 0 ? 2 * pt(-t, stat.DoF) : 2 * pt(t, stat.DoF)),
         lower: estimate.subtract(errMargin),
         upper: estimate.add(errMargin)
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

   if (isvector(X)) {
      X = reshape(X, X.length, 1);
   }

   if (!ismatrix(X)) {
      throw Error('lmpredict: argument "X" must be a matrix or a vector.');
   }

   if (!(m.class === 'lm' || m.class === 'pm') || !m.coeffs.estimate || m.coeffs.estimate.length < 1) {
      throw Error('lmpredict: argument "m" must be object with "lm" model.');
   }

   if (X.ncols !== (m.coeffs.estimate.length - 1)) {
      throw Error('lmpredict: number of columns in "X" does not match number of coefficients in model.');
   }


   // add column of ones for estimation of intercept
   const n = X.nrows;
   const Xr = cbind(Vector.ones(n), X);
   return reshape(Xr.dot(m.coeffs.estimate), n);
}


/**
 * Computes performance statistics for predicted and reference response values
 * @param {number[]} y — vector with reference response values
 * @param {number[]} yp — vector with predicted response values
 * @return JSON with statistics (adjusted R2, R2, s(e), F-value, p-value)
 */
export function regstat(y, yp, p) {

   const n = y.length;
   if (!p) p = 1;
   const e = y.subtract(yp);
   const SSe = ssq(e);
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


/**
 * Split matrix with data values into matrix of predictors (X) and and matrix with responses (Y).
 *
 * @param {Matrix} data - matrix with data values.
 *
 * @returns {Array} array with X and Y.
 */
export function splitregdata(data) {

   return [
      new Matrix(data.v.subarray(data.nrows), data.nrows, data.ncols - 1),
      new Matrix(data.getcolref(1), data.nrows, 1)
   ];
}


/**
 * Make predictions based on x-scores and y-loadings (for PCR or PLS1) and compute
 * performance statistics if reference y-values are provided.
 *
 * @param {Matrix} T - matrix with X-scores.
 * @param {Matrix} C - matrix with y-loadings.
 * @param {number} mY - number to use for centering y-values (from the model).
 * @param {number} sY - number to use for scaling y-values (from the model).
 * @param {Vector} [y] - vector with reference response values.
 */
export function reggetmainres(T, C, mY, sY, Yref) {

   const nrows = T.nrows;
   const ncomp = T.ncols;

   // prepare variables for predictions and statistics
   const Ypred = Matrix.zeros(nrows, ncomp);

   // if no reference values, compute predictions and return
   if (Yref === undefined || Yref === null || Yref === []) {
      for (let a = 1; a <= ncomp; a++) {
         // compute predictions
         const cind = Index.seq(1, a);
         const Ypreda = prep_unscale(tcrossprod(T.subset([], cind), C.subset([], cind)), mY, sY);
         Ypred.v.set(Ypreda.v, (a - 1) * nrows);
      }

      return {'Ypred': Ypred};
   }

   // otherwise compute statistics as well
   const rmse = Vector.zeros(ncomp);
   const r2 = Vector.zeros(ncomp);
   const bias = Vector.zeros(ncomp);

   // total y-variance
   const ssy = variance(Yref.v) * (Yref.nrows - 1);
   for (let a = 1; a <= ncomp; a++) {

      // compute predictions
      const cind = Index.seq(1, a);
      const Ypreda = prep_unscale(tcrossprod(T.subset([], cind), C.subset([], cind)), mY, sY);
      Ypred.v.set(Ypreda.v, (a - 1) * nrows);

      // compute performance statistics
      const e = Yref.subtract(Ypreda);
      const sse = ssq(e.v);
      bias.v[a - 1] = mean(e.v);
      rmse.v[a - 1] = Math.sqrt(sse / nrows);
      r2.v[a - 1] = 1 - sse / ssy;
   }

   return {
      "Yref": Yref,
      "Ypred": Ypred,
      "rmse": rmse,
      "r2": r2,
      "bias": bias
   };
}


/**
 * Implementation of SIMPLS algorithm.
 *
 * @param {Matrix} X - matrix with predictors.
 * @param {Matrix} Y - matrix with responses.
 * @param {number} ncomp - number of components.
 *
 * @return {JSON} JSON with decomposition results.
 *
 */
export function simpls(X, Y, ncomp) {

   const nobj  = X.nrows;
   const npred = X.ncols;
   const nresp = Y.ncols;

   // initial estimation
   let S = crossprod(X, Y);
   let M = crossprod(X);

   // prepare space for results
   const C = Matrix.zeros(nresp, ncomp)
   const R = Matrix.zeros(npred, ncomp)
   const V = Matrix.zeros(npred, ncomp)
   const P = Matrix.zeros(npred, ncomp)
   const T = Matrix.zeros(nobj, ncomp)
   const U = Matrix.zeros(nobj, ncomp)

   const xeigenvals = Vector.zeros(ncomp);
   const yeigenvals = Vector.zeros(ncomp);

   // loop for each components
   for (let a = 1; a <= ncomp; a++) {

      let r = rsvd(S, 1).U
      let t = X.dot(r);

      const tnorm = norm2(t.v);
      t = t.apply(v => v / tnorm, 0);
      r = r.apply(v => v / tnorm, 0);

      const p = crossprod(X, t);
      const c = crossprod(Y, t);
      let u = Y.dot(c);
      let v = p.copy()

      if (a > 1) {
         v = v.subtract(V.dot(crossprod(V, p)));
         u = u.subtract(T.dot(crossprod(T, u)));
      }

      const vnorm = norm2(v.v);
      v = v.apply(x => x / vnorm, 0);

      R.v.set(r.v, (a - 1) * npred);
      V.v.set(v.v, (a - 1) * npred);
      P.v.set(p.v, (a - 1) * npred);
      T.v.set(t.v, (a - 1) * nobj);
      U.v.set(u.v, (a - 1) * nobj);
      C.v.set(c.v, (a - 1) * nresp);


      xeigenvals.v[a - 1] = ssq(t.v) / (nobj - 1);
      yeigenvals.v[a - 1] = ssq(u.v) / (nobj - 1);

      M = M.subtract(tcrossprod(p))
      S = S.subtract(v.dot(crossprod(v, S)));
   }

   return {
      R: R, P: P, T: T, C: C, U: U,
      xeigenvals: xeigenvals,
      yeigenvals: yeigenvals
   };
}
