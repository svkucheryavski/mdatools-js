/********************************************************************/
/*    Methods for Preprocessing (work with vectors and matrices)    */
/********************************************************************/

import { isvector, Vector, Matrix, vector } from '../arrays/index.js';
import { mean, sd } from '../stat/index.js';

/**
 * Uncenter and unscale values in every column of a matrix.
 *
 * @param {Matrix} X - matrix with values.
 * @param {Vector} center - vector with numbers to use for uncentering.
 * @param {Vector} scale - vector with numbers to use for unscaling.
 *
 * @returns {Matrix} a matrix with uncentered and unscaled column values.
 *
 */
export function unscale(X, centerValues, scaleValues) {

   const Xp = Matrix.zeros(X.nrows, X.ncols);
   for (let c = 1; c <= X.ncols; c++) {
      const xc = X.getcolref(c);
      const xpc = Xp.getcolref(c);

      const cv = centerValues.v[c - 1];
      const sv = scaleValues.v[c - 1];

      for (let r = 0; r < X.nrows; r++) {
         xpc[r] = xc[r] * sv + cv;
      }
   }

   return Xp;
}


/**
 * Center and scale values in every column of a matrix.
 *
 * @param {Matrix} X - matrix with values.
 * @param {boolean|Vector} [center=true] - logical value or a vector with numbers to use for centering.
 * @param {boolean|Vector} [scale=true] - logical value or a vector with numbers to use for scaling.
 * @param {boolean} [full=true] - logical, influences the return values.
 *
 * @returns {Array|Matrix} either a matrix with centered and scaled column values (if 'full = false')
 * or an array with three elements: the matrix, vector with values for centring and vector with values
 * for scaling.
 *
 */
export function scale(X, center, scale, full) {

   if (center === undefined) {
      center = true;
   }

   if (scale === undefined) {
      scale = false;
   }

   if (full === undefined) {
      full = false;
   }

   function getStat(X, param, fun, alt) {

      if (typeof(param) === "boolean") {
         return param ? X.apply(fun, 2) : vector([alt]).rep(X.ncols);
      }

      if (!isvector(param)) {
         throw Error("scale: parameters 'center' and 'scale' must be boolean or vectors with numeric values.");
      }

      if (param.length !== X.ncols ) {
         throw Error("scale: number of values for centring and scaling must be the same as number of columns in the matrix.");
      }

      return param;
   }

   // prepare values for centering and scaling
   const centerValues = getStat(X, center, mean, 0);
   const scaleValues = getStat(X, scale, sd, 1);

   const Xp = Matrix.zeros(X.nrows, X.ncols);
   for (let c = 1; c <= X.ncols; c++) {
      const xc = X.getcolref(c);
      const xpc = Xp.getcolref(c);

      const cv = centerValues.v[c - 1];
      const sv = 1 / scaleValues.v[c - 1];

      for (let r = 0; r < X.nrows; r++) {
         xpc[r] = (xc[r] - cv) * sv;
      }
   }

   return full ? [Xp, centerValues, scaleValues] : Xp;
}

