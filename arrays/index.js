/* =================================================================== */
/*    Methods for Arrays (vectors, matrices, indices and factors, )    */
/* =================================================================== */

import { min, max, prod } from '../stat/index.js';
import { rnorm, runif } from '../distributions/index.js';
import { qr } from '../decomp/index.js';

export function isnumber(x) {
   return typeof(x) === 'number';
}


/**************************************************/
/*    Non-class methods for Vectors and Matrix    */
/**************************************************/

/**
 * Compute XY' product of two matrices.
 *
 * @param {Matrix} X - a matrix.
 * @param {Matrix} Y - a matrix.
 *
 * @returns {Matrix} result of the product.
 *
 */
export function tcrossprod(X, Y) {

   if (Y === undefined) {
      Y = X;
   }

   const nrx = X.nrows;
   const ncx = X.ncols;

   const nry = Y.nrows;
   const ncy = Y.ncols;

   // create vector for the product
   const nrows = nrx;
   const ncols = nry;
   const ninner = ncx;
   const out = new Float64Array(nrows * ncols);

   for (let i = 0; i < ninner; i++) {
      const xr = X.v.subarray(i * nrx, (i + 1) * nrx);
      const yc = Y.v.subarray(i * nry, (i + 1) * nry);

      for (let c = 0; c < ncols; c++) {
         const outc = out.subarray(c * nrows, (c + 1) * nrows);
         for (let r = 0; r < nrows; r++) {
            outc[r] += xr[r] * yc[c];
         }
      }
   }

   return new Matrix(out, nrows, ncols);

}


/**
 * Compute X'Y product of two matrices or a vector and a matrix.
 *
 * @param {Matrix|Vector} X - a matrix or a vector.
 * @param {Matrix} Y - a matrix.
 *
 * @returns {Matrix} result of the product.
 *
 */
export function crossprod(X, Y) {

   if (Y === undefined) {
      Y = X;
   }

   const nrx = X.nrows;
   const ncx = X.ncols;

   const nry = Y.nrows;
   const ncy = Y.ncols;

   // create vector for the product
   const nrows = ncx;
   const ncols = ncy;
   const ninner = nrx;
   const out = new Float64Array(nrows * ncols);

   for (let c = 0; c < ncols; c++) {
      // column c of Y
      const yc = Y.v.subarray(c * nry, (c + 1) * nry);

      // column c of out
      const outc = out.subarray(c * nrows, (c + 1) * nrows);

      for (let r = 0; r < nrows; r++) {

         // column r of X (row r of X')
         const xr = X.v.subarray(r * nrx, (r + 1) * nrx);

         let s = 0;
         for (let i = 0; i < ninner; i++) {
            s += xr[i] * yc[i];
         }
         outc[r] = s;
      }
   }

   return new Matrix(out, nrows, ncols);
}


/**
 * Bind several matrices or vectors into a single matrix row wise.
 *
 * @param  {...any} args - matrices or vectors separated by comma.
 *
 * @returns {Matrix} concatenated matrix.
 *
 */
export function rbind(...args) {

   if (args.length === 1) return args[0];

   // check that all matrices have the same number of rows
   const ncols = ismatrix(args[0]) ? args[0].ncols : args[0].length;
   const check = args.reduce( (acc, cur) => acc & ismatrix(cur) ? cur.ncols === ncols : cur.length === ncols, true);

   if (!check) {
      throw Error('rbind: all matrices must the same number of columns.');
   }

   // compute number of columns in outside matrix
   const nrows = args.reduce( (acc, cur) => acc + (ismatrix(cur) ? cur.nrows : 1), 0);
   const l = nrows * ncols;
   const out = new Matrix.valuesConstructor(l);

   let start = 0;
   for (const a of args) {
      out.set((ismatrix(a) ? a.t().v : a.v), start);
      start += a.v.length;
   }

   return new Matrix(out, ncols, nrows).t();

}


/**
 * Bind several matrices or vectors into a single matrix column wise.
 *
 * @param  {...any} args - matrices or vectors separated by comma.
 *
 * @returns {Matrix} concatenated matrix.
 *
 */
export function cbind(...args) {

   if (args.length === 1) return args[0];

   // check that all matrices have the same number of rows
   const nrows = ismatrix(args[0]) ? args[0].nrows : args[0].length;
   const check = args.reduce( (acc, cur) => acc & (ismatrix(cur) ? cur.nrows === nrows : cur.length === nrows), true);

   if (!check) {
      throw Error('cbind: all matrices must the same number of rows.');
   }

   // compute number of columns in outside matrix
   const ncols = args.reduce( (acc, cur) => acc + (ismatrix(cur) ? cur.ncols : 1), 0);
   const l = nrows * ncols;
   const out = new Matrix.valuesConstructor(l);

   let start = 0;
   for (const a of args) {
      out.set(a.v, start);
      start += a.v.length;
   }

   return new Matrix(out, nrows, ncols);

}


/**
 * Concatenate several vectors into one vector.
 *
 * @param {...Vector | ...Index} args - vectors separated by comma.
 *
 * @returns {Vector|Index} vector with concatenated values.
 *
 */
export function c(...args) {

   if (args.length === 1) return args[0];

   const l = args.reduce( (acc, cur) => acc + cur.length, 0);
   const out = new args[0].constructor.valuesConstructor(l);

   let start = 0;
   for (const a of args) {
      out.set(a.v, start);
      start += a.length;
   }

   return new args[0].constructor(out);
}


/**
 * Change dimension of matrices and vectors.
 * @param {Matrix|Vector} obj - a matrix or a vector.
 * @param {number} nrow - number of rows in the reshaped object.
 * @param {number} ncol - number of columns in the reshaped object, if undefined, a vector will be returned.
 *
 * @returns {Matrix|Vector} the reshaped object.
 *
 */
export function reshape(obj, nrow, ncol) {

   if (ncol === undefined) {
      if (nrow !== obj.v.length) {
         throw Error("reshape: provided dimension does not match number of values in the object.");
      }
      return new Vector(obj.v);
   }

   if (obj.v.length !== nrow * ncol) {
      throw Error("reshape: provided dimension does not match number of values in the object.");
   }

   return new Matrix(obj.v, nrow, ncol);
}



/*************************************************/
/*         Classes and type checkers             */
/*************************************************/


/**
 * Return 'true' of 'X' is a Matrix object, 'false' otherwise.
 *
 * @param {any} X - any object or variable.
 *
 * @returns {boolean}
 *
 */
export function ismatrix(X) {
   return X.constructor === Matrix;
}


/**
 * Create a Matrix object from array of values.
 *
 * @param {number[]} values - array with values.
 * @param {number} nrows - number of rows in matrix.
 * @param {number} ncols - number of columns in matrix.
 *
 * @returns {Matrix} the matrix.
 *
 */
export function matrix(values, nrows, ncols) {
   return new Matrix(new Float64Array(values), nrows, ncols);
}


/** Class representing a matrix */
export class Matrix {

   static valuesConstructor = Float64Array;
   static TOLERANCE = Math.pow(10, -9);

   /**
    * Constructor for a Matrix object.
    *
    * @param {Float64Array} values - typed array with values.
    * @param {number} nrows - number of rows in the matrix.
    * @param {number} ncols - number of columns in the matrix.
    *
    * @returns {Matrix} a Matrix object (see description).
    *
    * @description  Create a matrix from vector of values. Matrix is represented by an object with two
    * fields. Field 'v' is a typed Float64Array with all values, unfolded column by column. The second
    * field, 'dim', is an array with 2 values: number of rows and number of columns.
    *
    */
   constructor (values, nrows, ncols) {
      if (!ArrayBuffer.isView(values) || values.constructor !== Matrix.valuesConstructor) {
         throw Error('Matrix: parameter "values" must be Float64Array.')
      }

      if (values.length != nrows * ncols) {
         throw Error('Matrix: number elements in "values" is not equal to "nrows" x "ncols".')
      }

      this.v = values;
      this.nrows = nrows
      this.ncols = ncols;
   }

   /**
    * Compute inverse of the matrix.
    *
    * @returns inverse matrix.
    *
    */
   inv() {

      if (this.nrows !== this.ncols) {
         throw Error('inv: only squared matrices can be inverted.');
      }

      const n = this.ncols;

      if (this.isuppertriangular()) {

         if (this.islowertriangular()) {
            // it means diagonal
            return Matrix.diagm(this.diag().apply(v => Math.abs(v) > Number.EPSILON ? 1 / v : x));
         }

         // prepare matrices - we will transpose X to work with columns instead of rows
         const I = Matrix.eye(n);
         const Yt = Matrix.eye(n);
         const Xt = this.t();

         // solve for the last column
         const s = Xt.v[(n - 1) * n + (n - 1)];
         const c = n - 1;
         for (let r = 0; r < n; r++) {
            Yt.v[c * n + r] = I.v[c * n + r] / s;
         }

         // do backward substitution
         for (let c = n - 2; c >= 0; c--) {
            let s = I.getcolumn(c + 1);
            const xc = Xt.getcolref(c + 1);
            for (let r = c + 1; r < n; r++ ) {
               const yc = Yt.getcolumn(r + 1);
               s = s.subtract(yc.mult(xc[r]));
            }
            Yt.v.set(s.divide(xc[c]).v, c * n);
         }

         return Yt.t()
      }


      if (this.islowertriangular()) {

         // check if diagonal elements are not zeros
         const d = this.diag();
         if (prod(d) < Number.EPSILON) {
            throw Error('inv: the matrix is not invertible.');
         }

         // prepare matrices - we will transpose X to work with columns instead of rows
         const I = Matrix.eye(n);
         const Yt = Matrix.eye(n);
         const Xt = this.t();

         // solve for the first column
         const c = 0;
         const s = Xt.v[0];
         for (let r = 0; r < n; r++) {
            Yt.v[c * n + r] = I.v[c * n + r] / s;
         }

         // do forward substitution
         for (let c = 1; c < n; c++) {
            let s = I.getcolumn(c + 1);
            const xc = Xt.getcolref(c + 1);
            for (let r = 0; r < c; r++ ) {
               const yc = Yt.getcolumn(r + 1);
               s = s.subtract(yc.mult(xc[r]));
            }
            Yt.v.set(s.divide(xc[c]).v, c * n);
         }

         return Yt.t()
      }

      // invert matrix using QR transformation
      const r = qr(this);
      if (!r.R.isuppertriangular()) {
         throw Error('inv: QR decomposition of the matrix returned incorrect result.');
      }

      return r.R.inv().dot(r.Q.t());
   }


   /**
    * Check if matrix is lower triangular.
    *
    * @returns {boolean} true if matrix is lower triangular, false otherwise.
    */
   islowertriangular() {

      if (this.nrows !== this.ncols) return false;
      for (let j = 1; j < this.ncols; j++) {
         for (let i = 0; i < j; i++) {
            if (Math.abs(this.v[j * this.nrows + i]) > Matrix.TOLERANCE) return false;
         }
      }

      return true;
   }


   /**
    * Check if matrix is upper triangular.
    *
    * @returns {boolean} true if matrix is upper triangular, false otherwise.
    */
   isuppertriangular() {

      if (this.nrows !== this.ncols) return false;

      for (let j = 0; j < this.ncols; j++) {
         for (let i = j + 1; i < this.nrows; i++) {
            if (Math.abs(this.v[j * this.nrows + i]) > Matrix.TOLERANCE) return false;
         }
      }

      return true;
   }


   /**
    * Return vector with diagonal elements of matrix.
    *
    * @returns {Vector} vector with diagonal elements.
    *
    */
   diag() {

      const n = Math.min(this.nrows, this.ncols);
      const out = new Float64Array(n);
      for (let rc = 0; rc < n; rc++) {
         out[rc] = this.getcolref(rc + 1)[rc];
      }

      return new Vector(out);
   }


   /**
    * Transpose the matrix.
    *
    * @returns {Matrix} transposed matrix.
    *
    */
   t() {

      const ncols_out = this.nrows;
      const nrows_out = this.ncols;
      const out = new Matrix.valuesConstructor(this.v.length);

      if (this.ncols < this.nrows) {
         for(let i = 0; i < this.ncols; i++) {
            const x = this.v.subarray(i * this.nrows, (i + 1) * this.nrows)
            for(let j = 0; j < this.nrows; j++) {
               out[j * nrows_out + i] = x[j];
            }
         }

         return new Matrix(out, nrows_out, ncols_out);
      }

      for(let j = 0; j < this.nrows; j++) {
         const outx = out.subarray(j * nrows_out, (j + 1) * nrows_out)
         for(let i = 0; i < this.ncols; i++) {
            outx[i] = this.v[i * this.nrows + j];
         }
      }

      return new Matrix(out, nrows_out, ncols_out);
   }


   /**
    * Compute a dot product of the matrix with vector or another matrix.
    *
    * @param {Vector|Matrix} X - a vector or a matrix to compute a dot product with.
    *
    * @returns {Matrix} result of the dot product.
    *
    */
   dot(X) {

      if (isvector(X)) {
         return new Matrix(_dot(this.v, X.v, this.nrows, this.ncols, X.length, 1), this.nrows, 1);
      }

      if (ismatrix(X)) {
         return new Matrix(_dot(this.v, X.v, this.nrows, this.ncols, X.nrows, X.ncols), this.nrows, X.ncols);
      }

      throw Error("dot: argument 'X' must be a matrix or a vector.");
   }

   /**
    * Does mathematical operation between the matrix and another matrix, a vector or a number.
    *
    * @param {number|Vector|Matrix} x - a vector or a number to operate with.
    * @param {function} fun - function to apply.
    * @param {number} dims - if 'x' is a vector which dimension applies it to.
    *
    * @returns {Matrix} result of operation.
    *
    */
   op(X, fun, dims) {

      if (ismatrix(X)) {
         return new Matrix(_opvv(this.v, X.v, fun), this.nrows, this.ncols);
      }

      if (isnumber(X)) {
         return new Matrix(_opvn(this.v, X, fun), this.nrows, this.ncols);
      }

      if (!isvector(X)) {
         throw new Error('op: argument "x" must be a matrix, a vector or a number.');
      }

      if (dims === undefined) {
         dims = X.length === this.ncols ? 2 : 1;
      }

      if (dims === 2 && X.length === this.ncols) {

         const out = new Matrix.valuesConstructor(this.v.length);
         for (let c = 0; c < this.ncols; c++) {
            out.set(_opvn(this.getcolref(c + 1), X.v[c], fun), c * this.nrows);
         }

         return new Matrix(out, this.nrows, this.ncols);
      }

      if (dims === 1 && X.length === this.nrows) {

         const out = new Matrix.valuesConstructor(this.v.length);
         for (let c = 0; c < this.ncols; c++) {
            out.set(_opvv(this.getcolref(c + 1), X.v, fun), c * this.nrows);
         }

         return new Matrix(out, this.nrows, this.ncols);
      }

      throw new Error('op: length of vector "X" does not match number of rows nor number of columns of the maytrix.');

   }


   /**
    * Add a number or values from another matrix or vector.
    *
    * @param {number|Vector|Matrix} X - a matrix, a vector or a number to sum the values with.
    *
    * @returns {Matrix} result of operation.
    *
    */
   add(X) {
      return this.op(X, (a, b) => a + b);
   }


   /**
    * Subtract a number or values from another matrix or vector.
    *
    * @param {number|Vector|Matrix} X - a matrix, a vector or a number to subtract the values of.
    *
    * @returns {Matrix} result of operation.
    *
    */
   subtract(X) {
      return this.op(X, (a, b) => a - b);
   }


   /**
    * Multiply to a number or to values from another matrix or vector.
    *
    * @param {number|Vector|Matrix} X - a matrix, a vector or a number to multiply the values to.
    *
    * @returns {Matrix} result of operation.
    *
    */
   mult(X) {
      return this.op(X, (a, b) => a * b);
   }


   /**
    * Divide to a number or to values from another matrix or vector.
    *
    * @param {number|Vector|Matrix} X - a matrix, a vector or a number to divide the values to.
    *
    * @returns {Matrix} result of operation.
    *
    */
   divide(X) {
      return this.op(X, (a, b) => a / b);
   }


   /**
    * Applies a function to elements of the matrix.
    * @param {function} fun - function to apply (should return single value).
    * @param {number} dims - which elements to apply the function to (0 - every, 1 - rows, 2 - columns).
    *
    * @returns {Vector|Matrix} - result of applying the function (either array with one value for each
    * row/column or another Matrix with transformed values).
    */
   apply(fun, dims) {

      // by default function is applied to columns
      if (dims === undefined) {
         dims = 2;
      }

      // columns
      if (dims == 2) {
         const n = this.ncols;
         const out = new Matrix.valuesConstructor(n);
         for (let i = 0; i < n; i++) {
            out[i] = fun(this.getcolref(i + 1));
         }
         return new Vector(out);
      }

      // rows
      if (dims == 1) {
         return this.t().apply(fun, 2);
      }

      // individual values
      if (dims == 0) {
         const n = this.v.length;
         const out = new Matrix.valuesConstructor(n);
         for (let i = 0; i < n; i++) {
            out[i] = fun(this.v[i]);
         }
         return new Matrix(out, this.nrows, this.ncols);
      }
   }


   /**
    * Replace values of matrix specified by 'rind' and 'cind' with values from another matrix.
    *
    * @param {Matrix} - matrix with values to use as replacement.
    * @param {number | Array | Index} rind - row indices (as number or vector of indices).
    * @param {number | Array | Index} cind - column indices (as number or vector of indices).
    *
    * @description Row and column indices must start from 1. Empty array ([]) tells function to use
    * all elements (e.g. all rows or all columns). Size of matrix 'X' should correspond to the
    * number of indices.
    *
    * Function does not create a new matrix but changes the current one.
    *
    * @returns {Matrix} the matrix with replaced values.
    *
    */
   replace (X, rind, cind) {

      if (typeof(rind) === 'number') {
         rind = [rind];
      }

      if (rind.length === 0) {
         rind = Index.seq(1, this.nrows);
      }

      if (!isindex(rind)) {
         rind = index(rind);
      }

      if (typeof(cind) === 'number') {
         cind = [cind];
      }

      if (cind.length === 0) {
         cind = Index.seq(1, this.ncols);
      }

      if (!isindex(cind)) {
         cind = index(cind);
      }

      const nrows = rind.length;
      const ncols = cind.length;

      if (min(rind) < 1 || max(rind) > this.nrows) {
         throw Error('Matrix.replace: row indices do not match the dimension of matrix.');
      }

      if (min(cind) < 1 || max(cind) > this.ncols) {
         throw Error('Matrix.replace: column indices do not match the dimension of matrix.');
      }

      if (X.nrows !== nrows || X.ncols !== ncols) {
         throw Error('Matrix.replace: dimension of "X" does not match the indices.');
      }

      for (let ic = 0; ic < cind.length; ic++) {
         const xc = X.getcolref(ic + 1);
         const cv = (cind.v[ic] - 1) * this.nrows;
         for (let ir = 0; ir < rind.length; ir++) {
            const r = rind.v[ir];
            this.v[cv + r - 1] = xc[ir];
         }
      }

      return this;
   }


   /**
    * Create a deep copy of the matrix.
    *
    * @returns {Matrix}
    *
    */
   copy() {
      return new Matrix(this.v.slice(), this.nrows, this.ncols);
   }

   /**
    * Return values from a particular matrix row as vector.
    *
    * @param {number} ir - row index (starting from 1).
    *
    * @returns {Vector} vector with column values.
    *
    */
   getrow(ir) {

      if (ir < 1 || ir > this.nrows) {
         throw Error('getrow: wrong row index.');
      }

      const v = new Vector.valuesConstructor(this.ncols);
      for (let c = 0; c < this.ncols; c++) {
         v[c] = this.getcolref(c + 1)[ir - 1];
      }
      return new Vector(v);
   }


   /**
    * Return values from a particular matrix column as vector.
    *
    * @param {number} ic - column index (starting from 1).
    *
    * @returns {Vector} vector with column values.
    *
    */
   getcolumn(ic) {

      if (ic < 1 || ic > this.ncols) {
         throw Error('getcolumn: wrong column index.');
      }

      return new Vector(this.v.slice((ic - 1) * this.nrows, ic * this.nrows));
   }


   /**
    * Return a reference (not copy) to values of a particular matrix column.
    *
    * @param {number} ic - index of column (starting from 1).
    *
    * @returns {Float64Array} - reference to the column values as typed array.
    *
    */
   getcolref(ic) {
      return this.v.subarray((ic - 1) * this.nrows, ic * this.nrows);
   }


   /**
    * Create a subset of a matrix using row and column indices.
    *
    * @param {number|Index} rind - row indices to select (as Index or a single number).
    * @param {number|Index} cind - column indices to select (as Index or a single number).
    *
    * @description Row and column indices must start from 1. If 'rind' or 'cind' are null,
    * all items from this direction will be taken.
    *
    * @returns {Matrix} the subset of the matrix.
    *
    */
   subset(rind, cind) {

      if (typeof(rind) === 'number') {
         rind = index([rind]);
      }

      if (typeof(cind) === 'number') {
         cind = index([cind]);
      }

      if (Array.isArray(rind)) {
         rind = rind.length === 0 ? null : index(rind);
      }

      if (Array.isArray(cind)) {
         cind = cind.length === 0 ? null : index(cind);
      }

      if (rind !== null && !isindex(rind)) {
         throw Error('subset: parameter "rind" can be a number, null or instance of Index.');
      }

      if (cind !== null && !isindex(cind)) {
         throw Error('subset: parameter "cind" can be a number, null or instance of Index.');
      }


      // select all rows and a subset of columns
      if (rind === null) {

         const ncols_out = cind.length;
         const nrows_out = this.nrows;
         const out = new Matrix.valuesConstructor(nrows_out * ncols_out);
         for (let ic = 0; ic < cind.length; ic++) {
            out.set(this.v.slice((cind.v[ic] - 1) * this.nrows, cind.v[ic] * this.nrows), ic * this.nrows);
         }

         return new Matrix(out, nrows_out, ncols_out);
      }

      // select all columns and a subset of rows
      if (cind === null || cind === undefined) {
         return this.t().subset(cind, rind).t();
      }

      // select specific columns and rows
      const nrows_out = rind.length;
      const ncols_out = cind.length;
      const out = new Matrix.valuesConstructor(nrows_out * ncols_out);
      let xc_out = new Matrix.valuesConstructor(nrows_out)
      for (let ic = 0; ic < ncols_out; ic++) {

         // take the a whole column from matrix and pick row values from it
         const xc = this.v.subarray((cind.v[ic] - 1) * this.nrows, (cind.v[ic]) * this.nrows);
         for (let ir = 0; ir < nrows_out; ir++) {
            xc_out[ir] = xc[rind.v[ir] - 1];
         }
         out.set(xc_out, ic * nrows_out);
      }

      return new Matrix(out, nrows_out, ncols_out);
   }


   /**
    * Convert matrix to a text representation for showing the values in nice way.
    *
    * @param {number} ndec - number of decimals to round the values to.
    *
    * @returns {string} the string with matrix representation.
    *
    * TODO: 1) add workaround for large matrices (...)
    *
    */
   toString (ndec) {

      function leftpad(x, n) {
         return  " ".repeat(n - x.length) + x;
      }

      if (ndec === undefined) {
         ndec = 2;
      }

      const nvar = this.ncols;
      const ndigits = Math.round(Math.abs(Math.log10(max(this.v)))) + ndec + 3;

      let str = "";
      const Xt = this.t();
      for (let c = 1; c <= Xt.ncols; c++) {
         str += Array.from(Xt.getcolref(c)).map(v => leftpad(v.toFixed(ndec), ndigits)).join(" ") + "\n"
      }

      return str;
   }

   /**
    * Convert matrix values to a string which can be downloaded as CSV file.
    *
    * @param {string} sep - symbol to use for separation of values (by default ",").
    * @param {string} dec - symbold to use for separation of decimals (by default ".").
    * @param {Array} header - optional array of header elements (column names).
    * @param {Array} labels - optional array of labels (row names).
    *
    * @returns {string} - string with CSV data.
    */
   toCSV(sep, dec, header, labels) {

      // set default separator symbol
      if (sep === undefined) {
         sep = ",";
      }

      // set default decimals separator symbol
      if (dec === undefined) {
         dec = sep === ";" ? "," : "."
      }

      const hasLabels = labels !== undefined && labels !== null && labels.length > 0;
      const hasHeader = header !== undefined && header !== null && header.length > 0;

      if (hasLabels && labels.length !== this.nrows) {
         throw Error ("Matrix.toCSV: number of elements in labels array must match number of rows in the matrix.");
      }

      if (hasHeader && header.length !== this.ncols + (hasLabels ? 1 : 0)) {
         throw Error ("Matrix.toCSV: number of elements in header must match number of columns in the matrix.");
      }

      // transpose to make it faster
      const m = this.t();
      let rows = hasHeader ? header.join(sep) + "\n" : "";
      for (let c = 1; c <= m.ncols; c++) {
         rows = rows + (hasLabels ? labels[c - 1] + sep : "") + m.getcolref(c).join(sep) + "\n";
      }

      // change decimal separator if it is not a dot
      rows = rows.replace(/\./g, dec);

      return rows;
   }

   // Static methods //

   /**
    * Create a matrix with random values from normal distribution.
    *
    * @param {number} nrows - number of rows.
    * @param {number} ncols - number of columns.
    * @param {number} [mu=0] - mean (expected) value.
    * @param {number} [sigma=1] - standard deviation.
    *
    * @returns {Vector} matrix with values.
    *
    */
   static randn(nrows, ncols, mu, sigma) {
      return reshape(rnorm(nrows * ncols, mu, sigma), nrows, ncols);
   }


   /**
    * Create a matrix with random values from uniform distribution [a, b].
    *
    * @param {number} nrows - number of rows.
    * @param {number} ncols - number of columns.
    * @param {number} [a=0] - smallest possible value.
    * @param {number} [b=1] - largest possible value.
    *
    * @returns {Vector} matrix with values.
    *
    */
   static rand(nrows, ncols, a, b) {
      return reshape(runif(nrows * ncols, a, b), nrows, ncols);
   }


   /**
    * Create an identity matrix.
    *
    * @param {number} nrows - number of rows.
    * @param {number} ncols - number of columns (if undefined, will be same as 'nrows').
    *
    * @returns {Matrix} the matrix.
    *
    */
   static eye(nrows, ncols) {

      if (ncols === undefined) {
         ncols = nrows;
      }

      const out = new Float64Array(nrows * ncols);
      for (let rc = 0; rc < Math.min(nrows, ncols); rc++) {
         out[rc * nrows + rc] = 1;
      }

      return new Matrix(out, nrows, ncols);
   }


   /**
    * Create a diagonal matrix based on vector with values.
    *
    * @param {Vector} x - vector with values.
    *
    * @returns {Matrix} the matrix.
    *
    */
   static diagm(x) {

      if (!isvector(x)) {
         throw Error('diagm: the argument "x" must be a Vector.');
      }

      const n = x.length;
      const out = new Float64Array(n * n);
      for (let rc = 0; rc < n; rc++) {
         out[rc * n + rc] = x.v[rc];
      }

      return new Matrix(out, n, n);
   }


   /**
    * Create a Matrix object filled with zeros.
    *
    * @param {number} nrows - number of rows.
    * @param {number} ncols - number of columns.
    *
    * @returns {Matrix} the generated matrix.
    *
    */
   static zeros(nrows, ncols) {

      if (ncols === undefined) {
         ncols = nrows;
      }

      return new Matrix(new Float64Array(nrows * ncols), nrows, ncols);
   }


   /**
    * Create a Matrix object filled with ones.
    *
    * @param {number} nrows - number of rows.
    * @param {number} ncols - number of columns.
    *
    * @returns {Matrix} the generated matrix.
    *
    */
   static ones(nrows, ncols) {
      return Matrix.fill(1, nrows, ncols);
   }

   /**
    * Create a Matrix object filled with a manual value.
    *
    * @param {number} v - value to fill the matrix with.
    * @param {number} nrows - number of rows.
    * @param {number} ncols - number of columns.
    *
    * @returns {Matrix} the generated matrix.
    *
    */
   static fill(v, nrows, ncols) {

      if (ncols === undefined) {
         ncols = nrows;
      }

      const out = new Float64Array(nrows * ncols);
      for (let i = 0; i < out.length; i++) {
         out[i] = v;
      }

      return new Matrix(out, nrows, ncols);
   }


   /**
    * Create a matrix by applying function to all possible pairs of values from two vectors.
    *
    * @param {Array|Vector} x - vector with values.
    * @param {Array|Vector} y - vector with values.
    * @param {function} fun - function of two arguments which returns a single value.
    *
    * @returns {Matrix}
    */
   static outer(x, y, fun) {

      if (Array.isArray(x)) {
         x = vector(x);
      }

      if (Array.isArray(y)) {
         y = vector(y);
      }

      if (!isvector(x)) {
         throw Error('Matrix.outer: parameter "x" must be a vector or an array with numbers.');
      }

      if (!isvector(y)) {
         throw Error('Matrix.outer: parameter "y" must be a vector or an array with numbers.');
      }

      const out = Matrix.zeros(x.length, y.length);
      for (let c = 0; c < y.length; c++) {
         const yv = y.v[c]
         const outc = out.getcolref(c + 1);
         for (let r = 0; r < x.length; r++) {
            outc[r] = fun(x.v[r], yv);
         }
      }

      return out;
   }


   /**
    * Parse string with data from a CSV file and create a matrix with values.
    *
    * @param {string} strData - string with all data from the CSV file.
    * @param {string} sep - symbols used for separation of values (by default ",").
    * @param {boolean} hasHeader - logical, if first row must be treated as header or not.
    * @param {boolean} hasLabels - logical, if first column must be treated as object labels or not.
    *
    *
    * @returns {Object} JSON containing matrix with parsed values, header, values and decimal separators.
    */
   static parseCSV(strData, sep, hasHeader, hasLabels) {


      // function which returns true if at least one value in array can not be parsed to float
      function hasNaN(vec) {
         return vec.some(v => Number.isNaN(Number.parseFloat(v.trim())));
      }

      // set default separator symbol
      if (sep === undefined) {
         sep = ",";
      }

      // setup decimal separator
      const dec = sep === ";" ? "," : ".";

      // remove BOM symbol if any
      strData = strData.replace(/^\uFEFF/gm, "").replace(/^\u00EF?\u00BB\u00BF/gm,"");

      // remove \r symbols
      strData = strData.replace(/\r/gm, "");

      // split string with data values into array of rows
      let rows = strData.split(/\n/);

      // check if the first row contains header
      let header = [];
      let startRow = 0;
      const firstRow = rows[0].split(sep);
      if (hasHeader || (hasHeader === undefined && hasNaN(firstRow))) {
         header = firstRow;
         startRow = 1;
      }

      // filter empty rows out
      rows = rows.filter(v => v.trim().length > 1);

      // check if first column contains labels
      let labels = [];
      let startColumn = 0;
      if (hasLabels) {
         startColumn = 1;
      }

      // number of rows and columns in future matrix
      const nrows = rows.length - startRow;
      const ncols = firstRow.length - startColumn;


      if (nrows < 1) {
         throw Error("Data file should contain at least 1 row with values.");
      }

      if (ncols < 1) {
         throw Error("Data file should contain at least 1 column with values.");
      }

      // set rows of data as columns of matrix - it is faster
      const data = Matrix.zeros(ncols, nrows);

      // find symbol used for decimals separator
      for (let r = startRow; r < nrows + startRow; r++) {

         // parsing: trim spaces and replace non-numbers with dot (.)
         const rowValuesAll = rows[r].split(sep)

         let rowValues;
         if (hasLabels) {
            labels[r - startRow] = rowValuesAll[0];
            rowValues = rowValuesAll.slice(1).map(v => Number.parseFloat(v.trim().replace(dec, ".")));
         } else {
            rowValues = rowValuesAll.map(v => Number.parseFloat(v.trim().replace(dec, ".")));
         }

         if (rowValues.some(v => Number.isNaN(v))) {
            throw Error("Some of the values can not be converted to numbers, check your CSV file and try again.");
         }

         if (rowValues.length != ncols) {
            throw Error("Wrong number of values in row #" + r);
         }

         data.v.set(new Float64Array(rowValues), (r - startRow) * ncols);
      }

      return {values: data.t(), header: header, labels: labels, sep:sep, dec:dec};
   }

}


/**
 * Return 'true' of 'x' is a Vector object, 'false' otherwise.
 *
 * @param {any} x - any object or variable.
 *
 * @returns {boolean}
 *
 */
export function isvector(x) {
   return x.constructor === Vector;
}


/**
 * Create a Vector object from array of values.
 *
 * @param {number[]} values - array with values.
 *
 * @returns {Vector} the vector.
 *
 */
export function vector(values) {
   return new Vector(new Float64Array(values));
}


/** Class representing a vector */
export class Vector {

   static valuesConstructor = Float64Array;

   /**
    * Constructor for a Vector object.
    *
    * @param {Float64Array} values - typed array with values.
    *
    * @returns {Vector} a Vector object (see description).
    * @constructor
    *
    * @description  Create a vector of values. Vector is represented by an object with two
    * fields. Field 'v' is a typed Float64Array with values. The second field, 'length', is
    * the length of the vector.
    *
    */
   constructor(values) {

      if (!ArrayBuffer.isView(values) || values.constructor !== Float64Array) {
         throw Error('Vector: parameter "values" must be Float64Array.')
      }

      this.v = values;
      this.length = values.length;
   }

   /**
    * Returns a vector of indices for values for which function ´fun´ returns true.
    *
    * @param {function} fun - function with two arguments (value and its index).
    *
    * @returns {Index} a vector of indices.
    */
   which(fun) {
      const out = new Index.valuesConstructor(this.length);
      let n = 0;
      for (let i = 0; i < this.length; i++) {
         if (fun(this.v[i], i)) {
            out[n] = i + 1;
            n = n + 1;
         }
      }

      return new Index(out.subarray(0, n));
   }


   /**
    * Returns a subset of values for which function ´fun´ returns true.
    *
    * @param {function} fun - function with two arguments (value and its index).
    *
    * @returns {Vector} a subset.
    */
   filter(fun) {
      const out = new Vector.valuesConstructor(this.length);
      let n = 0;
      for (let i = 0; i < this.length; i++) {
         if (fun(this.v[i], i)) {
            out[n] = this.v[i];
            n = n + 1;
         }
      }

      return new Vector(out.subarray(0, n));
   }


   /**
    * Shuffles values in vector using Fisher–Yates algorithm.
    *
    * @returns {Vector} vector with shuffled values.
    */
   shuffle() {
      return new Vector(_shuffle(this.v));
   }


   /**
    * Sorts values in a vector.
    *
    * @param {boolean} decreasing - if true, values will be sorted in decreasing order, otherwise in increasing.
    *
    * @returns {Vector} vector with sorted values
    *
    */
   sort(decreasing) {
      return new Vector(_sort(this.v, decreasing));
   }


   /**
    * Replicate a vector 'n' times.
    *
    * @param {number} n - how many times the vector must be replicated.
    *
    * @returns {Vector} a vector with replicated values.
    *
    */
   rep(n) {
      return _rep(this, n);
   }


   /**
    * Replicate each value in the vector 'n' times.
    *
    * @param {number} n - how many times every value must be replicated.
    *
    * @returns {Vector} a vector with replicated values.
    *
    */
   repeach(n) {
      return _repeach(this, n);
   }


   /**
    * Replace values of vectors specified by 'ind' with values from another vector.
    *
    * @param {Vector} - vector with values to use as replacement.
    * @param {number | Array | Index} ind - indices (as number or vector of indices).
    *
    * @description Indices must start from 1. Empty array ([]) tells function to use
    * all elements. Length of vector 'x' should correspond to the
    * number of indices.
    *
    * Function does not create a new vector but changes the current one.
    *
    * @returns {Vector} the vector with replaced values.
    *
    */
   replace (x, ind) {

      if (typeof(ind) === 'number') {
         ind = [ind];
      }

      if (ind.length === 0) {
         ind = Index.seq(1, this.length);
      }

      if (!isindex(ind)) {
         ind = index(ind);
      }

      const n = ind.length;

      if (min(ind) < 1 || max(ind) > this.length) {
         throw Error('Vector.replace: indices do not match the lenght of the vector.');
      }

      if (x.length !== n) {
         throw Error('Vector.replace: length of "x" does not match the indices.');
      }

      for (let i = 0; i < ind.length; i++) {
         const r = ind.v[i];
         this.v[r - 1] = x.v[i];
      }

      return this;
   }


   /**
    * Create a subset of a vector using vector with indices.
    *
    * @param {number|Array|Index} ind - single index or vector with indices (must start from 1, not 0).
    *
    * @returns {Vector} a subset.
    */
   subset(ind) {

      if (typeof(ind) === 'number') {
         ind = index([ind]);
      }

      if (Array.isArray(ind)) {
         ind = typeof(ind[0]) === 'boolean' ? Index.bool2ind(ind) : index(ind);
      }

      if (!isindex(ind)) {
         throw Error('subset: parameter "ind" must be number, array of instance of class Index.');
      }

      const n = ind.length;
      const out = Vector.zeros(n);

      for (let i = 0; i < n; i++) {

         if (ind.v[i] < 1) {
            throw Error('subset: indices must start with 1 (not 0).');
         }

         if (ind.v[i] > this.length) {
            throw Error('subset: index exceeds the length of the vector.');
         }

         out.v[i] = this.v[ind.v[i] - 1];
      }

      return out;
   }


   /**
    * Create a subset of a vector values which are located between positions 'start' and 'end' (both included).
    *
    * @param {number} start - index of value to start (must start from 1, not 0).
    * @param {number} end - index of value to end.
    *
    * @returns {Vector} a subset.
    */
   slice(start, end) {

      if (start === null) {
         start = 1;
      }

      if (end === null || end === undefined) {
         end = this.length;
      }

      if (start < 1) {
         throw Error('slice: indices must start with 1 (not 0).');
      }

      if (end > this.length) {
         throw Error('slice: index exceeds the length of the vector.');
      }

      if (end < start) {
         throw Error('slice: "end" must not be smaller than "start".');
      }

      return new Vector(this.v.slice(start - 1, end));
   }


   /**
    * Make a deep copy of the vector.
    */
   copy() {
      return new Vector(this.v.slice());
   }


   /**
    * Compute a dot product with another vector.
    *
    * @param {Vector} x - a vector.
    *
    * @returns {number} result of the product.
    *
    */
   dot(x) {

      if (x === undefined) {
         x = this;
      }

      return _dot(this.v, x.v, 1, this.length, x.length, 1)[0];
   }


   /**
    * Does mathematical operation between vector values and either values from another vector or a number.
    *
    * @param {number|Vector} x - a vector or a number to operate with.
    * @param {function} fun - function to apply.
    *
    * @returns {Vector} result of operation.
    *
    */
   op(x, fun) {
      if (isvector(x)) {
         return new Vector(_opvv(this.v, x.v, fun));
      }

      if (isnumber(x)) {
         return new Vector(_opvn(this.v, x, fun));
      }

      throw new Error('op: argument "x" must be a vector or a number.');
   }


   /**
    * Add a number or values from another vector.
    *
    * @param {number|Vector} x - a vector or a number to sum the values with.
    *
    * @returns {Vector} result of operation.
    *
    */
   add(x) {
      return this.op(x, (a, b) => a + b);
   }


   /**
    * Subtract a number or values from another vector.
    *
    * @param {number|Vector} x - a vector or a number to subtract the values of.
    *
    * @returns {Vector} result of operation.
    *
    */
   subtract(x) {
      return this.op(x, (a, b) => a - b);
   }


   /**
    * Multiply to a number or to values from another vector.
    *
    * @param {number|Vector} x - a vector or a number to multiply the values to.
    *
    * @returns {Vector} result of operation.
    *
    */
   mult(x) {
      return this.op(x, (a, b) => a * b);
   }


   /**
    * Divide to a number or to values from another vector.
    *
    * @param {number|Vector} x - a vector or a number to divide the values to.
    *
    * @returns {Vector} result of operation.
    *
    */
   divide(x) {
      return this.op(x, (a, b) => a / b);
   }


   /**
    * Apply a function to each value of the vector.
    *
    * @param {function} fun - function to apply (should take one value and return one value).
    *
    * @returns {Vector} the transformed vector.
    */
   apply(fun) {

      const n = this.v.length;
      const out = Vector.zeros(n);
      for (let i = 0; i < n; i++) {
         out.v[i] = fun(this.v[i]);
      }

      return out;
   }


   // Static methods //


   /**
    * Concatenates numbers, arrays and vectors into a single vector.
    *
    * @param  {...any} args - numbers, arrays or/and vectors.
    *
    * @returns {Vector}
    *
    */
   static c(...args) {

      if (args.length === 1) return args[0];

      const l = args.reduce( (acc, cur) => acc + (typeof(cur) === 'number' ? 1 : cur.length), 0);
      const out = new Vector.valuesConstructor(l);

      let start = 0;
      for (const a of args) {

         if (typeof(a) === 'number') {
            out[start] = a;
            start += 1;
         } else {
            out.set(isvector(a) ? a.v : new Vector.valuesConstructor(a), start);
            start += typeof(a) === 'number' ? 1 : a.length;
         }
      }

      return new Vector(out);
   }


   /**
    * Generate sequence of values.
    *
    * @param {number} start - first value.
    * @param {number} end - last value.
    * @param {number} [by=1] - increment between values.
    *
    * @returns {Vector} vector with sequence values.
    *
    */
   static seq(start, end, by) {
      return _seq(start, end, by, Vector)
   }


   /**
    * Create a vector with random values from normal distribution.
    *
    * @param {number} length - number of values.
    * @param {number} [mu=0] - mean (expected) value.
    * @param {number} [sigma=1] - standard deviation.
    *
    * @returns {Vector} vector with values.
    *
    */
   static randn(length, mu, sigma) {
      return rnorm(length, mu, sigma);
   }


   /**
    * Create a vector with random values from uniform distribution [a, b].
    *
    * @param {number} length - number of values.
    * @param {number} [a=0] - smallest possible value.
    * @param {number} [b=1] - largest possible value.
    *
    * @returns {Vector} vector with values.
    *
    */
   static rand(length, a, b) {
      return runif(length, a, b);
   }


   /**
    * Create a Vector object filled with zeros.
    *
    * @param {number} n - length of the vector
    *
    * @returns {Vector} the generated vector.
    *
    */
   static zeros(n) {
      return new Vector(new Vector.valuesConstructor(n));
   }


   /**
    * Create a Vector object filled with ones.
    *
    * @param {number} n - length of the vector.
    *
    * @returns {Vector} the generated vector.
    *
    */
   static ones(n) {
      return Vector.fill(1, n);
   }


  /**
    * Create a Vector object filled with a repeated value.
    *
    * @param {number} v - value to fill the vector with.
    * @param {number} n - length of the vector.
    *
    * @returns {Vector} the generated vector.
    *
    */
   static fill(v, n) {

      const out = new Vector.valuesConstructor(n);
      for (let i = 0; i < out.length; i++) {
         out[i] = v;
      }

      return new Vector(out);
   }

}


/**
 * Return 'true' of 'x' is an Index object, 'false' otherwise.
 *
 * @param {any} x - any object or variable.
 *
 * @returns {boolean}
 *
 */
export function isindex(x) {
   return x.constructor === Index;
}


/**
 * Create an Index object from array of values.
 *
 * @param {number[]} values - array with values.
 *
 * @returns {Index} the vector.
 *
 */
export function index(values) {
   return new Index(new Int32Array(values));
}


/** Class representing a vector with indices */
export class Index {

   static valuesConstructor = Int32Array;

   /**
    * Constructor for an Index object.
    *
    * @param {Int32Array} values - typed array with index values.
    *
    * @returns {Index} an Index object (see description).
    * @constructor
    *
    * @description  Create a vector of values to be used as indices (e.g. for subsetting,
    * counting, etc.). Index is represented by an object with two fields. Field 'v' is a typed
    * Int32Array with index values. The second field, 'length', is the length of the vector.
    *
    */
   constructor(values) {

      if (!ArrayBuffer.isView(values) || values.constructor !== Int32Array) {
         throw Error('Index: parameter "values" must be Int32Array.')
      }

      this.v = values;
      this.length = values.length;
   }


   /**
    * Returns a vector of indices for values for which function ´fun´ returns true.
    *
    * @param {function} fun - function with two arguments (value and its index).
    *
    * @returns {Index} a vector of indices.
    */
   which(fun) {
      const out = new Index.valuesConstructor(this.length);
      let n = 0;
      for (let i = 0; i < this.length; i++) {
         if (fun(this.v[i], i)) {
            out[n] = i + 1;
            n = n + 1;
         }
      }

      return new Index(out.subarray(0, n));
   }


   /**
    * Returns a subset of indices for which function ´fun´ returns true.
    *
    * @param {function} fun - function with two arguments (value and its index).
    *
    * @returns {Index} a subset.
    */
   filter(fun) {
      const out = new Index.valuesConstructor(this.length);
      let n = 0;
      for (let i = 0; i < this.length; i++) {
         if (fun(this.v[i], i)) {
            out[n] = this.v[i];
            n = n + 1;
         }
      }

      return new Index(out.subarray(0, n));
   }



   /**
    * Shuffle indices.
    *
    * @returns {Index} vector with shuffled indices.
    */
   shuffle() {
      return new Index(_shuffle(this.v));
   }


   /**
    * Sorts indices.
    *
    * @param {boolean} decreasing - if true, values will be sorted in decreasing order, otherwise in increasing.
    *
    * @returns {Index} vector with sorted indices.
    *
    */
   sort(decreasing) {
      return new Index(_sort(this.v, decreasing));
   }


   /**
    * Replicate the index object 'n' times.
    *
    * @param {number} n - how many times the index must be replicated.
    *
    * @returns {Index} index with replicated values.
    *
    */
   rep(n) {
      return _rep(this, n);
   }


   /**
    * Replicate each index in the index 'n' times.
    *
    * @param {number} n - how many times every value must be replicated.
    *
    * @returns {Index} an index object with replicated values.
    *
    */
   repeach(n) {
      return _repeach(this, n);
   }

   /**
    * Create a subset of a index using another vector with indices.
    *
    * @param {number|Array|Index} ind - single index or vector with indices (must start from 1, not 0).
    *
    * @returns {Index} a subset.
    */
   subset(ind) {

     if (typeof(ind) === 'number') {
         ind = index([ind]);
      }

      if (Array.isArray(ind)) {
         ind = index(ind);
      }

      if (!isindex(ind)) {
         throw Error('subset: parameter "ind" must be number, array of instance of class Index.');
      }

      const n = ind.length;
      const out = new Index.valuesConstructor(n);

      for (let i = 0; i < n; i++) {

         if (ind.v[i] < 1) {
            throw Error('subset: indices must start with 1 (not 0).');
         }

         if (ind.v[i] > this.length) {
            throw Error('subset: index exceeds the length of the vector.');
         }

         out[i] = this.v[ind.v[i] - 1];
      }

      return new Index(out);
   }


   /**
    * Create a subset of a vector of indices which are located between positions 'start' and 'end' (both included).
    *
    * @param {number} start - index of value to start (must start from 1, not 0).
    * @param {number} end - index of value to end.
    *
    * @returns {Index} a subset.
    */
   slice(start, end) {

      if (start === null) {
         start = 1;
      }

      if (end === null || end === undefined) {
         end = this.length;
      }

      if (start < 1) {
         throw Error('slice: indices must start with 1 (not 0).');
      }

      if (end > this.length) {
         throw Error('slice: index exceeds the length of the vector.');
      }

      if (end < start) {
         throw Error('slice: "end" must not be smaller than "start".');
      }

      return new Index(this.v.slice(start - 1, end));
   }


   // Static methods //

   /**
    * Generate sequence of indices.
    *
    * @param {number} start - first value.
    * @param {number} end - last value.
    * @param {number} [by=1] - increment between values.
    *
    * @returns {Index} object with sequence values.
    *
    */
   static seq(start, end, by) {
      return _seq(start, end, by, Index);
   }


   /**
    * Create an Index object filled with ones.
    *
    * @param {number} n - length of the vector.
    *
    * @returns {Index} the generated vector of indices.
    *
    */
   static ones(n) {
      return Index.fill(1, n);
   }


  /**
    * Create an Index object filled with a repeated value.
    *
    * @param {number} v - value to fill the index vector with (must be integer).
    * @param {number} n - length of the vector.
    *
    * @returns {Vector} the generated vector of indices.
    *
    */
   static fill(v, n) {

      const out = new Index.valuesConstructor(n);
      for (let i = 0; i < out.length; i++) {
         out[i] = v;
      }

      return new Index(out);
   }


   /**
    * Convert array of logical values to vector of indices.
    *
    * @param {Array} x - array of logical values.
    *
    * @returns {Index} vector of indices where x is true.
    *
    */
   static bool2ind(x) {

      let n = 0;
      const out = new Index.valuesConstructor(x.length);

      for (let i = 0; i < x.length; i++) {
         if (x[i]) {
            out[n] = i + 1;
            n = n + 1;
         }
      }

      return new Index(out.subarray(0, n));
   }
}


/** Class representing a dataset */
export class Dataset {


   /**
    * Constructor for a Dataset object.
    *
    * @param {Matrix} values - matrix with data values.
    * @param {string} name - name of the dataset.
    * @param {Object} rowAttrs - JSON with rows attributes (labels, axis name, axis values and vector with excluded items).
    * @param {Object} colAttrs - JSON with columns attributes (same as for rows).
    *
    * @description  Dataset is a wrapper for Matrix objects letting them have additional attributes,
    * for plotting and other functionalities. The attributes are first of all associated with rows and columns.
    *
    * The attributes consist of:
    * - labels (short ID used e.g. for labeling points on scatter plots, e.g. "PC1", "Height")
    * - axisLabels (longer ID uses for labeling axes on scatter plots, e.g. "PC1 (30%)" or "Height, cm")
    * - axisValues (vector with numbers associated with rows or columns, e.g. time, wavelength, etc.)
    * - axisName (name of axis if axisValues are used for one of the axis, e.g. in line or bar plots.)
    *
    * @returns {Dataset} Dataset object (see description).
    * @constructor
    *
    */
   constructor(values, name, rowAttrs, colAttrs) {

      function processAttributes(attrs, n, name, label) {

         if (!attrs || attrs.constructor !== ({}).constructor) {
            attrs = {};
         }

         if (!attrs.axisValues || !isvector(attrs.axisValues) || attrs.axisValues.length !== n) {
            attrs.axisValues = Vector.seq(1, n);
         }

         if (!attrs.axisName || attrs.axisName === "") {
            attrs.axisName = name;
         }

         if (!attrs.labels || !Array.isArray(attrs.labels) || attrs.labels.length !== n) {
            attrs.labels = Array(n).fill().map((e, i) => label + (i + 1));
         }

         if (!attrs.axisLabels || !Array.isArray(attrs.axisLabels) || attrs.axisLabels.length !== n) {
            attrs.axisLabels = Array(n).fill().map((e, i) => label + (i + 1));
         }

         return attrs;
      }

      if (!ismatrix(values)) {
         throw Error('MDAData: parameter "values" must be an instance of Matrix class.')
      }

      if (values.nrows < 1 || values.ncols < 1) {
         throw Error('MDAData: parameter "values" must have at least one row and one column.')
      }

      this.values = values;
      this.name = name;
      this.rowAttrs = processAttributes(rowAttrs, values.nrows, "Objects", "O");
      this.colAttrs = processAttributes(colAttrs, values.ncols, "Variables", "X");
   }
}


/***********************************************/
/*    Generic methods                          */
/***********************************************/

/**
 * Generate sequence of values for vector or index.
 *
 * @param {number} start - first value.
 * @param {number} end - last value.
 * @param {number} [by=1] - increment between values.
 * @param {class} classRef - class reference
 *
 * @returns {Object} object with sequence values.
 *
 */
function _seq(start, end, by, classRef) {

   // by default increment is unit
   if (by === undefined) by = (start <= end) ? 1 : -1;

   // compute length of sequence
   const length = Math.floor((end - start) / by) + 1;

   if (length < 1) {
      throw Error('seq: wrong combination of parameters.');
   }

   const x = new classRef.valuesConstructor(length);
   for (let i = 0; i < length; i++) {
      x[i] = start + i * by;
   }

   return new classRef(x);
}


/**
 * Replicate object 'n' times.
 *
 * @param {Object} obj - object to replicate (Vector or Index).
 * @param {number} n - how many times the vector must be replicated.
 *
 * @returns {Object} an object with replicated values.
 *
 */
function _rep(obj, n) {

   const l = obj.length * n;
   const out = new obj.constructor.valuesConstructor(l);
   for (let i = 1; i <= n; i++) {
      for (let j = 0; j < obj.length; j++) {
         out[(i - 1) * obj.length + j] = obj.v[j];
      }
   }

   return new obj.constructor(out);
}


/**
 * Replicate each value in the object 'n' times.
 *
 * @param {Object} obj - object to replicate (Vector or Index).
 * @param {number} n - how many times every value must be replicated.
 *
 * @returns {Object} an object with replicated values.
 *
 */
function _repeach(obj, n) {

   if (n < 1 || Math.round(n) !== n) {
      throw Error('rep: parameter "n" must be an integer number > 1.');
   }

   const l = obj.length * n;
   const out = new obj.constructor.valuesConstructor(l);
   for (let j = 0; j < obj.length; j++) {
      for (let i = 0; i < n; i++) {
         out[j * n + i] = obj.v[j];
      }
   }

   return new obj.constructor(out);
}


/**
 * Does an arithmetic operation for individual values from two arrays.
 *
 * @param {Array|Float64Array} v1 - the first array.
 * @param {Array|Float64Array} v2 - the second array.
 * @param {function} fun - function to use.
 *
 * @description the function must take two numbers and return one.
 *
 * @returns array of the same type as v1 and v2.
 */
function _opvv(v1, v2, fun) {

   if (v1.length !== v2.length) {
      throw error("_opvv: arrays sizes do not match.");
   }

   if (v1.constructor !== v2.constructor) {
      throw error("_opvv: array types do not match.");
   }

   const out = new v1.constructor(v1.length);
   for (let i = 0; i < v1.length; i++) {
      out[i] = fun(v1[i], v2[i]);
   }

   return out;
}


/**
 * Does an arithmetic operation for individual value from an array and a number.
 *
 * @param {Array|Float64Array} v1 - an array.
 * @param {number} v2 - a number.
 * @param {function} fun - function to use.
 *
 * @description the function must take two numbers and return one.
 *
 * @returns array of the same type as v1.
 */
function _opvn(v1, v2, fun) {

   const out = new v1.constructor(v1.length);
   for (let i = 0; i < v1.length; i++) {
      out[i] = fun(v1[i], v2);
   }

   return out;
}


/**
 * Generic function for taking a dot product of two matrices, X and Y, represented as vectors
 * @param {TypedArray} xv - vector with values from X
 * @param {TypedArray} yv - vector with values from Y
 * @param {number} nrx - number of rows in X
 * @param {number} ncx - number of columns in X
 * @param {number} nry - number of rows in Y
 * @param {number} ncy - number of columns in Y
 *
 * @returns {TypedArray} - vector with result of the product
 */
export function _dot(xv, yv, nrx, ncx, nry, ncy) {

   if (ncx !== nry) {
      throw Error('_dot: matrix dimensions do not much.');
   }

   // create vector for the product
   const nrows = nrx;
   const ncols = ncy;
   const ninner = ncx;
   const out = new xv.constructor(nrows * ncols);

   for (let c = 0; c < ncols; c++) {
      const yc = yv.subarray(c * nry, (c + 1) * nry)
      const outc = out.subarray(c * nrows, (c + 1) * nrows);

      for (let i = 0; i < ninner; i++) {
         const yci = yc[i];
         const xr = xv.subarray(i * nrx, (i + 1) * nrx)
         for (let r = 0; r < nrows; r++) {
            outc[r] += xr[r] * yci;
         }
      }
   }

   return out;
}


/**
 * Shuffles values in vector x using Fisher–Yates algorithm.
 *
 * @param {Arrat|TypedArray} x - a vector with values.
 *
 * @returns {Array|TypedArray} ector with shuffled values.
 *
 */
export function _shuffle(x) {

  let y = x.slice();
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
 * Sorts values in a vector.
 *
 * @param {Array|TypedArray} x - vector with values.
 *
 * @returns {Array|TypedArray} vector with sorted values.
 *
 */
export function _sort(x, decreasing = false) {
   return decreasing ? x.slice().sort((a, b) => b - a) : x.slice().sort((a, b) => a - b);
}
