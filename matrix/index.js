import {max, min, rep, seq, subset, sum} from '../stat/index.js';

/**********************************************
 * Functions for manipulations with vectors   *
 **********************************************/

/* Simple functions for arithmetics */
const add = (a, b) => a + b;
const subtract = (a, b) => a - b;
const times = (a, b) => a * b;
const divide = (a, b) => a/b;

/**
 * Computes a Euclidean norm of a vector x
 * @param {Array} x — a vector of values
 * @returns a number (the norm)
 */
export function norm2(x) {

   if (!isvector(x)) {
      throw Error("Argument 'x' must be a vector.");
   }

   return Math.sqrt(sum(x.map(v => v**2)));
}

/**
 * Replaces subset of values in vector x, specified by indices, with values from vector y
 *
 * @param {Array} x — vector with values to be replaced
 * @param {Array} y — vector with values used for replacement
 * @param {Array} ind — vector of indices to select (starting from 1)
 */
export function vreplace(x, y, ind) {

   if (!isvector(x)) {
      throw Error("Argument 'x' must be a vector.");
   }

   if (!isvector(y)) {
      throw Error("Argument 'y' must be a vector.");
   }

   ind = processIndices(ind, x.length, true);

   if (ind.length !== y.length) {
      throw Error("Number of values in 'ind' should match the number of elements in 'y'.");
   }

   let z = subset(x, []);
   for (let i = 0; i < ind.length; i++) {
      z[ind[i] - 1] = y[i];
   }

   return z;
}

/**
 * Applies a function to each element of a vector
 *
 * @param {Array} x - a vector
 * @param {function} fun - a function which takes a numbers and returns a number
 * @returns {Array} - result of the operation
 */
export function vapply(x, fun) {

   if (!isvector(x)) {
      throw Error("Argument 'x' must be a vector.");
   }

   return x.map(v => fun(v));
}

/**
 * Does element by element division of two vectors, or a vector and a scalar
 * (one of the arguments must be a vector)
 *
 * @param {Array|number} x - a vector or a scalar
 * @param {Array|number} y - a vector or a scalar
 * @returns {Array} - result of the multiplication
 */
export function vdiv(x, y) {
   return vop(x, y, divide);
}


/**
 * Does element by element multiplication of two vectors, or a vector and a scalar
 * (one of the arguments must be a vector)
 *
 * @param {Array|number} x - a vector or a scalar
 * @param {Array|number} y - a vector or a scalar
 * @returns {Array} - result of the multiplication
 */
export function vmult(x, y) {
   return vop(x, y, times);
}


/**
 * Does element by element addition of two vectors, or a vector and a scalar
 * (one of the arguments must be a vector)
 *
 * @param {Array|number} x - a vector or a scalar
 * @param {Array|number} y - a vector or a scalar
 * @returns {Array} - result of the addition
 */
export function vadd(x, y) {
   return vop(x, y, add);
}


/**
 * Does element by element subtraction of two vectors, or a vector and a scalar
 * (one of the arguments must be a vector)
 *
 * @param {Array|number} x - a vector or a scalar
 * @param {Array|number} y - a vector or a scalar
 * @returns {Array} - result of the subtraction
 */
export function vsubtract(x, y) {
   return vop(x, y, subtract);
}


/**
 * Does element by element arithmetic operation for two vectors, or for a vector and a scalar
 * (one of the arguments must be a vector)
 *
 * @param {Array|number} x - a vector or a scalar
 * @param {Array|number} y - a vector or a scalar
 * @param {function} op - a function which takes two numbers and return a number
 * @returns {Array} - result of the operation
 */
export function vop(x, y, op) {

   // find the longest vector
   const n = x.length | y.length;

   if (n < 1) {
      throw Error("One of the arguments must be a vector.");
   }

   if (!isvector(y)) {
      y = rep(y, n);
   }

   if (!isvector(x)) {
      x = rep(x, n);
   }

   if (x.length !== y.length) {
      throw Error("Dimensions of 'x' and 'y' mismatch.");
   }

   let res = Array(n).fill(0);
   for (let i = 0; i < n; i++) {
      res[i] = op(x[i], y[i]);
   }

   return res;
}

/**
 * Checks if argument is a vector (1D Array)
 * @param {any} X - an object/variable
 * @returns {boolean} - result of check
 */
export function isvector(X) {

   if (!isarray(X)) return false;
   if (Array.isArray(X[0])) return false;

   return true;
}


/**
 * Computes a dot product of two vectors
 * @param {Array} x - a vector of values (1D Array)
 * @param {Array} y - a vector of values (same length as x)
 * @returns {Number} - result of dot product of the two vectors
 */
export function vdot(x, y) {

   if (!isvector(x)) {
      throw Error("Argument 'x' must be a vector of numbers.");
   }

   if (!isvector(y)) {
      throw Error("Argument 'y' must be a vector of numbers.");
   }

   if (x.length != y.length) {
      throw Error("Vectors 'x' and 'y' must have the same length.");
   }

   let res = 0;
   for (let i = 0; i < x.length; i++) {
      res = res + x[i] * y[i];
   }

   return res;
}


/**********************************************
 * Functions for manipulations with matrices  *
 **********************************************/


export function cbind(X, Y) {

   if (!isarray(X) || !isarray(Y)) {
      throw Error("Both 'X' and 'Y' must arrays (matrices or vectors).");
   }

   if (isvector(X)) {
      X = [X];
   }

   if (isvector(Y)) {
      Y = [Y];
   }

   if (nrow(X) !== nrow(Y)) {
      throw Error("Number of rows (or vector elements) in X and Y must be be the same.");
   }

   return X.concat(Y);
}

export function rbind(X, Y) {

   if (!isarray(X) || !isarray(Y)) {
      throw Error("Both 'X' and 'Y' must arrays (matrices or vectors).");
   }

   if (isvector(X)) {
      X = transpose([X]);
   }

   if (isvector(Y)) {
      Y = transpose([Y]);
   }

   if (ncol(X) !== ncol(Y)) {
      throw Error("Number of columns (or vector elements) in X and Y must be be the same.");
   }


   return transpose(cbind(transpose(X), transpose(Y)));
}
/**
 * Check row or column indices
 * @param {Array|number} ind — vector or a value with indices
 * @param {number} n — number of rows or columns in original matrix
 * @param {number} fill — logical, if 'true' and 'ind' is empty, will generate values from 1 to n
 * @returns array with indices
 */
function processIndices(ind, n, fill) {
   if (!Array.isArray(ind)) {
      ind = [ind];
   }

   if (ind.length > 0 && (min(ind) < 1 || max(ind) > n)) {
      throw Error("Wrong values for indices.");
   }

   if (ind.length === 0 && fill) {
      ind = seq(1, n);
   }

   return ind;
}

/**
 * Creates a subset of matrix X specified by row and column indices
 *
 * If all rows or all columns must be selected provide empty array, [], as indices.
 *
 * @param {Array} X — matrix with values
 * @param {Array} rowInd — vector of row indices to select (starting from 1)
 * @param {Array} colInd — vector of column indices to select (starting from 1)
 * @param {string} method - what to do with values ("select" or "remove")
 */
export function msubset(X, rowInd, colInd, method) {

   if (!ismatrix(X)) {
      throw Error("Argument 'X' must be a matrix.");
   }

   colInd = processIndices(colInd, ncol(X), method === "select");
   rowInd = processIndices(rowInd, nrow(X), false);

   if (method === "remove" || colInd.length === 0) {
      colInd = subset(seq(1, ncol(X)), colInd, "remove");
   }

   let Y = Array(colInd.length);
   for (let c = 0; c < colInd.length; c++) {
      Y[c] = subset(X[colInd[c] - 1], rowInd, method);
   }

   return Y;
}


/**
 * Replaces subset of values in matrix X, specified by row and column indices, with values from matrix Y
 *
 * If all rows or all columns must be taken provide empty array, [], as indices.
 *
 * @param {Array} X — matrix with values to be replaced
 * @param {Array} Y — matrix with values used for replacement
 * @param {Array} rowInd — vector of row indices to select (starting from 1)
 * @param {Array} colInd — vector of column indices to select (starting from 1)
 */
export function mreplace(X, Y, rowInd, colInd) {

   if (!ismatrix(X)) {
      throw Error("Argument 'X' must be a matrix.");
   }

   rowInd = processIndices(rowInd, nrow(X), true);
   colInd = processIndices(colInd, ncol(X), true);

   if (rowInd.length !== nrow(Y)) {
      throw Error("Number of values in 'rowInd' should match the number of rows in 'Y'.");
   }

   if (colInd.length !== ncol(Y)) {
      throw Error("Number of values in 'colInd' should match the number of columns in 'Y'.");
   }

   let Z = msubset(X, [], []);
   for (let c = 0; c < colInd.length; c++) {
      for (let r = 0; r < rowInd.length; r++) {
         Z[colInd[c] - 1][rowInd[r] - 1] = Y[c][r];
      }
   }

   return Z;
}


/**
 * Computes XY' product
 *
 * @param {Array} X - a matrix
 * @param {Array} Y - a matrix
 * @returns {Array} - result of the product
 */
export function tcrossprod(X, Y) {

   if (!Y) {
      Y = msubset(X, [], []);
   }

   if (!ismatrix(X)  || !ismatrix(Y)) {
      throw Error("Both arguments must be matrices (2D Arrays).");
   }

   return mdot(X, transpose(Y));
}


/**
 * Computes X'Y product
 *
 * @param {Array} X - a matrix
 * @param {Array} Y - a matrix
 * @returns {Array} - result of the product
 */
export function crossprod(X, Y) {

   if (!Y) {
      Y = msubset(X, [], []);
   }

   if (!ismatrix(X)  || !ismatrix(Y)) {
      throw Error("Both arguments must be matrices (2D Arrays).");
   }

   return mdot(transpose(X), Y);
}


/**
 * Does element by element operation of two matrices, a matrix and a scalar or a matrix and a vector
 *
 * if second argument is a vector, function checks its dimension. If it has the same number of elements
 * as number of rows in 'X' it will be applied to every column of 'X'. If it has the same number of
 * elements as number of columns in 'X', it will be applied to every row.
 *
 * @param {Array} X - a matrix
 * @param {Array|number} Y - a matrix, a vector or a scalar
 * @returns {Array} - result of the addition
 */
export function mop(X, Y, op) {

   if (!ismatrix(X)) {
      throw Error("Argument 'X' must be a matrix (2D Array).");
   }

   const nrows = nrow(X);
   const ncols = ncol(X);

   if (!ismatrix(Y)) {
      Y = tomatrix(Y, nrows, ncols);
   }

   if (nrow(X) !== nrow(Y) || ncol(X) !== ncol(Y)) {
      throw Error("Dimensions of 'X' and 'Y' mismatch.");
   }

   let res = zeros(nrows, ncols);
   for (let i = 0; i < nrows; i++) {
      for (let j = 0; j < ncols; j++) {
         res[j][i] = op(X[j][i], Y[j][i])
      }
   }

   return res;
}


/**
 * Does element by element division of two matrices, a matrix and a scalar or a matrix and a vector
 *
 * if second argument is a vector, function checks its dimension. If it has the same number of elements
 * as number of rows in 'X' it will be applied to every column of 'X'. If it has the same number of
 * elements as number of columns in 'X', it will be applied to every row.
 *
 * @param {Array} X - a matrix
 * @param {Array|number} Y - a matrix, a vector or a scalar
 * @returns {Array} - result of the multiplication
 */
export function mdiv(X, Y) {
   return mop(X, Y, divide);
}


/**
 * Does element by element addition of two matrices, a matrix and a scalar or a matrix and a vector
 *
 * if second argument is a vector, function checks its dimension. If it has the same number of elements
 * as number of rows in 'X' it will be applied to every column of 'X'. If it has the same number of
 * elements as number of columns in 'X', it will be applied to every row.
 *
 * @param {Array} X - a matrix
 * @param {Array|number} Y - a matrix, a vector or a scalar
 * @returns {Array} - result of the addition
 */
export function madd(X, Y) {
   return mop(X, Y, add);
}

/**
 * Does element by element subtraction of two matrices, a matrix and a scalar or a matrix and a vector
 *
 * if second argument is a vector, function checks its dimension. If it has the same number of elements
 * as number of rows in 'X' it will be applied to every column of 'X'. If it has the same number of
 * elements as number of columns in 'X', it will be applied to every row.
 *
 * @param {Array} X - a matrix
 * @param {Array|number} Y - a matrix, a vector or a scalar
 * @returns {Array} - result of the subtraction
 */
export function msubtract(X, Y) {
   return mop(X, Y, subtract);
}


/**
 * Does element by element multiplication of two matrices, a matrix and a scalar or a matrix and a vector
 *
 * if second argument is a vector, function checks its dimension. If it has the same number of elements
 * as number of rows in 'X' it will be applied to every column of 'X'. If it has the same number of
 * elements as number of columns in 'X', it will be applied to every row.
 *
 * @param {Array} X - a matrix
 * @param {Array|number} Y - a matrix, a vector or a scalar
 * @returns {Array} - result of the multiplication
 */
export function mmult(X, Y) {
   return mop(X, Y, times);
}


/**
 * Computes inner (dot) product of two matrices
 * @param {Array} X - a matrix (array of vectors of the same length)
 * @param {Array} Y - a matrix (array of vectors of the same length)
 * @returns {Array} - result of dot product
 */
export function mdot(X, Y) {

   if (isvector(X)) {
      X = [X];
   }

   if (!ismatrix(X)) {
      throw Error("Argument 'X' must be a vector or a matrix (1D or 2D Array).");
   }

   if (isvector(Y)) {
      Y = [Y];
   }

   if (!ismatrix(Y)) {
      throw Error("Argument 'Y' must be a vector or a matrix (1D or 2D Array).");
   }


   if (ncol(X) != nrow(Y)) {
      throw Error("Dimensions of 'X' and 'Y' mismatch.");
   }

   const n = nrow(X);
   const m = ncol(Y);
   let res = zeros(n, m);

   X = transpose(X);
   for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
         res[j][i] = vdot(X[i], Y[j])
      }
   }

   return res;
}

/**
 * Returns a transposed matrix
 * @param {Array} X - a vector or a matrix (1D or 2D Array)
 * @returns {Array} - a transposed
 */
export function transpose(X) {

   if (isvector(X)) {
      X = [X];
   }

   if (!ismatrix(X)) {
      throw Error("Argument 'X' must be a vector or a matrix (1D or 2D Array).");
   }

   return X[0].map((_, colIndex) => X.map(row => row[colIndex]));
}

/**
 * Creates an identity matrix of size 'n'
 * @param {number} n - number of rows and columns in the matrix
 */
export function eye(n) {
   let res = zeros(n, n);
   for (let i = 0; i < n; i++) {
      res[i][i] = 1;
   }

   return res;
}


/**
 * Returns a vector with values from main diagonal of squared matrix 'x'
 * @param {Array} x - a squared matrix
 * @return vector with diagonal elements
 */
export function diag(x) {

   if (!issquaredmat(x)) throw Error("Argument 'x' must be a squared matrix.");

   const n = nrow(x);
   let res = rep(0, n);
   for (let i = 0; i < n; i++) {
      res[i] = x[i][i];
   }

   return res;
}

/**
 * Creates a diagonal matrix filled with values from vector 'x'
 * @param {Array} x - a vector with values
 */
export function diagm(x) {
   if (!isvector(x)) throw Error("Argument 'x' must be a vector.");

   const n = x.length;
   let res = zeros(n, n);
   for (let i = 0; i < n; i++) {
      res[i][i] = x[i];
   }

   return res;
}

/**
 * Returns a matrix (2D Array) filled with zeros
 * @param {Number} n - number of rows
 * @param {Number} m - number of columns
 * @returns {Array} - the generated matrix
 */
export function zeros(n, m) {
   return matrix(n, m, 0);
}


/**
 * Creates a matrix (2D Array) filled with constant value
 * @param {Number} n - number of rows
 * @param {Number} m - number of columns
 * @param {Number} a - value
 * @returns {Array} - the generated matrix
 */
export function matrix(n, m, a) {
   return [...Array(m)].map(v => Array(n).fill(a));
}


/**
 * Returns number of rows in a matrix
 * @param {Array} X - a vector or a matrix (array of vectors of the same length)
 * @returns {Number} - number of rows
 */
export function nrow(X) {

   if (isvector(X)) {
      // if vector we treat it as column-vector
      return X.length
   }

   if (!ismatrix(X)) {
      throw Error("Argument 'X' must be a vector or a matrix (1D or 2D Array).");
   }

   return X[0].length;
}


/**
 * Returns number of columns in a matrix
 * @param {Array} X - a vector or a matrix (1D or 2D Array)
 * @returns {Number} - number of rows
 */
export function ncol(X) {


   if (isvector(X)) {
      // if vector we treat it as column-vector
      return 1
   }

   if (!ismatrix(X)) {
      throw Error("Argument 'X' must be a vector or a matrix (1D or 2D Array).");
   }

   return X.length;
}

/**
 * Converts a scalar or a vector into a matrix
 *
 * if 'x' is a scalar it returns a matrix filled with this value, if 'x' is a vector, function
 * checks its dimension. If it has the same number of elements  as 'nrows' it replicates 'x' as
 * columns of the matrix, if it has the same number of elements as 'ncols', it replicates 'x' as
 * rows of the matrix.
 * @param {Array|number} x - a scalar or a vector of values
 * @param {number} nrows - number of rows in final matrix
 * @param {number} ncols - number of columns in the final matrix
 */
export function tomatrix(x, nrows, ncols) {

   if (!isarray(x)) return matrix(nrows, ncols, x);

   if (!isvector(x)) {
      throw Error("Argument 'x' must me a scalar or a vector.");
   }

   // if number of elements in vector is the same as number of rows — replicate the vector column wise
   if (x.length === nrows) return Array(ncols).fill(x);

   // if number of elements in vector is the same as number of columns — replicate the vector row wise
   if (x.length === ncols) return transpose(Array(nrows).fill(x));

   // if number of elements in vector is the same as product of number of rows and columns — reshape the vector into matrix
   if (x.length === nrows * ncols) return Array(ncols).fill(null).map((v, i) => subset(x, vadd(seq(1, nrows), nrows * i )));

   throw Error("Number of elements in 'x' does not match neither 'nrows' nor 'ncols'.")
}


/**
 * Checks if argument is a non empty array
 * @param {any} X - an object/variable
 * @returns {boolean} - result of check
 */
export function isarray(X) {
   return Array.isArray(X) && X.length > 0;
}


/**
 * Checks if argument is a matrix (2D Array)
 * @param {any} X - an object/variable
 * @returns {boolean} - result of check
 */
export function ismatrix(X) {

   if (!isarray(X)) return false;
   if (!isarray(X[0])) return false;

   // check that all columns/vectors have the same length
   if (!X.every(v => v.length == X[0].length)) return false;

   return true;
}

/**
 * Return true if matrix is squared
 * @param {number[]} X - matrix to check (2D Array)
 * @returns logical value
 */
export function issquaredmat(X) {
   return ismatrix(X) && (nrow(X) === ncol(X));
}


/**
 * Return true if matrix is lower triangular
 * @param {number[]} X - matrix to check (2D Array)
 * @returns logical value
 */
export function islowertrianmat(X) {
   return isuppertrianmat(transpose(X));
}


/**
 * Return true if matrix is diagonal
 * @param {number[]} X - matrix to check (2D Array)
 * @returns logical value
 */
export function isdiagmat(X) {
   if (!issquaredmat(X)) throw Error("Argument 'X' must be a squared matrix.");

   const n = nrow(X);
   if (nrow(X) != ncol(X)) return false;

   for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
         if ((i !== j) && Math.abs(X[i][j]) > Number.EPSILON) return false;
      }
   }

   return true;
}

/**
 * Return true if matrix is upper triangular
 * @param {number[]} X - matrix to check (2D Array)
 * @returns logical value
 */
export function isuppertrianmat(X) {

   if (!issquaredmat) return false;

   const n = ncol(X);
   for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++)
         if (Math.abs(X[i][j]) > 10**(-10) )
            return false;

   return true;
}