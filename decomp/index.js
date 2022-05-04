import {rep, min, subset, seq} from '../stat/index.js';
import {nrow, ncol, eye, norm2, zeros, vadd, mdot, madd, transpose, msubset, mmult} from '../matrix/index.js';
import { issquaredmat, islowertrianmat, isuppertrianmat, mreplace, tcrossprod } from '../matrix/index.js';

/**********************************************
 * Functions for decompositions of matrices   *
 **********************************************/

/**
 * Computes QR decomposition of matrix X using Householder reflections
 * @param {number[]} X - matrix to decompose (2D Array)
 * @returns JSON with two matrices, Q and R
 */
export function qr(X) {

   const ncols = ncol(X);
   const nrows = nrow(X);
   const tMax = min([nrows, ncols]);

   let R = msubset(X, [], []);
   let Q = eye(nrows);

   for (let ic = 0; ic < tMax; ic++) {

      const x = subset(R[ic], seq(ic + 1, nrows));
      const n = x.length;

      let e = rep(0, n);
      e[0] = Math.sign(x[0]) * norm2(x);

      const v = vadd(e, x);
      const Qt = madd(eye(n), mmult(tcrossprod([v]), -2 / (norm2(v) ** 2)));
      const Qk = mreplace(eye(nrows), Qt, seq(nrows - n + 1, nrows), seq(nrows - n + 1, nrows));

      Q = mdot(Q, Qk)
      R = mdot(Qk, R);
   }

   return {Q, R};
}


/**
 * Computes Singular Value Decomposition of matrix X
 * @param {number[]} X — matrix to decompose
 * @param {number} n — number of components to compute
 */
export function svd(X, n) {
   if (!n) n = min([nrow(X) - 1, ncol(X)]);
}


/**
 * Computes inverse of a squared matrix X
 * @param {number[]} X — squared matrix
 * @returns inverse of X
 */
export function inv(X) {

   if (!issquaredmat(X)) {
      throw Error("Only squared matrices can be inverted.");
   }

   if (isuppertrianmat(X)) {
      return transpose(inv(transpose(X)));
   }

   if (islowertrianmat(X)) {
      const n = ncol(X);
      const I = eye(n);
      let Y = zeros(n, n);

      for (let k = 1; k <= n; k++) {
         for (let i = 1; i <= n; i++) {
            Y[i - 1][k - 1] = (
               I[i -  1][k - 1] - mdot(
                     msubset(X, k, seq(1, k - 1)),
                     msubset(Y, seq(1, k - 1), i)
                  )
               ) / X[k - 1][k - 1];
         }
      }
      return Y;
   }

   // invert matrix using QR transformation
   const r = qr(X);
   if (!isuppertrianmat(r.R)) {
      throw Error("QR decomposition of the matrix returned incorrect result.")
   }

   return mdot(inv(r.R), transpose(r.Q));
}


