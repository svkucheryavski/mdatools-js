/******************************************************************
 *  Tests for matrix functions                                    *
 ******************************************************************/

// import of functions to test
import {transpose, nrow, ncol, zeros, mdot, ismatrix, mmult, madd, mdiv, diag, eye, rbind, cbind} from '../matrix/index.js';
import {crossprod, tcrossprod, msubset, msubtract, mreplace} from '../matrix/index.js';
import {issquaredmat, islowertrianmat, isuppertrianmat} from '../matrix/index.js';
import {seq} from '../stat/index.js';

// import dependencies
import {default as chai} from 'chai';

const should = chai.should();
const expect = chai.expect;

/* Tests for operations with matrices */
describe('Tests for manipulations with matrices.', function () {
   it('mreplace() works correctly.', function () {
      const X1 = eye(5);
      const X2 = madd(zeros(3, 5), 1);
      const X3 = madd(zeros(5, 3), 1);

      const Y = [[1, 2, 3], [11, 12, 13], [21, 22, 23]];
      const E = [[1, 2], [3, 4]];

      expect(() => mreplace(X1, E, seq(2, 4), seq(2, 4))).
         to.throw(Error, "Number of values in 'rowInd' should match the number of rows in 'Y'.");
      expect(() => mreplace(X1, E, seq(2, 3), seq(2, 4))).
         to.throw(Error, "Number of values in 'colInd' should match the number of columns in 'Y'.");
      expect(() => mreplace(X1, Y, seq(2, 6), seq(2, 4))).
         to.throw(Error, "Wrong values for indices.");
      expect(() => mreplace(X1, Y, seq(2, 4), seq(2, 6))).
         to.throw(Error, "Wrong values for indices.");

      const Z1 = mreplace(X1, Y, seq(2, 4), seq(2, 4));
      expect(Z1).to.eql([[1, 0, 0, 0, 0], [0, 1, 2, 3, 0], [0, 11, 12, 13, 0], [0, 21, 22, 23, 0], [0, 0, 0, 0, 1]]);

      const Z2 = mreplace(X2, Y, [], seq(2, 4));
      expect(Z2).to.eql([[1, 1, 1], [1, 2, 3], [11, 12, 13], [21, 22, 23], [1, 1, 1]]);

      const Z3 = mreplace(X3, Y, seq(2, 4), []);
      expect(Z3).to.eql([[1, 1, 2, 3, 1], [1, 11, 12, 13, 1], [1, 21, 22, 23, 1]]);
   });

   it('msubset() works correctly.', function () {
      const X = [[1, 2, 3, 4, 5], [11, 12, 13, 14, 15], [21, 22, 23, 24, 25]];


      expect(() => msubset(X, [0, 1, 2] , [1])).to.throw(Error, "Wrong values for indices.");
      expect(() => msubset(X, [1, 1, 20], [1])).to.throw(Error, "Wrong values for indices.");
      expect(() => msubset(X, [1], [0, 1, 2] )).to.throw(Error, "Wrong values for indices.");
      expect(() => msubset(X, [1], [1, 1, 20])).to.throw(Error, "Wrong values for indices.");

      // colInd is empty
      const X11a = msubset(X, [1, 2, 3], []);
      expect(X11a).to.eql([[1, 2, 3], [11, 12, 13], [21, 22, 23]]);
      const X11b = msubset(X, [1, 2, 3], [], "select");
      expect(X11b).to.eql([[1, 2, 3], [11, 12, 13], [21, 22, 23]]);
      const X11c = msubset(X, [1, 2, 3], [], "remove");
      expect(X11c).to.eql([[4, 5], [14, 15], [24, 25]]);

      // colInd is single number
      const X12a = msubset(X, [1, 2, 3], 1);
      expect(X12a).to.eql([[1, 2, 3]]);
      const X12b = msubset(X, [1, 2, 3], [1]);
      expect(X12b).to.eql([[1, 2, 3]]);
      const X12c = msubset(X, [1, 2, 3], [1], "select");
      expect(X12c).to.eql([[1, 2, 3]]);
      const X12d = msubset(X, [1, 2, 3], [1], "remove");
      expect(X12d).to.eql([[14, 15], [24, 25]]);

      // both rowInd and colInd have several numbers
      const X13a = msubset(X, [1, 2, 4], [1, 3]);
      expect(X13a).to.eql([[1, 2, 4], [21, 22, 24]]);
      const X13b = msubset(X, [1, 2, 4], [1, 3], "select");
      expect(X13b).to.eql([[1, 2, 4], [21, 22, 24]]);
      const X13c = msubset(X, [1, 2, 4], [1, 3], "remove");
      expect(X13c).to.eql([[13, 15]]);

      // rowInd is empty
      const X21a = msubset(X, [], [1, 3]);
      expect(X21a).to.eql([[1, 2, 3, 4, 5], [21, 22, 23, 24, 25]]);
      const X21b = msubset(X, [], [1, 3], "select");
      expect(X21b).to.eql([[1, 2, 3, 4, 5], [21, 22, 23, 24, 25]]);
      const X21c = msubset(X, [], [1, 3], "remove");
      expect(X21c).to.eql([[11, 12, 13, 14, 15]]);

      // rowInd is single number
      const X22a = msubset(X, [1], [1, 3]);
      expect(X22a).to.eql([[1], [21]]);
      const X22b = msubset(X, 1, [1, 3]);
      expect(X22b).to.eql([[1], [21]]);
      const X22c = msubset(X, 1, [1, 3], "select");
      expect(X22c).to.eql([[1], [21]]);
      const X22d = msubset(X, 1, [1, 3], "remove");
      expect(X22d).to.eql([[12, 13, 14, 15]]);

      // both rowInd and colInd have several numbers
      const X23a = msubset(X, [1, 3], [1, 2]);
      expect(X23a).to.eql([[1, 3], [11, 13]]);
      const X23b = msubset(X, [1, 3], [1, 2], "select");
      expect(X23b).to.eql([[1, 3], [11, 13]]);
      const X23c = msubset(X, [1, 3], [1, 2], "remove");
      expect(X23c).to.eql([[22, 24, 25]]);

   });


   it('cbind() works correctly.', function () {
      const X = [[1, 2, 3, 4], [5, 6, 7, 8]];
      const x = [1, 2, 3, 4];
      const Y = [[11, 12, 13, 14], [15, 16, 17, 18], [19, 20, 21, 22]];
      const y = [11, 12, 13, 14];
      const E = [[1, 2, 3], [4, 5, 6]];
      const e = [1, 2, 3];

      expect(() => cbind(X, 2)).to.throw(Error, "Both 'X' and 'Y' must arrays (matrices or vectors).");
      expect(() => cbind(x, 2)).to.throw(Error, "Both 'X' and 'Y' must arrays (matrices or vectors).");
      expect(() => cbind(2, X)).to.throw(Error, "Both 'X' and 'Y' must arrays (matrices or vectors).");
      expect(() => cbind(2, x)).to.throw(Error, "Both 'X' and 'Y' must arrays (matrices or vectors).");

      expect(() => cbind(X, E)).to.throw(Error,  "Number of rows (or vector elements) in X and Y must be be the same.");
      expect(() => cbind(x, E)).to.throw(Error,  "Number of rows (or vector elements) in X and Y must be be the same.");
      expect(() => cbind(X, e)).to.throw(Error,  "Number of rows (or vector elements) in X and Y must be be the same.");
      expect(() => cbind(x, e)).to.throw(Error,  "Number of rows (or vector elements) in X and Y must be be the same.");

      // matrix and matrix
      const XY = cbind(X, Y);
      expect(nrow(XY)).to.equal(nrow(X));
      expect(ncol(XY)).to.equal(ncol(X) + ncol(Y));
      expect(XY).to.eql([[1, 2, 3, 4], [5, 6, 7, 8], [11, 12, 13, 14], [15, 16, 17, 18], [19, 20, 21, 22]]);

      // vector and matrix
      const xY = cbind(x, Y);
      expect(nrow(xY)).to.equal(nrow(x));
      expect(ncol(xY)).to.equal(ncol(x) + ncol(Y));
      expect(xY).to.eql([[1, 2, 3, 4], [11, 12, 13, 14], [15, 16, 17, 18], [19, 20, 21, 22]]);

      // matrix and vector
      const Xy = cbind(X, y);
      expect(nrow(Xy)).to.equal(nrow(X));
      expect(ncol(Xy)).to.equal(ncol(X) + ncol(y));
      expect(Xy).to.eql([[1, 2, 3, 4], [5, 6, 7, 8], [11, 12, 13, 14]]);

      // vector and vector
      const xy = cbind(x, y);
      expect(nrow(xy)).to.equal(nrow(x));
      expect(ncol(xy)).to.equal(ncol(x) + ncol(y));
      expect(xy).to.eql([[1, 2, 3, 4], [11, 12, 13, 14]]);
   });

   it('rbind() works correctly.', function () {
      const X = [[1, 2, 3, 4], [5, 6, 7, 8]];
      const x = [1, 2];
      const Y = [[11, 12], [15, 16]];
      const y = [11, 12];
      const E = [[1, 2, 3], [4, 5, 6], [1, 2, 3]];
      const e = [1, 2, 3];

      expect(() => rbind(X, 2)).to.throw(Error, "Both 'X' and 'Y' must arrays (matrices or vectors).");
      expect(() => rbind(x, 2)).to.throw(Error, "Both 'X' and 'Y' must arrays (matrices or vectors).");
      expect(() => rbind(2, X)).to.throw(Error, "Both 'X' and 'Y' must arrays (matrices or vectors).");
      expect(() => rbind(2, x)).to.throw(Error, "Both 'X' and 'Y' must arrays (matrices or vectors).");

      expect(() => rbind(X, E)).to.throw(Error,  "Number of columns (or vector elements) in X and Y must be be the same.");
      expect(() => rbind(x, E)).to.throw(Error,  "Number of columns (or vector elements) in X and Y must be be the same.");
      expect(() => rbind(X, e)).to.throw(Error,  "Number of columns (or vector elements) in X and Y must be be the same.");
      expect(() => rbind(x, e)).to.throw(Error,  "Number of columns (or vector elements) in X and Y must be be the same.");

      // matrix and matrix
      const XY = rbind(X, Y);
      expect(nrow(XY)).to.equal(nrow(X) + ncol(Y));
      expect(ncol(XY)).to.equal(ncol(X));
      expect(XY).to.eql([[1, 2, 3, 4, 11, 12], [5, 6, 7, 8, 15, 16]]);

      // vector and matrix
      const xY = rbind(x, Y);
      expect(nrow(xY)).to.equal(1 + nrow(Y)); // because x will be transposed to row vector
      expect(ncol(xY)).to.equal(nrow(x));
      expect(xY).to.eql([[1, 11, 12], [2, 15, 16]]);

      // matrix and vector
      const Xy = rbind(X, y);
      expect(nrow(Xy)).to.equal(nrow(X) + 1);
      expect(ncol(Xy)).to.equal(ncol(X));
      expect(Xy).to.eql([[1, 2, 3, 4, 11], [5, 6, 7, 8, 12]]);

      // vector and vector
      const xy = rbind(x, y);
      expect(nrow(xy)).to.equal(2);
      expect(ncol(xy)).to.equal(x.length);
      expect(xy).to.eql([[1, 11], [2, 12]]);
   });
});


/* Tests for operations with matrices */
describe('Tests for operations with matrices.', function () {

   it('tcrossprod() works correctly.', function () {
      const X = [[1, 2, 3], [6, 7, 8]];
      const Y = [[9, 8, 7], [5, 4, 3]];
      const E = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]];

      expect(() => tcrossprod(1, X)).to.throw(Error, "Both arguments must be matrices (2D Arrays).");
      expect(() => tcrossprod([1, 1], X)).to.throw(Error, "Both arguments must be matrices (2D Arrays).");
      expect(() => tcrossprod(X, 1)).to.throw(Error, "Both arguments must be matrices (2D Arrays).");
      expect(() => tcrossprod(X, [1, 1])).to.throw(Error, "Both arguments must be matrices (2D Arrays).");
      expect(() => tcrossprod(X, E)).to.throw(Error, "Dimensions of 'X' and 'Y' mismatch.");

      const XYt = tcrossprod(X, Y);

      expect(nrow(XYt)).to.equal(3);
      expect(ncol(XYt)).to.equal(3);
      expect(XYt[0][0]).to.equal(1 * 9 + 6 * 5);
      expect(XYt[1][0]).to.equal(1 * 8 + 6 * 4);
      expect(XYt[2][0]).to.equal(1 * 7 + 6 * 3);
      expect(XYt[0][1]).to.equal(2 * 9 + 7 * 5);
      expect(XYt[1][1]).to.equal(2 * 8 + 7 * 4);
      expect(XYt[2][1]).to.equal(2 * 7 + 7 * 3);
      expect(XYt[0][2]).to.equal(3 * 9 + 8 * 5);
      expect(XYt[1][2]).to.equal(3 * 8 + 8 * 4);
      expect(XYt[2][2]).to.equal(3 * 7 + 8 * 3);
   });

   it('crossprod() works correctly.', function () {
      const X = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]];
      const Y = [[9, 8, 7, 6, 6], [5, 4, 3, 2, 1]];
      const E = [[1, 2, 3], [1, 2, 3]];

      expect(() => crossprod(1, X)).to.throw(Error, "Both arguments must be matrices (2D Arrays).");
      expect(() => crossprod([1, 1], X)).to.throw(Error, "Both arguments must be matrices (2D Arrays).");
      expect(() => crossprod(X, 1)).to.throw(Error, "Both arguments must be matrices (2D Arrays).");
      expect(() => crossprod(X, [1, 1])).to.throw(Error, "Both arguments must be matrices (2D Arrays).");
      expect(() => crossprod(X, E)).to.throw(Error, "Dimensions of 'X' and 'Y' mismatch.");

      const XtY = crossprod(X, Y);

      expect(nrow(XtY)).to.equal(2);
      expect(ncol(XtY)).to.equal(2);
      expect(XtY[0][0]).to.equal(1 * 9 + 2 * 8 + 3 * 7 + 4 * 6 + 5 * 6);
      expect(XtY[1][0]).to.equal(1 * 5 + 2 * 4 + 3 * 3 + 4 * 2 + 5 * 1);
      expect(XtY[0][1]).to.equal(6 * 9 + 7 * 8 + 8 * 7 + 9 * 6 + 10 * 6);
      expect(XtY[1][1]).to.equal(6 * 5 + 7 * 4 + 8 * 3 + 9 * 2 + 10 * 1);
   });

   it('mdiv() works correctly.', function () {

      const X = [[1, 2, 3], [4, 5, 6]];
      const Y = [[3, 2, 1], [6, 5, 4]];
      const E = [[1, 2, 3, 4], [5, 6, 7, 8]];

      const z1 = 5;
      const z2 = [10, 20, 30];
      const z3 = [100, 200];
      const e = [1, 2, 3, 4];

      // matrix and matrix
      expect(() => mdiv(1, X)).to.throw(Error, "Argument 'X' must be a matrix (2D Array).");
      expect(() => mdiv(X, E)).to.throw(Error, "Dimensions of 'X' and 'Y' mismatch.");

      const resXY = mdiv(X, Y);
      const resYX = mdiv(Y, X);
      expect(nrow(resYX)).to.equal(nrow(X));
      expect(ncol(resYX)).to.equal(ncol(X));
      expect(resXY).to.eql([[1/3, 2/2, 3/1], [4/6, 5/5, 6/4]]);
      expect(resYX).to.eql([[3/1, 2/2, 1/3], [6/4, 5/5, 4/6]]);

      // matrix and scalar
      const resXz1 = mdiv(X, z1);
      expect(nrow(resXz1)).to.equal(nrow(X));
      expect(ncol(resXz1)).to.equal(ncol(X));
      expect(resXz1).to.eql([[1/5, 2/5, 3/5], [4/5, 5/5, 6/5]]);

      // matrix and vector with the same number of elements as number of rows
      expect(() => mdiv(X, e)).to.throw(Error, "Number of elements in 'x' does not match neither 'nrows' nor 'ncols'.");

      const resXz2 = mdiv(X, z2);
      expect(nrow(resXz2)).to.equal(nrow(X));
      expect(ncol(resXz2)).to.equal(ncol(X));
      expect(resXz2).to.eql([[1/10, 2/20, 3/30], [4/10, 5/20, 6/30]]);

      // matrix and vector with the same number of elements as number of columns
      const resXz3 = mdiv(X, z3);
      expect(nrow(resXz3)).to.equal(nrow(X));
      expect(ncol(resXz3)).to.equal(ncol(X));
      expect(resXz3).to.eql([[1/100, 2/100, 3/100], [4/200, 5/200, 6/200]]);

   });


   it('madd() works correctly.', function () {

      const X = [[1, 2, 3], [4, 5, 6]];
      const Y = [[3, 2, 1], [6, 5, 4]];
      const E = [[1, 2, 3, 4], [5, 6, 7, 8]];

      const z1 = 5;
      const z2 = [10, 20, 30];
      const z3 = [100, 200];
      const e = [1, 2, 3, 4];

      // matrix times matrix
      expect(() => madd(1, X)).to.throw(Error, "Argument 'X' must be a matrix (2D Array).");
      expect(() => madd(X, E)).to.throw(Error, "Dimensions of 'X' and 'Y' mismatch.");

      const resXY = madd(X, Y);
      const resYX = madd(Y, X);
      expect(nrow(resYX)).to.equal(nrow(X));
      expect(ncol(resYX)).to.equal(ncol(X));
      expect(resYX).to.eql([[4, 4, 4], [10, 10, 10]]);
      expect(resYX).to.eql(resXY);

      // matrix times scalar
      const resXz1 = madd(X, z1);
      expect(nrow(resXz1)).to.equal(nrow(X));
      expect(ncol(resXz1)).to.equal(ncol(X));
      expect(resXz1).to.eql([[6, 7, 8], [9, 10, 11]]);

      // matrix times vector with the same number of elements as number of rows
      expect(() => madd(X, e)).to.throw(Error, "Number of elements in 'x' does not match neither 'nrows' nor 'ncols'.");

      const resXz2 = madd(X, z2);
      expect(nrow(resXz2)).to.equal(nrow(X));
      expect(ncol(resXz2)).to.equal(ncol(X));
      expect(resXz2).to.eql([[11, 22, 33], [14, 25, 36]]);

      // matrix times vector with the same number of elements as number of columns
      const resXz3 = madd(X, z3);
      expect(nrow(resXz3)).to.equal(nrow(X));
      expect(ncol(resXz3)).to.equal(ncol(X));
      expect(resXz3).to.eql([[101, 102, 103], [204, 205, 206]]);

   });

   it('msubtract() works correctly.', function () {

      const X = [[1, 2, 3], [4, 5, 6]];
      const Y = [[3, 2, 1], [6, 5, 4]];
      const E = [[1, 2, 3, 4], [5, 6, 7, 8]];

      const z1 = 5;
      const z2 = [10, 20, 30];
      const z3 = [100, 200];
      const e = [1, 2, 3, 4];

      // matrix and matrix
      expect(() => msubtract(1, X)).to.throw(Error, "Argument 'X' must be a matrix (2D Array).");
      expect(() => msubtract(X, E)).to.throw(Error, "Dimensions of 'X' and 'Y' mismatch.");

      const resXY = msubtract(X, Y);
      expect(nrow(resXY)).to.equal(nrow(X));
      expect(ncol(resXY)).to.equal(ncol(X));
      expect(resXY).to.eql([[-2, 0, 2], [-2, 0, 2]]);

      const resYX = msubtract(Y, X);
      expect(nrow(resYX)).to.equal(nrow(X));
      expect(ncol(resYX)).to.equal(ncol(X));
      expect(resYX).to.eql([[2, 0, -2], [2, 0, -2]]);

      // matrix and scalar
      const resXz1 = msubtract(X, z1);
      expect(nrow(resXz1)).to.equal(nrow(X));
      expect(ncol(resXz1)).to.equal(ncol(X));
      expect(resXz1).to.eql([[-4, -3, -2], [-1, 0, 1]]);

      // matrix and vector with the same number of elements as number of rows
      expect(() => msubtract(X, e)).to.throw(Error, "Number of elements in 'x' does not match neither 'nrows' nor 'ncols'.");

      const resXz2 = msubtract(X, z2);
      expect(nrow(resXz2)).to.equal(nrow(X));
      expect(ncol(resXz2)).to.equal(ncol(X));
      expect(resXz2).to.eql([[-9, -18, -27], [-6, -15, -24]]);

      // matrix and vector with the same number of elements as number of columns
      const resXz3 = msubtract(X, z3);
      expect(nrow(resXz3)).to.equal(nrow(X));
      expect(ncol(resXz3)).to.equal(ncol(X));
      expect(resXz3).to.eql([[-99, -98, -97], [-196, -195, -194]]);

   });


   it('mmult() works correctly.', function () {

      const X = [[1, 2, 3], [4, 5, 6]];
      const Y = [[3, 2, 1], [6, 5, 4]];
      const E = [[1, 2, 3, 4], [5, 6, 7, 8]];

      const z1 = 5;
      const z2 = [10, 20, 30];
      const z3 = [100, 200];
      const e = [1, 2, 3, 4];

      // matrix times matrix
      expect(() => mmult(1, X)).to.throw(Error, "Argument 'X' must be a matrix (2D Array).");
      expect(() => mmult(X, E)).to.throw(Error, "Dimensions of 'X' and 'Y' mismatch.");

      const resXY = mmult(X, Y);
      const resYX = mmult(Y, X);
      expect(nrow(resYX)).to.equal(nrow(X));
      expect(ncol(resYX)).to.equal(ncol(X));
      expect(resYX).to.eql([[3, 4, 3], [24, 25, 24]]);
      expect(resYX).to.eql(resXY);

      // matrix times scalar
      const resXz1 = mmult(X, z1);
      expect(nrow(resXz1)).to.equal(nrow(X));
      expect(ncol(resXz1)).to.equal(ncol(X));
      expect(resXz1).to.eql([[5, 10, 15], [20, 25, 30]]);

      // matrix times vector with the same number of elements as number of rows
      expect(() => mmult(X, e)).to.throw(Error, "Number of elements in 'x' does not match neither 'nrows' nor 'ncols'.");

      const resXz2 = mmult(X, z2);
      expect(nrow(resXz2)).to.equal(nrow(X));
      expect(ncol(resXz2)).to.equal(ncol(X));
      expect(resXz2).to.eql([[10, 40, 90], [40, 100, 180]]);

      // matrix times vector with the same number of elements as number of columns
      const resXz3 = mmult(X, z3);
      expect(nrow(resXz3)).to.equal(nrow(X));
      expect(ncol(resXz3)).to.equal(ncol(X));
      expect(resXz3).to.eql([[100, 200, 300], [800, 1000, 1200]]);

   });

   it('mdot() works correctly.', function () {
      const X = [[1, 2, 3, 4], [1, 2, 3, 4]];
      const Y = [[1, 2], [3, 4], [5, 6], [7, 8]];
      const z = [1, 10];

      expect(() => mdot(1, X)).to.throw(Error, "Argument 'X' must be a vector or a matrix (1D or 2D Array).");
      expect(() => mdot(X, 1)).to.throw(Error, "Argument 'Y' must be a vector or a matrix (1D or 2D Array).");
      expect(() => mdot(X, [1])).to.throw(Error, "Dimensions of 'X' and 'Y' mismatch.");

      const resYX = mdot(Y, X);
      expect(nrow(resYX)).to.equal(2)
      expect(ncol(resYX)).to.equal(2)
      expect(resYX).to.eql([[50, 60], [50, 60]]);

      const resXY = mdot(X, Y);
      expect(nrow(resXY)).to.equal(4)
      expect(ncol(resXY)).to.equal(4)
      expect(resXY).to.eql([[3, 6, 9, 12], [7, 14, 21, 28], [11, 22, 33, 44], [15, 30, 45, 60]]);

      const resXz = mdot(X, z);
      expect(nrow(resXz)).to.equal(4)
      expect(ncol(resXz)).to.equal(1)
      expect(resXz).to.eql([[11, 22, 33, 44]]);

   });

   it('transpose() works correctly with 2D arrays.', function () {
      expect(() => transpose(1)).to.throw(Error, "Argument 'X' must be a vector or a matrix (1D or 2D Array).");

      const X = [[1, 3, 5, 7], [2, 4, 6, 8]];
      const Xt = transpose(X);

      expect(Xt.length).to.equal(4);
      expect(Xt[0].length).to.equal(2);
      expect(Xt[0]).to.eql([1, 2]);
      expect(Xt[1]).to.eql([3, 4]);
      expect(Xt[2]).to.eql([5, 6]);
      expect(Xt[3]).to.eql([7, 8]);
   });

   it('transpose() works correctly with 1D arrays.', function () {
      const X = [1, 3, 5, 7];
      const Xt = transpose(X);

      expect(nrow(X)).to.equal(4);
      expect(ncol(X)).to.equal(1);

      expect(nrow(Xt)).to.equal(1);
      expect(ncol(Xt)).to.equal(4);

      expect(Xt[0]).to.eql([1]);
      expect(Xt[1]).to.eql([3]);
      expect(Xt[2]).to.eql([5]);
      expect(Xt[3]).to.eql([7]);

   });

});


/* Tests for generation of matrices and checking its properties */
describe('Tests for generation of matrices and checking its properties.', function () {

   it('eye() works correctly.', function() {
      expect(eye(3)).to.eql([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
   });

   it('diag() works correctly.', function() {
      expect(() => diag(1)).to.throw(Error, "Argument 'x' must be a vector.");
      expect(() => diag([[1, 2], [1, 2]])).to.throw(Error, "Argument 'x' must be a vector.");
      expect(diag([1, 2, 3])).to.eql([[1, 0, 0], [0, 2, 0], [0, 0, 3]]);
   });

   it('zeros() works correctly.', function () {
      const X1 = zeros(5, 50);
      expect(nrow(X1)).to.equal(5);
      expect(ncol(X1)).to.equal(50);
      expect(X1[2][2]).to.equal(0);

      const X2 = zeros(50, 5);
      expect(nrow(X2)).to.equal(50);
      expect(ncol(X2)).to.equal(5);
      expect(X2[2][2]).to.equal(0);

      const X3 = zeros(50, 1);
      expect(nrow(X3)).to.equal(50);
      expect(ncol(X3)).to.equal(1);
      expect(X3[0][2]).to.equal(0);

      const X4 = zeros(1, 50)
      expect(nrow(X4)).to.equal(1);
      expect(ncol(X4)).to.equal(50);
      expect(X4[2][0]).to.equal(0);
   });

   it('nrow() and ncol() work correctly.', function () {
      expect(() => nrow(1)).to.throw(Error, "Argument 'X' must be a vector or a matrix (1D or 2D Array).");
      expect(() => ncol(1)).to.throw(Error, "Argument 'X' must be a vector or a matrix (1D or 2D Array).");

      const X = [[1, 3, 5, 7], [2, 4, 6, 8]];

      expect(nrow(X)).to.equal(4);
      expect(ncol(X)).to.equal(2);

      expect(nrow(transpose(X))).to.equal(2);
      expect(ncol(transpose(X))).to.equal(4);

      expect(nrow([1, 2, 3, 4])).to.equal(4)
      expect(ncol([1, 2, 3, 4])).to.equal(1)
   });

   it('ismatrix() works correctly.', function () {
      expect(ismatrix(1)).to.equal(false)
      expect(ismatrix([])).to.equal(false)
      expect(ismatrix([[], []])).to.equal(false)
      expect(ismatrix([[1, 2], [1]])).to.equal(false)

      expect(ismatrix([[1, 2]])).to.equal(true)
      expect(ismatrix([[1]])).to.equal(true)
      expect(ismatrix([[1, 2, 3], [1, 2, 3], [4, 5, 6]])).to.equal(true)
   });


   it('issquared() works correctly.', function () {
      const E1 = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]];
      const E2 = [1, 2, 3, 4];
      const X1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
      const X2 = [1];

      expect(issquaredmat(E1)).to.equal(false);
      expect(issquaredmat(transpose(E1))).to.equal(false);
      expect(issquaredmat(E2)).to.equal(false);
      expect(issquaredmat(transpose(E2))).to.equal(false);


      expect(issquaredmat(X1)).to.equal(true);
      expect(issquaredmat(transpose(X1))).to.equal(true);
      expect(issquaredmat(X2)).to.equal(true);
      expect(issquaredmat(transpose(X2))).to.equal(true);

   });


   it('islowertrianmat() works correctly.', function () {
      const E1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
      const E2 = [[1, 0, 0], [4, 5, 0], [7, 8, 9]];
      const X1 = transpose(E2);

      expect(islowertrianmat(E1)).to.equal(false);
      expect(islowertrianmat(transpose(E1))).to.equal(false);
      expect(islowertrianmat(E2)).to.equal(false);
      expect(islowertrianmat(X1)).to.equal(true);
   });

   it('isuppertrianmat() works correctly.', function () {
      const X1 = [[1, 0, 0], [4, 5, 0], [7, 8, 9]];
      const E1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
      const E2 = transpose(X1);

      expect(isuppertrianmat(E1)).to.equal(false);
      expect(isuppertrianmat(transpose(E1))).to.equal(false);
      expect(isuppertrianmat(E2)).to.equal(false);
      expect(isuppertrianmat(X1)).to.equal(true);
   });

});
