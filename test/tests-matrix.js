/******************************************************************
 *  Tests for matrix functions                                    *
 ******************************************************************/

// import of functions to test
import {transpose, nrow, ncol, zeros, mdot, ismatrix, mmult, madd, mdiv, diag, eye} from '../matrix/index.js';
import {tomatrix, crossprod, tcrossprod} from '../matrix/index.js';
import {seq} from '../stat/index.js';

// import dependencies
import {default as chai} from 'chai';

const should = chai.should();
const expect = chai.expect;

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



});
