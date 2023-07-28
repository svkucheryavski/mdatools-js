/****************************************************************/
/*  Tests for array methods (Index/Vector/Matrix classes)       */
/****************************************************************/

import {default as chai} from 'chai';
import {default as chaiAlmost} from 'chai-almost';

// import classes and related methods
import { Factor, isfactor, factor, Dataset, index, isindex, Index, ismatrix, matrix,
   Matrix, isvector, vector, Vector } from '../src/arrays/index.js';

// import non-class methods
import { tcrossprod, crossprod, rbind, cbind, c, reshape } from '../src/arrays/index.js';

// set up test settings
const expect = chai.expect;
chai.use(chaiAlmost(0.00001));


// function to test a matrix structure
function testMatrixStructure(X, nr, nc, values) {
   expect(X.constructor).equal(Matrix);
   expect(X.v.constructor).equal(Float64Array);
   expect(X.v.length).equal(nr * nc);
   expect(X.nrows).equal(nr);
   expect(X.ncols).equal(nc);
   expect(ismatrix(X)).to.be.true;
   expect(isvector(X)).to.be.false;

   if (values) {
      expect(X.v).to.deep.equal(new Float64Array(values));
   }
}

// function to test a vector structure
function testVectorStructure(x, length, values) {
   expect(x.constructor).equal(Vector);
   expect(x.v.constructor).equal(Vector.valuesConstructor);
   expect(x.v.length).equal(length);
   expect(ismatrix(x)).to.be.false;
   expect(isvector(x)).to.be.true;

   if (values) {
      expect(x.v).to.deep.almost.equal(new Vector.valuesConstructor(values));
   }
}

// function to test an index structure
function testIndexStructure(x, length, values) {
   expect(x.constructor).equal(Index);
   expect(x.v.constructor).equal(Index.valuesConstructor);
   expect(x.v.length).equal(length);
   expect(ismatrix(x)).to.be.false;
   expect(isvector(x)).to.be.false;
   expect(isindex(x)).to.be.true;

   if (values) {
      expect(x.v).to.deep.equal(new Index.valuesConstructor(values));
   }
}

// function to test an factor structure
function testFactorStructure(x, length, values, labels) {
   expect(x.constructor).equal(Factor);
   expect(x.v.constructor).equal(Factor.valuesConstructor);
   expect(x.v.length).equal(length);
   expect(ismatrix(x)).to.be.false;
   expect(isvector(x)).to.be.false;
   expect(isindex(x)).to.be.false;
   expect(isfactor(x)).to.be.true;

   if (labels) {
      expect(x.labels).to.deep.equal(labels);
      expect(x.nlevels).equal(labels.length);
   }

   if (values) {
      expect(x.v).to.deep.equal(new Factor.valuesConstructor(values));
   }
}

describe('Tests of non-class methods for matrices and vectors', function() {

   it('tests for method "tcrossprod"', function() {

      // both matrices are squared
      const X1 = reshape(Vector.seq(1, 16), 4, 4);
      const Y1 = reshape(Vector.seq(16, 1), 4, 4);

      // a. full indices
      const Z1a = tcrossprod(X1, Y1);
      testMatrixStructure(Z1a, 4, 4, [
         200, 240, 280, 320,
         172, 208, 244, 280,
         144, 176, 208, 240,
         116, 144, 172, 200
      ]);

      // matrices have different size
      const X2 = reshape(Vector.seq(1, 15), 3, 5);
      const Y2 = reshape(Vector.seq(15, 1), 3, 5);

      // a. full indices
      const Z2a1 = tcrossprod(X2, Y2);
      testMatrixStructure(Z2a1, 3, 3, [
         225, 270, 315,
         190, 230, 270,
         155, 190, 225
      ]);

      const X3 = reshape(Vector.seq(1, 15), 5, 3);
      const Y3 = reshape(Vector.seq(12, 1), 4, 3);

      // a. full indices
      const Z3a1 = tcrossprod(X3, Y3);
      testMatrixStructure(Z3a1, 5, 4, [
         104, 128, 152, 176, 200,
         86, 107, 128, 149, 170,
         68, 86, 104, 122, 140,
         50, 65, 80, 95, 110
      ]);

   });

   it('tests for method "crossprod"', function() {

      // both matrices are squared
      const X1 = reshape(Vector.seq(1, 16), 4, 4);
      const Y1 = reshape(Vector.seq(16, 1), 4, 4);

      // a. full indices
      const Z1a = crossprod(X1, Y1);
      testMatrixStructure(Z1a, 4, 4, [
         140, 372, 604, 836,
         100, 268, 436, 604,
         60, 164, 268, 372,
         20, 60, 100, 140
      ]);

      // matrices have different size
      const X2 = reshape(Vector.seq(1, 15), 3, 5);
      const Y2 = reshape(Vector.seq(15, 1), 3, 5);

      // a. full indices
      const Z2a1 = crossprod(X2, Y2);
      testMatrixStructure(Z2a1, 5, 5, [
            82, 208, 334, 460, 586,
            64, 163, 262, 361, 460,
            46, 118, 190, 262, 334,
            28, 73, 118, 163, 208,
            10, 28, 46, 64, 82
      ]);

      const X3 = reshape(Vector.seq(1, 15), 5, 3);
      const Y3 = reshape(Vector.seq(20, 1), 5, 4);

      // a. full indices
      const Z3a1 = crossprod(X3, Y3);
      testMatrixStructure(Z3a1, 3, 4, [
         260, 710, 1160,
         185, 510, 835,
         110, 310, 510,
         35, 110, 185
      ]);

   });

   it ('tests for method "rbind"', function() {
      const E1 = matrix([1, 2, 3, 4], 2, 2);
      const X1 = matrix([1, 2, 3], 1, 3);
      const X2 = matrix([10, 20, 30, 40, 50, 60], 2, 3);
      const X3 = matrix([11, 21, 31, 41, 51, 61, 71, 81, 91], 3, 3);

      expect(() => cbind(E1, X1).to.throw('rbind: all matrices must the same number of columns.'));
      expect(() => cbind(X1, E1).to.throw('rbind: all matrices must the same number of columns.'));

      const R1 = rbind(X1, X2);
      testMatrixStructure(R1, 3, 3, [1, 10, 20, 2, 30, 40, 3, 50, 60]);

      const R2 = rbind(X1, X2, X3);
      testMatrixStructure(R2, 6, 3, [
         1, 10, 20, 11, 21, 31,
         2, 30, 40, 41, 51, 61,
         3, 50, 60, 71, 81, 91]
      );

      // concatenate vectors into matrix
      const x1 = Vector.rand(100);
      const x2 = Vector.rand(100);
      const x3 = Vector.rand(100);
      const R4 = rbind(x1, x2, x3);
      testMatrixStructure(R4, 3, 100);

      // concatenate vectors and matrices into matrix
      const x4 = vector([1, 2, 3]);
      const x5 = vector([11, 12, 13]);
      const X7 = matrix([22, 23, 24, 25, 26, 27], 2, 3)
      const R5 = rbind(x4, x5, X7);
      testMatrixStructure(R5, 4, 3, [1, 11, 22, 23, 2, 12, 24, 25, 3, 13, 26, 27]);

   });

   it ('tests for method "cbind"', function() {

      const E1 = matrix([1, 2, 3, 4], 2, 2);
      const X1 = matrix([1, 2, 3], 3, 1);
      const X2 = matrix([10, 20, 30, 40, 50, 60], 3, 2);
      const X3 = matrix([11, 21, 31, 41, 51, 61, 71, 81, 91], 3, 3);

      expect(() => cbind(E1, X1).to.throw('cbind: all matrices must the same number of rows.'));
      expect(() => cbind(X1, E1).to.throw('cbind: all matrices must the same number of rows.'));

      const R1 = cbind(X1, X2);
      testMatrixStructure(R1, 3, 3, [1, 2, 3, 10, 20, 30, 40, 50, 60]);

      const R2 = cbind(X1, X2, X3);
      testMatrixStructure(R2, 3, 6, [1, 2, 3, 10, 20, 30, 40, 50, 60, 11, 21, 31, 41, 51, 61, 71, 81, 91]);

      // concatenate vectors into matrix
      const x1 = Vector.rand(100);
      const x2 = Vector.rand(100);
      const x3 = Vector.rand(100);
      const R4 = cbind(x1, x2, x3);
      testMatrixStructure(R4, 100, 3);

      // concatenate vectors and matrices into matrix
      const x4 = vector([1, 2, 3]);
      const x5 = vector([11, 12, 13]);
      const X7 = matrix([22, 23, 24, 25, 26, 27], 3, 2)
      const R5 = cbind(x4, x5, X7);
      testMatrixStructure(R5, 3, 4, [1, 2, 3, 11, 12, 13, 22, 23, 24, 25, 26, 27]);

   });

   it ('tests for method "c"', function() {

      const x1 = vector([9]);
      const x2 = vector([10, 20]);
      const x3 = vector([100, 200, 300]);

      const r1 = c(x1, x2);
      testVectorStructure(r1, 3, [9, 10, 20]);

      const r2 = c(x1, x2, x3);
      testVectorStructure(r2, 6, [9, 10, 20, 100, 200, 300]);

      const r3 = c(x1, x2, x3, x2, x1);
      testVectorStructure(r3, 9, [9, 10, 20, 100, 200, 300, 10, 20, 9]);

      const x4 = Vector.rand(10000);
      const x5 = Vector.rand(10000);
      const x6 = Vector.rand(10000);
      const r4 = c(x4, x5, x6);
      testVectorStructure(r4, 30000);

   });

});

describe('Tests of methods for class Matrix.', function () {

   it ('tests for method "inv".', function () {

      // errors
      const E = matrix([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5, 2);
      expect(() => E.inv()).to.throw(Error, 'inv: only squared matrices can be inverted.');

      // lower triangular
      const X1 = matrix([1, 0, 0, 0, 2, 3, 0, 0, 4, 5, 6, 0, 7, 8, 9, 10], 4, 4);
      expect(X1.dot(X1.inv())).to.be.deep.almost(Matrix.eye(4));

      // upper triangular
      const X2 = X1.t();
      expect(X2.dot(X2.inv())).to.be.deep.almost(Matrix.eye(4));

      // squared
      const X3 = matrix([1, 2, 7, 3, 9, 11, 12, 3, 1, 9, 0, 2, 11, 9, 1, 8], 4, 4);
      expect(X3.dot(X3.inv())).to.be.deep.almost(Matrix.eye(4));

      // diagonal
      const X4 = Matrix.diagm(vector([1, 2, 3, 4]));
      expect(X4.dot(X4.inv())).to.be.deep.almost(Matrix.eye(4));
   });

   it('tests for method "toCSV"', function() {

      // just values (sep = ",", dec = "."), no header
      const X1 = matrix([10.1, 5.5, 3.8, 1.9, 20.2, 6.61, 2.99, -1.991, 30.3, -7.22, 1.11, 1.9994], 4, 3);
      expect(X1.toCSV()).to.be.equal("10.1,20.2,30.3\n5.5,6.61,-7.22\n3.8,2.99,1.11\n1.9,-1.991,1.9994\n");

      // just values (sep = ";", dec = ","), no header
      const X2a = matrix([10.1, 5.5, 3.8, 1.9, 20.2, 6.61, 2.99, -1.991, 30.3, -7.22, 1.11, 1.9994], 4, 3);
      expect(X2a.toCSV(";", ",")).to.be.equal("10,1;20,2;30,3\n5,5;6,61;-7,22\n3,8;2,99;1,11\n1,9;-1,991;1,9994\n");

      // just values (sep = ";", dec = ","), no header + labels
      const X2b = matrix([10.1, 5.5, 3.8, 1.9, 20.2, 6.61, 2.99, -1.991, 30.3, -7.22, 1.11, 1.9994], 4, 3);
      const l2b = ["O1", "O2", "O3", "O4"];
      expect(X2b.toCSV(";", ",", null, l2b)).to.be.equal("O1;10,1;20,2;30,3\nO2;5,5;6,61;-7,22\nO3;3,8;2,99;1,11\nO4;1,9;-1,991;1,9994\n");

      // just values (sep = ",", dec = "."), with header
      const h3 = ["X1 cm", "Y2 kg", "Z3 m3"];
      const X3 = matrix([10.1, 5.5, 3.8, 1.9, 20.2, 6.61, 2.99, -1.991, 30.3, -7.22, 1.11, 1.9994], 4, 3);
      expect(X3.toCSV(",", ".", h3)).to.be.equal("X1 cm,Y2 kg,Z3 m3\n10.1,20.2,30.3\n5.5,6.61,-7.22\n3.8,2.99,1.11\n1.9,-1.991,1.9994\n");

      // just values (sep = ";", dec = ","), with header
      const h4a = ["X1 cm", "Y2 kg", "Z3 m3"];
      const X4a = matrix([10.1, 5.5, 3.8, 1.9, 20.2, 6.61, 2.99, -1.991, 30.3, -7.22, 1.11, 1.9994], 4, 3);
      expect(X4a.toCSV(";", ",", h4a)).to.be.equal("X1 cm;Y2 kg;Z3 m3\n10,1;20,2;30,3\n5,5;6,61;-7,22\n3,8;2,99;1,11\n1,9;-1,991;1,9994\n");

      // just values (sep = ";", dec = ","), with header + labels
      const h4b = ["", "X1 cm", "Y2 kg", "Z3 m3"];
      const l4b = ["O1", "O2", "O3", "O4"];
      const X4b = matrix([10.1, 5.5, 3.8, 1.9, 20.2, 6.61, 2.99, -1.991, 30.3, -7.22, 1.11, 1.9994], 4, 3);
      expect(X4b.toCSV(";", ",", h4b, l4b)).to.be.equal(";X1 cm;Y2 kg;Z3 m3\nO1;10,1;20,2;30,3\nO2;5,5;6,61;-7,22\nO3;3,8;2,99;1,11\nO4;1,9;-1,991;1,9994\n");

      // test both toCSV and parseCSV together
      const csvStr1A = "A1 cm,B2 g,C3 m3\n10.1,20.2,30.3\n5.5,6.61,-7.22\n3.8,2.99,1.11\n1.9, -1.991,1.9994\n \n\n\n";
      const csvStr1B = "A1 cm,B2 g,C3 m3\n10.1,20.2,30.3\n5.5,6.61,-7.22\n3.8,2.99,1.11\n1.9,-1.991,1.9994\n";
      const res1 = Matrix.parseCSV(csvStr1A);
      const csvStrOut1 = res1.values.toCSV(res1.sep, res1.dec, res1.header, res1.labels);
      expect(csvStrOut1).to.be.equal(csvStr1B);

      const csvStr2A = ",1,2,3\nO1,10.1,20.2,30.3\nO2,5.5,6.61,-7.22\nO3,3.8,2.99,1.11\nO4,1.9, -1.991,1.9994\n \n\n\n";
      const csvStr2B = ",1,2,3\nO1,10.1,20.2,30.3\nO2,5.5,6.61,-7.22\nO3,3.8,2.99,1.11\nO4,1.9,-1.991,1.9994\n";
      const res2 = Matrix.parseCSV(csvStr2A, ",", ".", true, true);
      const csvStrOut2 = res2.values.toCSV(res2.sep, res2.dec, res2.header, res2.labels);
      expect(csvStrOut2).to.be.equal(csvStr2B);
   });

   it ('tests for method "toString"', function() {
      const X1 = Matrix.rand(5, 4, -5, 10);
      //X1.toString(3));
   });

   it('tests for method "islowertriangular"', function () {

      const X1e = matrix([1, 2, 3, 0, 1, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 4, 4);
      const X2e = matrix([1, 0, 0, 0, 0, 1, 2, 0, 0, 0, 1, 2, 3, 0, 0], 5, 3);

      expect(X1e.islowertriangular()).to.be.false;
      expect(X2e.islowertriangular()).to.be.false;
      expect(X1e.t().islowertriangular()).to.be.false;
      expect(X2e.t().islowertriangular()).to.be.false;

      const X0 = matrix([1, 0, 2, 3], 2, 2);
      const X1 = matrix([1, 0, 0, 2, 3, 0, 3, 4, 5], 3, 3);
      const X2 = matrix([1, 0, 0, 2, 3, 0, 3, 4, 5], 3, 3);
      const X3 = Matrix.diagm(vector([1, 3, 5, 7, 9, 11, 13, 15]));
      const X4 = Matrix.eye(100);

      expect(X0.islowertriangular()).to.be.false;
      expect(X1.islowertriangular()).to.be.false;
      expect(X2.islowertriangular()).to.be.false;

      // diagonal matrices are both lower and upper triangular
      expect(X3.islowertriangular()).to.be.true;
      expect(X4.islowertriangular()).to.be.true;

      expect(X0.t().islowertriangular()).to.be.true;
      expect(X1.t().islowertriangular()).to.be.true;
      expect(X2.t().islowertriangular()).to.be.true;
      expect(X3.t().islowertriangular()).to.be.true;
      expect(X4.t().islowertriangular()).to.be.true;

   });

   it('tests for method "isuppertriangular"', function () {
      const X1e = matrix([1, 2, 3, 0, 1, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 4, 4);
      const X2e = matrix([1, 0, 0, 0, 0, 1, 2, 0, 0, 0, 1, 2, 3, 0, 0], 5, 3);

      expect(X1e.isuppertriangular()).to.be.false;
      expect(X2e.isuppertriangular()).to.be.false;

      const X0 = matrix([1, 0, 2, 3], 2, 2);
      const X1 = matrix([1, 0, 0, 2, 3, 0, 3, 4, 5], 3, 3);
      const X2 = matrix([1, 0, 0, 2, 3, 0, 3, 4, 5], 3, 3);
      const X3 = Matrix.diagm(vector([1, 3, 5, 7, 9, 11, 13, 15]));
      const X4 = Matrix.eye(100);

      expect(X0.isuppertriangular()).to.be.true;
      expect(X1.isuppertriangular()).to.be.true;
      expect(X2.isuppertriangular()).to.be.true;
      expect(X3.isuppertriangular()).to.be.true;
      expect(X4.isuppertriangular()).to.be.true;

      expect(X0.t().isuppertriangular()).to.be.false;
      expect(X1.t().isuppertriangular()).to.be.false;
      expect(X2.t().isuppertriangular()).to.be.false;

      expect(X3.t().isuppertriangular()).to.be.true;
      expect(X4.t().isuppertriangular()).to.be.true;

   });

   it('tests for method "diag"', function () {

      // squared matrix
      const X1 = matrix([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 4, 4);
      testVectorStructure(X1.diag(), 4, [1, 6, 11, 16]);

      // nrows > ncols
      const X2 = matrix([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 5, 3);
      testVectorStructure(X2.diag(), 3, [1, 7, 13]);

      // nrows < ncols
      const X3 = matrix([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 3, 5);
      testVectorStructure(X3.diag(), 3, [1, 5, 9]);

   });

   it ('tests for method "dot"', function() {

      // both matrices are squared
      const X1 = reshape(Vector.seq(1, 16), 4, 4);
      const Y1 = reshape(Vector.seq(16, 1), 4, 4);

      // a. full indices
      const Z1a = X1.dot(Y1)
      testMatrixStructure(Z1a, 4, 4, [386, 444, 502, 560, 274, 316, 358, 400, 162, 188, 214, 240, 50, 60, 70, 80]);

      // matrices have different size
      const X2 = reshape(Vector.seq(1, 15), 5, 3);
      const Y2 = reshape(Vector.seq(15, 1), 3, 5);

      // a. full indices
      const Z2a1 = X2.dot(Y2)
      testMatrixStructure(Z2a1, 5, 5, [
         242, 284, 326, 368, 410,
         188, 221, 254, 287, 320,
         134, 158, 182, 206, 230,
          80,  95, 110, 125, 140,
          26,  32,  38,  44,  50]);

      // matrix times vector
      const X6 = reshape(Vector.seq(1, 16), 4, 4);
      const y6 = Vector.seq(4, 1);
      const Z6 = X6.dot(y6)
      testMatrixStructure(Z6, 4, 1);

   });

   it ('tests for method "mult"', function() {
      const X = reshape(Vector.seq(1, 15), 5, 3);
      const Y = reshape(Vector.seq(150, 10, -10), 5, 3);

      // with a number
      const X1 = X.mult(10);
      testMatrixStructure(X1, 5, 3, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]);

      // with a vector (as long as column)
      const X2 = X.mult(vector([10, 20, 30]));
      testMatrixStructure(X2, 5, 3, [10, 20, 30, 40, 50, 120, 140, 160, 180, 200, 330, 360, 390, 420, 450]);

      // with a vector (as long as row)
      const X3 = X.mult(vector([10, 20, 30, 40, 50]));
      testMatrixStructure(X3, 5, 3, [10, 40, 90, 160, 250, 60, 140, 240, 360, 500, 110, 240, 390, 560, 750]);

      // with a another matrix
      const X4 = X.mult(Y);
      testMatrixStructure(X4, 5, 3, [150, 280, 390, 480, 550, 600, 630, 640, 630, 600, 550, 480, 390, 280, 150]);

   });

   it ('tests for method "divide"', function() {

      const X = reshape(Vector.seq(1, 15), 5, 3);
      const Y = reshape(Vector.seq(150, 10, -10), 5, 3);

      // with a number
      const X1 = X.divide(10);
      testMatrixStructure(X1, 5, 3, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]);

      // with a vector (as long as column)
      const X2 = X.divide(vector([5, 10, 100]));
      testMatrixStructure(X2, 5, 3, [0.2, 0.4, 0.6, 0.8, 1.0, 0.6, 0.7, 0.8, 0.9, 1.0, 0.11, 0.12, 0.13, 0.14, 0.15]);

      // with a vector (as long as row)
      const X3 = X.divide(vector([5, 10, 20, 50, 100]));
      testMatrixStructure(X3, 5, 3, [0.2, 0.2, 0.15, 0.08, 0.05, 1.2, 0.7, 0.4, 0.18, 0.10, 2.2, 1.2, 0.65, 0.28, 0.15]);

      // with a another matrix
      const X4 = X.divide(Y);
      testMatrixStructure(X4, 5, 3, [1/150, 2/140, 3/130, 4/120, 5/110, 0.06, 7/90, 0.1, 9/70, 10/60, 11/50, 0.3, 13/30, 0.7, 1.5]);

    });

   it ('tests for method "subtract"', function() {
      const X = reshape(Vector.seq(1, 15), 5, 3);
      const Y = reshape(Vector.seq(150, 10, -10), 5, 3);

      // with a number
      const X1 = X.subtract(10);
      testMatrixStructure(X1, 5, 3, [-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]);

      // with a vector (as long as column)
      const X2 = X.subtract(vector([10, 20, 30]));
      testMatrixStructure(X2, 5, 3, [-9, -8, -7, -6, -5, -14, -13, -12, -11, -10, -19, -18, -17, -16, -15]);

      // with a vector (as long as row)
      const X3 = X.subtract(vector([10, 20, 30, 40, 50]));
      testMatrixStructure(X3, 5, 3, [-9, -18, -27, -36, -45, -4, -13, -22, -31, -40, 1, -8, -17, -26, -35]);

      // with a another matrix
      const X4 = X.subtract(Y);
      testMatrixStructure(X4, 5, 3, [-149, -138, -127, -116, -105, -94, -83, -72, -61, -50, -39, -28, -17, -6, 5]);

   });

   it ('tests for method "add"', function() {
      const X = reshape(Vector.seq(1, 15), 5, 3);
      const Y = reshape(Vector.seq(150, 10, -10), 5, 3);

      // with a number
      const X1 = X.add(10);
      testMatrixStructure(X1, 5, 3, [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])

      // with a vector (as long as column)
      const X2 = X.add(vector([10, 20, 30]));
      testMatrixStructure(X2, 5, 3, [11, 12, 13, 14, 15, 26, 27, 28, 29, 30, 41, 42, 43, 44, 45]);

      // with a vector (as long as row)
      const X3 = X.add(vector([10, 20, 30, 40, 50]));
      testMatrixStructure(X3, 5, 3, [11, 22, 33, 44, 55, 16, 27, 38, 49, 60, 21, 32, 43, 54, 65]);

      // with a another matrix
      const X4 = X.add(Y);
      testMatrixStructure(X4, 5, 3, [151, 142, 133, 124, 115, 106, 97, 88, 79, 70, 61, 52, 43, 34, 25]);

   });

   it ('tests for method "t"', function() {
      const X = matrix([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], 6, 3);
      const Xt1 = X.t();
      testMatrixStructure(Xt1, 3, 6, [1, 7, 13, 2, 8, 14, 3, 9, 15, 4, 10, 16, 5, 11, 17, 6, 12, 18]);

      const Xt2 = Xt1.t();
      testMatrixStructure(Xt2, 6, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]);
   });

   it ('tests for method "getrow"', function() {
      const X1 = reshape(Vector.seq(1, 20), 5, 4);

      const x1 = X1.getrow(1);
      testVectorStructure(x1, 4, [1, 6, 11, 16]);

      const x3 = X1.getrow(3);
      expect(x3.length).equal(4);
      testVectorStructure(x3, 4, [3, 8, 13, 18]);

      const x5 = X1.getrow(5);
      testVectorStructure(x5, 4, [5, 10, 15, 20]);
   });

   it ('tests for method "getcolumn"', function() {
      const X1 = reshape(Vector.seq(1, 20), 5, 4);

      const x1 = X1.getcolumn(1);
      testVectorStructure(x1, 5, [1, 2, 3, 4, 5]);

      const x3 = X1.getcolumn(3);
      expect(x3.length).equal(5);
      testVectorStructure(x3, 5, [11, 12, 13, 14, 15]);

      const x4 = X1.getcolumn(4);
      testVectorStructure(x4, 5, [16, 17, 18, 19, 20]);
   });

   it ('tests for method "getcolref"', function() {
      const X = reshape(Vector.seq(1, 20), 5, 4);
      const x = X.getcolref(2);
      expect(x.length).to.be.equal(5);
      expect(x).to.be.deep.equal(X.v.subarray(5, 10));
      x[0] = -9;
      expect(X.v[5]).to.be.equal(-9);
   });

   it ('tests for method "replace"', function() {
      const X = reshape(Vector.seq(1, 20), 5, 4);

      // check for errors
      const Xe = X.copy();
      const Ye = matrix([96, 95, 94, 93, 92, 91], 3, 2);
      chai.expect(() => Xe.replace(Ye, [3, 4], [2, 3])).to.throw('Matrix.replace: dimension of "X" does not match the indices.');
      chai.expect(() => Xe.replace(Ye, [0, 3], [2, 3])).to.throw('Matrix.replace: row indices do not match the dimension of matrix.');
      chai.expect(() => Xe.replace(Ye, [1, 6], [2, 3])).to.throw('Matrix.replace: row indices do not match the dimension of matrix.');
      chai.expect(() => Xe.replace(Ye, [3, 4], [0, 3])).to.throw('Matrix.replace: column indices do not match the dimension of matrix.');
      chai.expect(() => Xe.replace(Ye, [3, 4], [2, 5])).to.throw('Matrix.replace: column indices do not match the dimension of matrix.');


      // replace different values
      const X1 = X.copy();
      const Y1 = matrix([96, 95, 94, 93, 92, 91], 3, 2);
      X1.replace(Y1, [2, 3, 4], [2, 3]);
      testMatrixStructure(X1, 5, 4, [1, 2, 3, 4, 5, 6, 96, 95, 94, 10, 11, 93, 92, 91, 15, 16, 17, 18, 19, 20]);

      const X2 = X.copy();
      const Y2 = matrix([96, 95, 94, 93, 92, 91], 3, 2);
      X2.replace(Y2, [1, 3, 5], [1, 4]);
      testMatrixStructure(X2, 5, 4, [96, 2, 95, 4, 94, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 93, 17, 92, 19, 91]);

      // replace whole column in matrix

      const Y3 = matrix([95, 94, 93, 92, 91], 5, 1);

      const X3a = X.copy();
      X3a.replace(Y3, [], 1);
      testMatrixStructure(X3a, 5, 4, [95, 94, 93, 92, 91, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]);

      const X3b = X.copy();
      X3b.replace(Y3, [], 3);
      testMatrixStructure(X3b, 5, 4, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 95, 94, 93, 92, 91, 16, 17, 18, 19, 20]);

      const X3c = X.copy();
      X3c.replace(Y3, [], 4);
      testMatrixStructure(X3c, 5, 4, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 95, 94, 93, 92, 91]);

      // replace whole row in matrix

      const Y4 = matrix([94, 93, 92, 91], 1, 4);

      const X4a = X.copy();
      X4a.replace(Y4, 1, []);
      testMatrixStructure(X4a, 5, 4, [94, 2, 3, 4, 5, 93, 7, 8, 9, 10, 92, 12, 13, 14, 15, 91, 17, 18, 19, 20]);

      const X4b = X.copy();
      X4b.replace(Y4, 3, []);
      testMatrixStructure(X4b, 5, 4, [1, 2, 94, 4, 5, 6, 7, 93, 9, 10, 11, 12, 92, 14, 15, 16, 17, 91, 19, 20]);

      const X4c = X.copy();
      X4c.replace(Y4, 5, []);
      testMatrixStructure(X4c, 5, 4, [1, 2, 3, 4, 94, 6, 7, 8, 9, 93, 11, 12, 13, 14, 92, 16, 17, 18, 19, 91]);

   });

   it ('tests for method "apply"', function() {
      const X = reshape(Vector.seq(1, 10), 5, 2);

      // dims = 0
      const X1 = X.apply(v => v * v, 0);
      testMatrixStructure(X1, 5, 2, [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]);

      // dims = 1
      const x2 = X.apply(v => v.reduce((p, v) => p + v), 1);
      testVectorStructure(x2, 5, [7, 9, 11, 13, 15]);

      // dims = 2
      const x3 = X.apply(v => v.reduce((p, v) => p + v), 2);
      testVectorStructure(x3, 2, [15, 40]);
   });

   it ('tests for method "copy"', function() {
      const X1 = Matrix.rand(100, 100);
      const X2 = X1.copy();

      expect(X2).not.equal(X1)
      expect(X2.v).not.equal(X1.v)
      expect(X2.v).to.deep.equal(X1.v)
   });

   it ('tests for method "subset"', function() {
      const X = reshape(Vector.seq(1, 24, 1), 6, 4);
      // consequent indices

      /// mid - mid
      const Xs1 = X.subset(Index.seq(3, 5), Index.seq(2, 3));
      testMatrixStructure(Xs1, 3, 2, [9, 10, 11, 15, 16, 17]);

      /// top - mid
      const Xs2 = X.subset(Index.seq(1,4), Index.seq(2,3));
      testMatrixStructure(Xs2, 4, 2, [7, 8, 9, 10, 13, 14, 15, 16]);

      /// bottom - mid
      const Xs3 = X.subset(Index.seq(3,6), Index.seq(2,3));
      testMatrixStructure(Xs3, 4, 2, [9, 10, 11, 12, 15, 16, 17, 18]);

      /// mid - left
      const Xs4 = X.subset(Index.seq(3,5), Index.seq(1,3));
      testMatrixStructure(Xs4, 3, 3, [3, 4, 5, 9, 10, 11, 15, 16, 17]);

      /// top - left
      const Xs5 = X.subset(Index.seq(1,4), Index.seq(1,3));
      testMatrixStructure(Xs5, 4, 3, [1, 2, 3, 4, 7, 8, 9, 10, 13, 14, 15, 16]);

      /// bottom - left
      const Xs6 = X.subset(Index.seq(3,6), Index.seq(1,3));
      testMatrixStructure(Xs6, 4, 3, [3, 4, 5, 6, 9, 10, 11, 12, 15, 16, 17, 18]);

      /// mid - right
      const Xs7 = X.subset(Index.seq(3,5), Index.seq(2,4));
      testMatrixStructure(Xs7, 3, 3, [9, 10, 11, 15, 16, 17, 21, 22, 23]);

      /// top - right
      const Xs8 = X.subset(Index.seq(1,4), Index.seq(2,4));
      testMatrixStructure(Xs8, 4, 3, [7, 8, 9, 10, 13, 14, 15, 16, 19, 20, 21, 22]);

      /// bottom - right
      const Xs9 = X.subset(Index.seq(3,6), Index.seq(2,4));
      testMatrixStructure(Xs9, 4, 3, [9, 10, 11, 12, 15, 16, 17, 18, 21, 22, 23, 24]);

      // consequent indices - one row

      /// mid - mid
      const Xs1a = X.subset(3, [2, 3]);
      testMatrixStructure(Xs1a, 1, 2, [9, 15]);

      /// top - mid
      const Xs2a = X.subset(1, [2, 3]);
      testMatrixStructure(Xs2a, 1, 2, [7, 13]);

      /// bottom - mid
      const Xs3a = X.subset(6, [2, 3]);
      testMatrixStructure(Xs3a, 1, 2, [12, 18]);

      /// mid - left
      const Xs4a = X.subset(4, Index.seq(1, 3));
      testMatrixStructure(Xs4a, 1, 3, [4, 10, 16]);

      /// top - left
      const Xs5a = X.subset(1, Index.seq(1, 3));
      testMatrixStructure(Xs5a, 1, 3, [1, 7, 13]);

      /// bottom - left
      const Xs6a = X.subset(6, Index.seq(1,3));
      testMatrixStructure(Xs6a, 1, 3, [6, 12, 18]);

      /// mid - right
      const Xs7a = X.subset(3, Index.seq(2,4));
      testMatrixStructure(Xs7a, 1, 3, [9, 15, 21]);

      /// top - right
      const Xs8a = X.subset(1, Index.seq(2,4));
      testMatrixStructure(Xs8a, 1, 3, [7, 13, 19]);

      /// bottom - right
      const Xs9a = X.subset(6, Index.seq(2,4));
      testMatrixStructure(Xs9a, 1, 3, [12, 18, 24]);

      // consequent indices - one column

      /// mid - mid
      const Xs1b = X.subset(Index.seq(3,5), 3);
      testMatrixStructure(Xs1b, 3, 1, [15, 16, 17]);

      /// top - mid
      const Xs2b = X.subset(Index.seq(1,3), 3);
      testMatrixStructure(Xs2b, 3, 1, [13, 14, 15]);

      /// bottom - mid
      const Xs3b = X.subset(Index.seq(3,6), 3);
      testMatrixStructure(Xs3b, 4, 1, [15, 16, 17, 18]);

      /// mid - left
      const Xs4b = X.subset(Index.seq(3,5), 1);
      testMatrixStructure(Xs4b, 3, 1, [3, 4, 5]);

      /// top - left
      const Xs5b = X.subset(Index.seq(1,3), 1);
      testMatrixStructure(Xs5b, 3, 1, [1, 2, 3]);

      /// bottom - left
      const Xs6b = X.subset(Index.seq(3,6), 1);
      testMatrixStructure(Xs6b, 4, 1, [3, 4, 5, 6]);

      /// mid - right
      const Xs7b = X.subset(Index.seq(2,4), 4);
      testMatrixStructure(Xs7b, 3, 1, [20, 21, 22]);

      /// top - right
      const Xs8b = X.subset(Index.seq(1,4), 4);
      testMatrixStructure(Xs8b, 4, 1, [19, 20, 21, 22]);

      /// bottom - right
      const Xs9b = X.subset(Index.seq(3,6), 4);
      testMatrixStructure(Xs9b, 4, 1, [21, 22, 23, 24]);

      // all rows

      /// all - mid
      const Xs1c = X.subset(null,Index.seq(2,3));
      testMatrixStructure(Xs1c, 6, 2, [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]);

      /// all - left
      const Xs2c = X.subset(null, Index.seq(1,2));
      testMatrixStructure(Xs2c, 6, 2, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);

      /// all - right
      const Xs3c = X.subset(null, Index.seq(3,4));
      testMatrixStructure(Xs3c, 6, 2, [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]);

      /// all - one column
      const Xs4c = X.subset(null, 3);
      testMatrixStructure(Xs4c, 6, 1, [13, 14, 15, 16, 17, 18]);

      /// all - columns 2 and 4
      const Xs5c = X.subset(null, [2, 4]);
      testMatrixStructure(Xs5c, 6, 2, [7, 8, 9, 10, 11, 12, 19, 20, 21, 22, 23, 24]);

      /// all - columns 1 and 3
      const Xs6c = X.subset(null, [1, 3]);
      testMatrixStructure(Xs6c, 6, 2, [1, 2, 3, 4, 5, 6, 13, 14, 15, 16, 17, 18]);

      // all columns

      /// mid - all
      const Xs1d = X.subset(Index.seq(2,4), null);
      testMatrixStructure(Xs1d, 3, 4, [2, 3, 4, 8, 9, 10, 14, 15, 16, 20, 21, 22]);

      /// top - all
      const Xs2d = X.subset(Index.seq(1,3), null);
      testMatrixStructure(Xs2d, 3, 4, [1, 2, 3, 7, 8, 9, 13, 14, 15, 19, 20, 21]);

      /// bottom - all
      const Xs3d = X.subset(Index.seq(3,6), null);
      testMatrixStructure(Xs3d, 4, 4, [3, 4, 5, 6, 9, 10, 11, 12, 15, 16, 17, 18, 21, 22, 23, 24]);

      /// one row - all
      const Xs4d = X.subset(3, null);
      testMatrixStructure(Xs4d, 1, 4, [3, 9, 15, 21]);

      /// rows 2 and 5 - all
      const Xs5d = X.subset([2, 5], null);
      testMatrixStructure(Xs5d, 2, 4, [2, 5, 8, 11, 14, 17, 20, 23]);

      /// rows 1 and 3 - all
      const Xs6d = X.subset([1, 3], null);
      testMatrixStructure(Xs6d, 2, 4, [1, 3, 7, 9, 13, 15, 19, 21]);

      /// rows 2 and 6 - all
      const Xs7d = X.subset([2, 6], null);
      testMatrixStructure(Xs7d, 2, 4, [2, 6, 8, 12, 14, 18, 20, 24]);

      // sparse rows and columns

      /// rows 2 and 6 - all
      const Xs1e = X.subset([2, 4, 6], [2, 4]);
      testMatrixStructure(Xs1e, 3, 2, [8, 10, 12, 20, 22, 24]);

      const Xs2e = X.subset([1, 6], [1, 4]);
      testMatrixStructure(Xs2e, 2, 2, [1, 6, 19, 24]);

      // large matrix 10.000 x 10.000
      const L = Matrix.rand(10000, 10000);
      const Ls1 = L.subset(Index.seq(1, 10000, 10), Index.seq(1, 10000, 5));
      testMatrixStructure(Ls1, 1000, 2000);

   });

});

describe('Tests of methods for class Vector.', function () {

   it ('tests for method "filter"', function() {
      const x1 = vector([-5, 1, -4, 2, -1, 3, -2, 4, 5, 0]);
      testVectorStructure(x1.filter(v => v > 0), 5, [1, 2, 3, 4, 5]);
      testVectorStructure(x1.filter(v => v < 0), 4, [-5, -4, -1, -2]);
      testVectorStructure(x1.filter(v => v == 0), 1, [0]);
      testVectorStructure(x1.filter(v => v == 10), 0, []);
   });

   it ('tests for methods "shuffle" and "sort".', function() {
      const x1 = vector([10, -8.2, 3, 1, -15.3, 100]);
      const x1a = x1.sort();
      const x1b = x1.sort(true);

      const x2 = x1.shuffle();
      const x2a = x2.sort();
      const x2b = x2.sort(true);

      expect(x2.v).to.be.not.deep.equal(x1.v);
      expect(x2a.v).to.be.deep.equal(x1a.v);
      expect(x2b.v).to.be.deep.equal(x1b.v);
   });

   it ('tests for method "dot"', function() {
      const x = vector([1, 2, 3, 4]);
      const y = vector([4, 3, 2, 1]);

      expect(x.dot(y)).to.be.equal(20);
      expect(x.dot()).to.be.equal(30);

   });

   it ('tests for arithmetic operations', function() {

      // vector and vector
      const x1 = vector([0, 2, 4, 6, 8]);
      const x2 = vector([5, 4, 3, 2, 1]);

      const x1x2add = x1.add(x2);
      testVectorStructure(x1x2add, 5, [5, 6, 7, 8, 9]);

      const x1x2sub = x1.subtract(x2);
      testVectorStructure(x1x2sub, 5, [-5, -2, 1, 4, 7]);

      const x1x2mult = x1.mult(x2);
      testVectorStructure(x1x2mult, 5, [0, 8, 12, 12, 8]);

      const x1x2div = x1.divide(x2);
      testVectorStructure(x1x2div, 5, [0, 0.5, 4/3, 3, 8]);

      // vector and number
      const x1xadd = x1.add(10);
      testVectorStructure(x1xadd, 5, [10, 12, 14, 16, 18]);

      const x1xsub = x1.subtract(2);
      testVectorStructure(x1xsub, 5, [-2, 0, 2, 4, 6]);

      const x1xmult = x1.mult(1.5);
      testVectorStructure(x1xmult, 5, [0, 3, 6, 9, 12]);

      const x1xdiv = x1.divide(2);
      testVectorStructure(x1xdiv, 5, [0, 1, 2, 3, 4]);

   });

   it ('tests for method "apply"', function() {
      const x1 = vector([1, 2, 3, 4, 5]);

      const x2 = x1.apply(v => 1 - v);
      testVectorStructure(x2, 5, [0, -1, -2, -3, -4]);

      const x3 = x1.apply(v => 1 / v);
      testVectorStructure(x3, 5, [1, 1/2, 1/3, 1/4, 1/5]);

      const x4 = x1.apply(v => v * v);
      testVectorStructure(x4, 5, [1, 4, 9, 16, 25]);

   });

   it ('tests for method "copy"', function() {
      const x1 = Vector.rand(100);
      const x2 = x1.copy();

      expect(x2).not.equal(x1)
      expect(x2.v).not.equal(x1.v)
      expect(x2.v).to.deep.equal(x1.v)
   });

   it ('tests for method "slice"', function() {
      const x1 = vector([9, 8, 7, 6, 5, 4, 3, 2, 1]);

      // errors
      chai.expect(() => x1.slice(0, 4)).to.
         throw('slice: indices must start with 1 (not 0).');

      chai.expect(() => x1.slice(5, 10)).to.
         throw('slice: index exceeds the length of the vector.');

      chai.expect(() => x1.slice(5, 1)).to.
         throw('slice: "end" must not be smaller than "start".');

      // simple tests

      const x21 = x1.slice(1, 5);
      testVectorStructure(x21, 5, [9, 8, 7, 6, 5]);

      const x22 = x1.slice(null, 5);
      testVectorStructure(x22, 5, [9, 8, 7, 6, 5]);

      const x31 = x1.slice(5, 9);
      testVectorStructure(x31, 5, [5, 4, 3, 2, 1]);

      const x32 = x1.slice(5);
      testVectorStructure(x32, 5, [5, 4, 3, 2, 1]);

      const x33 = x1.slice(5, 5);
      testVectorStructure(x33, 1, [5]);

      // subset of long vector
      const x6 = Vector.rand(1000000);
      testVectorStructure(x6.slice(500001, 600000), 100000)
   });

   it ('tests for method "subset"', function() {
      const x1 = vector([9, 8, 7, 6, 5, 4, 3, 2, 1]);

      // errors
      chai.expect(() => x1.subset(index([0, 1, 2]))).to.
         throw('subset: indices must start with 1 (not 0).');

      chai.expect(() => x1.subset(index([3, 5, 10]))).to.
         throw('subset: index exceeds the length of the vector.');

      // simple tests

      const x2 = x1.subset(index([1, 3, 5, 7, 9]));
      testVectorStructure(x2, 5, [9, 7, 5, 3, 1]);

      const x3 = x1.subset(Index.seq(1, 3));
      testVectorStructure(x3, 3, [9, 8, 7]);

      const x4 = x1.subset(Index.seq(6, 9));
      testVectorStructure(x4, 4, [4, 3, 2, 1]);

      const x5 = x1.subset(Index.seq(9, 1));
      testVectorStructure(x5, 9, [1, 2, 3, 4, 5, 6, 7, 8, 9]);

      // subset of long vector
      const x6 = Vector.rand(1000000);
      testVectorStructure(x6.subset(Index.seq(10, 1000000, 10)), 100000)
   });

   it ('tests for method "repeach"', function() {

      const x1 = vector([1, 5, 9]);
      testVectorStructure(x1.repeach(4), 12, [1, 1, 1, 1, 5, 5, 5, 5, 9, 9, 9, 9]);

      const x2 = vector([1.1, 5.2, 9.3]);
      const x3 = x2.repeach(1000000)
      testVectorStructure(x3, 3000000);
      expect(x3.v[0]).equal(1.1);
      expect(x3.v[999999]).equal(1.1);
      expect(x3.v[1000000]).equal(5.2);
      expect(x3.v[1999999]).equal(5.2);
      expect(x3.v[2000000]).equal(9.3);
      expect(x3.v[2999999]).equal(9.3);

   });

   it ('tests for method "rep"', function() {

      const x1 = vector([1]).rep(10);
      testVectorStructure(x1, 10, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);

      const x2 = vector([1, 5, 9]).rep(4);
      testVectorStructure(x2, 12, [1, 5, 9, 1, 5, 9, 1, 5, 9, 1, 5, 9]);

      const x3 = vector([1, 5, 9]).rep(1000000);
      testVectorStructure(x3, 3000000);
      expect(x3.v[99]).equal(1);
      expect(x3.v[100]).equal(5);
      expect(x3.v[101]).equal(9);

   });

});

describe('Tests of methods for class Index.', function () {

   it ('tests for methods "shuffle" and "sort".', function() {
      const x1 = index([10, 8, 3, 1, 15, 100]);
      const x1a = x1.sort();
      const x1b = x1.sort(true);

      const x2 = x1.shuffle();
      const x2a = x2.sort();
      const x2b = x2.sort(true);

      expect(x2.v).to.be.not.deep.equal(x1.v);
      expect(x2a.v).to.be.deep.equal(x1a.v);
      expect(x2b.v).to.be.deep.equal(x1b.v);
   });

   it ('tests for method "repeach"', function() {

      const x1 = index([1, 5, 9]);
      testIndexStructure(x1.repeach(4), 12, [1, 1, 1, 1, 5, 5, 5, 5, 9, 9, 9, 9]);

      const x2 = index([1.1, 5.0, 9.0]);
      const x3 = x2.repeach(1000000)
      testIndexStructure(x3, 3000000);
      expect(x3.v[0]).equal(1);
      expect(x3.v[999999]).equal(1);
      expect(x3.v[1000000]).equal(5);
      expect(x3.v[1999999]).equal(5);
      expect(x3.v[2000000]).equal(9);
      expect(x3.v[2999999]).equal(9);

   });

   it ('tests for method "rep"', function() {

      const x1 = index([1]);
      testIndexStructure(x1.rep(10), 10, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);

      const x2 = index([1, 5, 9]);
      testIndexStructure(x2.rep(4), 12, [1, 5, 9, 1, 5, 9, 1, 5, 9, 1, 5, 9]);

      const x3 = index([1, 5, 9]);
      const x4 = x3.rep(1000000)
      testIndexStructure(x4, 3000000);
      expect(x4.v[99]).equal(1);
      expect(x4.v[100]).equal(5);
      expect(x4.v[101]).equal(9);

   });

});

describe('Tests of methods for class Factor.', function () {

   it ('tests for methods "which".', function() {
      const x1 = [1, 2, 3, 3, 2, 1, 1, 2, 3, 3, 2, 1];
      const f1 = factor(x1);
      testIndexStructure(f1.which('1'), 4, [1, 6, 7, 12]);
      testIndexStructure(f1.which('3'), 4, [3, 4, 9, 10]);

      const x2 = ['red', 'blue', 'green', 'blue', 'green', 'red', 'red', 'blue'];
      const f2 = factor(x2);
      testIndexStructure(f2.which('blue'), 3, [2, 4, 8]);
      testIndexStructure(f2.which('red'), 3, [1, 6, 7]);

      const x3 = ['red', 'red', 'red', 'red', 'red', 'red', 'red', 'red', 'red'];
      const f3 = factor(x3);
      testIndexStructure(f3.which('red'), 9, [1, 2, 3, 4, 5, 6, 7, 8, 9]);
      testIndexStructure(f3.which('blue'), 0, []);
   });

});

describe('Tests of methods for generating vectors and matrices and static methods.', function () {

   it ('tests for static method "c"', function() {
      const x1 = Vector.c(1, [3, 4, 5], vector([20, 30, 50]), 55);
      testVectorStructure(x1, 8, [1, 3, 4, 5, 20, 30, 50, 55]);

      const x2 = Vector.c(vector([20, 30, 50]), 1, 10, [3, 4, 5], 55);
      testVectorStructure(x2, 9, [20, 30, 50, 1, 10, 3, 4, 5, 55]);

   });

   it ('tests for method "fill"', function() {

      // vector
      const x1 = Vector.fill(4, 3);
      testVectorStructure(x1, 3, [4, 4, 4]);

      const x2 = Vector.fill(-0.5, 1);
      testVectorStructure(x2, 1, [-0.5]);

      // matrix
      const X1 = Matrix.fill(4, 3, 4);
      testMatrixStructure(X1, 3, 4, [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]);

      const X2 = Matrix.fill(-0.5, 1, 1);
      testMatrixStructure(X2, 1, 1, [-0.5]);


   });

   it('tests for method "outer"', function() {
      const x = [-5, -3, 0, 3, 5];
      const y = [10, 0, 10];
      const Z = Matrix.outer(x, y, (x, y) => x**2 + y**2);

      testMatrixStructure(Z, 5, 3, [125, 109, 100, 109, 125, 25, 9, 0, 9, 25, 125, 109, 100, 109, 125]);
   });

   it('tests for method "parseCSV"', function() {

      // just values (sep = ",", dec = "."), no header
      const csvStr1 = "10.1,20.2,30.3\n5.5,6.61,-7.22\n3.8,2.99,1.11\n1.9,  -1.991,1.9994\n \n\n\n";
      const d1 = Matrix.parseCSV(csvStr1);
      expect(d1.header.length).to.be.equal(0);
      expect(d1.labels.length).to.be.equal(0);
      expect(d1.sep).to.be.equal(",");
      expect(d1.dec).to.be.equal(".");
      testMatrixStructure(d1.values.t(), 3, 4, [
         10.1, 20.2, 30.3, 5.5, 6.61, -7.22, 3.8, 2.99, 1.11, 1.9, -1.991, 1.9994
      ]);

      // just values (sep = ";", dec = ","), no header
      const csvStr2a = "10,1;20,2;30,3\n5,5;6,61;-7,22\n3,8;2,99;1,11\n1,9; -1,991;1,9994\n  \n\n\n";
      const d2a = Matrix.parseCSV(csvStr2a, ";");
      expect(d2a.header.length).to.be.equal(0);
      expect(d2a.labels.length).to.be.equal(0);
      expect(d2a.sep).to.be.equal(";");
      expect(d2a.dec).to.be.equal(",");
      testMatrixStructure(d2a.values.t(), 3, 4, [
         10.1, 20.2, 30.3, 5.5, 6.61, -7.22, 3.8, 2.99, 1.11, 1.9, -1.991, 1.9994
      ]);

      // just values (sep = ";", dec = ","), no header with labels
      const csvStr2b = "O1;10,1;20,2;30,3\nO2;5,5;6,61;-7,22\nO3;3,8;2,99;1,11\nO4;1,9; -1,991;1,9994\n  \n\n\n";
      const d2b = Matrix.parseCSV(csvStr2b, ";", false, true);
      expect(d2b.header.length).to.be.equal(0);
      expect(d2b.labels.length).to.be.equal(4);
      expect(d2b.labels).to.be.deep.equal(["O1", "O2", "O3", "O4"]);
      expect(d2b.sep).to.be.equal(";");
      expect(d2b.dec).to.be.equal(",");
      testMatrixStructure(d2b.values.t(), 3, 4, [
         10.1, 20.2, 30.3, 5.5, 6.61, -7.22, 3.8, 2.99, 1.11, 1.9, -1.991, 1.9994
      ]);

      // just values (sep = ",", dec = "."), with text header
      const csvStr3a = "A1 cm,B2 g,C3 m3\n10.1,20.2,30.3\n5.5,6.61,-7.22\n3.8,2.99,1.11\n1.9, -1.991,1.9994\n \n\n\n";
      const d3a = Matrix.parseCSV(csvStr3a);
      expect(d3a.header.length).to.be.equal(3);
      expect(d3a.header).to.be.deep.equal(["A1 cm", "B2 g", "C3 m3"]);
      expect(d3a.labels.length).to.be.equal(0);
      expect(d3a.sep).to.be.equal(",");
      expect(d3a.dec).to.be.equal(".");
      testMatrixStructure(d3a.values.t(), 3, 4, [
         10.1, 20.2, 30.3, 5.5, 6.61, -7.22, 3.8, 2.99, 1.11, 1.9, -1.991, 1.9994
      ]);

      // just values (sep = ",", dec = "."), with numeric header
      const csvStr3b = "1,2,3\n10.1,20.2,30.3\n5.5,6.61,-7.22\n3.8,2.99,1.11\n1.9, -1.991,1.9994\n \n\n\n";
      const d3b = Matrix.parseCSV(csvStr3b, ",", true);
      expect(d3b.header.length).to.be.equal(3);
      expect(d3b.header).to.be.deep.equal(["1", "2", "3"]);
      expect(d3b.labels.length).to.be.equal(0);
      expect(d3b.sep).to.be.equal(",");
      expect(d3b.dec).to.be.equal(".");
      testMatrixStructure(d3b.values.t(), 3, 4, [
         10.1, 20.2, 30.3, 5.5, 6.61, -7.22, 3.8, 2.99, 1.11, 1.9, -1.991, 1.9994
      ]);

      // just values (sep = ",", dec = "."), with numeric header and labels
      const csvStr3c = ",1,2,3\nO1,10.1,20.2,30.3\nO2,5.5,6.61,-7.22\nO3,3.8,2.99,1.11\nO4,1.9, -1.991,1.9994\n \n\n\n";
      const d3c = Matrix.parseCSV(csvStr3c, ",", true, true);
      expect(d3c.header.length).to.be.equal(4);
      expect(d3c.header).to.be.deep.equal(["", "1", "2", "3"]);
      expect(d3c.labels.length).to.be.equal(4);
      expect(d3c.labels).to.be.deep.equal(["O1", "O2", "O3", "O4"]);
      expect(d3c.sep).to.be.equal(",");
      expect(d3c.dec).to.be.equal(".");
      testMatrixStructure(d3c.values.t(), 3, 4, [
         10.1, 20.2, 30.3, 5.5, 6.61, -7.22, 3.8, 2.99, 1.11, 1.9, -1.991, 1.9994
      ]);
   });

   it('tests for method "seq"', function() {

      // default settings -> Index
      const x1 = Index.seq(1, 10);
      testIndexStructure(x1, 10, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

      // manual by -> Index
      const x2 = Index.seq(0, 10, 2);
      testIndexStructure(x2, 6, [0, 2, 4, 6, 8, 10]);

      // manual start/end/by which are not integer -> Vector
      const x31 = Vector.seq(0, 10, 2.1);
      testVectorStructure(x31, 5, [0, 2.1, 4.2, 6.3, 8.4]);

      const x32 = Vector.seq(0.1, 10, 2);
      testVectorStructure(x32, 5, [0.1, 2.1, 4.1, 6.1, 8.1]);

      const x33 = Vector.seq(0, 10.1, 2);
      testVectorStructure(x33, 6, [0, 2, 4, 6, 8, 10]);

      // manual start/end/by which are not integer but force to make it index
      const x41 = Index.seq(0, 10, 2.1);
      testIndexStructure(x41, 5, [0, 2, 4, 6, 8]);

      const x42 = Index.seq(0.1, 10, 2);
      testIndexStructure(x42, 5, [0, 2, 4, 6, 8]);

      const x43 = Index.seq(0, 10.1, 2);
      testIndexStructure(x43, 6, [0, 2, 4, 6, 8, 10]);

   });

   it('tests for method "randn"', function() {
      chai.use(chaiAlmost(0.5));

      // default settings
      const X1 = Matrix.randn(40, 10);
      testMatrixStructure(X1, 40, 10);

      // symmetry around mu
      expect(X1.v.filter(v => v < 0).length / 200).to.be.almost.equal(1);
      expect(X1.v.filter(v => v > 0).length / 200).to.be.almost.equal(1);

      // spread ± 4 sigma
      expect(X1.v.filter(v => v < -4).length).to.be.equal(0);
      expect(X1.v.filter(v => v >  4).length).to.be.equal(0);

      // with given mu and sigma
      const X2 = Matrix.randn(40, 10, 100, 20);
      testMatrixStructure(X1, 40, 10);

      // symmetry around mu
      expect(X2.v.filter(v => v < 100).length / 200).to.be.almost.equal(1);
      expect(X2.v.filter(v => v > 100).length / 200).to.be.almost.equal(1);

      // spread ± 4 sigma
      expect(X2.v.filter(v => v < 20).length).to.be.equal(0);
      expect(X2.v.filter(v => v > 180).length).to.be.equal(0);

      // small vector
      const x1 = Vector.randn(400);
      testVectorStructure(x1, 400);

      // symmetry around mu
      expect(x1.v.filter(v => v < 0).length / 200).to.be.almost.equal(1);
      expect(x1.v.filter(v => v > 0).length / 200).to.be.almost.equal(1);

      // spread ± 4 sigma
      expect(x1.v.filter(v => v < -4).length).to.be.equal(0);
      expect(x1.v.filter(v => v >  4).length).to.be.equal(0);

   });

   it('tests for method "rand"', function() {

      // nr > nc
      const X1 = Matrix.rand(40, 10);
      testMatrixStructure(X1, 40, 10);

      // nr < nc
      const X2 = Matrix.rand(10, 40);
      testMatrixStructure(X2, 10, 40);

      // with given range
      const X3 = Matrix.rand(10, 40, 5, 10);
      testMatrixStructure(X3, 10, 40);
      expect(X3.v.filter(v => v < 5).length).to.be.equal(0);
      expect(X3.v.filter(v => v > 10).length).to.be.equal(0);
      expect(X3.v.filter(v => v >= 5 && v <= 10).length).to.be.equal(400);

      // large matrices
      const X31 = Matrix.rand(1000, 5000);
      testMatrixStructure(X31, 1000, 5000);

      const X32 = Matrix.rand(1000, 1000);
      testMatrixStructure(X32, 1000, 1000);

      // small vector
      const x1 = Vector.rand(40);
      testVectorStructure(x1, 40);

      // large vector
      const x2 = Vector.rand(4000000);
      testVectorStructure(x2, 4000000);

      // vector with given range
      const x3 = Vector.rand(40, 5, 10);
      testVectorStructure(x3, 40);
      expect(x3.v.filter(v => v < 5).length).to.be.equal(0);
      expect(x3.v.filter(v => v > 10).length).to.be.equal(0);
      expect(x3.v.filter(v => v >= 5 && v <= 10).length).to.be.equal(40);

   });

   it('tests for method "diagm"', function() {

      // simple matrix
      const x1 = [9, 7, 5, 3, 1];

      chai.expect(() => Matrix.diagm(x1)).to.throw('diagm: the argument "x" must be a Vector.');

      const X1 = Matrix.diagm(vector(x1));
      testMatrixStructure(X1, 5, 5, [9, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 1]);

      // 1 x 1 matrix
      const x2 = [9];
      const X2 = Matrix.diagm(vector(x2));
      testMatrixStructure(X2, 1, 1, [9]);

      // large matrix
      const x3 = Vector.ones(5000);
      const X3 = Matrix.diagm(x3);
      expect(X3).to.be.deep.equal(Matrix.eye(5000));

   });

   it('tests for method "eye"', function() {

      // nr > nc
      const X11 = Matrix.eye(4, 3);
      const v11 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0];
      testMatrixStructure(X11, 4, 3, v11);

      const X12 = Matrix.eye(6, 2);
      const v12 = [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0];
      testMatrixStructure(X12, 6, 2, v12);

      // nr < nc
      const X21 = Matrix.eye(3, 4);
      const v21 = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0];
      testMatrixStructure(X21, 3, 4, v21);

      const X22 = Matrix.eye(2, 6);
      const v22 = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0];
      testMatrixStructure(X22, 2, 6, v22);

      // nr = 1
      const X3 = Matrix.eye(1, 12);
      const v3 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
      testMatrixStructure(X3, 1, 12, v3);

      // nc = 1
      const X4 = Matrix.eye(12, 1);
      const v4 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
      testMatrixStructure(X4, 12, 1, v4);

      // large matrix (10.000 x 5.000)
      const X5 = Matrix.eye(10000, 5000);
      testMatrixStructure(X5, 10000, 5000);

      // large matrix (10.000 x 10.000)
      const X6 = Matrix.eye(10000, 10000);
      testMatrixStructure(X6, 10000, 10000);
   });

   it('tests for method "ones"', function () {
      const values = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

      // nr > nc
      const X1 = Matrix.ones(4, 3);
      testMatrixStructure(X1, 4, 3, values);

      // nr < nc
      const X2 = Matrix.ones(3, 4);
      testMatrixStructure(X2, 3, 4, values);

      // nr = 1
      const X3 = Matrix.ones(1, 12);
      testMatrixStructure(X3, 1, 12, values);

      // nc = 1
      const X4 = Matrix.ones(12, 1);
      testMatrixStructure(X4, 12, 1, values);

      // large matrix (10.000 x 5.000)
      const X5 = Matrix.ones(10000, 5000);
      testMatrixStructure(X5, 10000, 5000);

      // nc = undefined -> return a squared matrix
      const x1 = Matrix.ones(3);
      testMatrixStructure(x1, 3, 3, [1, 1, 1, 1, 1, 1, 1, 1, 1]);
   });

   it('tests for method "zeros"', function () {
      const values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

      // nr > nc
      const X1 = Matrix.zeros(4, 3);
      testMatrixStructure(X1, 4, 3, values);

      // nr < nc
      const X2 = Matrix.zeros(3, 4);
      testMatrixStructure(X2, 3, 4, values);

      // nr = 1
      const X3 = Matrix.zeros(1, 12);
      testMatrixStructure(X3, 1, 12, values);

      // nc = 1
      const X4 = Matrix.zeros(12, 1);
      testMatrixStructure(X4, 12, 1, values);

      // large matrix (10.000 x 5.000)
      const X5 = Matrix.zeros(10000, 5000);
      testMatrixStructure(X5, 10000, 5000);

      // nc = undefined -> return a vector
      const x1 = Matrix.zeros(3);
      testMatrixStructure(x1, 3, 3, [0, 0, 0, 0, 0, 0, 0, 0, 0]);

   });

});

describe('Tests of constructors.', function () {

   it('tests for constructor "Dataset"', function () {

      // test for throwing errors
      expect(() => new Dataset({}, 'data', {}, {})).to.throw(Error);
      expect(() => new Dataset(matrix([], 0, 0), 'data', {}, {})).to.throw(Error);

      // test if all parameters except values are empty
      const values = matrix([1, 2, 3, 4], 2, 2);
      const mda1 = new Dataset(values, 'data', null, null);
      expect(mda1.rowAttrs.axisValues).to.deep.equal(vector([1, 2]));
      expect(mda1.rowAttrs.axisName).to.equal('Objects');
      expect(mda1.rowAttrs.labels).to.deep.equal(['O1', 'O2']);
      expect(mda1.rowAttrs.axisLabels).to.deep.equal(['O1', 'O2']);
      expect(mda1.colAttrs.axisValues).to.deep.equal(vector([1, 2]));
      expect(mda1.colAttrs.axisName).to.equal('Variables');
      expect(mda1.colAttrs.labels).to.deep.equal(['X1', 'X2']);
      expect(mda1.colAttrs.axisLabels).to.deep.equal(['X1', 'X2']);

      // test if some of the attributes are provided
      const rowAttrs = {
        axisValues: vector([10, 20]),
        axisName: 'Rows',
        labels: ['R1', 'R2'],
        axisLabels: ['Row 1', 'Row 2']
      };
      const colAttrs = {
        axisValues: vector([100, 200]),
        axisName: 'Cols',
        labels: ['C1', 'C2'],
        axisLabels: ['Col 1', 'Col 2']
      };

      const mda2 = new Dataset(values, 'data', rowAttrs, colAttrs);
      expect(mda2.rowAttrs.axisValues).to.deep.equal(vector([10, 20]));
      expect(mda2.rowAttrs.axisName).to.equal('Rows');
      expect(mda2.rowAttrs.labels).to.deep.equal(['R1', 'R2']);
      expect(mda2.rowAttrs.axisLabels).to.deep.equal(['Row 1', 'Row 2']);
      expect(mda2.colAttrs.axisValues).to.deep.equal(vector([100, 200]));
      expect(mda2.colAttrs.axisName).to.equal('Cols');
      expect(mda2.colAttrs.labels).to.deep.equal(['C1', 'C2']);
      expect(mda2.colAttrs.axisLabels).to.deep.equal(['Col 1', 'Col 2']);


      // if provided attributes are invalid default ones must be used
      const rowAttrsErr = {
        axisValues: [10],
        axisName: '',
        labels: 'invalid',
        axisLabels: {}
      };
      const colAttrsErr = {
        axisValues: null,
        axisName: 'Vars',
        labels: [],
        axisLabels: 123
      };

      const mda3 = new Dataset(values, 'data', rowAttrs, colAttrs);
      expect(mda1.rowAttrs.axisValues).to.deep.equal(vector([1, 2]));
      expect(mda1.rowAttrs.axisName).to.equal('Objects');
      expect(mda1.rowAttrs.labels).to.deep.equal(['O1', 'O2']);
      expect(mda1.rowAttrs.axisLabels).to.deep.equal(['O1', 'O2']);
      expect(mda1.colAttrs.axisValues).to.deep.equal(vector([1, 2]));
      expect(mda1.colAttrs.axisName).to.equal('Variables');
      expect(mda1.colAttrs.labels).to.deep.equal(['X1', 'X2']);
      expect(mda1.colAttrs.axisLabels).to.deep.equal(['X1', 'X2']);

   });

   it('tests for constructor "Index"', function () {
      const values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

      const x1 = index(values);
      testIndexStructure(x1, 12, values);

      const x2 = new Index(new Int32Array(values));
      testIndexStructure(x2, 12, values);
   });

   it('tests for constructor "Vector"', function () {
      const values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

      const x1 = vector(values);
      testVectorStructure(x1, 12, values);
   });

   it('tests for constructor "Matrix"', function () {
      const values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

      // nr > nc
      const X1 = matrix(values, 4, 3);
      testMatrixStructure(X1, 4, 3, values);

      // nr < nc
      const X2 = matrix(values, 3, 4);
      testMatrixStructure(X2, 3, 4, values);

      // nr = 1
      const X3 = matrix(values, 1, 12);
      testMatrixStructure(X3, 1, 12, values);

      // nc = 1
      const X4 = matrix(values, 12, 1);
      testMatrixStructure(X4, 12, 1, values);
   });

   it('tests for constructor "Factor"', function () {

      const x1 = [1, 2, 3, 3, 2, 1, 1, 2, 3, 3, 2, 1];
      const f1 = factor(x1);
      testFactorStructure(f1, 12, [0, 1, 2, 2, 1, 0, 0, 1, 2, 2, 1, 0], ['1', '2', '3']);

      const x2 = ['red', 'blue', 'green', 'blue', 'green', 'red', 'red', 'blue'];
      const f2 = factor(x2);
      testFactorStructure(f2, 8, [0, 1, 2, 1, 2, 0, 0, 1], ['red', 'blue', 'green']);

      const x3 = ['red', 'red', 'red', 'red', 'red', 'red', 'red', 'red', 'red'];
      const f3 = factor(x3);
      testFactorStructure(f3, 9, [0, 0, 0, 0, 0, 0, 0, 0, 0], ['red']);
   });

});
