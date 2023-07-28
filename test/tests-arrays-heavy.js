/****************************************************************/
/*  Tests for array methods (Index/Vector/Matrix classes)       */
/****************************************************************/

import {default as chai} from 'chai';
import {default as chaiAlmost} from 'chai-almost';

// import classes and related methods
import { isindex, Index, ismatrix, Matrix, isvector, Vector } from '../src/arrays/index.js';

// import non-class methods
import { tcrossprod, crossprod, rbind, cbind } from '../src/arrays/index.js';

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

describe('Tests of methods for matrices and vectors which are heavy in computation.', function() {

   it('tests for method "tcrossprod"', function() {

      // large matrices - nrows = ncols
      const X4 = Matrix.rand(1000, 1000);
      const Y4 = Matrix.rand(1000, 1000);
      const Z4 = tcrossprod(X4, Y4);
      testMatrixStructure(Z4, 1000, 1000);

      // large matrices - nrows > ncols
      const X5 = Matrix.rand(2000, 500);
      const Y5 = Matrix.rand(2000, 500);
      const Z5 = tcrossprod(X5, Y5);
      testMatrixStructure(Z5, 2000, 2000);

      // large matrices - ncols < nrows
      const X6 = Matrix.rand(500, 2000);
      const Y6 = Matrix.rand(500, 2000);
      const Z6 = tcrossprod(X6, Y6);
      testMatrixStructure(Z6, 500, 500);

      // large matrices - same matrix
      const Z6a = tcrossprod(X6);
      testMatrixStructure(Z6a, 500, 500);

   }).timeout(10000);

   it('tests for method "crossprod"', function() {

      // large matrices - nrows = ncols
      const X4 = Matrix.rand(1000, 1000);
      const Y4 = Matrix.rand(1000, 1000);
      const Z4 = crossprod(X4, Y4);
      testMatrixStructure(Z4, 1000, 1000);

      // large matrices - nrows > ncols
      const X5 = Matrix.rand(2000, 500);
      const Y5 = Matrix.rand(2000, 500);
      const Z5 = crossprod(X5, Y5);
      testMatrixStructure(Z5, 500, 500);

      // large matrices - ncols < nrows
      const X6 = Matrix.rand(500, 2000);
      const Y6 = Matrix.rand(500, 2000);
      const Z6 = crossprod(X6, Y6);
      testMatrixStructure(Z6, 2000, 2000);

      // large matrices - same matrix
      const Z6a = crossprod(X6);
      testMatrixStructure(Z6a, 2000, 2000);

   }).timeout(10000);

   it ('tests for method "rbind"', function() {

      const X4 = Matrix.rand(5000, 5000);
      const X5 = Matrix.rand(5000, 5000);
      const X6 = Matrix.rand(5000, 5000);

      const R3 = rbind(X4, X5, X6);
      testMatrixStructure(R3, 15000, 5000);

   }).timeout(10000);

   it ('tests for method "cbind"', function() {

      const X4 = Matrix.rand(5000, 5000);
      const X5 = Matrix.rand(5000, 5000);
      const X6 = Matrix.rand(5000, 5000);

      const R3 = cbind(X4, X5, X6);
      testMatrixStructure(R3, 5000, 15000);

   }).timeout(10000);

   it ('tests for method "dot"', function() {

      // large matrices - squared
      const X3 = Matrix.rand(2000, 2000);
      const Y3 = Matrix.rand(2000, 2000);
      const Z3 = X3.dot(Y3);
      testMatrixStructure(Z3, 2000, 2000);

      // large matrices - nrows > ncols
      const X4 = Matrix.rand(2000, 500);
      const Y4 = Matrix.rand(500, 2000);
      const Z4 = X4.dot(Y4);
      testMatrixStructure(Z4, 2000, 2000);

      // large matrices - ncols < nrows
      const X5 = Matrix.rand(500, 2000);
      const Y5 = Matrix.rand(2000, 500);
      const Z5 = X5.dot(Y5);
      testMatrixStructure(Z5, 500, 500);

   }).timeout(20000);

   it ('tests for method "mult"', function() {

      // difference two large matrices
      const L1 = Matrix.rand(10000, 10000);
      const L2 = Matrix.rand(10000, 10000);
      const L3 = L1.mult(L2);
      expect(L3.v[100]).to.be.almost.equal(L1.v[100] * L2.v[100]);
   }).timeout(10000);

   it ('tests for method "divide"', function() {

      // difference two large matrices
      const L1 = Matrix.rand(10000, 10000);
      const L2 = Matrix.rand(10000, 10000);
      const L3 = L1.divide(L2);
      expect(L3.v[100]).to.be.almost.equal(L1.v[100] / L2.v[100]);
   }).timeout(10000);

   it ('tests for method "subtract"', function() {

      // difference two large matrices
      const L1 = Matrix.rand(10000, 10000);
      const L2 = Matrix.rand(10000, 10000);
      const L3 = L1.subtract(L2);
      expect(L3.v[100]).to.be.almost.equal(L1.v[100] - L2.v[100]);

   }).timeout(10000);

   it ('tests for method "add"', function() {

      // sum of two large matrices
      const L1 = Matrix.rand(10000, 10000);
      const L2 = Matrix.rand(10000, 10000);
      const L3 = L1.add(L2);
      expect(L3.v[100]).to.be.almost.equal(L1.v[100] + L2.v[100]);
   }).timeout(10000);

   it ('tests for method "t"', function() {

      // large matrix - squared
      const X1 = Matrix.rand(10000, 10000);

      const X1t1 = X1.t();
      expect(X1t1.v).to.not.deep.equal(X1.v);
      expect(X1t1.nrows).to.be.equal(10000);
      expect(X1t1.ncols).to.be.equal(10000);

      const X1t2 = X1t1.t();
      expect(X1t2.v).to.deep.equal(X1.v);
      expect(X1t2.nrows).to.be.equal(X1.nrows);
      expect(X1t2.ncols).to.be.equal(X1.ncols);

      // large matrix - nrows > ncols
      const X2 = Matrix.rand(10000, 5000);

      const X2t1 = X2.t();
      expect(X2t1.v).to.not.deep.equal(X2.v);
      expect(X2t1.nrows).to.be.equal(5000)
      expect(X2t1.ncols).to.be.equal(10000);

      const X2t2 = X2t1.t();
      expect(X2t2.v).to.deep.equal(X2.v);
      expect(X2t2.nrows).to.be.equal(X2.nrows);
      expect(X2t2.ncols).to.be.equal(X2.ncols);

      // large matrix - ncols > nrows
      const X3 = Matrix.rand(5000, 10000);

      const X3t1 = X3.t();
      expect(X3t1.v).to.not.deep.equal(X3.v);
      expect(X3t1.nrows).to.be.equal(10000);
      expect(X3t1.ncols).to.be.equal(5000);

      const X3t2 = X3t1.t();
      expect(X3t2.v).to.deep.equal(X3.v);
      expect(X3t2.nrows).to.be.equal(X3.nrows);
      expect(X3t2.ncols).to.be.equal(X3.ncols);

   }).timeout(10000);

});

