/******************************************************************
 *  Tests for decomposition methods                               *
 ******************************************************************/

// import of functions to test
import {qr, inv} from '../decomp/index.js';
import {mdot, eye, nrow, tomatrix, transpose, tcrossprod} from '../matrix/index.js';
import {seq} from '../stat/index.js';

// import dependencies
import {default as chai} from 'chai';
import {default as chaiAlmost} from 'chai-almost';
const should = chai.should();
const expect = chai.expect;

chai.use(chaiAlmost(0.00001));
describe('Tests for QR decomposition.', function () {

   it('qr() works correctly for squared matrix.', function () {
      const X = [[12, 6, -4], [-51, 167, 24], [4, -68, -41]];
      const res = qr(X)
      it('qr() works correctly.', function () {
         expect(mdot(res.Q, res.R)).to.be.deep.almost(X)
         expect(tcrossprod(res.Q, res.Q)).to.be.deep.almost(eye(nrow(X)))
      });
   });


   it('qr() works correctly for matrix with more columns than rows.', function () {
      const X = transpose(tomatrix(seq(1, 20), 4, 5));
      const res = qr(X)
      it('qr() works correctly.', function () {
         expect(mdot(res.Q, res.R)).to.be.deep.almost(X)
         expect(tcrossprod(res.Q, res.Q)).to.be.deep.almost(eye(nrow(X)))
      });
   });


   it('qr() works correctly for matrix with more rows than columns.', function () {
      const X = transpose(tomatrix(seq(1, 20), 5, 4));
      const res = qr(X)
      it('qr() works correctly.', function () {
         expect(mdot(res.Q, res.R)).to.be.deep.almost(X)
         expect(tcrossprod(res.Q, res.Q)).to.be.deep.almost(eye(nrow(X)))
      });
   });

});

describe('Tests for matrix inversion.', function () {

   it('inv() works correctly.', function () {

      // errors
      const E = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]];
      expect(() => inv(E)).to.throw(Error, "Only squared matrices can be inverted.");

      // lower triangular
      const X1 = [[1, 0, 0, 0], [2, 3, 0, 0], [4, 5, 6, 0], [7, 8, 9, 10]];
      expect(mdot(X1, inv(X1))).to.be.deep.almost(eye(4))

      // upper triangular
      const X2 = transpose(X1);
      expect(mdot(X2, inv(X2))).to.be.deep.almost(eye(4))

      // squared
      const X3 = [[1, 2, 7, 3], [9, 11, 12, 3], [1, 9, 0, 2], [11, 9, 1, 8]]
      expect(mdot(X3, inv(X3))).to.be.deep.almost(eye(4))

   });
});
