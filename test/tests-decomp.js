/*****************************************************************
 *  Tests for methods for decompositions of a matrix             *
 *****************************************************************/

//   Tests of randomized SVD decomposition.
//     ✓ rsvd() works correctly for m > n. (7789ms)

//   Tests of LU and QR decompositions.
//     ✓ qr() works correctly. (4100ms)

//   Tests of SVD decomposition and related methods.
//     ✓ svd() works correctly. (16686ms)
//     ✓ rot() works correctly.
//     ✓ tests for method "bidiag" (1645ms)
//     ✓ tests for method "householder" (49ms)


// import dependencies
import {default as chai} from 'chai';
import {default as chaiAlmost} from 'chai-almost';
import { crossprod, tcrossprod, reshape, isvector, vector, Vector,
   ismatrix, matrix, Matrix, Index } from '../src/arrays/index.js';

// import methods to test
import { rsvd, qr, lu, svd, rot, bidiag, householder } from '../src/decomp/index.js';

// set up test settings
const expect = chai.expect;
chai.use(chaiAlmost(0.0001));

const ZERO = Math.pow(10.0, -6);

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
      expect(X.v).to.deep.almost.equal(new Float64Array(values));
   }
}

// computes sum of squared differences between two matrices
function ssqdiff(A, B) {
   return A.subtract(B).v.map(v => v ** 2).reduce((p, s) => p + s);
}

// test SVD results
function svdtests(X, r, ncomp) {
   // checks that:
   // - V and Y and s have proper dimensions
   // - V and U has orthogonal columns
   // - (T = XV) == (US)
   // - X ≈ TV'

   const m = X.nrows;
   const n = X.ncols;

   if (ncomp === undefined) {
      ncomp = Math.min(m, n);
   }

   // check size of U and V
   testMatrixStructure(r.U, m, ncomp);
   testMatrixStructure(r.V, n, ncomp);

   // check orthogonality of V and U
   const A = r.s.v.filter(v => v > Math.pow(10,-12)).length;
   const ind = Index.seq(1, A);
   expect(crossprod(r.V.subset([], ind)).apply(Math.abs, 0)).to.be.deep.almost.equal(Matrix.eye(A));
   expect(crossprod(r.U.subset([], ind)).apply(Math.abs, 0)).to.be.deep.almost.equal(Matrix.eye(A));

   // check that T = US
   const T = X.dot(r.V);
   const US = r.U.dot(Matrix.diagm(r.s));
   expect(T).to.be.deep.almost.equal(US);

   // check that X ≈ USV
   const R = tcrossprod(r.U.dot(Matrix.diagm(r.s)), r.V);
   expect(X.v).to.be.deep.almost.equal(R.v);
}

// create simulated dataset as a linear combination of 6 sinus curves
function simdata(m, n) {
   let x = new Float64Array(n);
   x = x.map((v, i) => i + 1.);
   const s1 = x.map(v => 9. * Math.sin(v));
   const s2 = x.map(v => 8. * Math.sin(v / 2.));
   const s3 = x.map(v => 7. * Math.sin(v / 3.));
   const s4 = x.map(v => 6. * Math.sin(v / 4.));
   const s5 = x.map(v => 5. * Math.sin(v / 5.));
   const s6 = x.map(v => 4. * Math.sin(v / 6.));
   const S = new Matrix(new Float64Array([...s1, ...s2, ...s3, ...s4, ...s5, ...s6]), n, 6);

   const C = Matrix.rand(m, 6);
   return tcrossprod(C, S);
}


describe('Tests for matrix decompositions.', function () {

   it('tests for method "rsvd".', function() {

      // simple example to compare with R results
      const A0a = matrix([1, 3, 17, 19, 10, 14, 7, 13, 9, 11, 2, 15, 8, 6, 4, 12, 22, 18], 6, 3);
      const r0a = rsvd(A0a, 3);
      svdtests(A0a, r0a)

      expect(r0a.s).to.be.deep.almost.equal(vector([
         47.82344, 15.49189, 12.07974
      ]));

      expect(r0a.V.t().apply(x => Math.abs(x), 0).v).to.be.deep.almost.equal(new Float64Array([
         0.6074766,  0.4602873,  0.64738,
         0.4757956,  0.4417727,  0.76056,
         0.6360745,  0.7700471,  0.04936
      ]));

      expect(r0a.U.t().apply(x => Math.abs(x), 0).v).to.be.deep.almost.equal(new Float64Array([
         0.1887493, 0.16832558, 0.4198321,
         0.2472474, 0.16160874, 0.6822448,
         0.3586852, 0.56291730, 0.3280726,
         0.5103920, 0.28172128, 0.2766423,
         0.4395333, 0.73939460, 0.3200999,
         0.5664784, 0.05100958, 0.2676873
      ]));

      // simple example to compare with R results
      const A0b = matrix([1, 3, 17, 19, 10, 14, 7, 13, 9, 11, 2, 15, 8, 6, 4, 12, 22, 18, 11, 12], 5, 4);
      const r0b = rsvd(A0b, 4);
      svdtests(A0b, r0b)

      expect(r0b.s).to.be.deep.almost.equal(vector([
         50.4985905, 16.4651258, 10.7821742, 0.7326014
      ]));

      // simple example to compare with R results
      const A0c = reshape(Vector.seq(1, 20), 5, 4);
      const r0c = rsvd(A0c, 4);
      svdtests(A0c, r0c)

      expect(r0c.s).to.be.deep.almost.equal(vector([
         53.49077, 2.955910, 0, 0
      ]));

      // simulated data m > n
      const A1 = simdata(1000, 100)
      const r1a = rsvd(A1, 70)
      svdtests(A1, r1a, 70)
      const r1b = rsvd(A1, 10)
      svdtests(A1, r1b, 10)

      // simulated data m < n
      const A2 = simdata(100, 1000)
      const r2a = rsvd(A2, 70)
      svdtests(A2, r2a, 70)
      const r2b = rsvd(A2, 10)
      svdtests(A2, r2b, 10)

      // simulated data m = n
      const A3 = simdata(100, 100)
      const r3a = rsvd(A3, 70)
      svdtests(A3, r3a, 70)
      const r3b = rsvd(A3, 10)
      svdtests(A3, r3b, 10)

   }).timeout(200000);

   it('tests for method "qr.', function() {

      // test function, checks that:
      // - Q and R has proper dimensions
      // - Q has orthogonal columns
      // - R is upper triangular
      // - X ≈ QR
      function qrtests(X, r) {

         const m = X.nrows;
         const n = X.ncols;

         if (m > n) {
            testMatrixStructure(r.Q, m, n)
            testMatrixStructure(r.R, n, n)
            expect(crossprod(r.Q).v).to.be.deep.almost.equal(Matrix.eye(n).v);
            expect(r.R.isuppertriangular()).to.be.true;
         } else {
            testMatrixStructure(r.Q, m, m)
            testMatrixStructure(r.R, m, n)
            expect(crossprod(r.Q).v).to.be.deep.almost.equal(Matrix.eye(m).v);
         }
         expect(ssqdiff(X, r.Q.dot(r.R)) < ZERO).to.be.true;
      }

      // small squared matrix
      const X1 = matrix([4, 6, 3, 3], 2, 2);
      const r1 = qr(X1);
      qrtests(X1, r1);

      // small matrix m > n
      const X2a = matrix([4, 5, 6, 3, 3, 3], 3, 2);
      const r2a = qr(X2a);
      qrtests(X2a, r2a);

      // small matrix n > m
      const X2b = matrix([4, 5, 6, 3, 3, 3], 2, 3);
      const r2b = qr(X2b);
      qrtests(X2b, r2b);

      // small matrix from rsvd example
      const X2c = matrix([
         20.248701174781164,    22.01952684298161,
         23.790352511182057,     25.5611781793825,
         27.332003847582946,  -1.9975236505272767,
         -1.0679127357664608, -0.13830182100564326,
         0.7913090937551726,   1.7209200085159875,
         18.675243127797636,   20.372992503051968,
            22.0707418783063,   23.768491253560633,
         25.466240628814965,    7.761140001162658,
         7.8419852095081035,    7.922830417853547,
            8.00367562619899,    8.084520834544435
      ], 5, 4);

      const r2c = qr(X2c);
      qrtests(X2c, r2c);


      // small matrix from paper describing algoritm
      // https://www.math.usm.edu/lambers/mat610/sum10/lecture9.pdf
      const X2d = matrix([
         0.8147, 0.9058, 0.1270, 0.9134, 0.6324,
         0.0975, 0.2785, 0.5469, 0.9575, 0.9649,
         0.1576, 0.9706, 0.9572, 0.4854, 0.8003
      ], 5, 3)

      const r2d = qr(X2d);
      qrtests(X2a, r2a);

      expect(r2d.Q.v).to.be.deep.almost.eql(new Float64Array([
          0.4927,  0.5478,  0.0768,  0.5524,  0.3824,
         -0.4807, -0.3583,  0.4754,  0.3391,  0.5473,
          0.1780, -0.5777, -0.6343,  0.4808,  0.0311
         // -0.7033,  0.4825, -0.4317,  0.2769, -0.0983,
         //  0.0000,  0.0706, -0.4235, -0.5216,  0.7373
      ]));

      expect(r2d.R.v).to.be.deep.almost(new Float64Array([
          1.6536, 0.0000,  0.0000,
          1.1405, 0.9661,  0.0000,
          1.2569, 0.6341, -0.8816
      ]));


      // middle size random matrix m = n
      const n3 = 150
      const m3 = 150
      const X3 = Matrix.rand(m3, n3);
      const r3 = qr(X3);
      qrtests(X3, r3)

      // large size random matrix m > n
      const n4 = 500
      const m4 = 1000
      const X4 = Matrix.rand(m4, n4);
      const r4 = qr(X4);
      qrtests(X4, r4)

      // large size random matrix m < n
      const n5 = 1000
      const m5 = 500
      const X5 = Matrix.rand(m5, n5);
      const r5 = qr(X5);
      qrtests(X5, r5)

   }).timeout(20000);

   it('tests for method "lu".', function() {
      const X1 = matrix([4, 6, 3, 3], 2, 2);
      const r1 = lu(X1);
      // check that U is upper triangular
      expect(r1.U.isuppertriangular()).to.be.true;
      // check that L is lower triangular
      expect(r1.L.islowertriangular()).to.be.true;
      // check that X ≈ LU
      expect(ssqdiff(X1, r1.L.dot(r1.U)) < ZERO).to.be.true;

      const X2 = matrix([8, 4, 6, 2, 9, 7, 9, 4, 9], 3, 3);
      const r2 = lu(X2);
      // check that U is upper triangular
      expect(r2.U.isuppertriangular()).to.be.true;
      // check that L is lower triangular
      expect(r2.L.islowertriangular()).to.be.true;
      // check that X ≈ LU
      expect(ssqdiff(X2, r2.L.dot(r2.U)) < ZERO).to.be.true;

      const X3 = Matrix.rand(1000, 1000);
      const r3 = lu(X3);
      // check that U is upper triangular
      expect(r3.U.isuppertriangular()).to.be.true;
      // check that L is lower triangular
      expect(r3.L.islowertriangular()).to.be.true;
      // check that X ≈ LU
      expect(ssqdiff(X3, r3.L.dot(r3.U)) < ZERO).to.be.true;

      const X4 = Matrix.rand(1000, 500);
      const r4 = lu(X4);
      // check that X ≈ LU
      expect(ssqdiff(X4, r4.L.dot(r4.U)) < ZERO).to.be.true;

      const X5 = Matrix.rand(500, 1000);
      const r5 = lu(X5);
      // check that X ≈ LU
      expect(ssqdiff(X5, r5.L.dot(r5.U)) < ZERO).to.be.true;

   }).timeout(20000);

   it(' tests for method "svd".', function() {

      // simple example to compare with R results
      const A0a = matrix([1, 3, 17, 19, 10, 14, 7, 13, 9, 11, 2, 15, 8, 6, 4, 12, 22, 18], 6, 3);
      const r0a = svd(A0a);
      svdtests(A0a, r0a)

      expect(r0a.s).to.be.deep.almost.equal(vector([
         47.82344, 15.49189, 12.07974
      ]));

      expect(r0a.V.t().apply(x => Math.abs(x), 0).v).to.be.deep.almost.equal(new Float64Array([
         0.6074766,  0.4602873,  0.64738,
         0.4757956,  0.4417727,  0.76056,
         0.6360745,  0.7700471,  0.04936
      ]));

      expect(r0a.U.t().apply(x => Math.abs(x), 0).v).to.be.deep.almost.equal(new Float64Array([
         0.1887493, 0.16832558, 0.4198321,
         0.2472474, 0.16160874, 0.6822448,
         0.3586852, 0.56291730, 0.3280726,
         0.5103920, 0.28172128, 0.2766423,
         0.4395333, 0.73939460, 0.3200999,
         0.5664784, 0.05100958, 0.2676873
      ]));

      // simple example to compare with R results
      const A0b = matrix([1, 3, 17, 19, 10, 14, 7, 13, 9, 11, 2, 15, 8, 6, 4, 12, 22, 18, 11, 12], 5, 4);
      const r0b = svd(A0b);
      svdtests(A0b, r0b)

      expect(r0b.s).to.be.deep.almost.equal(vector([
         50.4985905, 16.4651258, 10.7821742, 0.7326014
      ]));

      // simple example to compare with R results
      const A0c = reshape(Vector.seq(1, 20), 5, 4);
      const r0c = svd(A0c);
      svdtests(A0c, r0c)

      expect(r0c.s).to.be.deep.almost.equal(vector([
         53.49077, 2.955910, 0, 0
      ]));

      // simulated data m > n
      const A1 = simdata(200, 100)
      const r1a = svd(A1)
      svdtests(A1, r1a)
      const r1b = svd(A1, 10)
      svdtests(A1, r1b, 10)

      // simulated data m < n
      const A2 = simdata(100, 200)
      const r2a = svd(A2)
      svdtests(A2, r2a)
      const r2b = svd(A2, 10)
      svdtests(A2, r2b, 10)

      // simulated data m = n
      const A3 = simdata(200, 200)
      const r3a = svd(A3)
      svdtests(A3, r3a)
      const r3b = svd(A3, 10)
      svdtests(A3, r3b, 10)

   }).timeout(10000000);

   it(' tests for method "rot".', function () {
      expect(rot(1, 0)).to.be.deep.almost([1, 0, 1]);

      const r1 = rot(1, 2);
      const r12 = matrix([r1[0], -r1[1], r1[1], r1[0]], 2, 2).dot(matrix([1, 2], 2, 1));
      expect(r12.v).to.be.deep.almost(new Float64Array([r1[2], 0]));

      const r2 = rot(0, 1);
      const r22 = matrix([r2[0], -r2[1], r2[1], r2[0]], 2, 2).dot(matrix([0, 1], 2, 1));
      expect(r22.v).to.be.deep.almost(new Float64Array([r2[2], 0]));
   });

   it ('tests for method "bidiag"', function () {
      const A1 = reshape(Vector.seq(1, 12), 4, 3);
      const [B1, V1, U1] = bidiag(A1);
      const R1 = U1.dot(tcrossprod(B1, V1));
      expect(R1.v).to.be.deep.almost.equal(A1.v);

      // best time so far 1.42 for 201 x 200
      const A2 = Matrix.rand(201, 200);
      const [B2, V2, U2] = bidiag(A2);
      const R2 = U2.dot(tcrossprod(B2, V2));
      expect(R2.v).to.be.deep.almost.equal(A2.v);

   }).timeout(20000);

   it ('tests for method "householder"', function() {
      const H1 = householder([1, 2, 3], 2);
      testMatrixStructure(H1, 2, 2, [
         0.5547002, 0.8320503,
         0.8320503, -0.5547002
      ]);

      const H2 = householder([1, 2, 3], 1);
      testMatrixStructure(H2, 3, 3, [
         0.2672612,  0.5345225,  0.8017837,
         0.5345225,  0.6100735, -0.5848898,
         0.8017837, -0.5848898,  0.1226653
      ]);

      const H3 = householder(Vector.seq(1, 10000).v, 5000);
      testMatrixStructure(H3, 5001, 5001);
   });

});