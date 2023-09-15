/****************************************************************/
/*  Tests for array methods (Index/Vector/Matrix classes)       */
/****************************************************************/


// Speed tests for operations with two 10.000 x 10.000 matrices:
// ---------------------------------------------------------------
// sum - done in: 0.315 s.
// difference - done in: 1.244 s.
// division - done in: 1.276 s.
// multiplication - done in: 1.268 s.

// Speed tests for operations with two 2.000 x 2.000 matrices:
// -------------------------------------------------------------
// tcrossprod - done in: 10.69 s.
// crossprod - done in: 9.536 s.
// dot - done in: 9.25 s.

// Speed tests for decomposition of 200 x 200 matrix:
// ----------------------------------------------------
// bidiag - done in: 1.967 s.
// svd - done in: 342.243 s.
// rsvd - done in: 3.082 s.
// qr - done in: 0.053 s.
// lu - done in: 0.023 s.

// Speed tests for modelling with 500 x 1000 matrix:
// ---------------------------------------------------
// pcafit - done in: 16.385 s.
// pcapredict - done in: 0.049 s.
// pcrfit - done in: 10.749 s.
// pcrpredict - done in: 0.055 s.
// plsfit - done in: 3.486 s.
// plspredict - done in: 0.036 s.

// import  methods
import { Matrix, tcrossprod, crossprod } from '../src/arrays/index.js';
import { bidiag, rsvd, svd, qr, lu } from '../src/decomp/index.js';
import { pcafit, pcapredict, plsfit, plspredict, pcrfit, pcrpredict } from '../src/models/index.js';

function measure(f, msg = '') {
   const start = Date.now();
   const out = f()
   console.log(msg + ' - done in: ' + (Date.now() - start) / 1000 + ' s.')
   return out;
}

const X1 = Matrix.rand(10000, 10000);
const Y1 = Matrix.rand(10000, 10000);
const X2 = Matrix.rand(2000, 2000);
const Y2 = Matrix.rand(2000, 2000);
const X3 = Matrix.rand(200, 200);
const X4 = Matrix.rand(500, 1000);
const Y4 = Matrix.rand(500, 1);

console.log('\nSpeed tests for operations with two 10.000 x 10.000 matrices:');
console.log('---------------------------------------------------------------');
measure( () => X1.add(Y1), 'sum');
measure( () => X1.subtract(Y1), 'difference');
measure( () => X1.divide(Y1), 'division');
measure( () => X1.mult(Y1), 'multiplication');
measure( () => X1.t(), 'transposition');


console.log('\nSpeed tests for operations with two 2.000 x 2.000 matrices:');
console.log('-------------------------------------------------------------');
measure( () => tcrossprod(X2, Y2), 'tcrossprod');
measure( () => crossprod(X2, Y2), 'crossprod');
measure( () => X2.dot(Y2), 'dot');

console.log('\nSpeed tests for decomposition of 2000 x 2000 matrix:');
console.log('----------------------------------------------------');
measure( () => qr(X2), 'qr');
measure( () => lu(X2), 'lu');

console.log('\nSpeed tests for decomposition of 200 x 200 matrix:');
console.log('----------------------------------------------------');
measure( () => bidiag(X3), 'bidiag');
measure( () => svd(X3, 20), 'svd');
measure( () => rsvd(X3, 20), 'rsvd');


console.log('\nSpeed tests for modelling with 500 x 1000 matrix:');
console.log('---------------------------------------------------');

const mpca = measure( () => pcafit(X4, 20), 'pcafit');
measure( () => pcapredict(mpca, X4), 'pcapredict');
const mpcr = measure( () => pcrfit(X4, Y4, 20), 'pcrfit');
measure( () => pcrpredict(mpcr, X4), 'pcrpredict');
const mpls = measure( () => plsfit(X4, Y4, 20), 'plsfit');
measure( () => plspredict(mpls, X4, Y4), 'plspredict');

console.log('');
