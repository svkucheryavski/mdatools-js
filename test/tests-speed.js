/****************************************************************/
/*  Tests for array methods (Index/Vector/Matrix classes)       */
/****************************************************************/

// import  methods
import { Matrix, tcrossprod, crossprod } from '../arrays/index.js';
import { bidiag, rsvd, svd, qr, lu } from '../decomp/index.js';
import { pcafit, pcapredict, plsfit, plspredict, pcrfit, pcrpredict } from '../models/index.js';
import { pcvpca, pcvpcr, pcvpls } from '../pcv/index.js';

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


console.log('\nSpeed tests for operations with two 2.000 x 2.000 matrices:');
console.log('-------------------------------------------------------------');
measure( () => tcrossprod(X2, Y2), 'tcrossprod');
measure( () => crossprod(X2, Y2), 'crossprod');
measure( () => X2.dot(Y2), 'dot');


console.log('\nSpeed tests for decomposition of 200 x 200 matrix:');
console.log('----------------------------------------------------');
measure( () => bidiag(X3), 'bidiag');
measure( () => svd(X3, 20), 'svd');
measure( () => rsvd(X3, 20), 'rsvd');
measure( () => qr(X3), 'qr');
measure( () => lu(X3), 'lu');


console.log('\nSpeed tests for modelling with 500 x 1000 matrix:');
console.log('---------------------------------------------------');

const mpca = measure( () => pcafit(X4, 20), 'pcafit');
measure( () => pcapredict(mpca, X4), 'pcapredict');
measure( () => pcvpca(X4, mpca, 20, {type: 'ven', nseg: 10}), 'pcvpca');
const mpcr = measure( () => pcrfit(X4, Y4, 20), 'pcrfit');
measure( () => pcrpredict(mpcr, X4), 'pcrpredict');
measure( () => pcvpcr(X4, Y4, mpcr, 20, {type: 'ven', nseg: 10}), 'pcvpcr');
const mpls = measure( () => plsfit(X4, Y4, 20), 'plsfit');
measure( () => plspredict(mpls, X4, Y4), 'plspredict');
measure( () => pcvpls(X4, Y4, mpls, 20, {type: 'ven', nseg: 10}), 'pcvpls');

console.log('');