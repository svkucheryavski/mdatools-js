

// import dependencies
import {default as chai} from 'chai';
import {default as chaiAlmost} from 'chai-almost';
import { quantile } from 'mdatools/stat';
import { crossprod, tcrossprod, reshape, isvector, vector, Vector,
   ismatrix, matrix, Matrix, Index } from '../src/arrays/index.js';

// import methods to test
import { rsvd, qr, lu, svd, rot, bidiag, householder } from '../src/decomp/index.js';
import { pchisq, qchisq, gamma, igamma } from '../src/distributions/index.js';

// set up test settings
const expect = chai.expect;
chai.use(chaiAlmost(0.0001));

const ZERO = Math.pow(10.0, -6);

describe('Temporary tests.', function () {

   it ('tests for method "bidiag"', function () {
   }).timeout(20000);



});