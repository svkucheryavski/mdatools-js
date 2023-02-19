/*****************************************************************
 *  Tests for preprocessing methods                              *
 *****************************************************************/

// import dependencies
import { default as chai} from 'chai';
import { default as chaiAlmost} from 'chai-almost';
import { vector, ismatrix, Vector, Matrix } from '../arrays/index.js';
import { sd, mean } from '../stat/index.js';

// import methods to test
import { scale, unscale } from '../prep/index.js';


// set up test settings
const expect = chai.expect;
chai.use(chaiAlmost(0.0001));

describe('Tests of preprocessing methods.', function () {

   it ('tests for method "scale" for matrices', function () {

      const X1 = Matrix.rand(500, 10, 0, 2);
      const mX1 = X1.apply(mean, 2);
      const sX1 = X1.apply(sd, 2);

      // short version
      const X1p11 = scale(X1);
      const X1p12 = scale(X1, true, false);
      const X1p13 = scale(X1, true, true);
      const X1p14 = scale(X1, false, true);

      expect(X1p11.v).to.be.deep.almost.equal(X1p12.v);
      expect(X1p12.apply(mean, 2)).to.be.deep.almost.equal(Vector.zeros(10));
      expect(X1p12.apply(sd, 2)).to.be.deep.almost.equal(sX1);
      expect(X1p13.apply(mean, 2)).to.be.deep.almost.equal(Vector.zeros(10));
      expect(X1p13.apply(sd, 2)).to.be.deep.almost.equal(Vector.ones(10));
      expect(X1p14.apply(mean, 2)).to.be.deep.almost.equal(mX1.divide(sX1));
      expect(X1p14.apply(sd, 2)).to.be.deep.almost.equal(Vector.ones(10));

      // full version
      const [X1p22, mp22, sp22] = scale(X1,  true, false, true);
      const [X1p23, mp23, sp23] = scale(X1,  true,  true, true);
      const [X1p24, mp24, sp24] = scale(X1, false,  true, true);

      expect(ismatrix(X1p22))
      expect(mp22).to.be.deep.almost.equal(mX1);
      expect(sp22).to.be.deep.almost.equal(Vector.ones(10));
      expect(mp23).to.be.deep.almost.equal(mX1);
      expect(sp23).to.be.deep.almost.equal(sX1);
      expect(mp24).to.be.deep.almost.equal(Vector.zeros(10));
      expect(sp24).to.be.deep.almost.equal(sX1);
   });

   it ('tests for method "unscale" for matrices', function () {

      const X1 = Matrix.rand(500, 5, 0, 2);
      const [X1s, mX1, sX1] = scale(X1, true, true, true);
      const X1u = unscale(X1s, mX1, sX1);
      expect(X1u.v).to.be.deep.almost.equal(X1.v);
      expect(X1s.v).to.be.deep.almost.not.equal(X1.v);

      const mX2 = vector([1, 2, 3, 4, 5]);
      const sX2 = vector([0.5, 0.4, 0.3, 0.2, 0.1]);
      const X2 = Matrix.rand(500, 5, 0, 2);
      const X2s = scale(X2, mX2, sX2);
      const X2u = unscale(X2s, mX2, sX2);
      expect(X2u.v).to.be.deep.almost.equal(X2.v);
      expect(X2s.v).to.be.deep.almost.not.equal(X2.v);

   });
});

