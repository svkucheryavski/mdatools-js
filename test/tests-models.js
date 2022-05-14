/******************************************************************
 *  Tests for modelling methods                                   *
 ******************************************************************/

// import of functions to test
import {lmfit, lmpredict, regstat, polyfit, polypredict} from '../models/index.js';

// import dependencies
import {default as chai} from 'chai';
import {default as chaiAlmost} from 'chai-almost';
const should = chai.should();
const expect = chai.expect;

chai.use(chaiAlmost(0.001));

describe('Tests for polynomial regression.', function () {

   it('polyfit() works correctly.', function () {
      const x = [1, 2, 3, 4, 5];
      const y = [11, 14, 19, 26, 35];

      expect(() => polyfit(x, y, 5)).to.throw(Error, "Polynomial degree 'd' must a positive value smaller than number of measurements.");
      expect(() => polyfit([x, x], y, 1)).to.throw(Error, "Argument 'x' must be a vector.");

      const m1 = polyfit(x, y, 3);

      // check class and polynomial degree
      m1.class.should.be.equal("pm");
      m1.pdegree.should.be.equal(3);

      // check coefficients and related statistics
      m1.coeffs.estimate.length.should.be.equal(4);
      expect(m1.coeffs.estimate).to.be.deep.almost([10, 0, 1, 0]);
      expect(m1.coeffs.p).to.be.deep.almost([0, 0.912, 0, 0.931]);
   });

   it('polypredict() works correctly.', function () {
      const x = [1, 2, 3, 4, 5];
      const y = [11, 14, 19, 26, 35];
      const m = polyfit(x, y, 3);

      const yp = polypredict(m, x);
      expect(yp).to.be.deep.almost(y);

   });
});

describe('Tests for linear regression.', function () {


   it('lmfit() raises errors correctly.', function () {
      expect(() => lmfit(1, 2)).to.throw(Error, "Argument 'X' must be a matrix or a vector.");
      expect(() => lmfit([1], 2)).to.throw(Error, "Argument 'y' must be a vector.");
      expect(() => lmfit([1], [2])).to.throw(Error, "Number of measurements must be larger than number of predictors.");
      expect(() => lmfit([1, 2], [2, 3, 4])).to.throw(Error, "Arguments 'X' and 'y' must have the same number of measurements.");
   });


   it('lmfit() works correctly for single predictor.', function () {
      const X = [1, 2, 3, 4, 5];
      const y =[10, 19, 31, 39, 55];
      const m = lmfit(X, y);

      m.class.should.be.equal("lm");

      // check coefficients and related statistics
      m.coeffs.estimate.length.should.be.equal(2);
      expect(m.coeffs.estimate).to.be.deep.almost([-2.2, 11.0]);
      expect(m.coeffs.se).to.be.deep.almost([2.3295, 0.7024]);
      expect(m.coeffs.tstat).to.be.deep.almost([-0.944, 15.661]);
      expect(m.coeffs.p).to.be.deep.almost([0.4146, 0.000566]);

      // check predictions
      m.fitted.length.should.be.equal(y.length);
      expect(m.fitted).to.be.deep.almost([8.8, 19.8, 30.8, 41.8, 52.8]);

      // check performance statistics
      m.stat.DoF.should.be.equal(3);
      expect(m.stat.se).to.be.almost(2.221);
      expect(m.stat.R2).to.be.almost(0.9879);
      expect(m.stat.R2adj).to.be.almost(0.9839);
      expect(m.stat.Fstat).to.be.almost(245.270);
      expect(m.stat.p).to.be.almost(0.0005658);

   });


   it('lmfit() works correctly for multiple predictors.', function () {
      const X = [[1, 2, 3, 4, 5], [10, 20, 10, 20, 10]];
      const y =[10, 19, 31, 39, 55];
      const m = lmfit(X, y);

      m.class.should.be.equal("lm");

      // check coefficients and related statistics
      m.coeffs.estimate.length.should.be.equal(3);
      expect(m.coeffs.estimate).to.be.deep.almost([2.0, 11.0, -0.3]);
      expect(m.coeffs.se).to.be.deep.almost([2.3381, 0.4472, 0.1291]);
      expect(m.coeffs.tstat).to.be.deep.almost([0.855, 24.597, -2.324]);
      expect(m.coeffs.p).to.be.deep.almost([0.48245, 0.00165, 0.14576]);

      // check predictions
      m.fitted.length.should.be.equal(y.length);
      expect(m.fitted).to.be.deep.almost([10, 18, 32, 40, 54]);

      // check performance statistics
      m.stat.DoF.should.be.equal(2);
      expect(m.stat.se).to.be.almost(1.414);
      expect(m.stat.R2).to.be.almost(0.9967);
      expect(m.stat.R2adj).to.be.almost(0.9935);
      expect(m.stat.Fstat).to.be.almost(305.2);
      expect(m.stat.p).to.be.almost(0.003266);

   });


   it('lmpredict() raises errors correctly.', function () {

      const X = [[1, 2, 3, 4, 5], [10, 20, 10, 20, 10]];
      const y = [10, 19, 31, 39, 55];
      const m = lmfit(X, y);

      expect(() => lmpredict({}, 2)).to.throw(Error, "Argument 'X' must be a matrix or a vector.");
      expect(() => lmpredict({}, [2, 3])).to.throw(Error, "Argument 'm' must be object with MLR model returned by method 'lmfit()'.");
      expect(() => lmpredict(m, [2, 3])).to.throw(Error, "Number of columns in 'X' do not match number of coefficients in model 'm'");
   })


   it('lmpredict() works correctly.', function () {

      const X1 = [1, 2, 3, 4, 5];
      const y1 =[10, 19, 31, 39, 55];
      const m1 = lmfit(X1, y1);

      const yp1 = lmpredict(m1, X1)
      yp1.should.be.eql(m1.fitted);

      const X2 = [[1, 2, 3, 4, 5], [10, 20, 10, 20, 10]];
      const y2 =[10, 19, 31, 39, 55];
      const m2 = lmfit(X2, y2);

      const yp2 = lmpredict(m2, X2)
      yp2.should.be.eql(m2.fitted);
   });

});
