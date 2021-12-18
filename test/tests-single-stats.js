/******************************************************************
 *  Tests for functions which return statistic as a single value  *
 ******************************************************************/

// import of functions to test
import {min, max, sum, prod, mean, sd, skewness, kurtosis, cov, cor} from '../stat/index.js';

// import dependencies
import {runif} from '../stat/index.js';
import {default as chai} from 'chai';

const should = chai.should();
const expect = chai.expect;

describe('Simple test for functions computing single statistic.', function () {

   // mix of negative and positive numbers - main dataset for tests
   const mn1 = -2.008;     // min
   const mx1 =  2.001;     // max
   const sm1 = -0.007;     // sum
   const av1 = -0.001;     // average/mean
   const sd1 = 1.733785;   // std (/n-1)
   const sd1b = 1.605173;  // std biased (/n)


   const x1 = [mx1, 2, 1, 0, -1, -2, mn1];

   // only negative  numbers
   const mx2 = -0.001;
   const mn2 =  -2.003;
   const sm2 = -5.004;
   const av2 = -1.251;
   const x2 = [mx2, -1, -2, mn2];

   // only positive  numbers
   const mn3 = 0.001;
   const mx3 =  2.003;
   const sm3 = 5.004;
   const av3 = 1.251;
   const x3 = [mn3, 1, 2, mx3];

   it('min() returns correct results.', function () {
      min(x1).should.be.a('number');
      min(x1).should.equal(mn1);
      min(x2).should.equal(mn2);
      min(x3).should.equal(mn3);
   });

   it('max() returns correct results.', function () {
      max(x1).should.be.a('number');
      max(x1).should.equal(mx1);
      max(x2).should.equal(mx2);
      max(x3).should.equal(mx3);
   });

   it('sum() returns correct results.', function () {
      sum(x1).should.be.a('number');
      sum(x1).should.be.closeTo(sm1, 0.0000001);
      sum(x2).should.be.closeTo(sm2, 0.0000001);
      sum(x3).should.be.closeTo(sm3, 0.0000001);
   });

   it('prod() returns correct results.', function () {
      prod([1, 2, 3, 4, 5]).should.be.a('number');
      prod([1, 2, 3, 4, 5]).should.be.closeTo(120, 0.0000000001);
      prod([10]).should.be.a('number');
      prod([10]).should.be.closeTo(10, 0.0000000001);
   });

   it('mean() returns correct results.', function () {
      mean(x1).should.be.a('number');
      mean(x1).should.be.closeTo(av1, 0.0000001);
      mean(x2).should.be.closeTo(av2, 0.0000001);
      mean(x3).should.be.closeTo(av3, 0.0000001);
   });

   it('sd() returns correct results.', function () {
      sd(x1).should.be.a('number');
      sd(x1).should.be.closeTo(sd1, 0.000001);
      sd(x1, true).should.be.closeTo(sd1b, 0.000001);
      sd(x1, false, mean(x1)).should.be.closeTo(sd1, 0.000001);
   });

   it('skewness() returns correct results.', function() {
      skewness([0, 1, 2]).should.be.closeTo(0, 0.0000001);
      skewness([-10, 1, 2]).should.be.closeTo(-0.6892055, 0.0000001);
      skewness([10, 1, 2]).should.be.closeTo(0.6745555, 0.0000001);
   })

   it('kurtosis() returns correct results.', function() {
      kurtosis([1, 2, 3]).should.be.closeTo(1.5, 0.0000001);
   })

   it('cov() returns correct results.', function() {
      expect(() => cov([1, 2], [1, 2, 3]).to.throw(Error, "Vectors 'x' and 'y' must have the same length."));
      expect(() => cov([1], [1]).to.throw(Error, "Vectors 'x' and 'y' must have at least two values."));

      cov([1, 2, 3], [1, 2, 3]).should.equal(1);
      cov([1, 2, 3], [3, 2, 1]).should.equal(-1);

      cov([1, 2, 3], [10, 20, 30]).should.equal(10);
      cov([1, 2, 3], [30, 20, 10]).should.equal(-10);

      cov([1, 2, 1, 2], [10, 10, 20, 20]).should.equal(0);
   })

   it('cor() returns correct results.', function() {
      cor([1, 2, 3], [1, 2, 3]).should.equal(1);
      cor([1, 2, 3], [3, 2, 1]).should.equal(-1);

      cor([1, 2, 3], [10, 20, 30]).should.equal(1);
      cor([1, 2, 3], [30, 20, 10]).should.equal(-1);

      cor([1, 2, 1, 2], [10, 10, 20, 20]).should.equal(0);
   })
});

describe('Functions computing single statistic work with n = 1 000 000.', function () {
   const xxl = runif(1000000, 10, 20);

   it('min() is fast enough.', function() {
      min(xxl).should.be.closeTo(10, 0.01);
   })

   it('max() is fast enough.', function() {
      max(xxl).should.be.closeTo(20, 0.01);
   })

   it('mean() is fast enough.', function() {
      mean(xxl).should.be.closeTo(15, 0.01);
   })

   it('skewness() is fast enough.', function() {
      skewness(xxl).should.be.closeTo(0, 0.01);
   })

   it('kurtosis() is fast enough.', function() {
      kurtosis(xxl).should.be.closeTo(1.8, 0.01);
   })

})


