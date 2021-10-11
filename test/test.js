
import {min, max, sum, prod, mean, sd, quantile, skewness, kurtosis} from '../src/index.js';
import {range, mrange, split, count, mids, diff, sort, getOutliers, seq, ppoints} from '../src/index.js';
import {runif, rnorm, dnorm, dunif, pnorm, punif} from '../src/index.js';
import {rep, subset, expandGrid} from '../src/index.js';
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
   const Q1 = -1.5;        // 25% percentile
   const Q2 = 0;           // 50% percentile
   const Q3 = 1.5;         // 75% percentile
   const P10 = -2.0032;    // 10% percentile
   const P90 = 2.0004;     // 90% percentile


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


describe('Simple test for functions computing vectors with statistics.', function () {

   const x = [2.001, 2, 1, 0, -1, -2, -2.008];
   const Q1 = -1.5;        // 25% percentile for x1
   const Q2 = 0;           // 50% percentile for x1
   const Q3 = 1.5;         // 75% percentile for x1
   const P10 = -2.0032;    // 10% percentile for x1
   const P90 = 2.0004;     // 90% percentile for x1

   const x1 = [-10, -2, 0, 2, 10, 20, 50, 100, 150];
   const x2 = [150, -2, 100, 2, 50, 0, 10, 20, -10];
   const x3 = [150, 100, 50, 20, 10, 2, 0, -2, -10];

   const y = runif(1000000, 10, 20);

   it('quantile() returns correct results.', function () {

      expect(() => quantile(x, -0.01)).to.throw(Error, "Parameter 'p' must be between 0 and 1 (both included).");
      expect(() => quantile(x, 1.00001)).to.throw(Error, "Parameter 'p' must be between 0 and 1 (both included).");

      quantile(x, 0.01).should.be.a('number');
      quantile(x, 0).should.equal(min(x));
      quantile(x, 1).should.equal(max(x));

      quantile(x, 0.25).should.be.closeTo(Q1, 0.000001);
      quantile(x, 0.50).should.be.closeTo(Q2, 0.000001);
      quantile(x, 0.75).should.be.closeTo(Q3, 0.000001);
      quantile(x, 0.10).should.be.closeTo(P10, 0.000001);
      quantile(x, 0.90).should.be.closeTo(P90, 0.000001);
   });

   it('quantile() works with vectors of probabilities.', function () {
      expect(() => quantile(x, [0.2, 0.4, -0.01])).to.throw(Error, "Parameter 'p' must be between 0 and 1 (both included).");
      expect(() => quantile(x, [0.2, 1.00001, 0.4])).to.throw(Error, "Parameter 'p' must be between 0 and 1 (both included).");

      const p = quantile(x, [0.25, 0.50, 0.75, 0.10, 0.90]);
      expect(p).to.eql([Q1, Q2, Q3, P10, P90]);
   });

   it('quantile() works with large vectors (n = 1 000 000).', function () {
      const p = quantile(y, [0.25, 0.50, 0.75]);
      p[0].should.be.closeTo(12.5, 0.1);
      p[1].should.be.closeTo(15.0, 0.1);
      p[2].should.be.closeTo(17.5, 0.1);
   });

   it('range() returns correct results.', function () {
      const r = range(x1);
      r.should.be.a('Array');
      r.should.have.lengthOf(2);
      r[0].should.equal(min(x1));
      r[1].should.equal(max(x1));
   });

   it('range() works with large vectors (n = 1 000 000).', function () {
      const r = range(y);
      r.should.be.a('Array');
      r.should.have.lengthOf(2);
      r[0].should.equal(min(y));
      r[1].should.equal(max(y));
   });

   it('mrange() returns correct results.', function () {
      const dx = (max(x1) - min(x1));
      const margin1 = 0.1;
      const mr1 = mrange(x1, margin1);
      mr1.should.be.a('Array');
      mr1.should.have.lengthOf(2);
      mr1[0].should.equal(min(x1) - dx * margin1);
      mr1[1].should.equal(max(x1) + dx * margin1);

      const margin2 = 1.1;
      const mr2 = mrange(x1, margin2);
      mr2.should.be.a('Array');
      mr2.should.have.lengthOf(2);
      mr2[0].should.equal(min(x1) - dx * margin2);
      mr2[1].should.equal(max(x1) + dx * margin2);
   });

   it('split() returns correct results.', function () {
      const s1 = split(x1, 2);
      s1.should.be.a('Array');
      s1.should.have.lengthOf(3);
      s1[0].should.equal(min(x1));
      s1[1].should.equal(min(x1) + (max(x1) - min(x1)) / 2);
      s1[2].should.equal(max(x1));

      const s2 = split(x1, 4);
      s2.should.be.a('Array');
      s2.should.have.lengthOf(5);
      s2[0].should.equal(min(x1));
      s2[1].should.equal(min(x1) + (max(x1) - min(x1)) * 0.25);
      s2[2].should.equal(min(x1) + (max(x1) - min(x1)) * 0.50);
      s2[3].should.equal(min(x1) + (max(x1) - min(x1)) * 0.75);
      s2[4].should.equal(max(x1));
   });

   it('split() works with large vectors (n = 1 000 000).', function () {
      const s1 = split(y, 2);
      s1.should.be.a('Array');
      s1.should.have.lengthOf(3);
      s1[0].should.equal(min(y));
      s1[1].should.equal(min(y) + (max(y) - min(y)) / 2);
      s1[2].should.equal(max(y));

      const s2 = split(y, 100);
      s2.should.be.a('Array');
      s2.should.have.lengthOf(101);

   });

   it('count() returns correct results.', function () {
      const bins1 = [-20, 0, 200];
      const c1 = count(x1, bins1);
      c1.should.be.a('Array');
      c1.should.have.lengthOf(2);
      c1[0].should.equal(2);
      c1[1].should.equal(7);

      const bins2 = [-20, 0, 50, 200];
      const c2 = count(x1, bins2);
      c2.should.be.a('Array');
      c2.should.have.lengthOf(3);
      c2[0].should.equal(2);
      c2[1].should.equal(4);
      c2[2].should.equal(3);
   });

   it('count() works with large vectors (n = 1 000 000).', function () {
      const bins = [10, 12, 14, 16, 18, 20];
      const c = count(y, bins);
      c.should.be.a('Array');
      c.should.have.lengthOf(5);
      c[0].should.be.closeTo(200000, 2000);
      c[1].should.be.closeTo(200000, 2000);
      c[2].should.be.closeTo(200000, 2000);
      c[3].should.be.closeTo(200000, 2000);
      c[4].should.be.closeTo(200000, 2000);
   });

   it('mids() returns correct results.', function () {
      const m1 = mids(x1);
      m1.should.be.a('Array');
      m1.should.have.lengthOf(x1.length - 1);
      m1[0].should.equal(0.5 * x1[0] + 0.5 * x1[1]);
      m1[1].should.equal(0.5 * x1[1] + 0.5 * x1[2]);
      m1[7].should.equal(0.5 * x1[7] + 0.5 * x1[8]);
   });

   it('diff() returns correct results.', function () {
      const d1 = diff(x1);
      d1.should.be.a('Array');
      d1.should.have.lengthOf(x1.length - 1);
      d1[0].should.equal(x1[1] - x1[0]);
      d1[1].should.equal(x1[2] - x1[1]);
      d1[7].should.equal(x1[8] - x1[7]);
   });

   it('sort() returns correct results.', function () {
      const s1 = sort(x2);
      s1.should.be.a('Array');
      s1.should.have.lengthOf(x2.length);
      expect(s1).to.eql(x1);

      const s2 = sort(x2, true);
      s2.should.be.a('Array');
      s2.should.have.lengthOf(x2.length);
      expect(s2).to.eql(x3);
   });

   it('getOutliers() returns correct results.', function () {
      const x4 = [-100, -2, -1, 0, 1, 2, 3, 50, 100];
      const o1 = getOutliers(x4);
      expect(o1).to.eql([-100, 50, 100]);

      const o2 = getOutliers(x4, quantile(x4, 0.25));
      expect(o2).to.eql([-100, 50, 100]);

      const o3 = getOutliers(x4, quantile(x4, 0.25), quantile(x4, 0.75));
      expect(o2).to.eql([-100, 50, 100]);
   });

   it('seq() returns correct results.', function () {
      const s1 = seq(1, 10, 10);
      expect(s1).to.eql([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

      const s2 = seq(-5, 4, 10);
      expect(s2).to.eql([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]);

      const s3 = seq(0, 1, 11);
      expect(s3).to.eql([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]);
   });

   it('ppoints() works correctly.', function () {

      const p1 = ppoints(1);
      expect(p1).to.have.lengthOf(1);
      p1[0].should.be.closeTo(0.5, 0.0000001);

      const p2 = ppoints(3);
      expect(p2).to.have.lengthOf(3);
      p2[0].should.be.closeTo(0.1923077, 0.0000001);
      p2[1].should.be.closeTo(0.5000000, 0.0000001);
      p2[2].should.be.closeTo(0.8076923, 0.0000001);

      const p3 = ppoints(5);
      expect(p3).to.have.lengthOf(5);
      p3[0].should.be.closeTo(0.1190476, 0.0000001);
      p3[1].should.be.closeTo(0.3095238, 0.0000001);
      p3[2].should.be.closeTo(0.5000000, 0.0000001);
      p3[3].should.be.closeTo(0.6904762, 0.0000001);
      p3[4].should.be.closeTo(0.8809524, 0.0000001);

      const p4 = ppoints(11);
      expect(p4).to.have.lengthOf(11);
      p4[ 0].should.be.closeTo(0.04545455, 0.0000001);
      p4[ 3].should.be.closeTo(0.31818182, 0.0000001);
      p4[ 5].should.be.closeTo(0.50000000, 0.0000001);
      p4[ 7].should.be.closeTo(0.68181818, 0.0000001);
      p4[10].should.be.closeTo(0.95454545, 0.0000001);
   });
});

describe('Tests for theoretical distribution functions.', function () {

   it('runif() works correctly (n = 1 000 000).', function () {
      const n = 1000000;
      const r1 = runif(n);

      expect(r1).to.have.lengthOf(n)
      expect(min(r1)).to.be.above(0);
      expect(max(r1)).to.be.below(1);

      const r2 = runif(n, 10, 20);
      expect(r2).to.have.lengthOf(n)
      expect(min(r2)).to.be.above(10);
      expect(max(r2)).to.be.below(20);
   });

   it('rnorm() works correctly (n = 1 000 000).', function () {
      const n = 1000000;
      const r1 = rnorm(n);
      expect(r1).to.have.lengthOf(n)
      sd(r1).should.be.closeTo(1, 0.01);
      mean(r1).should.be.closeTo(0, 0.01);
      min(r1).should.be.above(-6);
      max(r1).should.be.below(6);
   });

   it('dnorm() works correctly (n = 1 000 000).', function () {
      const n = 1000000;

      // standardized distribution for ± 3 sigma
      const x1 = seq(-3, 3, n);
      const d1 = dnorm(x1);
      expect(d1).to.have.lengthOf(n);
      d1[0].should.be.closeTo(0.004431848, 0.00000001);
      d1[n-1].should.be.closeTo(0.004431848, 0.00000001);
      d1[n/2].should.be.closeTo(0.3989423, 0.0000001);

      // distribution with mu = 10 and sigma = 10, for ± 3 sigma
      const mu = 10;
      const sigma = 10
      const x2 = seq(mu - 3 * sigma, mu + 3 * sigma, n);
      const d2 = dnorm(x2, mu, sigma);
      expect(d2).to.have.lengthOf(n);
      d2[0].should.be.closeTo(0.0004431848, 0.00000001);
      d2[n-1].should.be.closeTo(0.0004431848, 0.00000001);
      d2[n/2].should.be.closeTo(0.03989423, 0.0000001);

      // distribution with mu = 10 and sigma = 10, for ± 6 sigma should have area of one
      const x3 = seq(mu - 6 * sigma, mu + 6 * sigma, n);
      const d3 = dnorm(x3, mu, sigma);
      expect(d3).to.have.lengthOf(n);
      (sum(d3) * 12 * sigma/n).should.be.closeTo(1.0, 0.00001);

      // if values are far from mean density is 0
      dnorm([mu - 6 * sigma], mu, sigma)[0].should.be.closeTo(0.0, 0.0000001);
      dnorm([mu + 6 * sigma], mu, sigma)[0].should.be.closeTo(0.0, 0.0000001);
   });


   it('dunif() works correctly (n = 1 000 000).', function () {
      const n = 1000000;

      // standardized distribution for a = 0, b = 1
      const x1 = seq(0, 1, n);
      const d1 = dunif(x1);

      expect(d1).to.have.lengthOf(n);
      d1[0].should.be.closeTo(1,   0.0000000001);
      d1[n-1].should.be.closeTo(1, 0.0000000001);
      d1[n/2].should.be.closeTo(1, 0.0000000001);

      // distribution with mu = 10 and sigma = 10, for ± 3 sigma
      const a = 10;
      const b = 100;
      const x2 = seq(a, b, n);
      const d2 = dunif(x2, a, b);

      expect(d2).to.have.lengthOf(n);
      d2[0].should.be.closeTo(1 / (b - a),   0.00000001);
      d2[n-1].should.be.closeTo(1 / (b - a), 0.00000001);
      d2[n/2].should.be.closeTo(1 / (b - a), 0.00000001);
      (sum(d2) * (b - a)/n).should.be.closeTo(1.0, 0.000001);

      dunif([a - 0.0000001], a, b)[0].should.be.closeTo(0.0, 0.0000001);
      dunif([b + 0.0000001], a, b)[0].should.be.closeTo(0.0, 0.0000001);
   });


   it('pnorm() works correctly (n = 1 000 000).', function () {
      const n = 1000000;

      // standardized distribution for ± 3 sigma
      const x1 = seq(-3, 3, n);
      const p1 = pnorm(x1);

      expect(p1).to.have.lengthOf(n);
      p1[  0].should.be.closeTo(0.00134996, 0.00001);
      p1[n-1].should.be.closeTo(0.998650, 0.00001);
      p1[n/2].should.be.closeTo(0.5, 0.00001);

     // distribution with mu = 10 and sigma = 10, for ± 3 sigma
      const mu = 10;
      const sigma = 10
      const x2 = seq(mu - 3 * sigma, mu + 3 * sigma, n);
      const p2 = pnorm(x2, mu, sigma);
      expect(p2).to.have.lengthOf(n);
      p2[  0].should.be.closeTo(0.001350, 0.000001);
      p2[n-1].should.be.closeTo(0.998650, 0.000001);
      p2[n/2].should.be.closeTo(0.5, 0.00001);

   });

   it('punif() works correctly (n = 1 000 000).', function () {
      const n = 1000000;

      // standardized distribution for a = 0, b = 1
      const x1 = seq(0, 1, n);
      const p1 = punif(x1);

      expect(p1).to.have.lengthOf(n);
      p1[0].should.be.closeTo(0, 0.00001);
      p1[n-1].should.be.closeTo(1, 0.00001);
      p1[n/2].should.be.closeTo(0.5, 0.00001);

      // outside the range
      punif([-1])[0].should.be.closeTo(0, 0.00001);
      punif([ 1])[0].should.be.closeTo(1, 0.00001);

      // distribution with mu = 10 and sigma = 10, for ± 3 sigma
      const a = 10;
      const b = 100;
      const x2 = seq(a, b, n);
      const p2 = punif(x2, a, b);

      expect(p2).to.have.lengthOf(n);
      punif([a - 10], a, b)[0].should.be.closeTo(0.0, 0.00001);
      punif([b + 10], a, b)[0].should.be.closeTo(1.0, 0.00001);
   });

});

describe('Tests for replicate functions.', function () {

   it('rep() works correctly with one value.', function () {

      const y1 = rep(10, 3);
      y1.should.be.a('Array');
      expect(y1).to.have.lengthOf(3);
      expect(y1).to.eql([10, 10, 10]);

      const y2 = rep(20, 1);
      y2.should.be.a('Array');
      expect(y2).to.have.lengthOf(1);
      expect(y2).to.eql([20]);

      const y3 = rep(30, 1000000);
      y3.should.be.a('Array');
      expect(y3).to.have.lengthOf(1000000);
      expect(y3[1]).to.eql(30);
      expect(y3[100]).to.eql(30);
      expect(y3[10000]).to.eql(30);
      expect(y3[100000]).to.eql(30);

   });

   it('rep() works correctly with vector of numbers.', function () {

      const y1 = rep([10, 20], 3);
      y1.should.be.a('Array');
      expect(y1).to.have.lengthOf(6);
      expect(y1).to.eql([10, 20, 10, 20, 10, 20]);

      const y2 = rep([10, 20], 1);
      y2.should.be.a('Array');
      expect(y2).to.have.lengthOf(2);
      expect(y2).to.eql([10, 20]);

      const y3 = rep([30, 40], 1000000);
      y3.should.be.a('Array');
      expect(y3).to.have.lengthOf(2000000);
      expect(y3[0]).to.eql(30);
      expect(y3[1]).to.eql(40);
      expect(y3[100]).to.eql(30);
      expect(y3[101]).to.eql(40);
      expect(y3[10000]).to.eql(30);
      expect(y3[10001]).to.eql(40);
      expect(y3[100000]).to.eql(30);
      expect(y3[100001]).to.eql(40);
   });


   it('rep() works correctly with vector of numbers and vector of times.', function () {
      expect(() => rep([10, 20, 30], [10, 20])).to.throw(Error, "Parameter 'n' should be a single value or a vector of the same length as x.")

      const z1 = rep([10, 20, 30], [1, 2, 3])
      expect(z1).to.eql([10, 20, 20, 30, 30, 30]);

      const z2 = rep([10, 20, 30], [3, 2, 1])
      expect(z2).to.eql([10, 10, 10, 20, 20, 30]);

      const z3 = rep([10, 20, 30], [3, 3, 3])
      expect(z3).to.eql([10, 10, 10, 20, 20, 20, 30, 30, 30]);

   });
});

describe('Tests for subset() function.', function () {
   const x = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];

   it('subset() works correctly with one index.', function () {

      // wrong index
      expect(() => subset(x, 0)).to.throw(Error, "Parameter 'indices' must have values between 1 and 'x.length'.");
      expect(() => subset(x, 11)).to.throw(Error, "Parameter 'indices' must have values between 1 and 'x.length'.");

      // index as a vector
      expect(subset(x, [1])).to.eql([10]);
      expect(subset(x, [3])).to.eql([30]);
      expect(subset(x, [10])).to.eql([100]);

      // index as a value
      expect(subset(x, 1)).to.eql([10]);
      expect(subset(x, 3)).to.eql([30]);
      expect(subset(x, 10)).to.eql([100]);
   })

   it('subset() works correctly with vector of indices.', function () {
      // wrong index
      expect(() => subset(x, [0, 1, 2])).to.throw(Error, "Parameter 'indices' must have values between 1 and 'x.length'.");
      expect(() => subset(x, [8, 9, 11])).to.throw(Error, "Parameter 'indices' must have values between 1 and 'x.length'.");

      // index as a vector
      expect(subset(x, [1, 2, 3])).to.eql([10, 20, 30]);
      expect(subset(x, [3, 7, 9])).to.eql([30, 70, 90]);
      expect(subset(x, [10, 9, 8, 7, 6, 1])).to.eql([100, 90, 80, 70, 60, 10]);
   });


   it('subset() works correctly with repeated indices.', function () {
      expect(subset(x, [1, 1, 2, 2, 3, 3])).to.eql([10, 10, 20, 20, 30, 30]);
   });

   const y = sort(runif(1000000, 10, 20));
   it('subset() works correctly with large vectors (n = 1 000 000).', function () {
      const z = subset(y, [1, 100, 1000, 10000, 100000, 1000000]);
      z[0].should.be.closeTo(10.0, 0.01);
      z[5].should.be.closeTo(20.0, 0.01);
   })
});

describe('Tests for expandGrid() function.', function () {
   it('expandGrid() works correctly with numeric vectors.', function () {

      const z1 = expandGrid([100, 200, 300], [10, 20]);
      z1.should.be.a('Array');
      expect(z1).to.have.lengthOf(2);
      expect(z1[0]).to.eql([100, 200, 300, 100, 200, 300]);
      expect(z1[1]).to.eql([10, 10, 10, 20, 20, 20]);

      const z2 = expandGrid([100, 200, 300], [10, 20], [-20, +20]);
      z2.should.be.a('Array');
      expect(z2).to.have.lengthOf(3);
      expect(z2[0]).to.eql([100, 200, 300, 100, 200, 300, 100, 200, 300, 100, 200, 300]);
      expect(z2[1]).to.eql([10, 10, 10, 20, 20, 20, 10, 10, 10, 20, 20, 20]);
      expect(z2[2]).to.eql([-20, -20, -20, -20, -20, -20, 20, 20, 20, 20, 20, 20]);

   })

   it('expandGrid() works correctly with text vectors.', function () {
      const z1 = expandGrid([100, 200, 300], ["A", "B"]);
      z1.should.be.a('Array');
      expect(z1).to.have.lengthOf(2);
      expect(z1[0]).to.eql([100, 200, 300, 100, 200, 300]);
      expect(z1[1]).to.eql(["A", "A", "A", "B", "B", "B"]);
   })

});
