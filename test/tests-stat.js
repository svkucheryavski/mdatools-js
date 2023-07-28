/****************************************************************/
/*  Tests for methods computing various statistics              */
/****************************************************************/

// import dependencies
import {default as chai} from 'chai';
import {default as chaiAlmost} from 'chai-almost';
import { isvector, vector, Vector } from '../src/arrays/index.js';

// import methods to test
import {
   norm2, quantile, count, mids, split, diff, ppoints, cumsum,
   cor, cov, skewness, kurtosis, rank, mean, sd, sum, prod, min, minind, max, maxind,
   median, iqr
} from '../src/stat/index.js';

// set up test settings
const should = chai.should();
const expect = chai.expect;
chai.use(chaiAlmost(0.00001));



describe('Tests of stat methods.', function () {

   // mix of negative and positive numbers - main dataset for tests
   const mn1 = -2.008;     // min
   const mx1 =  2.001;     // max
   const sm1 = -0.007;     // sum
   const av1 = -0.001;     // average/mean
   const sd1 = 1.733785;   // std (/n-1)
   const sd1b = 1.605173;  // std biased (/n)


   const x1 = vector([mx1, 2, 1, 0, -1, -2, mn1]);

   // only negative  numbers
   const mx2 = -0.001;
   const mn2 =  -2.003;
   const sm2 = -5.004;
   const av2 = -1.251;
   const x2 = vector([mx2, -1, -2, mn2]);

   // only positive  numbers
   const mn3 = 0.001;
   const mx3 =  2.003;
   const sm3 = 5.004;
   const av3 = 1.251;
   const x3 = vector([mn3, 1, 2, mx3]);

   // vectors for percentiles, etc.
   const xp = vector([2.001, 2, 1, 0, -1, -2, -2.008]);
   const Q1 = -1.5;        // 25% percentile for x1
   const Q2 = 0;           // 50% percentile for x1
   const Q3 = 1.5;         // 75% percentile for x1
   const P10 = -2.0032;    // 10% percentile for x1
   const P90 = 2.0004;     // 90% percentile for x1

   const xp1 = vector([-10, -2, 0, 2, 10, 20, 50, 100, 150]);
   const xp2 = vector([150, -2, 100, 2, 50, 0, 10, 20, -10]);
   const xp3 = vector([150, 100, 50, 20, 10, 2, 0, -2, -10]);

   // large random vector
   const yp = Vector.rand(1000000, 10, 20);

   it('tests for method "iqr".', function () {
      iqr(xp).should.be.a('number');
      iqr(xp).should.equal(quantile(xp, 0.75) - quantile(xp, 0.25));
   });

   it('tests for method "median".', function () {
      median(xp).should.be.a('number');
      median(xp).should.equal(quantile(xp, 0.5));
      median(yp).should.equal(quantile(yp, 0.5));
   });

   it('tests for method "norm2".', function () {
      const x1 = vector([1, 2, 3, 4]);
      const x2 = vector([1, -2, -3, 4]);
      const x3 = vector([0, 0, 0, 0]);
      const x4 = vector([1, 1]);

      expect(norm2(x1)).to.equal(Math.sqrt(30));
      expect(norm2(x2)).to.equal(Math.sqrt(30));
      expect(norm2(x3)).to.equal(0);
      expect(norm2(x4)).to.equal(Math.sqrt(2));
   });

   it('tests for method "quantile".', function () {

      expect(() => quantile(xp, -0.01)).to.throw(Error, "Parameter 'p' must be between 0 and 1 (both included).");
      expect(() => quantile(xp, 1.00001)).to.throw(Error, "Parameter 'p' must be between 0 and 1 (both included).");

      quantile(xp, 0.01).should.be.a('number');
      quantile(xp, 0).should.equal(min(xp));
      quantile(xp, 1).should.equal(max(xp));

      quantile(xp, 0.25).should.be.closeTo(Q1, 0.000001);
      quantile(xp, 0.50).should.be.closeTo(Q2, 0.000001);
      quantile(xp, 0.75).should.be.closeTo(Q3, 0.000001);
      quantile(xp, 0.10).should.be.closeTo(P10, 0.000001);
      quantile(xp, 0.90).should.be.closeTo(P90, 0.000001);

      expect(() => quantile(xp, [0.2, 0.4, -0.01])).to.throw(Error, "Parameter 'p' must be between 0 and 1 (both included).");
      expect(() => quantile(xp, [0.2, 1.00001, 0.4])).to.throw(Error, "Parameter 'p' must be between 0 and 1 (both included).");

      const p1 = quantile(xp, [0.25, 0.50, 0.75, 0.10, 0.90]);
      expect(p1).to.deep.equal(vector([Q1, Q2, Q3, P10, P90]));

      const p2 = quantile(yp, [0.25, 0.50, 0.75]);
      p2.v[0].should.be.closeTo(12.5, 0.1);
      p2.v[1].should.be.closeTo(15.0, 0.1);
      p2.v[2].should.be.closeTo(17.5, 0.1);

   });

   it('tests for method "count".', function () {

      const bins1 = [-20, 0, 200];
      const c1 = count(xp1, bins1);
      isvector(c1).should.be.true;
      c1.v.should.have.lengthOf(2);
      c1.v[0].should.equal(2);
      c1.v[1].should.equal(7);

      const bins2 = [-20, 0, 50, 200];
      const c2 = count(xp1, bins2);
      isvector(c2).should.be.true;
      c2.v.should.have.lengthOf(3);
      c2.v[0].should.equal(2);
      c2.v[1].should.equal(4);
      c2.v[2].should.equal(3);

      const bins = [10, 12, 14, 16, 18, 20];
      const c3 = count(yp, bins);
      isvector(c3).should.be.true;
      c3.v.should.have.lengthOf(5);
      c3.v[0].should.be.closeTo(200000, 2000);
      c3.v[1].should.be.closeTo(200000, 2000);
      c3.v[2].should.be.closeTo(200000, 2000);
      c3.v[3].should.be.closeTo(200000, 2000);
      c3.v[4].should.be.closeTo(200000, 2000);

   });

   it ('tests for method "mids".', function () {
      const m1 = mids(xp1);
      isvector(m1).should.be.true;
      m1.should.have.lengthOf(xp1.length - 1);
      m1.v[0].should.equal(0.5 * xp1.v[0] + 0.5 * xp1.v[1]);
      m1.v[1].should.equal(0.5 * xp1.v[1] + 0.5 * xp1.v[2]);
      m1.v[7].should.equal(0.5 * xp1.v[7] + 0.5 * xp1.v[8]);
   });

   it ('tests for method "split".', function () {
      const s1 = split(xp1, 2);
      isvector(s1).should.be.true;
      s1.v.should.have.lengthOf(3);
      s1.v[0].should.equal(min(xp1));
      s1.v[1].should.equal(min(xp1) + (max(xp1) - min(xp1)) / 2);
      s1.v[2].should.equal(max(xp1));

      const s2 = split(xp1, 4);
      isvector(s2).should.be.true;
      s2.v.should.have.lengthOf(5);
      s2.v[0].should.equal(min(xp1));
      s2.v[1].should.equal(min(xp1) + (max(xp1) - min(xp1)) * 0.25);
      s2.v[2].should.equal(min(xp1) + (max(xp1) - min(xp1)) * 0.50);
      s2.v[3].should.equal(min(xp1) + (max(xp1) - min(xp1)) * 0.75);
      s2.v[4].should.equal(max(xp1));

      const s3 = split(yp, 2);
      isvector(s1).should.be.true;
      s3.v.should.have.lengthOf(3);
      s3.v[0].should.equal(min(yp));
      s3.v[1].should.equal(min(yp) + (max(yp) - min(yp)) / 2);
      s3.v[2].should.equal(max(yp));

      const s4 = split(yp, 100);
      isvector(s4).should.be.true;
      s4.v.should.have.lengthOf(101);

   });

   it ('tests for method "diff".', function () {
      const d1 = diff(x1);

      isvector(d1).should.be.true;
      d1.v.should.have.lengthOf(x1.length - 1);
      d1.v[0].should.equal(x1.v[1] - x1.v[0]);
      d1.v[1].should.equal(x1.v[2] - x1.v[1]);
      d1.v[5].should.equal(x1.v[6] - x1.v[5]);
   });

   it ('tests for method "ppoints".', function () {

      const p1 = ppoints(1);

      expect(p1.v).to.have.lengthOf(1);
      p1.v[0].should.be.closeTo(0.5, 0.0000001);

      const p2 = ppoints(3);
      expect(p2.v).to.have.lengthOf(3);
      p2.v[0].should.be.closeTo(0.1923077, 0.0000001);
      p2.v[1].should.be.closeTo(0.5000000, 0.0000001);
      p2.v[2].should.be.closeTo(0.8076923, 0.0000001);

      const p3 = ppoints(5);
      expect(p3.v).to.have.lengthOf(5);
      p3.v[0].should.be.closeTo(0.1190476, 0.0000001);
      p3.v[1].should.be.closeTo(0.3095238, 0.0000001);
      p3.v[2].should.be.closeTo(0.5000000, 0.0000001);
      p3.v[3].should.be.closeTo(0.6904762, 0.0000001);
      p3.v[4].should.be.closeTo(0.8809524, 0.0000001);

      const p4 = ppoints(11);
      expect(p4.v).to.have.lengthOf(11);
      p4.v[ 0].should.be.closeTo(0.04545455, 0.0000001);
      p4.v[ 3].should.be.closeTo(0.31818182, 0.0000001);
      p4.v[ 5].should.be.closeTo(0.50000000, 0.0000001);
      p4.v[ 7].should.be.closeTo(0.68181818, 0.0000001);
      p4.v[10].should.be.closeTo(0.95454545, 0.0000001);
   });

   it ('tests for method "cumsum".', function () {
      const x1 = vector([1, 10, 20, 100, 300]);
      const x2 = vector([1, -10, 20, -100, 300]);

      expect(cumsum(x1)).to.be.deep.equal(vector([1, 11, 31, 131, 431]));
      expect(cumsum(x2)).to.be.deep.equal(vector([1, -9, 11, -89, 211]));
   });

   it ('tests for method "rank".', function () {
      // i:       1   2  3  4  5  6  7  8
      // sorted: -3   0  1  2  3  4  8  9
      // rank:    5   1  2  6  3  4  8  7
      const x1 = vector([3, -3, 0, 4, 1, 2, 9, 8]);
      expect(rank(x1)).to.eql(vector([5, 1, 2, 6, 3, 4, 8, 7]));

      //  i:       1   2   3   4   5   6   7
      // sorted:  -2  -2  -2   1   3  10  11
      // rank:     1   5   1   4   6   7   1
      const x2 = vector([-2,  3, -2, 1, 10, 11, -2]);
      expect(rank(x2)).to.eql(vector([1, 5, 1, 4, 6, 7, 1]));
   });

   it ('tests for methods "minind" and "min"', function () {

      // normal cases
      const x1l = vector([10, 1, 2, 3, 100, 0, 5, 500, 200]);
      const x2l = vector([-30, -10, -2, -1000, -100, -200]);
      const x3l = vector([0, 1, 2, 3, 100, 10, 5, 500, 200]);
      const x4l = vector([-1000, -10, -20, -100, -200]);
      const x5l = vector([10, 11, 2, 3, 100, 10, 5, 500, 1]);
      const x6l = vector([-10, -15, -20, -100, -200, -1000]);

      expect(minind(x1l)).equal(6)
      expect(minind(x2l)).equal(4)
      expect(minind(x3l)).equal(1)
      expect(minind(x4l)).equal(1)
      expect(minind(x5l)).equal(9)
      expect(minind(x6l)).equal(6)

      expect(min(x1l)).equal(0)
      expect(min(x2l)).equal(-1000)
      expect(min(x3l)).equal(0)
      expect(min(x4l)).equal(-1000)
      expect(min(x5l)).equal(1)
      expect(min(x6l)).equal(-1000)

      // large vector case
      const x7l = Vector.rand(1000000);
      x7l.v[99999] = -0.01;
      expect(minind(x7l)).equal(100000);
      expect(min(x7l)).almost.equal(-0.01);

      // repeated min values - return first index
      const x8l = [-1, 3, -5, 10, -11, 3, -2, -11, 3, 2, -11, -10, 1];
      expect(minind(x8l)).equal(5);
      expect(min(x8l)).almost.equal(-11);

   });

   it ('tests for methods "maxind" and "max"', function () {

      // normal cases
      const x1 = vector([10, 1, 2, 3, 100, 0, 5, 500, 200]);
      const x2 = vector([-30, -10, -2, -1, -100, -200]);
      const x3 = vector([1000, 1, 2, 3, 100, 10, 5, 500, 200]);
      const x4 = vector([-1, -10, -20, -100, -200]);
      const x5 = vector([0, 1, 2, 3, 100, 10, -5, 500, 2000]);
      const x6 = vector([-10, -15, -20, -100, -200, -0.00001]);

      expect(maxind(x1)).equal(8)
      expect(maxind(x2)).equal(4)
      expect(maxind(x3)).equal(1)
      expect(maxind(x4)).equal(1)
      expect(maxind(x5)).equal(9)
      expect(maxind(x6)).equal(6)

      expect(max(x1)).equal(500)
      expect(max(x2)).equal(-1)
      expect(max(x3)).equal(1000)
      expect(max(x4)).equal(-1)
      expect(max(x5)).equal(2000)
      expect(max(x6)).equal(-0.00001)


      // large vector case
      const x7 = Vector.rand(1000000);
      x7.v[99999] = 1.0;
      expect(maxind(x7.v)).equal(100000);
      expect(max(x7.v)).equal(1.0);

      // repeated min values - return first index
      const x8 = [-1, 3, -5, 10, 11, 3, -2, 11, 3, 2, 11, -10, 1];
      expect(maxind(x8)).equal(5);
      expect(max(x8)).almost.equal(11);

   });

   it ('tests for methods "min" and "max" (additional).', function () {
      min(x1).should.be.a('number');
      min(x1).should.equal(mn1);
      min(x2).should.equal(mn2);
      min(x3).should.equal(mn3);

      max(x1).should.be.a('number');
      max(x1).should.equal(mx1);
      max(x2).should.equal(mx2);
      max(x3).should.equal(mx3);
   });

   it ('tests for methods "sum" and "prod".', function () {
      sum(x1).should.be.a('number');
      sum(x1).should.be.closeTo(sm1, 0.0000001);
      sum(x2).should.be.closeTo(sm2, 0.0000001);
      sum(x3).should.be.closeTo(sm3, 0.0000001);

      prod(vector([1, 2, 3, 4, 5])).should.be.a('number');
      prod(vector([1, 2, 3, 4, 5])).should.be.closeTo(120, 0.0000000001);
      prod(vector([10])).should.be.a('number');
      prod(vector([10])).should.be.closeTo(10, 0.0000000001);
   });

   it ('tests for method "mean".', function () {
      mean(x1).should.be.a('number');
      mean(x1).should.be.closeTo(av1, 0.0000001);
      mean(x2).should.be.closeTo(av2, 0.0000001);
      mean(x3).should.be.closeTo(av3, 0.0000001);
   });

   it ('tests for method "sd".', function () {
      sd(x1).should.be.a('number');
      sd(x1).should.be.closeTo(sd1, 0.000001);
      sd(x1, true).should.be.closeTo(sd1b, 0.000001);
      sd(x1, false, mean(x1)).should.be.closeTo(sd1, 0.000001);
   });

   it ('tests for method "skewness".', function() {
      skewness([0, 1, 2]).should.be.closeTo(0, 0.0000001);
      skewness([-10, 1, 2]).should.be.closeTo(-0.6892055, 0.0000001);
      skewness([10, 1, 2]).should.be.closeTo(0.6745555, 0.0000001);
   });

   it ('tests for method "kurtosis".', function() {
      kurtosis([1, 2, 3]).should.be.closeTo(1.5, 0.0000001);
   });

   it ('tests for method "cov".', function() {
      expect(() => cov(vector([1, 2]), vector([1, 2, 3])).to.throw(Error, "Vectors 'x' and 'y' must have the same length."));
      expect(() => cov(vector([1]), vector([1])).to.throw(Error, "Vectors 'x' and 'y' must have at least two values."));

      cov(vector([1, 2, 3]), vector([1, 2, 3])).should.equal(1);
      cov(vector([1, 2, 3]), vector([3, 2, 1])).should.equal(-1);
      cov(vector([1, 2, 3]), vector([10, 20, 30])).should.equal(10);
      cov(vector([1, 2, 3]), vector([30, 20, 10])).should.equal(-10);
      cov(vector([1, 2, 1, 2]), vector([10, 10, 20, 20])).should.equal(0);
   });

   it ('tests for method "cor".', function() {
      cor(vector([1, 2, 3]), vector([1, 2, 3])).should.equal(1);
      cor(vector([1, 2, 3]), vector([3, 2, 1])).should.equal(-1);
      cor(vector([1, 2, 3]), vector([10, 20, 30])).should.equal(1);
      cor(vector([1, 2, 3]), vector([30, 20, 10])).should.equal(-1);
      cor(vector([1, 2, 1, 2]), vector([10, 10, 20, 20])).should.equal(0);
   });

});

