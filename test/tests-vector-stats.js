/******************************************************************
 *  Tests for functions which return statistic as a vector        *
 ******************************************************************/

// import of functions to test
import {quantile, range, mrange, split, count, mids, diff, getOutliers, ppoints, rank, cumsum} from '../stat/index.js';

// import dependencies
import {runif, seq, min, max} from '../stat/index.js';
import {default as chai} from 'chai';

const should = chai.should();
const expect = chai.expect;

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

   it('getOutliers() returns correct results.', function () {
      const x4 = [-100, -2, -1, 0, 1, 2, 3, 50, 100];
      const o1 = getOutliers(x4);
      expect(o1).to.eql([-100, 50, 100]);

      const o2 = getOutliers(x4, quantile(x4, 0.25));
      expect(o2).to.eql([-100, 50, 100]);

      const o3 = getOutliers(x4, quantile(x4, 0.25), quantile(x4, 0.75));
      expect(o2).to.eql([-100, 50, 100]);
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

   it('rank() works correctly.', function () {
      // i:       1   2  3  4  5  6  7  8
      // sorted: -3   0  1  2  3  4  8  9
      // rank:    5   1  2  6  3  4  8  7
      const x1 = [3, -3, 0, 4, 1, 2, 9, 8];
      expect(rank(x1)).to.eql([5, 1, 2, 6, 3, 4, 8, 7]);

      //  i:       1   2   3   4   5   6   7
      // sorted:  -2  -2  -2   1   3  10  11
      // rank:     1   5   1   4   6   7   1
      const x2 = [-2,  3, -2, 1, 10, 11, -2];
      expect(rank(x2)).to.eql([1, 5, 1, 4, 6, 7, 1]);
   });

   it('cumsum() returns correct results.', function () {
      cumsum([1, 2, 3]).should.be.a('Array');
      expect(cumsum([1, 2, 3])).to.eql([1, 3, 6]);
      expect(cumsum([1])).to.eql([1]);
      cumsum(seq(1, 1000000))[999999].should.be.equal(1000000 * 1000001 / 2);
   });

});

