// import dependencies
import {default as chai} from 'chai';
import {sum, sd, mean, min, max} from '../stat/index.js';
import { isvector, vector, Vector } from '../arrays/index.js';

// import of functions to test
import {igamma, gamma, beta, runif, dunif, punif, rnorm, dnorm, pnorm, qnorm, dt, pt, qt, df, pf, pchisq, qchisq} from '../distributions/index.js';

const should = chai.should();
const expect = chai.expect;

describe('Tests for helper functions.', function () {

   it ('tests for "igamma" function.', function () {
      igamma(1, 1).should.be.closeTo(0.632120, 0.00001)
      igamma(1, 2).should.be.closeTo(0.264241, 0.00001)
      igamma(2, 1).should.be.closeTo(0.864665, 0.00001)
      igamma(2, 2).should.be.closeTo(0.593994, 0.00001)

      igamma( 5,  5).should.be.closeTo(13.42816, 0.00001)
      igamma( 5, 10).should.be.closeTo(11549.765, 0.001)
      igamma(10,  5).should.be.closeTo(23.29793, 0.00001)
      igamma(10, 10).should.be.closeTo(196706.465212, 0.00001)
   });

   it ('tests for "gamma" function.', function () {
      expect(() => gamma(-1)).to.throw(Error, "gamma: the function only works with arguments > 0.");
      expect(() => gamma(0)).to.throw(Error, "gamma: the function only works with arguments > 0.");
      gamma(1).should.be.closeTo(1, 0.000001);
      gamma(4).should.be.closeTo(6, 0.000001);
      gamma(0.5).should.be.closeTo(1.772454, 0.000001);
      gamma(1.5).should.be.closeTo(0.8862269, 0.000001);

      expect(() => gamma(vector([1, 2, -1, 3]))).to.throw(Error, "gamma: the function only works with arguments > 0.");
      expect(() => gamma(vector([1, 2, 0, 3]))).to.throw(Error, "gamma: the function only works with arguments > 0.");
      const g = gamma(vector([1, 4, 0.5, 1.5]));
      g.v[0].should.be.closeTo(1, 0.000001);
      g.v[1].should.be.closeTo(6, 0.000001);
      g.v[2].should.be.closeTo(1.772454, 0.000001);
      g.v[3].should.be.closeTo(0.8862269, 0.000001);
   });

   it ('tests for "beta".', function () {
      beta(1, 1).should.be.closeTo(1, 0.000001);
      beta(0.5, 0.5).should.be.closeTo(3.141593, 0.000001);
      beta(0.5, 1.5).should.be.closeTo(1.570796, 0.000001);
   });

});

describe('Tests for theoretical distribution functions.', function () {

   it ('tests for method "dunif".', function () {

      const n = 1000000;

      // standardized distribution for a = 0, b = 1
      const x1 = Vector.seq(0, 1, 1/(n - 1));
      const d1 = dunif(x1);

      expect(d1.v).to.have.lengthOf(n);
      d1.v[0].should.be.closeTo(1,   0.0000000001);
      d1.v[n-1].should.be.closeTo(1, 0.0000000001);
      d1.v[n/2].should.be.closeTo(1, 0.0000000001);

      // distribution with mu = 10 and sigma = 10, for ± 3 sigma
      const a = 10;
      const b = 100;
      const x2 = Vector.seq(a, b, (b - a) / (n - 1));
      const d2 = dunif(x2, a, b);

      expect(d2.v).to.have.lengthOf(n);
      d2.v[0].should.be.closeTo(1 / (b - a),   0.00000001);
      d2.v[n-1].should.be.closeTo(1 / (b - a), 0.00000001);
      d2.v[n/2].should.be.closeTo(1 / (b - a), 0.00000001);
      (sum(d2) * (b - a)/n).should.be.closeTo(1.0, 0.000001);

      dunif(a - 0.0000001, a, b).should.be.closeTo(0.0, 0.0000001);
      dunif(b + 0.0000001, a, b).should.be.closeTo(0.0, 0.0000001);
   });

   it('tests for method "punif".', function () {

      const n = 1000000;

      // standardized distribution for a = 0, b = 1
      const x1 = Vector.seq(0, 1, 1 / (n - 1));
      const p1 = punif(x1);

      expect(p1.v).to.have.lengthOf(n);
      p1.v[0].should.be.closeTo(0, 0.00001);
      p1.v[n-1].should.be.closeTo(1, 0.00001);
      p1.v[n/2].should.be.closeTo(0.5, 0.00001);

      // outside the range
      punif(-1).should.be.closeTo(0, 0.00001);
      punif( 1).should.be.closeTo(1, 0.00001);

      // distribution with mu = 10 and sigma = 10, for ± 3 sigma
      const a = 10;
      const b = 100;
      const x2 = Vector.seq(a, b, (b - a) / (n - 1));
      const p2 = punif(x2, a, b);

      expect(p2.v).to.have.lengthOf(n);
      punif(a - 1, a, b).should.be.closeTo(0.0, 0.00001);
      punif(b + 1, a, b).should.be.closeTo(1.0, 0.00001);
   });

   it('tests for method "runif".', function () {

      const n = 1000000;
      const r1 = runif(n);

      expect(r1.v).to.have.lengthOf(n)
      expect(min(r1)).to.be.above(0);
      expect(max(r1)).to.be.below(1);

      const r2 = runif(n, 10, 20);
      expect(r2.v).to.have.lengthOf(n)
      expect(min(r2)).to.be.above(10);
      expect(max(r2)).to.be.below(20);
   });

   it ('tests for method "dnorm".', function () {

      const n = 1000000;

      // standardized distribution for ± 3 sigma
      const x1 = Vector.seq(-3, 3, 6 / (n - 1));
      const d1 = dnorm(x1);
      expect(d1.v).to.have.lengthOf(n);
      d1.v[0].should.be.closeTo(0.004431848, 0.00000001);
      d1.v[n-1].should.be.closeTo(0.004431848, 0.00000001);
      d1.v[n/2].should.be.closeTo(0.3989423, 0.0000001);

      // distribution with mu = 10 and sigma = 10, for ± 3 sigma
      const mu = 10;
      const sigma = 10
      const x2 = Vector.seq(mu - 3 * sigma, mu + 3 * sigma, 6 * sigma / (n - 1));
      const d2 = dnorm(x2, mu, sigma);
      expect(d2.v).to.have.lengthOf(n);
      d2.v[0].should.be.closeTo(0.0004431848, 0.00000001);
      d2.v[n-1].should.be.closeTo(0.0004431848, 0.00000001);
      d2.v[n/2].should.be.closeTo(0.03989423, 0.0000001);

      // distribution with mu = 10 and sigma = 10, for ± 6 sigma should have area of one
      const x3 = Vector.seq(mu - 6 * sigma, mu + 6 * sigma, 12 * sigma / (n - 1));
      const d3 = dnorm(x3, mu, sigma);
      expect(d3.v).to.have.lengthOf(n);
      (sum(d3) * 12 * sigma / n).should.be.closeTo(1.0, 0.00001);

      // if values are far from mean density is 0
      dnorm(mu - 6 * sigma, mu, sigma).should.be.closeTo(0.0, 0.0000001);
      dnorm(mu + 6 * sigma, mu, sigma).should.be.closeTo(0.0, 0.0000001);
   });

   it ('tests for method "pnorm".', function () {

      const n = 1000000;

      // standardized distribution for ± 3 sigma
      const x1 = Vector.seq(-3, 3, 6 / (n - 1));
      const p1 = pnorm(x1);

      expect(p1.v).to.have.lengthOf(n);
      p1.v[  0].should.be.closeTo(0.00134996, 0.00001);
      p1.v[n-1].should.be.closeTo(0.998650, 0.00001);
      p1.v[n/2].should.be.closeTo(0.5, 0.00001);

     // distribution with mu = 10 and sigma = 10, for ± 3 sigma
      const mu = 10;
      const sigma = 10
      const x2 = Vector.seq(mu - 3 * sigma, mu + 3 * sigma, 6 * sigma / (n - 1));
      const p2 = pnorm(x2, mu, sigma);
      expect(p2.v).to.have.lengthOf(n);
      p2.v[  0].should.be.closeTo(0.001350, 0.000001);
      p2.v[n-1].should.be.closeTo(0.998650, 0.000001);
      p2.v[n/2].should.be.closeTo(0.5, 0.00001);

   });

   it ('tests for method "qnorm".', function () {

      const n = 1000000;

      // border cases
      qnorm(0).should.be.equal(-Infinity);
      qnorm(1).should.be.equal(Infinity);
      qnorm(vector([0, 0, 1, 1])).should.be.eql(vector([-Infinity, -Infinity, Infinity, Infinity]));

      // middle point and border cases
      qnorm(0.5).should.be.equal(0);
      qnorm(vector([0.5, 0.5])).should.be.eql(vector([0, 0]));
      qnorm(vector([0, 0.5, 1])).should.be.eql(vector([-Infinity, 0, Infinity]));

      // other cases
      qnorm(0.9999).should.be.closeTo( 3.719016, 0.00001);
      qnorm(0.0001).should.be.closeTo(-3.719016, 0.00001);

      qnorm(0.975).should.be.closeTo( 1.959964, 0.00001);
      qnorm(0.025).should.be.closeTo(-1.959964, 0.00001);

      qnorm(0.840).should.be.closeTo( 0.9944579, 0.00001);
      qnorm(0.160).should.be.closeTo(-0.9944579, 0.00001);

      qnorm(0.750).should.be.closeTo( 0.6744898, 0.00001);
      qnorm(0.250).should.be.closeTo(-0.6744898, 0.00001);

      // cases with non standard distribution
      qnorm(0.975, 10, 2).should.be.closeTo(13.91993, 0.00001);
      qnorm(0.025, 10, 2).should.be.closeTo( 6.080072, 0.00001);

      // errors
      expect(() => qnorm(-0.0001)).to.throw(Error, 'Parameter "p" must be between 0 and 1.');
      expect(() => qnorm( 1.0001)).to.throw(Error, 'Parameter "p" must be between 0 and 1.');

      // long vectors
      const p = Vector.seq(0.0001, 0.9999, 0.9998 / (n - 1));
      const q = qnorm(p);
      expect(q.v).to.have.lengthOf(n);
      q.v[0].should.be.equal(qnorm(0.0001));
      q.v[n-1].should.be.closeTo(qnorm(0.9999), 0.000000001);
   });

   it('tests for method "rnorm".', function () {
      const n = 1000000;

      const r1 = rnorm(n);
      expect(r1.v).to.have.lengthOf(n)
      sd(r1).should.be.closeTo(1, 0.01);
      mean(r1).should.be.closeTo(0, 0.01);
      min(r1).should.be.above(-6);
      max(r1).should.be.below(6);

      const r2 = rnorm(n, 10, 5);
      expect(r2.v).to.have.lengthOf(n)
      sd(r2).should.be.closeTo(5, 0.01);
      mean(r2).should.be.closeTo(10, 0.01);
      min(r2).should.be.above(-20);
      max(r2).should.be.below(40);
   });

   it ('tests for method "dt".', function () {

      const n = 100000;

      //  distribution for DoF = 1
      const x1 = Vector.seq(-5, 5, 10/(n - 1));
      const d1 = dt(x1, 1);
      expect(d1.v).to.have.lengthOf(n);
      d1.v[0].should.be.closeTo(0.01224269, 0.00000001);
      d1.v[n-1].should.be.closeTo(0.01224269, 0.00000001);
      d1.v[n/2].should.be.closeTo(0.31830989, 0.0000001);

      //  distribution for DoF = 3
      const x2 = Vector.seq(-5, 5, 10/(n - 1));
      const d2 = dt(x2, 3);
      expect(d2.v).to.have.lengthOf(n);
      d2.v[0].should.be.closeTo(0.004219354, 0.00000001);
      d2.v[n-1].should.be.closeTo(0.004219354, 0.00000001);
      d2.v[n/2].should.be.closeTo(0.3675526, 0.0000001);

      //  distribution for DoF = 30
      const x3 = Vector.seq(-3, 3, 6/(n - 1));
      const d3 = dt(x3, 30);
      expect(d3.v).to.have.lengthOf(n);
      d3.v[0].should.be.closeTo(0.006779063, 0.00000001);
      d3.v[n-1].should.be.closeTo(0.006779063, 0.00000001);
      d3.v[n/2].should.be.closeTo(0.3956322, 0.0000001);

   });

   it ('tests for mehtod "pt".', function () {

      const n = 10000;

      //  distribution for DoF = 1
      const t1 = Vector.seq(-5, 5, 10 / (n - 1));
      const p1 = pt(t1, 1);
      expect(p1.v).to.have.lengthOf(n);
      p1.v[0].should.be.closeTo(0.06283296, 0.001);
      p1.v[n-1].should.be.closeTo(0.937167, 0.001);
      p1.v[n/2].should.be.closeTo(0.5, 0.001);

      //  distribution for DoF = 3
      const t2 = Vector.seq(-5, 5, 10 / (n - 1));
      const p2 = pt(t2, 3);
      expect(p2.v).to.have.lengthOf(n);
      p2.v[0].should.be.closeTo(0.007696219, 0.001);
      p2.v[n-1].should.be.closeTo(0.9923038, 0.001);
      p2.v[n/2].should.be.closeTo(0.5, 0.001);

      //  distribution for DoF = 30
      const t3 = Vector.seq(-5, 5, 10 / (n - 1));
      const p3 = pt(t3, 30);
      expect(p3.v).to.have.lengthOf(n);
      p3.v[0].should.be.closeTo(0.00001164834, 0.001);
      p3.v[n-1].should.be.closeTo(0.9999884, 0.001);
      p3.v[n/2].should.be.closeTo(0.5, 0.001);

   });

   it ('tests for method "qt".', function () {

      const n = 1000000;

      // border cases
      qt(0, 1).should.be.equal(-Infinity);
      qt(1, 1).should.be.equal(Infinity);
      qt(vector([0, 0, 1, 1]), 1).should.be.eql(vector([-Infinity, -Infinity, Infinity, Infinity]));

      // middle point and border cases
      qt(0.5, 1).should.be.equal(0);
      qt(vector([0.5, 0.5]), 1).should.be.eql(vector([0, 0]));
      qt(vector([0, 0.5, 1]), 1).should.be.eql(vector([-Infinity, 0, Infinity]));

      // other fixed cases
      const dof = [1, 2, 3, 4, 10, 30, 100];
      let p, expected;

      p = 0.9999
      expected = [3183.098757, 70.700071, 22.203742, 13.033672, 5.693820, 4.233986, 3.861600];
      for (let i = 0; i < dof.length; i++) {
         qt(p, dof[i]).should.be.closeTo(expected[i], 0.00001);
         qt(1 - p, dof[i]).should.be.closeTo(-expected[i], 0.00001);
      }

      p = 0.99
      expected = [31.820516, 6.964557, 4.540703, 3.746947, 2.763769, 2.457262, 2.364217];
      for (let i = 0; i < dof.length; i++) {
         qt(p, dof[i]).should.be.closeTo(expected[i], 0.0001);
         qt(1 - p, dof[i]).should.be.closeTo(-expected[i], 0.0001);
      }

      p = 0.95
      expected = [6.313752, 2.919986, 2.353363, 2.131847, 1.812461, 1.697261, 1.660234];
      for (let i = 0; i < dof.length; i++) {
         qt(p, dof[i]).should.be.closeTo(expected[i], 0.0001);
         qt(1 - p, dof[i]).should.be.closeTo(-expected[i], 0.0001);
      }

      p = 0.85
      expected = [1.962611, 1.386207, 1.249778, 1.189567, 1.093058, 1.054662, 1.041836];
      for (let i = 0; i < dof.length; i++) {
         qt(p, dof[i]).should.be.closeTo(expected[i], 0.0001);
         qt(1 - p, dof[i]).should.be.closeTo(-expected[i], 0.0001);
      }

      p = 0.75
      expected = [1.0000000, 0.8164966, 0.7648923, 0.7406971, 0.6998121, 0.6827557, 0.6769510];
      for (let i = 0; i < dof.length; i++) {
         qt(p, dof[i]).should.be.closeTo(expected[i], 0.0001);
         qt(1 - p, dof[i]).should.be.closeTo(-expected[i], 0.0001);
      }

      // errors
      expect(() => qt(-0.0001, 1)).to.throw(Error, 'Parameter "p" must be between 0 and 1.');
      expect(() => qt( 1.0001, 1)).to.throw(Error, 'Parameter "p" must be between 0 and 1.');
      expect(() => qt(0.2)).to.throw(Error, 'Parameter "dof" (degrees of freedom) must be an integer number >= 1.');
      expect(() => qt(0.2, -1)).to.throw(Error, 'Parameter "dof" (degrees of freedom) must be an integer number >= 1.');
      expect(() => qt(0.2, 0.5)).to.throw(Error, 'Parameter "dof" (degrees of freedom) must be an integer number >= 1.');

      // long vectors
      p = Vector.seq(0.0001, 0.9999, 0.9998/(n - 1));
      const q = qt(p, 10);
      expect(q.v).to.have.lengthOf(n);
      q.v[0].should.be.equal(qt(0.0001, 10));
      q.v[n - 1].should.be.equal(qt(0.9998999999999999, 10));
   });

   it ('tests for method "df".', function () {

      const n = 10000;

      //  distribution for DoF = 1, 2
      const F1 = Vector.seq(0.001, 10, 10/n);
      const d1 = df(F1, 1, 2);
      expect(d1.v).to.have.lengthOf(n);
      d1.v[0].should.be.closeTo(11.17196, 0.001);
      d1.v[n-1].should.be.closeTo(0.007607258, 0.001);
      d1.v[n/2].should.be.closeTo(0.02414726, 0.001);

      //  distribution for DoF = 3, 10
      const F2 = Vector.seq(0.001, 10, 10/n);
      const d2 = df(F2, 3, 10);
      expect(d2.v).to.have.lengthOf(n);
      d2.v[0].should.be.closeTo(0.07019374, 0.001);
      d2.v[n-1].should.be.closeTo(0.0008585295, 0.001);
      d2.v[n/2].should.be.closeTo(0.01288309, 0.001);

   });

   it ('tests for method "pf".', function () {

      const n = 10000;

      //  distribution for DoF = 1, 2
      const F1 = Vector.seq(0, 10, 10/n);
      const p1 = pf(F1, 1, 2);
      expect(p1.v).to.have.lengthOf(n + 1);
      p1.v[0].should.be.closeTo(0, 0.001);
      p1.v[n-1].should.be.closeTo(0.9128709, 0.001);
      p1.v[n/2].should.be.closeTo(0.8451543, 0.001);

      //  distribution for DoF = 3, 10
      const F2 = Vector.seq(0, 10, 10/n);
      const p2 = pf(F2, 3, 10);
      expect(p2.v).to.have.lengthOf(n + 1);
      p2.v[0].should.be.closeTo(0, 0.001);
      p2.v[n].should.be.closeTo(0.9976484, 0.001);
      p2.v[n/2].should.be.closeTo(0.9773861, 0.001);

   });

   it ('tests for method "pchisq".', function () {

      // single value
      const x1 = 10
      pchisq(x1,   0).should.be.equal(1);
      pchisq(x1,   1).should.be.closeTo(0.9984346, 0.0001);
      pchisq(x1,   5).should.be.closeTo(0.9247648, 0.0001);
      pchisq(x1,  10).should.be.closeTo(0.5595067, 0.0001);
      pchisq(x1, 100).should.be.closeTo(0.0000000, 0.0001);

      const x2 = 0
      pchisq(x2, 0).should.be.equal(0);
      pchisq(x2, 5).should.be.equal(0);
      pchisq(x2, 10).should.be.equal(0);
      pchisq(x2, 100).should.be.equal(0);

      const x3 = 3.16
      pchisq(x3,   0).should.be.equal(1);
      pchisq(x3,   1).should.be.closeTo(0.9245368, 0.0001);
      pchisq(x3,   5).should.be.closeTo(0.3246659, 0.0001);
      pchisq(x3,  10).should.be.closeTo(0.0225961, 0.0001);
      pchisq(x3, 100).should.be.closeTo(0.0000000, 0.0001);

      // vector with values
      const x4 = vector([10, 0, 3.16]);
      const p4 = pchisq(x4, 5)

      expect(isvector(p4)).to.be.true;
      p4.length.should.be.equal(3)
      p4.v[0].should.be.closeTo(0.9247648, 0.0001);
      p4.v[1].should.be.closeTo(0.0000000, 0.0001);
      p4.v[2].should.be.closeTo(0.3246659, 0.0001);
   });

   it ('tests for method "qchisq".', function () {

      // single value
      const p1 = 0.05
      qchisq(p1, 0).should.be.equal(0);
      qchisq(p1, 5).should.be.closeTo(1.15, 0.01);
      qchisq(p1, 10).should.be.closeTo(3.94, 0.01);
      qchisq(p1, 100).should.be.closeTo(77.93, 0.01);

      const p2 = 0.01
      qchisq(p2, 0).should.be.equal(0);
      qchisq(p2, 5).should.be.closeTo(0.55, 0.02);
      qchisq(p2, 10).should.be.closeTo(2.56, 0.02);
      qchisq(p2, 100).should.be.closeTo(70.07, 0.02);

      // vector with values
      const p3 = vector([0.01, 0.05]);
      const q3 = qchisq(p3, 5)
      q3.v[0].should.be.closeTo(0.55, 0.02);
      q3.v[1].should.be.closeTo(1.15, 0.02);

      // array with values
      const p4 = [0.01, 0.05];
      const q4 = qchisq(p3, 5)
      q4.v[0].should.be.closeTo(0.55, 0.02);
      q4.v[1].should.be.closeTo(1.15, 0.02);
   });

});

