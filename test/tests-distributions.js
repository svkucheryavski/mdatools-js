/******************************************************************
 *  Tests for functions for theoretical distributions             *
 ******************************************************************/

// import of functions to test
import {runif, dunif, punif, rnorm, dnorm, pnorm, qnorm, dt, pt, qt, df, pf} from '../stat/index.js';

// import dependencies
import {seq, sum, sd, mean, min, max} from '../stat/index.js';
import {default as chai} from 'chai';

const should = chai.should();
const expect = chai.expect;

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
      dnorm([mu - 6 * sigma], mu, sigma).should.be.closeTo(0.0, 0.0000001);
      dnorm([mu + 6 * sigma], mu, sigma).should.be.closeTo(0.0, 0.0000001);
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


   it('dt() works correctly (n = 100 000).', function () {
      const n = 100000;

      //  distribution for DoF = 1
      const x1 = seq(-5, 5, n);
      const d1 = dt(x1, 1);
      expect(d1).to.have.lengthOf(n);
      d1[0].should.be.closeTo(0.01224269, 0.00000001);
      d1[n-1].should.be.closeTo(0.01224269, 0.00000001);
      d1[n/2].should.be.closeTo(0.31830989, 0.0000001);

      //  distribution for DoF = 3
      const x2 = seq(-5, 5, n);
      const d2 = dt(x2, 3);
      expect(d2).to.have.lengthOf(n);
      d2[0].should.be.closeTo(0.004219354, 0.00000001);
      d2[n-1].should.be.closeTo(0.004219354, 0.00000001);
      d2[n/2].should.be.closeTo(0.3675526, 0.0000001);

      //  distribution for DoF = 30
      const x3 = seq(-3, 3, n);
      const d3 = dt(x3, 30);
      expect(d3).to.have.lengthOf(n);
      d3[0].should.be.closeTo(0.006779063, 0.00000001);
      d3[n-1].should.be.closeTo(0.006779063, 0.00000001);
      d3[n/2].should.be.closeTo(0.3956322, 0.0000001);
   });


   it('df() works correctly (n = 10 000).', function () {
      const n = 10000;

      //  distribution for DoF = 1, 2
      const F1 = seq(0.001, 10, n);
      const d1 = df(F1, 1, 2);
      expect(d1).to.have.lengthOf(n);
      d1[0].should.be.closeTo(11.17196, 0.001);
      d1[n-1].should.be.closeTo(0.007607258, 0.001);
      d1[n/2].should.be.closeTo(0.02414726, 0.001);

      //  distribution for DoF = 3, 10
      const F2 = seq(0.001, 10, n);
      const d2 = df(F2, 3, 10);
      expect(d2).to.have.lengthOf(n);
      d2[0].should.be.closeTo(0.07019374, 0.001);
      d2[n-1].should.be.closeTo(0.0008585295, 0.001);
      d2[n/2].should.be.closeTo(0.01288309, 0.001);

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

   it('pt() works correctly (n = 10 000).', function () {
      const n = 10000;

      //  distribution for DoF = 1
      const t1 = seq(-5, 5, n);
      const p1 = pt(t1, 1);
      expect(p1).to.have.lengthOf(n);
      p1[0].should.be.closeTo(0.06283296, 0.001);
      p1[n-1].should.be.closeTo(0.937167, 0.001);
      p1[n/2].should.be.closeTo(0.5, 0.001);

      //  distribution for DoF = 3
      const t2 = seq(-5, 5, n);
      const p2 = pt(t2, 3);
      expect(p2).to.have.lengthOf(n);
      p2[0].should.be.closeTo(0.007696219, 0.001);
      p2[n-1].should.be.closeTo(0.9923038, 0.001);
      p2[n/2].should.be.closeTo(0.5, 0.001);

      //  distribution for DoF = 30
      const t3 = seq(-5, 5, n);
      const p3 = pt(t3, 30);
      expect(p3).to.have.lengthOf(n);
      p3[0].should.be.closeTo(0.00001164834, 0.001);
      p3[n-1].should.be.closeTo(0.9999884, 0.001);
      p3[n/2].should.be.closeTo(0.5, 0.001);

   });

   it('pf() works correctly (n = 10 000).', function () {
      const n = 10000;

      //  distribution for DoF = 1, 2
      const F1 = seq(0, 10, n);
      const p1 = pf(F1, 1, 2);
      expect(p1).to.have.lengthOf(n);
      p1[0].should.be.closeTo(0, 0.001);
      p1[n-1].should.be.closeTo(0.9128709, 0.001);
      p1[n/2].should.be.closeTo(0.8451543, 0.001);

      //  distribution for DoF = 3, 10
      const F2 = seq(0, 10, n);
      const p2 = pf(F2, 3, 10);
      expect(p2).to.have.lengthOf(n);
      p2[0].should.be.closeTo(0, 0.001);
      p2[n-1].should.be.closeTo(0.9976484, 0.001);
      p2[n/2].should.be.closeTo(0.9773861, 0.001);

   });

   it('qnorm() works correctly (n = 1 000 000).', function () {
      const n = 1000000;

      // border cases
      qnorm(0).should.be.equal(-Infinity);
      qnorm(1).should.be.equal(Infinity);
      qnorm([0, 0, 1, 1]).should.be.eql([-Infinity, -Infinity, Infinity, Infinity]);

      // middle point and border cases
      qnorm(0.5).should.be.equal(0);
      qnorm([0.5, 0.5]).should.be.eql([0, 0]);
      qnorm([0, 0.5, 1]).should.be.eql([-Infinity, 0, Infinity]);

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
      expect(() => qnorm(-0.0001)).to.throw(Error, "Parameter 'p' must be between 0 and 1.");
      expect(() => qnorm( 1.0001)).to.throw(Error, "Parameter 'p' must be between 0 and 1.");

      // long vectors
      const p = seq(0.0001, 0.9999, n);
      const q = qnorm(p);
      expect(q).to.have.lengthOf(n);
      q[0].should.be.equal(qnorm(0.0001));
      q[n-1].should.be.equal(qnorm(0.9999));
   });

   it('qt() works correctly (n = 1 000 000).', function () {
      const n = 1000000;

      // border cases
      qt(0, 1).should.be.equal(-Infinity);
      qt(1, 1).should.be.equal(Infinity);
      qt([0, 0, 1, 1], 1).should.be.eql([-Infinity, -Infinity, Infinity, Infinity]);

      // middle point and border cases
      qt(0.5, 1).should.be.equal(0);
      qt([0.5, 0.5], 1).should.be.eql([0, 0]);
      qt([0, 0.5, 1], 1).should.be.eql([-Infinity, 0, Infinity]);

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
      expect(() => qt(-0.0001, 1)).to.throw(Error, "Parameter 'p' must be between 0 and 1.");
      expect(() => qt( 1.0001, 1)).to.throw(Error, "Parameter 'p' must be between 0 and 1.");
      expect(() => qt(0.2)).to.throw(Error, "Parameter 'dof' (degrees of freedom) must be an integer number >= 1.");
      expect(() => qt(0.2, -1)).to.throw(Error, "Parameter 'dof' (degrees of freedom) must be an integer number >= 1.");
      expect(() => qt(0.2, 0.5)).to.throw(Error, "Parameter 'dof' (degrees of freedom) must be an integer number >= 1.");

      // long vectors
      p = seq(0.0001, 0.9999, n);
      const q = qt(p, 10);
      expect(q).to.have.lengthOf(n);
      q[0].should.be.equal(qt(0.0001, 10));
      q[n-1].should.be.equal(qt(0.9999, 10));
   });
});

