/******************************************************************
 *  Tests for helper functions                                    *
 ******************************************************************/

// import of functions to test
import {integrate, beta, ibeta, gamma} from '../stat/index.js';

// import dependencies
import {dnorm} from '../stat/index.js';
import {default as chai} from 'chai';

const should = chai.should();
const expect = chai.expect;

describe('Tests for helper functions.', function () {

   it('gamma() works correctly for single argument.', function () {
      expect(() => gamma(-1)).to.throw(Error, "Gamma function only works with arguments > 0.");
      expect(() => gamma(0)).to.throw(Error, "Gamma function only works with arguments > 0.");
      gamma(1).should.be.closeTo(1, 0.000001);
      gamma(4).should.be.closeTo(6, 0.000001);
      gamma(0.5).should.be.closeTo(1.772454, 0.000001);
      gamma(1.5).should.be.closeTo(0.8862269, 0.000001);
   });

   it('gamma() works correctly for vector argument.', function () {
      expect(() => gamma([1, 2, -1, 3])).to.throw(Error, "Gamma function only works with arguments > 0.");
      expect(() => gamma([1, 2, 0, 3])).to.throw(Error, "Gamma function only works with arguments > 0.");
      const g = gamma([1, 4, 0.5, 1.5]);
      g[0].should.be.closeTo(1, 0.000001);
      g[1].should.be.closeTo(6, 0.000001);
      g[2].should.be.closeTo(1.772454, 0.000001);
      g[3].should.be.closeTo(0.8862269, 0.000001);
   });

   it('beta() works correctly for single argument.', function () {
      beta(1, 1).should.be.closeTo(1, 0.000001);
      beta(0.5, 0.5).should.be.closeTo(3.141593, 0.000001);
      beta(0.5, 1.5).should.be.closeTo(1.570796, 0.000001);
   });


   it('integrate() works correctly', function() {

      integrate(x => x**2, -1, 1).should.be.closeTo(0.6666667, 0.00001);
      integrate(x => x**2, -1, 0).should.be.closeTo(0.3333333, 0.00001);
      integrate(x => x**2,  0, 1).should.be.closeTo(0.3333333, 0.00001);

      integrate(Math.sin,  0, Math.PI).should.be.closeTo(2, 0.00001);
      integrate(Math.sin,  0, 2 * Math.PI).should.be.closeTo(0, 0.00001);
      integrate((x) => Math.exp(-x), 0, Infinity).should.be.closeTo(1, 0.00001);

      [0.1, 0.5, 1, 5, 10].map(a =>
         integrate((x) => 1/Math.sqrt(a**2 - x**2), 0, a).should.be.closeTo(Math.PI/2, 0.00001)
      );

      integrate((x) => Math.exp(-(x**2)), -Infinity, Infinity).should.be.closeTo(Math.sqrt(Math.PI), 0.0001);

      // integrate over CDF of normal distribution
      integrate((x) => dnorm(x, 0, 1), -Infinity, 0).should.be.closeTo(0.5, 0.00001);
      integrate((x) => dnorm(x, 0, 1), -1.96, 1.96).should.be.closeTo(0.95, 0.00001);
      integrate((x) => dnorm(x, 0, 1), -Infinity, 1.96).should.be.closeTo(0.975, 0.00001);
   });
});