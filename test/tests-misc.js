/****************************************************************/
/*  Tests for methods computing various statistics              */
/****************************************************************/

// import dependencies
import {default as chai} from 'chai';
import {default as chaiAlmost} from 'chai-almost';
import { vector } from '../src/arrays/index.js';
import { quantile } from '../src/stat/index.js';

// import methods to test
import { closestind, closestindleft, closestindright, getoutliers, expandgrid, integrate, round } from '../src/misc/index.js';

// set up test settings
const should = chai.should();
const expect = chai.expect;
chai.use(chaiAlmost(0.00001));

describe('Tests for misc extra functions.', function () {

   it('tests for method "getoutliers"".', function () {

      const x4 = vector([-100, -2, -1, 0, 1, 2, 3, 50, 100]);
      const o1 = getoutliers(x4);
      expect(o1).to.eql(vector([-100, 50, 100]));

      const o2 = getoutliers(x4, quantile(x4, 0.25));
      expect(o2).to.eql(vector([-100, 50, 100]));

      const o3 = getoutliers(x4, quantile(x4, 0.25), quantile(x4, 0.75));
      expect(o3).to.eql(vector([-100, 50, 100]));

   });

   it('tests for method "integrate".', function() {

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
   });

   it('tests for method "expandgrid".', function () {

      const z1 = expandgrid(vector([100, 200, 300]), vector([10, 20]));
      z1.should.be.a('Array');
      expect(z1).to.have.lengthOf(2);
      expect(z1[0]).to.eql(vector([100, 200, 300, 100, 200, 300]));
      expect(z1[1]).to.eql(vector([10, 10, 10, 20, 20, 20]));

      const z2 = expandgrid(vector([100, 200, 300]), vector([10, 20]), vector([-20, +20]));
      z2.should.be.a('Array');
      expect(z2).to.have.lengthOf(3);
      expect(z2[0]).to.eql(vector([100, 200, 300, 100, 200, 300, 100, 200, 300, 100, 200, 300]));
      expect(z2[1]).to.eql(vector([10, 10, 10, 20, 20, 20, 10, 10, 10, 20, 20, 20]));
      expect(z2[2]).to.eql(vector([-20, -20, -20, -20, -20, -20, 20, 20, 20, 20, 20, 20]));

   });

   it ('tests for method "closestindleft".', function() {
      const x1 = vector([-200, -100, -50, -0.5, 0.1, 1.5, 20, 500, 1000]);

      closestindleft(x1, -120).should.equal(1);
      closestindleft(x1, -101).should.equal(1);
      closestindleft(x1, -99).should.equal(2);
      closestindleft(x1, -51).should.equal(2);
      closestindleft(x1, -50).should.equal(3);
      closestindleft(x1, -49).should.equal(3);
      closestindleft(x1, 0).should.equal(4);
      closestindleft(x1, 0.099999).should.equal(4);
      closestindleft(x1, 0.1).should.equal(5);
      closestindleft(x1, 0.11).should.equal(5);
      closestindleft(x1, 499).should.equal(7);
      closestindleft(x1, 501).should.equal(8);
      closestindleft(x1, 2000).should.equal(9);
   });

   it ('tests for method "closestindright".', function() {
      const x1 = vector([-200, -100, -50, -0.5, 0.1, 1.5, 20, 500, 1000]);

      closestindright(x1, -1200).should.equal(1);
      closestindright(x1, -120).should.equal(2);
      closestindright(x1, -101).should.equal(2);
      closestindright(x1, -99).should.equal(3);
      closestindright(x1, -51).should.equal(3);
      closestindright(x1, -50).should.equal(3);
      closestindright(x1, -49).should.equal(4);
      closestindright(x1, 0).should.equal(5);
      closestindright(x1, 0.099999).should.equal(5);
      closestindright(x1, 0.1).should.equal(5);
      closestindright(x1, 0.11).should.equal(6);
      closestindright(x1, 499).should.equal(8);
      closestindright(x1, 501).should.equal(9);
      closestindright(x1, 2000).should.equal(9);
   });


   it ('tests for method "closestind".', function() {
      const x1 = vector([-200, -0.5, -0.45, 1.2, -5.1, 4.2, -10.0]);

      closestind(x1, -120).should.equal(1);
      closestind(x1, -20).should.equal(7);
      closestind(x1, -7).should.equal(5);
      closestind(x1, 0).should.equal(3);
      closestind(x1, 3).should.equal(6);
      closestind(x1, 100000000).should.equal(6);
      closestind(x1, -100000000).should.equal(1);
   });

   it('tests for method "round".', function () {
      expect(round([1.23456789, 2.3456789])).be.eql([1, 2])
      expect(round([1.23456789, 2.3456789], 2)).be.eql([1.23, 2.35])
   });

});