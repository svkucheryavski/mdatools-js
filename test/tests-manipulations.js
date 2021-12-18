/******************************************************************
 *  Tests for functions for manipulation of values                *
 ******************************************************************/

// import of functions to test
import {seq, rep, sort, subset, expandGrid, shuffle, round, scale} from '../stat/index.js';

// import dependencies
import {runif, rnorm, mean, sd} from '../stat/index.js';
import {default as chai} from 'chai';

const should = chai.should();
const expect = chai.expect;


describe('Tests for functions for manipulation of values.', function () {

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


   it('subset() works correctly with one index.', function () {
      const x = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];

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
   });


   it('subset() works correctly with vector of indices.', function () {
      const x = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];

      // wrong index
      expect(() => subset(x, [0, 1, 2])).to.throw(Error, "Parameter 'indices' must have values between 1 and 'x.length'.");
      expect(() => subset(x, [8, 9, 11])).to.throw(Error, "Parameter 'indices' must have values between 1 and 'x.length'.");

      // index as a vector
      expect(subset(x, [1, 2, 3])).to.eql([10, 20, 30]);
      expect(subset(x, [3, 7, 9])).to.eql([30, 70, 90]);
      expect(subset(x, [10, 9, 8, 7, 6, 1])).to.eql([100, 90, 80, 70, 60, 10]);
   });


   it('subset() works correctly with repeated indices.', function () {
      const x = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
      expect(subset(x, [1, 1, 2, 2, 3, 3])).to.eql([10, 10, 20, 20, 30, 30]);
   });


   it('subset() works correctly with large vectors (n = 1 000 000).', function () {
      const y = sort(runif(1000000, 10, 20));
      const z = subset(y, [1, 100, 1000, 10000, 100000, 1000000]);
      z[0].should.be.closeTo(10.0, 0.01);
      z[5].should.be.closeTo(20.0, 0.01);
   });


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

   });


   it('expandGrid() works correctly with text vectors.', function () {
      const z1 = expandGrid([100, 200, 300], ["A", "B"]);
      z1.should.be.a('Array');
      expect(z1).to.have.lengthOf(2);
      expect(z1[0]).to.eql([100, 200, 300, 100, 200, 300]);
      expect(z1[1]).to.eql(["A", "A", "A", "B", "B", "B"]);
   });


   it('shuffle() works correctly with any vectors.', function () {

      const x1 = [1, 2, 3, 4, 5, 6, 7, 8, 9];
      const z1 = shuffle(x1);
      z1.should.be.a('Array');
      expect(z1).to.have.lengthOf(x1.length);
      expect(z1).to.not.eql(x1);
      expect(sort(z1)).to.eql(x1);

      const x2 = ["a", "b", "c", "d", "e", "f"];
      const z2 = shuffle(x2);
      z2.should.be.a('Array');
      expect(z2).to.have.lengthOf(x2.length);
      expect(z2).to.not.eql(x2);

   });


   it('seq() returns correct results.', function () {
      const s1 = seq(1, 10, 10);
      expect(s1).to.eql([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

      const s2 = seq(-5, 4, 10);
      expect(s2).to.eql([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]);

      const s3 = seq(0, 1, 11);
      expect(s3).to.eql([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]);
   });


   it('sort() returns correct results.', function () {
      const x1 = [-10, -2, 0, 2, 10, 20, 50, 100, 150];
      const x2 = [150, -2, 100, 2, 50, 0, 10, 20, -10];
      const x3 = [150, 100, 50, 20, 10, 2, 0, -2, -10];

      const s1 = sort(x2);
      s1.should.be.a('Array');
      s1.should.have.lengthOf(x2.length);
      expect(s1).to.eql(x1);

      const s2 = sort(x2, true);
      s2.should.be.a('Array');
      s2.should.have.lengthOf(x2.length);
      expect(s2).to.eql(x3);
   });


   it('round() works correctly for single argument.', function () {
      round(1.23456789).should.be.equal(1)
      round(1.23456789, 2).should.be.equal(1.23)

      round(1.5000001).should.be.equal(2)
      round(1.5000001, 2).should.be.equal(1.50)
      round(1.4999999).should.be.equal(1)
      round(1.4999999, 2).should.be.equal(1.50)

      round(-1.23456789).should.be.equal(-1)
      round(-1.23456789, 2).should.be.equal(-1.23)
   });


   it('round() works correctly for vectors.', function () {
      expect(round([1.23456789, 2.3456789])).be.eql([1, 2])
      expect(round([1.23456789, 2.3456789], 2)).be.eql([1.23, 2.35])
   });


   it('scale() works correctly with default arguments.', function () {
      const x = rnorm(1000, 100, 2);
      mean(scale(x)).should.be.closeTo(0, 0.0000001);
      sd(scale(x)).should.be.closeTo(1, 0.0000001);
   });

   it('scale() works correctly with default arguments.', function () {
      const x = rnorm(1000, 100, 2);

      mean(scale(x, 0, sd(x))).should.be.closeTo(mean(x) / sd(x), 0.0000001);
      sd(scale(x, 0, sd(x))).should.be.closeTo(1, 0.0000001);

      mean(scale(x, mean(x), 1)).should.be.closeTo(0, 0.0000001);
      sd(scale(x, mean(x), 1)).should.be.closeTo(sd(x), 0.0000001);

      mean(scale(x, 0, 1)).should.be.closeTo(mean(x), 0.0000001);
      sd(scale(x, 0, 1)).should.be.closeTo(sd(x), 0.0000001);
   });

});