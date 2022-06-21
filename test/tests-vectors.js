/******************************************************************
 *  Tests for vector functions                                    *
 ******************************************************************/

// import of functions to test
import {vdot, isvector, vadd, vdiv, vmult, vapply, vsubtract, norm2, vreplace} from '../matrix/index.js';

// import dependencies
import {default as chai} from 'chai';

const should = chai.should();
const expect = chai.expect;

/* Tests for operations with vectors */
describe('Tests for operations with vectors.', function () {

   it('vreplace() works correctly', function() {
      const x = [10, 20, 30, 40, 50];
      const y = [14, 15, 11];
      const ind1 = [4, 5, 1];
      const ind2 = [3, 3, 3];
      const inde1 = [1, 2, 3, 4, 5];
      const inde2 = [1, 2, 55];

      // vector plus vector
      expect(() => vreplace(1, y)).to.throw(Error, "Argument 'x' must be a vector.");
      expect(() => vreplace(x, 2)).to.throw(Error, "Argument 'y' must be a vector.");
      expect(() => vreplace(x, y, inde1)).to.throw(Error, "Number of values in 'ind' should match the number of elements in 'y'.");
      expect(() => vreplace(x, y, inde2)).to.throw(Error, "Wrong values for indices");

      let res1 = vreplace(x, y, ind1);
      expect(res1).to.eql([11, 20, 30, 14, 15]);

      let res2 = vreplace(x, y, ind2);
      expect(res2).to.eql([10, 20, 11, 40, 50]);
   });

   it('vnorm2() works correctly.', function () {
      const x1 = [1, 2, 3, 4];
      const x2 = [1, -2, -3, 4];
      const x3 = [0, 0, 0, 0];
      const x4 = [1, 1];

      expect(norm2(x1)).to.equal(Math.sqrt(30));
      expect(norm2(x2)).to.equal(Math.sqrt(30));
      expect(norm2(x3)).to.equal(0);
      expect(norm2(x4)).to.equal(Math.sqrt(2));
   });

   it('vapply() works correctly.', function () {

      const x = [1, 4, 9, 16];

      // vector plus vector
      expect(() => vapply(1, Math.sqrt)).to.throw(Error, "Argument 'x' must be a vector.");

      // built in function
      const res1 = vapply(x, Math.sqrt);
      expect(isvector(res1)).to.true;
      expect(res1.length).to.equal(x.length);
      expect(res1).to.eql([1, 2, 3, 4]);


      // user defined function
      const f = a => Math.sqrt(a) * 2;
      const res2 = vapply(x, f);
      expect(isvector(res2)).to.true;
      expect(res2.length).to.equal(x.length);
      expect(res2).to.eql([2, 4, 6, 8]);
   });

   it('vdiv() works correctly.', function () {

      const x = [1, 2, 3, 4];
      const y = [7, 5, 3, 1];
      const e = [1, 2, 3]
      const z = 5;

      // vector plus vector
      expect(() => vdiv(1, 1)).to.throw(Error, "One of the arguments must be a vector.");
      expect(() => vdiv(x, e)).to.throw(Error, "Dimensions of 'x' and 'y' mismatch.");

      const resxy = vdiv(x, y);
      const resyx = vdiv(y, x);

      expect(isvector(resxy)).to.true;
      expect(isvector(resyx)).to.true;
      expect(resxy.length).to.equal(x.length);
      expect(resxy).to.eql([1/7, 2/5, 3/3, 4/1]);
      expect(resyx).to.eql([7/1, 5/2, 3/3, 1/4]);

      // vector and number
      const resxz = vdiv(x, z);
      expect(resxz.length).to.equal(x.length);
      expect(isvector(resxz)).to.be.true;
      expect(resxz).to.eql([1/5, 2/5, 3/5, 4/5]);

      // number and vector
      const reszx = vdiv(z, x);
      expect(reszx.length).to.equal(x.length);
      expect(isvector(reszx)).to.be.true;
      expect(reszx).to.eql([5/1, 5/2, 5/3, 5/4]);
   });

   it('vmult() works correctly.', function () {

      const x = [1, 2, 3, 4];
      const y = [7, 5, 3, 1];
      const e = [1, 2, 3]
      const z = 5;

      // vector and vector
      expect(() => vmult(1, 1)).to.throw(Error, "One of the arguments must be a vector.");
      expect(() => vmult(x, e)).to.throw(Error, "Dimensions of 'x' and 'y' mismatch.");

      const resxy = vmult(x, y);
      const resyx = vmult(y, x);

      expect(isvector(resxy)).to.true;
      expect(isvector(resyx)).to.true;
      expect(resxy.length).to.equal(x.length);
      expect(resxy).to.eql([7, 10, 9, 4]);
      expect(resyx).to.eql(resxy);

      // vector and number
      const resxz = vmult(x, z);
      expect(resxz.length).to.equal(x.length);
      expect(isvector(resxz)).to.be.true;
      expect(resxz).to.eql([5, 10, 15, 20]);

      // number and vector
      const reszx = vmult(z, x);
      expect(reszx.length).to.equal(x.length);
      expect(isvector(reszx)).to.be.true;
      expect(reszx).to.eql([5, 10, 15, 20]);
   });


   it('vadd() works correctly.', function () {

      const x = [1, 2, 3, 4];
      const y = [7, 5, 3, 1];
      const e = [1, 2, 3]
      const z = 5;

      // vector and vector
      expect(() => vadd(1, 1)).to.throw(Error, "One of the arguments must be a vector.");
      expect(() => vadd(x, e)).to.throw(Error, "Dimensions of 'x' and 'y' mismatch.");

      const resxy = vadd(x, y);
      const resyx = vadd(y, x);

      expect(isvector(resxy)).to.true;
      expect(isvector(resyx)).to.true;
      expect(resxy.length).to.equal(x.length);
      expect(resxy).to.eql([8, 7, 6, 5]);
      expect(resyx).to.eql(resxy);

      // vector and number
      const resxz = vadd(x, z);
      expect(resxz.length).to.equal(x.length);
      expect(isvector(resxz)).to.be.true;
      expect(resxz).to.eql([6, 7, 8, 9]);

      // number and vector
      const reszx = vadd(z, x);
      expect(reszx.length).to.equal(x.length);
      expect(isvector(reszx)).to.be.true;
      expect(reszx).to.eql([6, 7, 8, 9]);
   });


   it('vsubtract() works correctly.', function () {

      const x = [1, 2, 3, 4];
      const y = [7, 5, 3, 1];
      const e = [1, 2, 3]
      const z = 5;

      // vector and vector
      expect(() => vsubtract(1, 1)).to.throw(Error, "One of the arguments must be a vector.");
      expect(() => vsubtract(x, e)).to.throw(Error, "Dimensions of 'x' and 'y' mismatch.");

      const resxy = vsubtract(x, y);
      const resyx = vsubtract(y, x);

      expect(isvector(resxy)).to.true;
      expect(isvector(resyx)).to.true;
      expect(resxy.length).to.equal(x.length);
      expect(resxy).to.eql([-6, -3, 0, 3]);
      expect(resyx.length).to.equal(x.length);
      expect(resyx).to.eql([6, 3, 0, -3]);

      // vector and number
      const resxz = vsubtract(x, z);
      expect(resxz.length).to.equal(x.length);
      expect(isvector(resxz)).to.be.true;
      expect(resxz).to.eql([-4, -3, -2, -1]);

      // number and vector
      const reszx = vsubtract(z, x);
      expect(reszx.length).to.equal(x.length);
      expect(isvector(reszx)).to.be.true;
      expect(reszx).to.eql([4, 3, 2, 1]);
   });

   it('vdot() works correctly.', function () {
      expect(() => vdot(1, 1)).to.throw(Error, "Argument 'x' must be a vector of numbers.");
      expect(() => vdot([[1, 2]], 1)).to.throw(Error, "Argument 'x' must be a vector of numbers.");
      expect(() => vdot([1, 2], 1)).to.throw(Error, "Argument 'y' must be a vector of numbers.");
      expect(() => vdot([1, 2], [[1, 2]])).to.throw(Error, "Argument 'y' must be a vector of numbers.");
      expect(() => vdot([1, 2, 3], [4, 5, 6, 7])).to.throw(Error, "Vectors 'x' and 'y' must have the same length.");

      expect(vdot([1, 2, 3], [3, 2, 1])).to.equal(10);
      expect(vdot([1, 2, 1], [1, -1, 1])).to.equal(0);
   });

   it('isvector() works correctly.', function () {
      expect(isvector(1)).to.equal(false)
      expect(isvector([])).to.equal(false)
      expect(isvector([[1, 2]])).to.equal(false)
      expect(isvector([1, 2])).to.equal(true)
      expect(isvector([1])).to.equal(true)
   });


});
