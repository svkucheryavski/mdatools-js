// /******************************************************************
//  *  Tests for functions for hypothesis testing                    *
//  ******************************************************************/

// import dependencies
import {default as chai} from 'chai';
import {pnorm} from '../distributions/index.js';
import {vector} from '../arrays/index.js';

// import of functions to test
import {getpvalue, ttest, ttest2} from '../tests/index.js';

const should = chai.should();
const expect = chai.expect;

describe('Tests of methods for hypothesis testing.', function () {

   it('tests for method "getpvalue".', function () {

      // left tail
      getpvalue(pnorm, 0, "left").should.be.closeTo(0.5, 0.00001);
      getpvalue(pnorm, 10, "left", [10, 1]).should.be.closeTo(0.5, 0.00001);
      getpvalue(pnorm, 8.04, "left", [10, 1]).should.be.closeTo(0.025, 0.00001);
      getpvalue(pnorm, 11.96, "left", [10, 1]).should.be.closeTo(0.975, 0.00001);

      // right tail
      getpvalue(pnorm, 0, "right").should.be.closeTo(0.5, 0.00001);
      getpvalue(pnorm, 10, "right", [10, 1]).should.be.closeTo(0.5, 0.00001);
      getpvalue(pnorm, 8.04, "right", [10, 1]).should.be.closeTo(0.975, 0.00001);
      getpvalue(pnorm, 11.96, "right", [10, 1]).should.be.closeTo(0.025, 0.00001);

      // both tails
      getpvalue(pnorm, 0, "both").should.be.closeTo(1.0, 0.00001);
      getpvalue(pnorm, 10, "both", [10, 1]).should.be.closeTo(1.0, 0.00001);
      getpvalue(pnorm, 8.04, "both", [10, 1]).should.be.closeTo(0.05, 0.00001);
      getpvalue(pnorm, 11.96, "both", [10, 1]).should.be.closeTo(0.05, 0.00001);
   });

   it('tests for method "ttest".', function () {

      let res = ttest(vector([-2, -1, 0, 1, 2]));

      // exact numbers
      res.effectObserved.should.be.equal(0);
      res.alpha.should.be.equal(0.05);
      res.tail.should.be.equal("both");
      res.DoF.should.be.equal(4);

      // estimated numbers
      res.se.should.be.closeTo(0.7071068, 0.00001);
      res.tValue.should.be.closeTo(0, 0.00001);
      res.pValue.should.be.closeTo(1, 0.00001);
      res.ci[0].should.be.closeTo(-1.963243, 0.00001);
      res.ci[1].should.be.closeTo(1.963243, 0.00001);

      // same but for left tail
      res = ttest(vector([-2, -1, 0, 1, 2]), 0, 0.05, "left");

      // exact numbers
      res.effectObserved.should.be.equal(0);
      res.alpha.should.be.equal(0.05);
      res.tail.should.be.equal("left");
      res.DoF.should.be.equal(4);

      // estimated numbers
      res.se.should.be.closeTo(0.7071068, 0.00001);
      res.tValue.should.be.closeTo(0, 0.00001);
      res.pValue.should.be.closeTo(0.5, 0.00001);
      res.ci[0].should.be.closeTo(-1.963243, 0.00001);
      res.ci[1].should.be.closeTo(1.963243, 0.00001);

      // same but for right tail
      res = ttest(vector([-2, -1, 0, 1, 2]), 0, 0.05, "right");

      // exact numbers
      res.effectObserved.should.be.equal(0);
      res.alpha.should.be.equal(0.05);
      res.tail.should.be.equal("right");
      res.DoF.should.be.equal(4);

      // estimated numbers
      res.se.should.be.closeTo(0.7071068, 0.00001);
      res.tValue.should.be.closeTo(0, 0.00001);
      res.pValue.should.be.closeTo(0.5, 0.00001);
      res.ci[0].should.be.closeTo(-1.963243, 0.00001);
      res.ci[1].should.be.closeTo(1.963243, 0.00001);

      // changed mu and alpha - both tails
      res = ttest(vector([-3, -2, -1, 0, 1, 2]), 2, 0.01, "both");

      // exact numbers
      res.effectExpected.should.be.equal(2.0);
      res.effectObserved.should.be.equal(-0.5);
      res.alpha.should.be.equal(0.01);
      res.tail.should.be.equal("both");
      res.DoF.should.be.equal(5);

      // estimated numbers
      res.se.should.be.closeTo(0.7637626, 0.0001);
      res.tValue.should.be.closeTo(-3.2733, 0.0001);
      res.pValue.should.be.closeTo( 0.02212, 0.0001);
      res.ci[0].should.be.closeTo(-3.5796, 0.0001);
      res.ci[1].should.be.closeTo( 2.5796, 0.0001);


      // changed mu and alpha - left tail
      res = ttest(vector([-3, -2, -1, 0, 1, 2]), 2, 0.01, "left");

      // exact numbers
      res.effectExpected.should.be.equal(2.0);
      res.effectObserved.should.be.equal(-0.5);
      res.alpha.should.be.equal(0.01);
      res.tail.should.be.equal("left");
      res.DoF.should.be.equal(5);

      // estimated numbers
      res.se.should.be.closeTo(0.7637626, 0.0001);
      res.tValue.should.be.closeTo(-3.2733, 0.0001);
      res.pValue.should.be.closeTo( 0.01106, 0.0001);
      res.ci[0].should.be.closeTo(-3.5796, 0.0001);
      res.ci[1].should.be.closeTo( 2.5796, 0.0001);

      // changed mu and alpha - right tail
      // changed mu and alpha - left tail
      res = ttest(vector([-3, -2, -1, 0, 1, 2]), 2, 0.01, "right");

      // exact numbers
      res.effectExpected.should.be.equal(2.0);
      res.effectObserved.should.be.equal(-0.5);
      res.alpha.should.be.equal(0.01);
      res.tail.should.be.equal("right");
      res.DoF.should.be.equal(5);

      // estimated numbers
      res.se.should.be.closeTo(0.7637626, 0.0001);
      res.tValue.should.be.closeTo(-3.2733, 0.0001);
      res.pValue.should.be.closeTo( 0.9889, 0.0001);
      res.ci[0].should.be.closeTo(-3.5796, 0.0001);
      res.ci[1].should.be.closeTo( 2.5796, 0.0001);
   });

   it('tests for method "ttest2".', function () {
      let res = ttest2(vector([-2, -1, 0, 1, 2]), vector([-3, -2, -1, 0, 1]));

      // exact numbers
      res.effectObserved.should.be.equal(1);
      res.alpha.should.be.equal(0.05);
      res.tail.should.be.equal("both");
      res.DoF.should.be.equal(8);

      // estimated numbers
      res.se.should.be.closeTo(1, 0.00001);
      res.tValue.should.be.closeTo(1, 0.00001);
      res.pValue.should.be.closeTo(0.3466, 0.0001);
      res.ci[0].should.be.closeTo(-1.306004, 0.00001);
      res.ci[1].should.be.closeTo(3.306004, 0.00001);

      res = ttest2(vector([-2, -1, 0, 1, 2]), vector([-3, -2, -1, 0, 1]), 0.05, "left");

      // exact numbers
      res.effectObserved.should.be.equal(1);
      res.alpha.should.be.equal(0.05);
      res.tail.should.be.equal("left");
      res.DoF.should.be.equal(8);

      // estimated numbers
      res.se.should.be.closeTo(1, 0.00001);
      res.tValue.should.be.closeTo(1, 0.00001);
      res.pValue.should.be.closeTo(0.8267, 0.0001);
      res.ci[0].should.be.closeTo(-1.306004, 0.00001);
      res.ci[1].should.be.closeTo(3.306004, 0.00001);

      res = ttest2(vector([-2, -1, 0, 1, 2]), vector([-3, -2, -1, 0, 1]), 0.05, "right");

      // exact numbers
      res.effectObserved.should.be.equal(1);
      res.alpha.should.be.equal(0.05);
      res.tail.should.be.equal("right");
      res.DoF.should.be.equal(8);

      // estimated numbers
      res.se.should.be.closeTo(1, 0.00001);
      res.tValue.should.be.closeTo(1, 0.00001);
      res.pValue.should.be.closeTo(0.1733, 0.0001);
      res.ci[0].should.be.closeTo(-1.306004, 0.00001);
      res.ci[1].should.be.closeTo(3.306004, 0.00001);

      res = ttest2(vector([-2, -1, 0, 1, 2]), vector([-3, -2, -1, 0, 1]), 0.01);

      // exact numbers
      res.effectObserved.should.be.equal(1);
      res.alpha.should.be.equal(0.01);
      res.tail.should.be.equal("both");
      res.DoF.should.be.equal(8);

      // estimated numbers
      res.se.should.be.closeTo(1, 0.00001);
      res.tValue.should.be.closeTo(1, 0.00001);
      res.pValue.should.be.closeTo(0.3466, 0.0001);
      res.ci[0].should.be.closeTo(-2.355387, 0.00001);
      res.ci[1].should.be.closeTo(4.355387, 0.00001);
   });

});

