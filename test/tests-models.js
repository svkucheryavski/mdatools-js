/****************************************************************/
/*  Tests for modelling methods                                 */
/****************************************************************/

// import dependencies
import {default as chai} from 'chai';
import {default as chaiAlmost} from 'chai-almost';
import { factor, cbind, vector, matrix, tcrossprod, Vector, Matrix } from '../src/arrays/index.js';
import { variance, mean, sd, sum } from '../src/stat/index.js';
import { svd } from '../src/decomp/index.js';
import { scale as prep_scale } from '../src/prep/index.js';

// import of functions to test
import {simcapredict, getclassres, simpls, plsfit, plspredict, splitregdata, pcrfit, pcrpredict, pcafit, pcapredict, lmfit,
   lmpredict, polyfit, polypredict, getsimcaparams} from '../src/models/index.js';

// set up test settings
const expect = chai.expect;
const should = chai.should();
chai.use(chaiAlmost(0.001));

function testClassResObject(x, cPred, cRef, className, stat) {

   expect(x.class.includes('classres'));
   expect(x.cPred).to.be.deep.equal(cPred);
   expect(x.className).to.be.equal(className);

   if (cRef) {
      expect(x.cRef).to.be.deep.equal(cRef);
      expect(x.TP).to.be.deep.equal(vector(stat[0]));
      expect(x.FP).to.be.deep.equal(vector(stat[1]));
      expect(x.TN).to.be.deep.equal(vector(stat[2]));
      expect(x.FN).to.be.deep.equal(vector(stat[3]));
      expect(x.sensitivity).to.be.deep.equal(vector(stat[4]));
      expect(x.specificity).to.be.deep.equal(vector(stat[5]));
      expect(x.accuracy).to.be.deep.equal(vector(stat[6]));
   } else {
      expect(x.cRef === undefined).to.be.true;
      expect(x.TP === undefined).to.be.true;
      expect(x.FP === undefined).to.be.true;
      expect(x.TN === undefined).to.be.true;
      expect(x.FP === undefined).to.be.true;
      expect(x.sensitivity === undefined).to.be.true;
      expect(x.specificity === undefined).to.be.true;
      expect(x.accuracy === undefined).to.be.true;
   }
}

describe('Tests for classification methods.', function () {

   it ('tests for method "getclassres" - one predicted item.', function () {

      // perfect case
      const cp1 = [factor(['none', 'red', 'none', 'red', 'none', 'red', 'none', 'red'])];
      const cr1 = factor(['blue', 'red', 'blue', 'red', 'blue', 'red', 'green', 'red']);
      const r1a = getclassres(cp1, 'red', cr1);
      const r1b = getclassres(cp1, 'red');
      testClassResObject(r1a, cp1, cr1, 'red', [[4], [0], [4], [0], [1], [1], [1]])
      testClassResObject(r1b, cp1, null, 'red')

      // one false negative
      const cp2 =[factor(['none', 'red', 'none', 'none', 'none', 'red', 'none', 'red'])];
      const cr2 = factor(['blue', 'red', 'green', 'red', 'blue', 'red', 'green', 'red']);
      const r2a = getclassres(cp2, 'red', cr2);
      const r2b = getclassres(cp2, 'red');
      testClassResObject(r2a, cp2, cr2, 'red', [[3], [0], [4], [1], [0.75], [1], [0.875]])
      testClassResObject(r2b, cp2, null, 'red')

      // one false positive
      const cp3 = [factor(['red', 'red', 'none', 'red', 'none', 'red', 'none', 'red'])];
      const cr3 = factor(['blue', 'red', 'blue', 'red', 'blue', 'red', 'green', 'red']);
      const r3a = getclassres(cp3, 'red', cr3);
      const r3b = getclassres(cp3, 'red');
      testClassResObject(r3a, cp3, cr3, 'red', [[4], [1], [3], [0], [1], [0.75], [0.875]])
      testClassResObject(r3b, cp3, null, 'red')

      // everything is wrong
      const cp4 = [factor(['red', 'none', 'red', 'none', 'red', 'none', 'red', 'none'])];
      const cr4 = factor(['blue', 'red', 'blue', 'red', 'blue', 'red', 'green', 'red']);
      const r4a = getclassres(cp4, 'red', cr4);
      const r4b = getclassres(cp4, 'red');
      testClassResObject(r4a, cp4, cr4, 'red', [[0], [4], [0], [4], [0], [0], [0]])
      testClassResObject(r4b, cp4, null, 'red')

      // no class items in predictions (all none)
      const cp5 = [factor(['none', 'none', 'none', 'none', 'none', 'none', 'none', 'none'])];
      const cr5 = factor(['blue', 'red', 'blue', 'red', 'blue', 'red', 'green', 'red']);
      const r5a = getclassres(cp5, 'red', cr5);
      const r5b = getclassres(cp5, 'red');
      testClassResObject(r5a, cp5, cr5, 'red', [[0], [0], [4], [4], [0], [1], [0.5]])
      testClassResObject(r5b, cp5, null, 'red')

      // no class items in references
      const cp6 = [factor(['red', 'none', 'red', 'none', 'red', 'none', 'red', 'none'])];
      const cr6 = factor(['blue', 'green', 'blue', 'green', 'blue', 'green', 'green', 'blue']);
      const r6a = getclassres(cp6, 'red', cr6);
      const r6b = getclassres(cp6, 'red');
      testClassResObject(r6a, cp6, cr6, 'red', [[0], [4], [4], [0], [NaN], [0.5], [0.5]])
      testClassResObject(r6b, cp6, null, 'red')

      // only class items in references
      const cp7 = [factor(['red', 'none', 'red', 'none', 'red', 'none', 'red', 'none'])];
      const cr7 = factor(['red', 'red', 'red', 'red', 'red', 'red', 'red', 'red']);
      const r7a = getclassres(cp7, 'red', cr7);
      const r7b = getclassres(cp7, 'red');
      testClassResObject(r7a, cp7, cr7, 'red', [[4], [0], [0], [4], [0.5], [NaN], [0.5]])
      testClassResObject(r7b, cp7, null, 'red')

   });

   it ('tests for method "getclassres" - several predicted items.', function () {

      // perfect case
      const cp1 = [
         factor(['none', 'red', 'none', 'red', 'none', 'red', 'none', 'red']),
         factor(['none', 'red', 'none', 'none', 'none', 'red', 'none', 'red']),
         factor(['red', 'none', 'red', 'none', 'red', 'none', 'red', 'none']),
         factor(['none', 'none', 'none', 'none', 'none', 'none', 'none', 'none'])
      ];
      const cr1 = factor(['blue', 'red', 'blue', 'red', 'blue', 'red', 'green', 'red']);
      const r1a = getclassres(cp1, 'red', cr1);
      const r1b = getclassres(cp1, 'red');
      testClassResObject(r1a, cp1, cr1, 'red', [
         [4, 3, 0, 0],
         [0, 0, 4, 0],
         [4, 4, 0, 4],
         [0, 1, 4, 4],
         [1, 0.75, 0, 0],
         [1, 1, 0, 1],
         [1, 0.875, 0, 0.5]
      ])
      testClassResObject(r1b, cp1, null, 'red')
   });
});

describe('Tests for SIMCA methods.', function () {

   it ('tests for method "simcapredict".', function () {
      const X1 = Matrix.rand(50, 10);
      const X2 = Matrix.rand(20, 10);
      const mpca = pcafit(X1, 10);
      const params = getsimcaparams('red', 0.05, 'classic');
      const c = simcapredict(mpca, params, X2)
   });

});

describe('Tests for PLS methods.', function () {

   it ('tests for method "pldpredict"', function () {
      // common dataset for all tests
      const data = matrix([
         32, 150, 41, 28000, 119,
         35, 160, 48, 31000, 129,
         36, 166, 47, 28000, 112,
         37, 166, 49, 14000, 123,
         42, 175, 67, 38000, 105,
         43, 180, 80, 30000, 129,
         43, 181, 75, 31000, 105,
         44, 180, 81, 42000, 113
      ], 5, 8).t();

      const [Xc, Yc] = splitregdata(data);

      // make PCA and PCR for centered and scaled data and check all results in full
      const m1 = plsfit(Xc, Yc, 3, true, true);
      const r1 = plspredict(m1, Xc, Yc);
      const r2 = plspredict(m1, Xc);

      const T1 = matrix([
         -0.4945617, -0.36754034, -0.1250141,
         -0.3220733, -0.02656872, -0.5115363,
         -0.1747736, -0.20483128,  0.2758381,
         -0.3630906,  0.53394896,  0.4310806,
          0.2781758, -0.39131972,  0.1470429,
          0.2420359,  0.59936775, -0.3803569,
          0.3655882,  0.01375190,  0.4564504,
          0.4686992, -0.15680855, -0.2935047,
      ], 3, 8).t();

      const H1 = matrix([
         1.7121391, 2.6577404, 2.767140,
         0.7261183, 0.7310596, 2.562745,
         0.2138206, 0.5075116, 1.040118,
         0.9228437, 2.9185541, 4.219368,
         0.5416725, 1.6135903, 1.764942,
         0.4100696, 2.9247616, 3.937461,
         0.9355833, 0.9369071, 2.395336,
         1.5377529, 1.7098753, 2.312890,
      ], 3, 8).t();

      const Q1 = matrix([
         0.8783125, 0.15842635, 9.720942e-02,
         1.1479065, 1.16544180, 3.649739e-02,
         0.7186410, 0.38216058, 6.778351e-02,
         2.0996044, 0.79275237, 8.243227e-05,
         1.0857301, 0.09480393, 3.913798e-03,
         3.0896422, 0.61662640, 2.667597e-03,
         0.8971517, 0.90605443, 1.124068e-02,
         0.4379797, 0.36701856, 1.254689e-03,
      ], 3, 8).t();

      const U1 = matrix([
         -3.950369, -0.16641101, -0.0064092242,
         -2.257354, -0.03450792, -0.0046740695,
         -1.693015, -0.12839530, -0.0102507432,
         -1.128677,  0.29291384,  0.0188942539,
          1.693015, -0.03033479,  0.0206833688,
          2.257354,  0.15737118, -0.0109039174,
          2.257354, -0.03229076, -0.0069754088,
          2.821692, -0.05834525, -0.0003642595,
      ], 3, 8).t();

      const Yp1 = matrix([
         33.25851, 32.25875, 32.19505,
         35.26097, 35.18870, 34.92805,
         36.97101, 36.41384, 36.55440,
         34.78479, 36.23720, 36.45686,
         42.22941, 41.16497, 41.23990,
         41.80985, 43.44021, 43.24640,
         43.24420, 43.28161, 43.51419,
         44.44125, 44.01471, 43.86515,
      ], 3, 8).t();

      // check main outcomes
      expect(r1.T).to.be.deep.almost.equal(T1);
      expect(r1.H).to.be.deep.almost.equal(H1);
      expect(r1.Q).to.be.deep.almost.equal(Q1);
      expect(r1.U).to.be.deep.almost.equal(U1);
      expect(r1.Ypred).to.be.deep.almost.equal(Yp1);

      expect(r2.T).to.be.deep.almost.equal(T1);
      expect(r2.H).to.be.deep.almost.equal(H1);
      expect(r2.Q).to.be.deep.almost.equal(Q1);
      expect(r2.Ypred).to.be.deep.almost.equal(Yp1);

   });

   it ('tests for method "plsfit"', function () {

      // common dataset for all tests
      const data = matrix([
         32, 150, 41, 28000, 119,
         35, 160, 48, 31000, 129,
         36, 166, 47, 28000, 112,
         37, 166, 49, 14000, 123,
         42, 175, 67, 38000, 105,
         43, 180, 80, 30000, 129,
         43, 181, 75, 31000, 105,
         44, 180, 81, 42000, 113
      ], 5, 8).t();

      const [Xc, Yc] = splitregdata(data);

      // make PCA and PCR for centered and scaled data and check all results in full
      const ncomp = 3;
      const m1 = plsfit(Xc, Yc, ncomp, true, true);

      const R1 = matrix([
          0.15492668,  0.1881235,  0.12558132,
          0.15317320,  0.1316393, -0.06823508,
          0.08218702, -0.2824183, -0.32337181,
         -0.05900391,  0.2131759, -0.34074718
      ], 3, 4).t();

      const P1 = matrix([
          2.481456,  0.7967174,  0.3398910,
          2.526248,  0.6219986, -0.3463264,
          1.828851, -1.5105447, -1.1672083,
         -1.326946,  1.6025908, -1.6324190
      ], 3, 4).t();

      const C1 = matrix([
         2.559598, 0.5997323, 0.1123446
      ], 3, 1).t();

      const yeigenvals1 = vector([
         6.5515441996, 0.0230428629, 0.0001600688
      ]);

      const xeigenvals1 = vector([
         0.1428571, 0.1428571, 0.1428571
      ]);

      const Nq1 = vector([
         4, 4, 1
      ]);

      const Nh1 = vector([
         6, 6, 13
      ]);

      const q01 = vector([
         1.29437100, 0.56041055, 0.02758119
      ]);

      const h01 = vector([
         0.875, 1.750, 2.625
      ]);

      // check that the model outcomes are correct
      expect(m1.P).to.be.deep.almost.equal(P1);
      expect(m1.R).to.be.deep.almost.equal(R1);
      expect(m1.C).to.be.deep.almost.equal(C1);
      expect(m1.yeigenvals).to.be.deep.almost.equal(yeigenvals1);
      expect(m1.xeigenvals).to.be.deep.almost.equal(xeigenvals1);
      expect(m1.qParams['classic'][1]).to.be.deep.almost.equal(Nq1);
      expect(m1.hParams['classic'][1]).to.be.deep.almost.equal(Nh1);
      expect(m1.qParams['classic'][0]).to.be.deep.almost.equal(q01);
      expect(m1.hParams['classic'][0]).to.be.deep.almost.equal(h01);
   });

   it ('tests for method "simpls"', function () {

      // common dataset for all tests
      const data = matrix([
         32, 150, 41, 28000, 119,
         35, 160, 48, 31000, 129,
         36, 166, 47, 28000, 112,
         37, 166, 49, 14000, 123,
         42, 175, 67, 38000, 105,
         43, 180, 80, 30000, 129,
         43, 181, 75, 31000, 105,
         44, 180, 81, 42000, 113
      ], 5, 8).t();

      const [Xc, Yc] = splitregdata(data);

      const Xp = prep_scale(Xc, true, true);
      const Yp = prep_scale(Yc, true, true);

      const R1 = matrix([
          0.15492668,  0.1881235,  0.12558132,
          0.15317320,  0.1316393, -0.06823508,
          0.08218702, -0.2824183, -0.32337181,
         -0.05900391,  0.2131759, -0.34074718
      ], 3, 4).t();

      const P1 = matrix([
          2.481456,  0.7967174,  0.3398910,
          2.526248,  0.6219986, -0.3463264,
          1.828851, -1.5105447, -1.1672083,
         -1.326946,  1.6025908, -1.6324190
      ], 3, 4).t();

      const T1 = matrix([
         -0.4945617, -0.36754034, -0.1250141,
         -0.3220733, -0.02656872, -0.5115363,
         -0.1747736, -0.20483128,  0.2758381,
         -0.3630906,  0.53394896,  0.4310806,
          0.2781758, -0.39131972,  0.1470429,
          0.2420359,  0.59936775, -0.3803569,
          0.3655882,  0.01375190,  0.4564504,
          0.4686992, -0.15680855, -0.2935047
      ], 3, 8).t();

      const C1 = matrix([
         2.559598, 0.5997323, 0.1123446
      ], 3, 1).t();

      const U1 = matrix([
         -3.950369, -0.16641101, -0.0064092242,
         -2.257354, -0.03450792, -0.0046740695,
         -1.693015, -0.12839530, -0.0102507432,
         -1.128677,  0.29291384,  0.0188942539,
          1.693015, -0.03033479,  0.0206833688,
          2.257354,  0.15737118, -0.0109039174,
          2.257354, -0.03229076, -0.0069754088,
          2.821692, -0.05834525, -0.0003642595
      ], 3, 8).t();

      // partial decomposition - compare with results from R
      const m1 = simpls(Xp, Yp, 3);

      // decomposition of X
      expect(m1.R).to.be.deep.almost.equal(R1);
      expect(m1.P).to.be.deep.almost.equal(P1);
      expect(m1.T).to.be.deep.almost.equal(T1);

      // decomposition of Y
      expect(m1.C).to.be.deep.almost.equal(C1);
      expect(m1.U).to.be.deep.almost.equal(U1);

      // full decomposition - check that X = TP' and Y = UC'
      const m2 = simpls(Xp, Yp, 4);
      // residual variance for X is almost 0
      expect(tcrossprod(m2.T, m2.P)).to.be.deep.almost.equal(Xp);
      // residual variance for Y is below 5%
      expect(sum(Yp.subtract(tcrossprod(m2.T, m2.C)).apply(v => v * v, 0).v) / sum(Yp.apply(v => v * v, 0).v) < 0.05).to.been.true;

   });
});

describe('Tests for PCR methods.', function () {

   it ('tests for method "pcrfit"', function () {

      // common dataset for all PCR tests
      const data = matrix([
         36, 166, 47, 28000, 112,
         43, 180, 80, 30000, 129,
         44, 180, 81, 42000, 113,
         36, 166, 49, 14000, 123,
         43, 181, 75, 31000, 105,
         42, 175, 67, 38000, 105,
         35, 160, 48, 31000, 129
      ], 5, 7).t();

      const [Xc, Yc] = splitregdata(data);

      // make PCA and PCR for centered and scaled data and check all results in full
      const ncomp = 3;
      const m1 = pcrfit(Xc, Yc, ncomp, true, true);
      const m2 = pcafit(Xc, ncomp, true, true);
      const C1 = matrix([0.5953372, -0.2370964, 0.1571044], 1, ncomp);

      // check that PCA part is correct
      expect(m1.P).to.be.deep.almost.equal(m2.P);
      expect(m1.eigenvals).to.be.deep.almost.equal(m2.eigenvals);
      expect(m1.mX).to.be.deep.almost.equal(m2.mX);
      expect(m1.sX).to.be.deep.almost.equal(m2.sX);
      expect(m1.center).to.be.deep.almost.equal(m2.center);
      expect(m1.scale).to.be.deep.almost.equal(m2.scale);
      expect(m1.h0).to.be.deep.almost.equal(m2.h0);
      expect(m1.q0).to.be.deep.almost.equal(m2.q0);
      expect(m1.Nq).to.be.deep.almost.equal(m2.Nq);
      expect(m1.Nh).to.be.deep.almost.equal(m2.Nh);

      // check that PCR part is correct
      expect(m1.C.v.map(v => Math.abs(v))).to.be.deep.almost.equal(C1.v.map(v => Math.abs(v)));
      expect(m1.mY).to.be.deep.equal(Yc.apply(mean, 2));
      expect(m1.sY).to.be.deep.equal(Yc.apply(sd, 2));


      // make PCR for centered and non-scaled data and check only results PCR results
      const m3 = pcrfit(Xc, Yc, ncomp, true, false);
      const C3 = matrix([0.0002877219, -0.2121022329, 0.0296248313], 1, ncomp);
      expect(m3.C.v.map(v => Math.abs(v))).to.be.deep.almost.equal(C3.v.map(v => Math.abs(v)));
      expect(m3.mY).to.be.deep.equal(Yc.apply(mean));
      expect(m3.sY).to.be.deep.equal(vector([1]));
   });

   it ('tests for method "pcrpredict"', function () {

      // common dataset for all PCR tests
      const data = matrix([
         36, 166, 47, 28000, 112,
         43, 180, 80, 30000, 129,
         44, 180, 81, 42000, 113,
         36, 166, 49, 14000, 123,
         43, 181, 75, 31000, 105,
         42, 175, 67, 38000, 105,
         35, 160, 48, 31000, 129
      ], 5, 7).t();

      const [Xc, Yc] = splitregdata(data);

      // make PCA and PCR for centered and scaled data and check all results in full
      const m1 = pcrfit(Xc, Yc, 3, true, true);
      const m2 = pcafit(Xc, 3, true, true);

      const r1 = pcrpredict(m1, Xc, Yc);
      const r2 = pcapredict(m2, Xc);

      // check that PCA part is correct
      expect(r2.T).to.be.deep.almost.equal(r1.T);
      expect(r2.H).to.be.deep.almost.equal(r1.H);
      expect(r2.Q).to.be.deep.almost.equal(r1.Q);
      expect(r2.expvar).to.be.deep.almost.equal(r1.expvar);
      expect(r2.cumexpvar).to.be.deep.almost.equal(r1.cumexpvar);

   });

});

describe('Tests for PCA methods.', function () {

   it ('tests for method "pcafit"', function () {

      const X1 = Matrix.rand(50, 10);
      const mX1 = X1.apply(mean, 2);
      const m11 = pcafit(X1);
      const m12 = svd(X1.subtract(mX1));
      const T1 = m12.U.dot(Matrix.diagm(m12.s));

      expect(m11.P).to.be.deep.almost.equal(m12.V);
      expect(m11.mX).to.be.deep.almost.equal(mX1);
      expect(m11.sX).to.be.deep.almost.equal(Vector.ones(X1.ncols));
      expect(m11.eigenvals).to.be.deep.almost.equal(T1.apply(variance, 2));
      expect(m11.center).to.be.true;
      expect(m11.scale).to.be.false;
      expect(m11.ncomp).to.be.equal(10);

      const X2 = Matrix.rand(10, 50);
      const mX2 = X2.apply(mean, 2);
      const sX2 = X2.apply(sd, 2);
      const m21 = pcafit(X2, 5, true, true);
      const m22 = svd(X2.subtract(mX2).divide(sX2), 5);
      const T2 = m22.U.dot(Matrix.diagm(m22.s));

      expect(m21.P).to.be.deep.almost.equal(m22.V);
      expect(m21.mX).to.be.deep.almost.equal(mX2);
      expect(m21.sX).to.be.deep.almost.equal(sX2);
      expect(m21.eigenvals).to.be.deep.almost.equal(T2.apply(variance, 2));
      expect(m21.center).to.be.true;
      expect(m21.scale).to.be.true;
      expect(m21.ncomp).to.be.equal(5);

      // compare with results from R
      const X3 = matrix([
         166, 180, 157, 180, 166, 181, 175, 160,
         47, 80, 47, 81, 49, 75, 67, 48,
         28000, 30000, 32000, 42000, 14000, 31000, 38000, 31000,
         112, 129, 127, 113, 123, 105, 105, 129
      ], 8, 4);

      const P3 = matrix([
          0.5732978, -0.3559463, -0.2788432,  0.6832850,
          0.5707159, -0.4525553,  0.0850746, -0.6798818,
          0.4182815,  0.4024368,  0.7935761,  0.1825437,
         -0.4131024, -0.7117166,  0.5340840,  0.1938042
      ], 4, 4).t();

      const eigenvals3 = vector([2.594793132, 0.714955083, 0.682758553, 0.007493231]);

      // moments based
      const h03m = vector([0.875000, 1.750000,  2.625000,  3.500000]);
      const q03m = vector([1.229556, 0.6039703, 0.006556578, 0.0]);
      const Nh3m = vector([6.0, 6.0, 12.0, 16.0]);
      const Nq3m = vector([4.0, 3.0, 1.0, 4.0]);

      // robust based
      const h03r = vector([1.041937, 1.484588, 2.282291, 3.934952]);
      const q03r = vector([0.977954, 0.671042, 0.005573, 0.0]);
      const Nh3r = vector([13.0, 19.0, 20.0, 11.0]);
      const Nq3r = vector([11.0, 5.0, 2.0, 7.0]);

      const m3 = pcafit(X3, 4, true, true);

      expect(m3.P.apply(Math.abs, 0)).to.be.deep.almost.equal(P3.apply(Math.abs, 0));
      expect(m3.eigenvals).to.be.deep.almost.equal(eigenvals3);

      expect(m3.qParams['classic'][1].slice(1, 3)).to.be.deep.almost.equal(Nq3m.slice(1, 3));
      expect(m3.hParams['classic'][1]).to.be.deep.almost.equal(Nh3m);
      expect(m3.hParams['classic'][0]).to.be.deep.almost.equal(h03m);
      expect(m3.qParams['classic'][0]).to.be.deep.almost.equal(q03m);

      chai.use(chaiAlmost(0.01));
      expect(m3.qParams['robust'][1].slice(1, 3)).to.be.deep.almost.equal(Nq3r.slice(1, 3));
      expect(m3.hParams['robust'][1]).to.be.deep.almost.equal(Nh3r);
      expect(m3.hParams['robust'][0]).to.be.deep.almost.equal(h03r);
      expect(m3.qParams['robust'][0]).to.be.deep.almost.equal(q03r);
      chai.use(chaiAlmost(0.001));

   });

   it ('tests for method "pcapredict"', function () {

      const X1 = Matrix.rand(50, 10);
      const m11 = pcafit(X1);
      const mX1 = X1.apply(mean, 2);
      const m12 = svd(X1.subtract(mX1));
      const T1 = m12.U.dot(Matrix.diagm(m12.s));

      const r1 = pcapredict(m11, X1);
      const q12 = X1.subtract(mX1)
         .subtract(tcrossprod(r1.T.subset([], [1, 2]), m11.P.subset([], [1, 2])))
         .apply(v => v * v, 0)
         .apply(sum, 1);
      const h12 = m12.U.subset([], [1, 2])
         .apply(v => v * v, 0)
         .apply(sum, 1)
         .apply(v => v * (X1.nrows -1));

      expect(r1.T.apply(Math.abs, 0)).to.be.deep.almost.equal(T1.apply(Math.abs, 0));
      expect(r1.Q.getcolumn(2)).to.be.deep.almost.equal(q12);
      expect(r1.H.getcolumn(2)).to.be.deep.almost.equal(h12);

      const X2 = Matrix.rand(10, 50);
      const mX2 = X2.apply(mean, 2);
      const sX2 = X2.apply(sd, 2);
      const m21 = pcafit(X2, 5, true, true);
      const m22 = svd(X2.subtract(mX2).divide(sX2), 5);
      const T2 = m22.U.dot(Matrix.diagm(m22.s));

      const r2 = pcapredict(m21, X2);
      const q22 = X2.subtract(mX2).divide(sX2).
         subtract(tcrossprod(r2.T.subset([], [1, 2]), m21.P.subset([], [1, 2]))).apply(v => v * v, 0).apply(sum, 1);
      const h22 = m22.U.subset([], [1, 2]).apply(v => v * v, 0).apply(sum, 1).apply(v => v * (X2.nrows -1));

      expect(r2.T.apply(Math.abs, 0)).to.be.deep.almost.equal(T2.apply(Math.abs, 0));
      expect(r2.Q.getcolumn(2)).to.be.deep.almost.equal(q22);
      expect(r2.H.getcolumn(2)).to.be.deep.almost.equal(h22);


      // compare with results from R
      const X3 = matrix([
         166, 180, 157, 180, 166, 181, 175, 160,
         47, 80, 47, 81, 49, 75, 67, 48,
         28000, 30000, 32000, 42000, 14000, 31000, 38000, 31000,
         112, 129, 127, 113, 123, 105, 105, 129
      ], 8, 4);

      const T3 = matrix([
         -0.7231736,  0.87047760, -0.51644930,  0.143706174,
          0.7468125, -1.68232514,  0.32979598,  0.062205787,
         -1.6567570,  0.36752606,  0.90839404, -0.126899054,
          2.0362791, -0.01931637,  0.67077094, -0.014230821,
         -1.8047465, -0.63358644, -1.29333991, -0.049380702,
          1.6340299,  0.12882816, -0.87065320, -0.076003459,
          1.3393512,  0.92824983, -0.06167086,  0.002243734,
         -1.5717957,  0.04014630,  0.83315232,  0.058358340
      ], 4, 8).t();

      const H3 = matrix([
         0.2015498, 1.2613804, 1.652031, 4.408047,
         0.2149415, 4.1735369, 4.332840, 4.849247,
         1.0578276, 1.2467561, 2.455353, 4.604408,
         1.5979819, 1.5985038, 2.257498, 2.284524,
         1.2552483, 1.8167267, 4.266682, 4.592103,
         1.0290045, 1.0522181, 2.162474, 2.933374,
         0.6913312, 1.8965087, 1.902079, 1.902751,
         0.9521151, 0.9543694, 1.971043, 2.425546
      ], 4, 8).t();

      const Q3 = matrix([
         1.0451026, 0.28737135, 2.065146e-02, 0.0,
         2.9428528, 0.11263495, 3.869560e-03, 0.0,
         0.9763585, 0.84128309, 1.610337e-02, 0.0,
         0.4505093, 0.45013617, 2.025163e-04, 0.0,
         2.0765984, 1.67516659, 2.438454e-03, 0.0,
         0.7804102, 0.76381352, 5.776526e-03, 0.0,
         0.8654561, 0.00380833, 5.034343e-06, 0.0,
         0.6991602, 0.69754849, 3.405696e-03, 0.0
      ], 4, 8).t();

      const cumexpvar3 = vector([64.86983, 82.74371, 99.81267, 100.00000]);
      const expvar3 = vector([64.8698283, 17.8738771, 17.0689638, 0.1873308]);
      const m3 = pcafit(X3, 4, true, true);
      const r3 = pcapredict(m3, X3)

      expect(r3.T.apply(Math.abs, 0)).to.be.deep.almost.equal(T3.apply(Math.abs, 0));
      expect(r3.Q).to.be.deep.almost.equal(Q3);
      expect(r3.H).to.be.deep.almost.equal(H3);
      expect(r3.expvar).to.be.deep.almost.equal(expvar3);
      expect(r3.cumexpvar).to.be.deep.almost.equal(cumexpvar3)

      // same model new prediction set
      const X4 = matrix([
         168, 52, 23500, 100,
         166, 49, 14000, 123
      ], 4, 2).t();

      const T4 = matrix([
         -0.1700773,  1.2564700, -1.603759, -0.2578108,
         -1.8047465, -0.6335864, -1.293340, -0.0493807
      ], 4, 2).t();

      const H4 = matrix([
         0.01114782, 2.219282, 5.986415, 14.856606,
         1.25524835, 1.816727, 4.266682,  4.592103
      ], 4, 2).t();

      const Q4 = matrix([
         4.217225, 2.638508, 0.066466391, 1.195617e-30,
         2.076598, 1.675167, 0.002438454, 1.103173e-30
      ], 4, 2).t();

      const cumexpvar4 = vector([34.30150,  54.97142,  99.28073, 100.00000]);
      const expvar4 = vector([34.3015044, 20.6699134, 44.3093144, 0.7192678]);
      const r4 = pcapredict(m3, X4);

      expect(r4.T.apply(Math.abs, 0)).to.be.deep.almost.equal(T4.apply(Math.abs, 0));
      expect(r4.Q).to.be.deep.almost.equal(Q4);
      expect(r4.H).to.be.deep.almost.equal(H4);
      expect(r4.expvar).to.be.deep.almost.equal(expvar4);
      expect(r4.cumexpvar).to.be.deep.almost.equal(cumexpvar4);

   });

});

describe('Tests for polynomial regression methods.', function () {

   it ('tests for method "polyfit".', function () {

      const x = vector([1, 2, 3, 4, 5]);
      const y = vector([11, 14, 19, 26, 35]);

      expect(() => polyfit(x, y, 5)).to.throw(Error, 'polyfit: polynomial degree "d" must a positive value smaller than number of measurements.');
      expect(() => polyfit(cbind(x, x), y, 1)).to.throw(Error, 'polymat: argument "x" must be a vector.');

      const m1 = polyfit(x, y, 3);

      // check class and polynomial degree
      m1.class.should.be.equal("pm");
      m1.pdegree.should.be.equal(3);

      // check coefficients and related statistics
      m1.coeffs.estimate.length.should.be.equal(4);
      expect(m1.coeffs.estimate).to.be.deep.almost(vector([10, 0, 1, 0]));
      expect(m1.coeffs.p.apply(v => v < 0.10)).to.be.deep.almost(vector([1, 0, 1, 0]));
   });

   it ('tests for method "polypredict".', function () {
      const x = vector([1, 2, 3, 4, 5]);
      const y = vector([11, 14, 19, 26, 35]);
      const m = polyfit(x, y, 3);

      const yp = polypredict(m, x);
      expect(yp).to.be.deep.almost(y);
   });

});

describe('Tests for lm (MLR) methods.', function () {

   it ('tests for method "lmfit".', function () {

      // errors
      expect(() => lmfit(1, 2)).to.throw(Error,
         'lmfit: argument "X" must be a matrix or a vector.');
      expect(() => lmfit(vector([1]), 2)).to.throw(Error,
         'lmfit: argument "y" must be a vector.');
      expect(() => lmfit(vector([1]), vector([2]))).to.throw(Error,
         'lmfit: number of objects must be larger than number of predictors.');
      expect(() => lmfit(vector([1, 2]), vector([2, 3, 4]))).to.throw(Error,
         'lmfit: arguments "X" and "y" must have the same number of objects.');

      // vector - vector
      const X = vector([1, 2, 3, 4, 5]);
      const y = vector([10, 19, 31, 39, 55]);
      const m = lmfit(X, y);

      m.class.should.be.equal('lm');

      // check coefficients and related statistics
      m.coeffs.estimate.length.should.be.equal(2);
      expect(m.coeffs.estimate).to.be.deep.almost(vector([-2.2, 11.0]));
      expect(m.coeffs.se).to.be.deep.almost(vector([2.3295, 0.7024]));
      expect(m.coeffs.tstat).to.be.deep.almost(vector([-0.944, 15.661]));
      expect(m.coeffs.p).to.be.deep.almost(vector([0.4146, 0.000566]));

      // check predictions
      m.fitted.length.should.be.equal(y.length);
      expect(m.fitted).to.be.deep.almost(vector([8.8, 19.8, 30.8, 41.8, 52.8]));

      // check performance statistics
      m.stat.DoF.should.be.equal(3);
      expect(m.stat.se).to.be.almost(2.221);
      expect(m.stat.R2).to.be.almost(0.9879);
      expect(m.stat.R2adj).to.be.almost(0.9839);
      expect(m.stat.Fstat).to.be.almost(245.270);
      expect(m.stat.p).to.be.almost(0.0005658);

      // matrix - vector
      const X2 = matrix([1, 2, 3, 4, 5, 10, 20, 10, 20, 10], 5, 2);
      const y2 = vector([10, 19, 31, 39, 55]);
      const m2 = lmfit(X2, y2);

      m.class.should.be.equal("lm");

      // check coefficients and related statistics
      m2.coeffs.estimate.length.should.be.equal(3);
      expect(m2.coeffs.estimate).to.be.deep.almost(vector([2.0, 11.0, -0.3]));
      expect(m2.coeffs.se).to.be.deep.almost(vector([2.3381, 0.4472, 0.1291]));
      expect(m2.coeffs.tstat).to.be.deep.almost(vector([0.855, 24.597, -2.324]));
      expect(m2.coeffs.p).to.be.deep.almost(vector([0.48245, 0.00165, 0.14576]));

      // check predictions
      m2.fitted.length.should.be.equal(y.length);
      expect(m2.fitted).to.be.deep.almost(vector([10, 18, 32, 40, 54]));

      // check performance statistics
      m2.stat.DoF.should.be.equal(2);
      expect(m2.stat.se).to.be.almost(1.414);
      expect(m2.stat.R2).to.be.almost(0.9967);
      expect(m2.stat.R2adj).to.be.almost(0.9935);
      expect(m2.stat.Fstat).to.be.almost(305.2);
      expect(m2.stat.p).to.be.almost(0.003266);

   });

   it('tests for method "lmpredict".', function () {

      // vector - vector
      const X1 = vector([1, 2, 3, 4, 5]);
      const y1 = vector([10, 19, 31, 39, 55]);
      const m1 = lmfit(X1, y1);

      // errors
      expect(() => lmpredict({}, 2)).to.throw(Error, 'lmpredict: argument "X" must be a matrix or a vector.');
      expect(() => lmpredict({}, vector([2, 3]))).to.throw(Error, 'lmpredict: argument "m" must be object with "lm" model.');
      expect(() => lmpredict(m1, matrix([2, 3], 1, 2))).to.throw(Error, 'lmpredict: number of columns in "X" does not match number of coefficients in model.');

      const yp1 = lmpredict(m1, X1)
      yp1.should.be.eql(m1.fitted);

      // matrix - vector
      const X2 = matrix([1, 2, 3, 4, 5, 10, 20, 10, 20, 10], 5, 2);
      const y2 = vector([10, 19, 31, 39, 55]);
      const m2 = lmfit(X2, y2);

      const yp2 = lmpredict(m2, X2)
      yp2.should.be.eql(m2.fitted);
   });

});
