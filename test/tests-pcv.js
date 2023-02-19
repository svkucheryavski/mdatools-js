/*****************************************************************
 *  Tests for PCV methods                                        *
 *****************************************************************/

// import dependencies
import {default as chai} from 'chai';
import {default as chaiAlmost} from 'chai-almost';

import { matrix, vector, index, c, Matrix, Index } from '../arrays/index.js';
import { pcafit, pcapredict, pcrfit, pcrpredict, plsfit, plspredict, splitregdata } from '../models/index.js';

// import methods to test
import { cv2obs, crossval, pcvpca, pcvpcr, pcvpls } from '../pcv/index.js';

// set up test settings
const expect = chai.expect;
chai.use(chaiAlmost(0.00001));

describe('Tests for PCV methods', function () {

   // common dataset for PCR and PLS tests
   const data = matrix([
      32, 150, 41, 28000, 119,
      35, 160, 48, 31000, 129,
      36, 166, 47, 28000, 112,
      37, 166, 49, 14000, 123,
      42, 175, 67, 38000, 105,
      43, 180, 80, 30000, 129,
      43, 181, 75, 31000, 105,
      44, 180, 81, 42000, 113,
   ], 5, 8).t();

   it ('tests for method "pcvpls"', function () {

      const D1 = matrix([
         0.7618305, 0.9701346,  0.8583589,
         0.8664892, 1.1233015, -0.4189311,
         0.9040871, 1.0111721,  1.4835077,
         0.9153578, 0.4453001, -1.0219994,
      ], 3, 4);

      const T1 = matrix([
         -0.4395929, -0.47753680, -0.1098689,
         -0.2967576, -0.25378922,  0.2791594,
         -0.1757461, -0.17091191,  0.5104952,
         -0.3086492,  0.03666759, -2.1360181,
          0.2527472, -0.40208568,  0.2000121,
          0.1879328,  0.80050607,  0.3795580,
          0.3488912,  0.11874715,  0.8917092,
          0.4663885,  0.04006119,  1.4380749,
      ], 3, 8).t();

      const H1 = matrix([
         1.3526935, 2.9489832,  3.033481,
         0.6164554, 1.0673182,  1.612828,
         0.2162068, 0.4206830,  2.244920,
         0.6668503, 0.6762619, 32.614276,
         0.4471680, 1.5788783,  1.858912,
         0.2472311, 4.7329009,  5.741351,
         0.8520756, 0.9507819,  6.516799,
         1.5226279, 1.5338622, 16.010279,
      ], 3, 8).t();

      const Q1 = matrix([
          1.5700065,  0.3147137, 0.271188024,
          0.9251551,  0.4339503, 0.090147424,
          1.5304402,  1.2197429, 0.087902017,
         19.5613906, 19.4285563, 0.010954067,
          1.2592216,  0.1816409, 0.008551453,
          3.8916987,  0.6137465, 0.002013308,
          3.3392149,  3.4253307, 0.018918789,
          8.7232886,  8.8057658, 0.013770933,
      ], 3, 8).t();

      const Ypred1 = matrix([
         33.89666, 32.59770, 32.54171,
         35.55487, 34.86453, 35.00677,
         36.95972, 36.49482, 36.75494,
         35.41682, 35.51656, 34.42815,
         41.93420, 40.84048, 40.94240,
         41.18176, 43.35924, 43.55264,
         43.05036, 43.37337, 43.82774,
         44.41442, 44.52339, 45.25616,
      ], 3, 8).t()

      const rmse1 = vector([1.162827, 0.7671318, 1.181358]);
      const r21 = vector([0.9248796, 0.967306, 0.9224663]);
      const eigenvals1 = vector([2.5423400, 0.8669532, 0.5592240]);

      const [X, Y] = splitregdata(data);
      const m = plsfit(X, Y, 3, true, true);
      const [Xpv, D] = pcvpls(X, Y, m, {"ven": 4});

      const r = plspredict(m, Xpv, Y);

      expect(D).to.be.deep.almost.equal(D1);
      expect(r.T).to.be.deep.almost.equal(T1);
      expect(r.H).to.be.deep.almost.equal(H1);
      // expect(r.Q).to.be.deep.almost.equal(Q1);
      expect(r.Ypred).to.be.deep.almost.equal(Ypred1);
      expect(r.rmse).to.be.deep.almost.equal(rmse1);
      expect(r.r2).to.be.deep.almost.equal(r21);

   });


   it ('tests for method "pcvpcr"', function () {

      const D1 = matrix([
         0.8928038, 0.6953515,  1.7268124,
         0.9272071, 1.4768482, -0.4511441,
         0.9676971, 1.1075906,  1.1945457,
         1.1402709, 0.6762249, -0.7272937
      ], 3, 4);

      const T1 = matrix([
         -1.5959914, -0.35669193,  1.7114956,
         -1.1768841, -0.90612631, -0.4232573,
         -0.6259745, -0.77332366, -0.4885416,
         -1.3751630,  0.30118095,  1.3175077,
          1.2243539, -0.61895968,  0.4291630,
          0.5703641,  1.67461147, -0.6487102,
          1.4017631, -0.12549636, -1.2603507,
          1.9661328,  0.07996545, -0.8937680
      ], 3, 8).t();

      const P1 = matrix([
          0.5638443,  0.3451625, -0.36608984,
          0.5779670,  0.3938996, -0.01666786,
          0.4691121, -0.2891090,  0.80985904,
         -0.3577254,  0.8013251,  0.45807067
      ], 3, 4).t();

      const Ypred1 = matrix([
         34.74990, 34.24415, 32.50785,
         35.86598, 34.58119, 35.01058,
         37.33304, 36.23655, 36.73217,
         35.33797, 35.76501, 34.42841,
         42.26043, 41.38281, 40.94743,
         40.51887, 42.89329, 43.55140,
         42.73287, 42.55493, 43.83354,
         44.23578, 44.34916, 45.25588
      ], 3, 8).t()

      const eigenvals1 = vector([2.5423400, 0.8669532, 0.5592240]);

      const [X, Y] = splitregdata(data);
      const m = pcrfit(X, Y, 3, true, true);
      const [Xpv, D] = pcvpcr(X, Y, m, {"ven": 4});
      const r = pcrpredict(m, Xpv, Y);

      expect(D).to.be.deep.almost.equal(D1);
      expect(m.P.apply(Math.abs, 0)).to.be.deep.almost.equal(P1.apply(Math.abs, 0));
      expect(m.eigenvals).to.be.deep.almost.equal(eigenvals1);
      expect(r.T.apply(Math.abs, 0)).to.be.deep.almost.equal(T1.apply(Math.abs, 0));
      expect(r.Ypred).to.be.deep.almost.equal(Ypred1);


   });

   it ('tests for method "pcvpca"', function () {

      // comparing outcomes with R
      const X1 = matrix([1, 3, 17, 19, 10, 14, 7, 13, 9, 11, 2, 15, 8, 6, 4, 12,
         22, 18, 4, 14, 13, 12, 1, 15], 8, 3);

      // first case - all possible components + full cross-validation
      const A11 = 3;
      const cv11 = {type: "loo"};
      const m11 = pcafit(X1, A11);
      const X11pv = pcvpca(X1, m11, A11, cv11);
      const r11 = pcapredict(m11, X11pv);

      expect(r11.T.t().v.map(x => Math.abs(x))).to.be.deep.almost.equal(new Float64Array([
         10.0140380, 8.5235218, 3.2171212,
          8.6658232, 4.0965210, 1.7038996,
         11.5878439, 1.8647639, 3.9084240,
          0.7731364, 9.8942609, 4.5041217,
          0.6373488, 0.3702545, 0.4877993,
          2.8319600, 1.4993523, 2.7862511,
          5.6623197, 9.4108933, 6.3367557,
          1.7167852, 4.7241645, 1.0080517
      ]));

      expect(r11.H.t().v).to.be.deep.almost.equal(new Float64Array([
         1.46251181, 3.58303765,  5.59816235,
         1.09521796, 1.58503715,  2.15030603,
         1.95833139, 2.05982837,  5.03403003,
         0.00871752, 2.86612471,  6.81603569,
         0.00592428, 0.00992562,  0.05625425,
         0.11696479, 0.18258125,  1.69407924,
         0.46759481, 3.05263393, 10.87073001,
         0.04298460, 0.69439613,  0.89224461
      ]));

      expect(r11.Q.t().v).to.be.deep.almost.equal(new Float64Array([
          83.0002919, 10.3498690,  0,
          19.6847583,  2.9032738,  0,
          18.7531225, 15.2757780,  0,
         118.1835101, 20.2871123,  0,
           0.3750365,  0.2379482,  0,
          10.0112525,  7.7631951,  0,
         128.7193857, 40.1544729,  0,
          23.3338984,  1.0161683,  0
      ]));

      // second case - 2 components + ven with 3 segments
      const A12 = 2;
      const cv12 = {type: "ven", nseg: 3};
      const m12 = pcafit(X1, A12);

      const X12pv = pcvpca(X1, m12, A12, cv12);
      const r12 = pcapredict(m12, X12pv);

      expect(r12.T.t().v.map(x => Math.abs(x))).to.be.deep.almost.equal(new Float64Array([
         12.2937342,  3.5829202,
          8.8832034,  3.6204597,
         11.5446749,  1.5220728,
          1.1951282, 10.6782087,
          0.6186727,  0.4293741,
          2.6220429,  1.3324750,
          7.0616804,  8.9658336,
          2.2733024,  4.4926312
      ]));

      expect(r12.H.t().v).to.be.deep.almost.equal(new Float64Array([
         2.20418755, 2.57888410,
         1.15085371, 1.53344303,
         1.94376754, 2.01138772,
         0.02083099, 3.34897591,
         0.00558217, 0.01096334,
         0.10026757, 0.15209070,
         0.72727227, 3.07358973,
         0.07536945, 0.66449378
      ]));

      expect(r12.Q.t().v).to.be.deep.almost.equal(new Float64Array([
          32.1453500, 19.308033,
          15.8699472,  2.762219,
          19.7517308, 17.435025,
         117.3529185,  3.328778,
           0.3984941,  0.214132,
          11.1561408,  9.380651,
         110.9139196, 30.527747,
          21.1133461,  0.929611
      ]));

      // third case - 1 component + ven with 2 segments
      const A13 = 1;
      const cv13 = {type: "ven", nseg: 2};
      const m13 = pcafit(X1, A13);
      const X13pv = pcvpca(X1, m13, A13, cv13);
      const r13 = pcapredict(m13, X13pv);

      expect(r13.T.t().v.map(x => Math.abs(x))).to.be.deep.almost.equal(new Float64Array([
         9.39633799,
         8.75468266,
         4.50645582,
         0.01524855,
         0.64466043,
         2.54815594,
         3.92591475,
         2.36235379
      ]));

      expect(r13.H.t().v).to.be.deep.almost.equal(new Float64Array([
         1.28765100,
         1.11779387,
         0.29617687,
         0.00000339,
         0.00606098,
         0.09469628,
         0.22478257,
         0.08138995
      ]));

      expect(r13.Q.t().v).to.be.deep.almost.equal(new Float64Array([
          94.9900823,
          18.1367816,
         132.7231059,
         118.7810175,
           0.3656629,
          11.5381513,
         145.3684434,
          20.7005346
      ]));

   }).timeout(30000);

});

describe('Tests of cross-validation methods.', function () {

   it ('tests for method "cv2obs"', function () {
      const out1 = cv2obs(index([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]), 3);
      expect(out1[1]).to.be.deep.equal(index([3, 7, 11]));
      expect(out1[0]).to.be.deep.equal(index([1, 2, 4, 5, 6, 8, 9, 10, 12]));

      const out2 = cv2obs(index([1, 3, 2, 4, 3, 2, 1, 3, 1, 2, 4, 4]), 3);
      expect(out2[1]).to.be.deep.equal(index([2, 5, 8]));
      expect(out2[0]).to.be.deep.equal(index([1, 3, 4, 6, 7, 9, 10, 11, 12]));

      const cv3 = crossval("rand", 4, 1000);
      const out3 = cv2obs(cv3, 2);
      expect(out3[1].length).equal(250);
      expect(out3[0].length).equal(750);
      expect(c(out3[0], out3[1]).sort()).to.be.deep.equal(Index.seq(1, 1000));
   });

   it ('tests for method "crossval"', function () {
      const cv1 = crossval("loo", 100, 10);
      expect(cv1).to.be.deep.equal(index([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));

      const cv2 = crossval("loo", 0, 5);
      expect(cv2).to.be.deep.equal(index([1, 2, 3, 4, 5]));

      const cv3 = crossval("ven", 4, 12);
      expect(cv3).to.be.deep.equal(index([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]));

      const cv4 = crossval("ven", 4, 10);
      expect(cv4).to.be.deep.equal(index([1, 2, 3, 4, 1, 2, 3, 4, 1, 2]));

      const cv5 = crossval("ven", 6, 10);
      expect(cv5).to.be.deep.equal(index([1, 2, 3, 4, 5, 6, 1, 2, 3, 4]));

      const cv6 = crossval("ven", 3, 10);
      expect(cv6).to.be.deep.equal(index([1, 2, 3, 1, 2, 3, 1, 2, 3, 1]));

      const cv7 = crossval("rand", 4, 12);
      const cv7v = index([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]);
      expect(cv7).not.to.be.deep.equal(cv7v)
      expect(cv7.sort()).to.be.deep.equal(cv7v.sort())

      const cv8 = crossval("rand", 4, 1000);
      const cv8v = crossval("ven", 4, 1000);
      expect(cv8).not.to.be.deep.equal(cv8v)
      expect(cv8.sort()).to.be.deep.equal(cv8v.sort())
   });

});

