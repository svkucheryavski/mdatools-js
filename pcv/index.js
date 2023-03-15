/************************************************************/
/*    Methods for Procrustes Cross Validation               */
/************************************************************/


// import methods to test
import { _dot, reshape, tcrossprod, Vector, Index, Matrix, isindex} from '../arrays/index.js';
import { ssq, norm2, max } from '../stat/index.js';
import { rsvd } from '../decomp/index.js';
import { simpls } from '../models/index.js';
import { scale as prep_scale, unscale as prep_unscale } from '../prep/index.js';


/**
 * Compute PV-set using PLS.
 *
 * @param {Matrix} data - matrix with calibration set (first column with y the rest is X).
 * @param {JSON} m - object with global PCR model.
 * @param {number} ncomp - number of components to use.
 * @param {JSON} cv - object with type of splits ('ven', 'rand', 'loo') and number of segments.
 *
 * @return {Matrix} matrix with PV-set.
 *
 */
export function pcvpls(X, Y, m, ncomp, cv) {

   const nobs = X.nrows;
   const nvar = X.ncols;

   // get cross-validation parameters and adjusted number of components
   const [cvNcomp, cvInd, cvNSeg] = getcvparams(nobs, nvar, ncomp, cv, m);

   // autoscale the calibration set
   const Xp = prep_scale(X, m.mX, m.sX);
   const Yp = prep_scale(Y, m.mY, m.sY);

   // get main parameters of the global model
   const cind = Index.seq(1, cvNcomp);
   const C = m.C.subset([], cind);
   const R = m.R.subset([], cind);
   const P = m.P.subset([], cind);

   const PRM = Matrix.eye(nvar).subtract(tcrossprod(R, P));

   // prepare empty matrix for pseudo-validation set and scalars
   const Xpv = Matrix.zeros(nobs, nvar);
   const D = Matrix.zeros(cvNcomp, cvNSeg);

   // loop for computing the PV set
   for (let k = 1; k <= cvNSeg; k++) {
      const [indc, indk] = cv2obs(cvInd, k);
      const [Xpvk, dk] = getplsXpvk(Xp, Yp, indc, indk, P, R, C, PRM);
      Xpv.replace(Xpvk, indk, []);
      D.v.set(dk.v, (k - 1) * cvNcomp)
   }

   // ucnenter and unscale the values and return
   return [prep_unscale(Xpv, m.mX, m.sX), D];
}


/**
 * Compute part of PV-set for k-th segment.
 *
 * @param {Matrix} X - matrix with predictors.
 * @param {Matrix} Y - matrix with responses.
 * @param {Index} indc - vector with indices of rows for local calibration set.
 * @param {Index} indk - vector with indices of rows for local validation set.
 * @param {Matrix} P - matrix with X-loadings from global model.
 * @param {Matrix} R - matrix with weights from global model.
 * @param {Matrix} P - matrix with Y-loadings from global model.
 * @param {Matrix} PRM - projection matrix for the global model (I - PP').
 *
 * @return {Array} array with PV-set matrix for k-th segment and vector with scalars (ck/c).
 *
 */
export function getplsXpvk(X, Y, indc, indk, P, R, C, PRM) {

   const ncomp = P.ncols;
   const nvar = P.nrows;

   const Xc = X.subset(indc, []);
   const Xk = X.subset(indk, []);
   const Yc = Y.subset(indc, []);

   // get outcomes for local model
   const mk = simpls(Xc, Yc, ncomp);
   const Pk = mk.P;
   const Rk = mk.R;
   const Ck = mk.C;

   // normalize weights for computing angles
   const a = getdirections(R, Rk);
   const Pka = Pk.dot(Matrix.diagm(a, nvar));
   const Rka = Rk.dot(Matrix.diagm(a, nvar));
   const Cka = Ck.dot(Matrix.diagm(a, nvar));

   // compute scores for PV subset (only for one Y!)
   const Tk = Xk.dot(Rka);
   const dk = Cka.divide(C);
   const Dk = Matrix.diagm(reshape(dk, dk.v.length));


   // compute first part of PV subset
   const Tpvk = Tk.dot(Dk);
   const Xpvk = tcrossprod(Tpvk, P);

   // get scores and residuals by projection local validation set to the local model
   const Ek = Xk.subtract(tcrossprod(Tk, Pka));
   const qkn = Ek.apply(norm2, 1);

   // add orthogonal part if necessary and return
   return P.nrows > P.ncols ? [Xpvk.add(getxpvorth(qkn, Xk, PRM)), dk] : [Xpvk, dk];
}


/**
 * Compute PV-set using PCR.
 *
 * @param {Matrix} data - matrix with calibration set (first column with y the rest is X).
 * @param {JSON} m - object with global PCR model.
 * @param {number} ncomp - number of components to use.
 * @param {JSON} cv - object with type of splits ('ven', 'rand', 'loo') and number of segments.
 *
 * @return {Matrix} matrix with PV-set.
 *
 */
export function pcvpcr(X, Y, m, ncomp, cv) {

   const nobs = X.nrows;
   const nvar = X.ncols;

   // get cross-validation parameters and adjusted number of components
   const [cvNcomp, cvInd, cvNSeg] = getcvparams(nobs, nvar, ncomp, cv, m);


   // autoscale the calibration set
   const Xp = prep_scale(X, m.mX, m.sX);
   const Yp = prep_scale(Y, m.mY, m.sY);

   // get main parameters of the global model
   const cind = Index.seq(1, cvNcomp);
   const C = m.C.subset([], cind);
   const P = m.P.subset([], cind);
   const PRM = Matrix.eye(nvar).subtract(tcrossprod(P));

   // prepare empty matrix for pseudo-validation set and scalars
   const Xpv = Matrix.zeros(nobs, nvar);
   const D = Matrix.zeros(cvNcomp, cvNSeg);

   // loop for computing the PV set
   for (let k = 1; k <= cvNSeg; k++) {
      const [indc, indk] = cv2obs(cvInd, k);
      const [Xpvk, dk] = getpcrXpvk(Xp, Yp, indc, indk, P, C, PRM);
      Xpv.replace(Xpvk, indk, []);
      D.v.set(dk.v, (k - 1) * cvNcomp)
   }

   // uncenter and unscale the values and return
   return [prep_unscale(Xpv, m.mX, m.sX), D];
}


/**
 * Compute part of PV-set for k-th segment.
 *
 * @param {Matrix} X - matrix with predictors.
 * @param {Matrix} Y - matrix with responses.
 * @param {Index} indc - vector with indices of rows for local calibration set.
 * @param {Index} indk - vector with indices of rows for local validation set.
 * @param {Matrix} P - matrix with X-loadings from global model.
 * @param {Matrix} P - matrix with Y-loadings from global model.
 * @param {Matrix} PRM - projection matrix for the global model (I - PP').
 *
 * @return {Array} array with PV-set matrix for k-th segment and vector with scalars (ck/c).
 *
 */
export function getpcrXpvk(X, Y, indc, indk, P, C, PRM) {

   const ncomp = P.ncols;
   const nvar = P.nrows;

   const Xc = X.subset(indc, []);
   const Xk = X.subset(indk, []);
   const Yc = Y.subset(indc, []);
   const nrows = Xc.nrows;

   // get loadings for local model and rotation matrix between global and local models
   const Pk = rsvd(Xc, ncomp).V;

   // correct direction of loadings for local model
   const a = getdirections(P, Pk);
   const Pka = Pk.dot(Matrix.diagm(a, nvar));
   const Tc = Xc.dot(Pka);

   // compute Y-loadings for local model
   const Ck = Matrix.zeros(1, ncomp);
   for (let a = 1; a <= ncomp; a++) {
      const ta = Tc.getcolref(a);
      const eigena = ssq(ta);
      Ck.v[a - 1] = _dot(Yc.v, ta, 1, nrows, nrows, 1)[0] / eigena;
   }

   // compute scores for PV subset
   const Tk = Xk.dot(Pka);
   const dk = Ck.divide(C);
   const Dk = Matrix.diagm(reshape(dk, dk.v.length));

   // compute first part of PV subset
   const Tpvk = Tk.dot(Dk);
   const Xpvk = tcrossprod(Tpvk, P);

   // get scores and residuals by projection local validation set to the local model
   const Ek = Xk.subtract(tcrossprod(Tk, Pka));
   const qkn = Ek.apply(norm2, 1);

   // add orthogonal part if necessary and return
   return P.nrows  > P.ncols ? [Xpvk.add(getxpvorth(qkn, Xk, PRM)), dk] : [Xpvk, dk];
}


/**
 * Compute PV-set using PCA.
 *
 * @param {Matrix} X - matrix with calibration set.
 * @param {JSON} m - object with global PCA model.
 * @param {number} ncomp - number of components to use.
 * @param {JSON} cv - object with type of splits ('ven', 'rand', 'loo') and number of segments.
 *
 * @return {Matrix} matrix with PV-set.
 *
 */
export function pcvpca(X, m, ncomp, cv) {

   const nobs = X.nrows;
   const nvar = X.ncols;

   // get cross-validation parameters and adjusted number of components
   const [cvNcomp, cvInd, cvNSeg] = getcvparams(nobs, nvar, ncomp, cv, m);

   // autoscale the calibration set
   const Xp = prep_scale(X, m.mX, m.sX);

   // create a global model
   const P = m.P.subset([], Index.seq(1, cvNcomp));
   const PRM = Matrix.eye(nvar).subtract(tcrossprod(P));

   // prepare empty matrix for pseudo-validation set
   const Xpv = Matrix.zeros(nobs, nvar);

   // loop for computing the PV set
   for (let k = 1; k <= cvNSeg; k++) {
      const [indc, indk] = cv2obs(cvInd, k);
      Xpv.replace(getpcaXpvk(Xp, indc, indk, P, PRM), indk, []);
   }

   // ucenter and unscale the values and return
   return prep_unscale(Xpv, m.mX, m.sX);
}


/**
 * Compute part of PV-set for k-th segment.
 *
 * @param {Matrix} X - matrix with calibration set.
 * @param {Index} indc - vector with indices of rows for local calibration set.
 * @param {Index} indk - vector with indices of rows for local validation set.
 * @param {Matrix} P - matrix with PCA loadings from global model
 * @param {Matrix} PRM - projection matrix for the global model (I - PP')
 *
 * @return {Matrix} matrix with PV-set for k-th segment.
 *
 */
export function getpcaXpvk(X, indc, indk, P, PRM) {

   const ncomp = P.ncols;
   const nvar = P.nrows;
   const Xc = X.subset(indc, []);
   const Xk = X.subset(indk, []);

   // get loadings for local model and rotation matrix between global and local models
   const Pk = rsvd(Xc, ncomp).V;

   // correct direction of loadings for local model
   const a = getdirections(P, Pk);
   const Pka = Pk.dot(Matrix.diagm(a, nvar));

   // get scores and residuals by projection local validation set to the local model
   const Tk = Xk.dot(Pka);
   const Ek = Xk.subtract(tcrossprod(Tk, Pka));
   const qkn = Ek.apply(norm2, 1);

   // compute Xpvk
   const Xpvk = tcrossprod(Tk, P);

   // add orthogonal part if necessary and return
   return P.nrows > P.ncols ? Xpvk.add(getxpvorth(qkn, Xk, PRM)) : Xpvk;
}


/**
 * Compute orthogonal part of PV-set.
 *
 * @param {Vector} qkn - vector with orthogonal distances for k-th segment (square root of qk).
 * @param {Matrix} Xk - matrix with local validation set for k-th segment.
 * @param {Matrix} PRM - projection matrix for global model.
 *
 * @returns {Matrix} matrix with orthogonal part values.
 *
 */
export function getxpvorth(qkn, Xk, PRM) {


   const nobj = qkn.length;

   // project Xk to random vectors
   const Z = Matrix.rand(nobj, nobj, -1, 1)
   const X1 = Z.dot(Xk);

   // normalize columns of X1 to norm2
   for (let c = 1; c <= X1.ncols; c++) {
      const x = X1.getcolref(c);
      const nx = norm2(x);
      for (let r = 1; r <= nobj; r++) {
         x[r - 1] = nx == 0 ? x[r - 1] : x[r - 1] / nx;
      }
   }

   // compute X2 and scale its rows to unit length
   const X2 = X1.dot(PRM);

   // normalize rows of X2 to sqrt(qk/norm2)
   const nr = X2.apply(norm2, 1);
   for (let c = 1; c <= X1.ncols; c++) {
      const x = X2.getcolref(c);
      for (let r = 1; r <= nobj; r++) {
         x[r - 1] = nr.v[r - 1] == 0 ? x[r - 1] : x[r - 1] * qkn.v[r - 1] / nr.v[r - 1];
      }
   }

   return X2;
}


/**
 * Return two vector with observation indices (for local training and for local validation subsets).
 *
 * @param {Index} cv - vector with CV segments indices (e.g. from 'crossval()' method).
 * @param {number} k - which segment to use for validation.
 *
 * @return {Array} - array with two Int32Array with indices (first for "cal" and second for "val").
 *
 */
export function cv2obs(cv, k) {

   if (!isindex(cv)) {
      throw Error('cv2obs: parameter "cv" must be instance of Index.');
   }

   return [
      cv.which(v => v !== k),
      cv.which(v => v === k)
   ];
}


/**
 * Generate indices for cross-validation.
 *
 * @param {string} type - cross-validation type ("loo", "ven", "rand").
 * @param {number} nseg - number of segments.
 * @param {number} nobs - number of observations.
 *
 * @return {Index} vector of length 'nobs' with segment indices for each observation.
 *
 */
export function crossval(type, nseg, nobs) {

   if (type == "loo") {
      return Index.seq(1, nobs);
   }

   if (type == "ven") {
      const nrep = Math.ceil(nobs/nseg);
      return new Index(Index.seq(1, nseg).rep(nrep).v.subarray(0, nobs));
   }

   if (type == "rand") {
      return crossval("ven", nseg, nobs).shuffle();
   }
}


/**
 * Check if direction of vectors from 'X' and 'Xk' are same or opposite.
 *
 * @param {Matrix} X - matrix with vectors for global model (e.g. loadings).
 * @param {Matrix} Xk - matrix with vectors for local model (e.g. loadings).
 *
 * @returns {Vector} vector with '1' if directions are the same and '-1' if opposite.
 *
 */
export function getdirections(X, Xk) {

   const a = Vector.zeros(X.ncols);

   for (let c = 1; c <= X.ncols; c++) {
      const xc = X.getcolref(c);
      const xck = Xk.getcolref(c);
      const cnorm = norm2(xc);
      const cnormk = norm2(xck);

      let s = 0;
      for (let r = 0; r < xc.length; r++) {
         s += xc[r] / cnorm * xck[r] / cnormk;
      }

      a.v[c - 1] = (Math.acos(s) < (Math.PI / 2.0)) * 2 - 1;
   }

   return a;
}


/**
 * Computes cross-validation parameters for a model.
 *
 * @param {number} nobs - number of observations.
 * @param {number} nvar - number of predictors.
 * @param {number} ncomp - number of components specified by user.
 * @param {Object} cv - JSON wiht user defines cross-validation parameters.
 *
 * @returns {Array} array with four elements: adjusted number of components, type, vector with
 * indices and number of segments in cross-validation.
 *
 */
export function getcvparams(nobs, nvar, ncomp, cv, m) {

   const nseg = cv === undefined ? 4 : cv.nseg;
   const cvType = cv === undefined ? "ven" : cv.type;
   const cvInd = crossval(cvType, nseg, nobs);
   const cvNSeg = max(cvInd);

   if (ncomp !== undefined) {
      ncomp = m.ncomp;
   }

   // correct maximum number of components
   const maxNcomp = Math.min(m.ncomp, nobs - Math.ceil(nobs / nseg) - 1 , nvar);
   if (ncomp > maxNcomp) {
      ncomp = maxNcomp;
   }

   return [ncomp, cvInd, cvNSeg]
}
