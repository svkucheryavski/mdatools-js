/************************************************************/
/*    Methods for computing matrix decompositions           */
/************************************************************/


import { ssq } from '../stat/index.js';
import { crossprod, Vector, Matrix, Index } from '../arrays/index.js';


/**
 * Randomized SVD decomposition.
 *
 * @param {Matrix} X - matrix to decompose.
 * @param {number} [ncomp] - number of components.
 * @param {number} [pa=1.5] - oversampling factor (ncomp * pa + pb).
 * @param {number} [pb=10] - oversampling increment (ncomp * pa + pb).
 * @param {number} [its=4] - number of iterations
 *
 * @returns {JSON} JSON with three fields, 's' - vector with singular values,
 * 'U', 'V' - matrices with left and right singular vectors.
 *
 */
export function rsvd(X, ncomp, pa, pb, its) {

   const m = X.nrows;
   const n = X.ncols;

   if (its === undefined) {
      its = 3;
   }

   if (pb === undefined) {
      pb = 10;
   }

   if (pa === undefined) {
      pa = 1;
   }

   if (its < 1 || its > 10) {
      throw new Error('rsvd: wrong value for parameter "its" (must be between 1 and 10).');
   }

   if (pa < 1 || pa > 5) {
      throw new Error('rsvd: wrong value for parameter "pa" (must be between 1 and 5).');
   }

   if (pb < 1 || pb > 100) {
      throw new Error('rsvd: wrong value for parameter "pb" (must be between 1 and 100).');
   }


   if (ncomp === undefined) {
      ncomp = Math.round(Math.min(m - 1, n) / 1.5);
   }

   if (m < n) {
      const res = rsvd(X.t(), ncomp);
      return {s: res.s, V: res.U, U: res.V}
   }

   // the more the better but slower
   const l = Math.round(pa * ncomp + pb);
   let Q = qr(X.dot(Matrix.rand(n, l, -1, 1))).Q;

   for (let it = 1; it <= its; it++) {
      Q = lu(crossprod(X, Q)).L;
      Q = it < its ? lu(X.dot(Q)).L : qr(X.dot(Q)).Q;
   }

   const res = svd(crossprod(Q, X), ncomp);
   return {s: res.s, V: res.V, U: Q.dot(res.U)};
}

/**
 * QR decomposition
 *
 * @param {Matrix} X - matrix to decompose.
 *
 * @returns {JSON} JSON with two matrices, Q and R, so X = QR.
 *
 */
export function qr(X) {

   const m = X.nrows;
   const n = X.ncols;

   if (m < n) {
      const res = qr(X.subset([], Index.seq(1, m)));
      return {Q:res.Q, R:crossprod(res.Q, X)}
   }

   let Q = Matrix.eye(m);
   let Rt = X.t();

   for (let j = 1; j <= n; j++) {
      for (let i = m; i >= (j + 1); i--) {

         const r = Rt.v.subarray((i - 2) * n, i * n);
         const q =  Q.v.subarray((i - 2) * m, i * m);
         const rc = r.slice();
         const qc = q.slice();

         const [c, s, temp] = rot(rc[j - 1], rc[j - 1 + n]);

         // apply transformation to columns i and i - 1 of matrix Rt
         for (let k = 0; k < n; k++) {

            r[k]     = rc[k] *  c + rc[k + n] * s;
            r[k + n] = rc[k] * -s + rc[k + n] * c;

            q[k]     = qc[k] *  c + qc[k + m] * s;
            q[k + m] = qc[k] * -s + qc[k + m] * c;
         }

         for (let k = n; k < m; k++) {
            q[k]     = qc[k] *  c + qc[k + m] * s;
            q[k + m] = qc[k] * -s + qc[k + m] * c;
         }
      }
   }

   const ind = Index.seq(1, n);
   return {
      Q: m === n ? Q : Q.subset([], ind),
      R: m === n ? Rt.t() : Rt.subset([], ind).t()
   }
}

/**
 * LU decomposition.
 *
 * @param {Matrix} X - matrix to decompose.
 *
 * @returns {JSON} JSON with two matrices, L and U, so X = LU.
 *
 */
export function lu(X) {

   const nr = X.nrows;
   const nc = X.ncols
   const n = Math.min(nr, nc)

   const Lv = new Float64Array(nr * nc);
   const Uv = new Float64Array(nc * nc);
   const Xv = X.v.slice();

   const s = Math.round(Math.pow(n, 0.5285425));
   let z = 1;

   for (let c = 1; c <= n; c++) {

      if (c == z + s) {
         const nro = nr - c + 1;
         const nco = nc - c + 1;
         const ni = c - z;
         const Sv = new Float64Array(nro * nco);

         for (let i = 0; i < nro; i++) {
            for (let j = 0; j < nco; j++) {
               let s = 0;
               for (let k = 0; k < ni; k++) {
                  s += Lv[(z - 1 + k) * nr + (c - 1) + i] * Uv[(c - 1 + j) * nc + (z - 1 + k)];
               }
               Sv[j * nro + i] = s;
            }
         }

         for (let i = c; i <= nr; i++) {
            for (let j = c; j <= nc; j++) {
               Xv[(j - 1) * nr + (i - 1)] = Xv[(j - 1) * nr + (i - 1)] - Sv[(j - c) * nro + (i - c)];
            }
         }

         z = c;
      }

      for (let i = c; i <= nr; i++) {
         let acc = Xv[(c - 1) * nr + (i - 1)];
         for (let k = z; k <= c - 1; k++) {
            acc = acc - Lv[(k - 1) * nr + (i - 1)] * Uv[(c - 1) * nc + (k - 1)];
         }
         Lv[(c - 1) * nr + (i - 1)] = acc;
      }

      for (let i = c; i <= nc; i++) {
         let acc = Xv[(i - 1) * nr + (c - 1)];
         for (let k = z; k <= c - 1; k++) {
            acc = acc - Lv[(k - 1) * nr + (c - 1)] * Uv[(i - 1) * nc + (k - 1)];
         }
         const l = Lv[(c - 1) * nr + (c - 1)];
         Uv[(i - 1) * nc + (c - 1)] = (l == 0 ? acc : acc / l);
      }
   }

   return {L: new Matrix(Lv, nr, nc), U: new Matrix(Uv, nc, nc)}
}


/**
 * Singular value decomposition
 *
 * @param {Matrix} X - matrix to decompose.
 * @param {number} [ncomp] - number of components.
 *
 * @returns {JSON} JSON with three fields, 's' - vector with singular values,
 * 'U', 'V' - matrices with left and right singular vectors.
 *
 */
export function svd(X, ncomp) {

   if (X.nrows < X.ncols) {
      const res = svd(X.t(), ncomp);
      return {s: res.s, V: res.U, U: res.V}
   }

   const m = X.nrows;
   const n = X.ncols;

   if (!ncomp) {
      ncomp = Math.min(m, n);
   }

   let [B, V, U] = bidiag(X);

   // get diagonal elements of B
   let d = new Float64Array(n);
   let e = new Float64Array(n - 1);
   for (let i = 0; i < n - 1; i++) {
      d[i] = B.v[i * n + i];
      e[i] = B.v[(i + 1) * n + i];
   }
   d[n - 1] = B.v[(n - 1) * n + n - 1];

   const maxit = 500 * n * n;
   const thresh = Math.pow(10, -64);
   let Gt = Matrix.eye(n);
   let P = Matrix.eye(n);

   for (let it = 0; it < maxit; it++) {

      // find first nonzero element in e from bottom
      let iU = 0;
      for (let i = n - 1; i >= 1; i--) {
         if (Math.abs(e[i - 1]) > thresh) {
            iU = i
            break;
         }
      }

      // find first nonzero element in e from top
      let iL = iU + 1;
      for (let i = 1; i <= n - 1; i++) {
         if (Math.abs(e[i - 1]) > thresh) {
            iL = i;
            break;
         }
      }

      // check the convergence and return result
      if ((iU == iL && Math.abs(e[iU - 1]) <= thresh) || (iU < iL)) {

         const s = d.slice(0, ncomp);
         const Uout = Matrix.zeros(m, ncomp);
         const Vout = Matrix.zeros(n, ncomp);

         for (let k = 1; k <= ncomp; k++) {

            const pk = P.getcolref(k);
            const gtk = Gt.getcolref(k);

            const uk = Uout.getcolref(k);
            const vk = Vout.getcolref(k);

            const sign = Math.sign(s[k - 1]);

            for (let i = 0; i < n; i++) {

               const ui = U.getcolref(i + 1);
               const vi = V.getcolref(i + 1);

               const pki = pk[i] * sign;
               const gtki = gtk[i];

               for (let r = 0; r < n; r++) {
                  uk[r] += ui[r] * pki;
                  vk[r] += vi[r] * gtki;
               }

               for (let r = n; r < m; r++) {
                  uk[r] += ui[r] * pki;
               }
            }

            s[k - 1] = Math.abs(s[k - 1]);
         }

         return {s: new Vector(s), U: Uout, V: Vout};
      }

      // re-sweep
      const [rd, re, rG, rPt] = vsweep(d.subarray(iL - 1, iU + 1), e.subarray(iL - 1, iU));

      // replace elements of d and e
      d.set(rd, iL - 1);
      e.set(re, iL - 1);

      // now we need to compute:
      // G = rG' * G = (G' * rG)' -> this should be done only fo selected columns in G
      // P =            P  * rP'  -> only for selected columns in P

      // get local copy of selected columns from G' and P
      const l = iU - iL + 2;
      const lGt = Matrix.zeros(n, l);
      const lP = Matrix.zeros(n, l)
      for (let c = iL; c <= iU + 1; c++) {
         lGt.v.set(Gt.v.subarray((c - 1) * n, c * n), (c - iL) * n);
         lP.v.set(P.v.subarray((c - 1) * n, c * n), (c - iL) * n);
      }

      // compute dot(lG', rG) and save back to G
      // compute dot(P, rP') and save back to P
      for (let c = 1; c <= l; c++) {
         const newgc = lGt.dot(new Vector(rG.getcolref(c)));
         const newpc = lP.dot(new Vector(rPt.getcolref(c)));
         Gt.v.set(newgc.v, (c - 1 + iL - 1) * n);
         P.v.set(newpc.v, (c - 1 + iL - 1) * n);
      }
   }

   throw Error("svd: can not converge.")
}


/**
 * Applies givens rotations to diagonal elements of bidiagonalized matrix.
 *
 * @param {Float64Array} d - vector with main diagonal elements.
 * @param {Float64Array} e - vector with diagonal elements on top of main diagonal.
 *
 * @returns {Array} array with vectors 'd' and 'e' after rotation as well as right
 * transformation matrices G and P'.
 *
 */
export function vsweep(d, e) {

   const n = d.length;

   let cold = 1;
   let sold = 1;

   let c = 1;
   let s = 0;
   let r = 0;

   let G = Matrix.eye(n);
   let Pt = Matrix.eye(n);

   for (let k = 1; k <= (n - 1); k++) {

      [c, s, r] = rot(c * d[k - 1], e[k - 1]);

      // get two columns from G and rotate their values
      const gk = G.v.subarray((k - 1) * n, (k + 1) * n);
      const gkc = gk.slice();

      for (let i = 0; i < n; i++) {
         gk[i]     =  gkc[i] * c + gkc[i + n] * s;
         gk[i + n] = -gkc[i] * s + gkc[i + n] * c;
      }

      if (k != 1) {
         e[k - 2] = r * sold;
      }

      [cold, sold, r] = rot(cold * r, d[k] * s);
      d[k - 1] = r;

      // get two columns from Pt and rotate their values
      const ptk = Pt.v.subarray((k - 1) * n, (k + 1) * n);
      const ptkc = ptk.slice();

      for (let i = 0; i < n; i++) {
         ptk[i]     =  ptkc[i] * cold + ptkc[i + n] * sold;
         ptk[i + n] = -ptkc[i] * sold + ptkc[i + n] * cold;
      }

   }

   const h = c * d[n - 1];
   e[n - 2] = h * sold;
   d[n - 1] = h * cold;

   return [d, e, G, Pt];
}


/**
 * Generate c, s, r values for Givens rotations, so [c s; -s c][f; g] = [r; 0].
 *
 * @param {number} f - first element of vector.
 * @param {number} g - second element of vector.
 *
 * @return array with values [c, s, r].
 *
 */
export function rot(f, g) {

   if (f == 0) {
      return [0, 1, g];
   }

   if (Math.abs(f) > Math.abs(g)) {
      const t = g / f;
      const t1 = Math.sqrt(1 + t * t);
      const c = 1 / t1
      return [c, t * c, f * t1];
   }

   const t = f / g;
   const t1 = Math.sqrt(1 + t * t)
   const s = 1 / t1
   return [t * s, s, g * t1];
}


/**
 * Golub-Reinsch bidiagonalization of matrix A
 *
 * @param {Matrix} A - matrix with values.
 *
 * @returns {Array} array with three matrices [B, V, U].
 *
 */
export function bidiag(A) {

   const m = A.nrows;
   const n = A.ncols;

   if (m < n) {
      throw Error("bidiag: number of rows in A must not be smaller number of columns.");
   }

   let Ut = Matrix.eye(m);
   let V = Matrix.eye(n);
   let B = A.copy();

   for (let k = 1; k <= (m > n ? n : n - 1); k++) {

      const mk = m - k + 1;

      // compute:
      // B  = H * B
      // U' = H' * U';
      const H1 = householder(B.getcolref(k), k);

      for (let c = 1; c <= n; c++) {

         // get c-th column of B (starting from (k-1) row) and make a copy
         const bc = B.v.subarray((c - 1) * m + k - 1, c * m)
         const bcc = bc.slice();

         // get c-th column of U' (starting from (k-1) row) and make a copy
         const utc = Ut.v.subarray((c - 1) * m + k - 1, c * m)
         const utcc = utc.slice();

         for (let r = k; r <= m; r++) {
            // because H is symmetric we take column instead of row
            const hr = H1.getcolref(r - k + 1);
            let sb = 0;
            let sut = 0;
            for (let i = 0; i < mk; i++) {
               sb += hr[i] * bcc[i];
               sut += hr[i] * utcc[i]
            }
            bc[r - k] = sb;
            utc[r - k] = sut;
         }
      }


      for (let c = n + 1; c <= m; c++) {

         // get c-th column of U' and make a copy
         const utc = Ut.v.subarray((c - 1) * m + k - 1, c * m)
         const utcc = utc.slice();

         for (let r = k; r <= m; r++) {
            const hr = H1.getcolref(r - k + 1);
            let sut = 0;
            for (let i = 0; i < mk; i++) {
               sut += hr[i] * utcc[i];
            }
            utc[r - k] = sut;
         }
      }

      if (k < n - 1) {

         // compute:
         // B = B * H'
         // V = V * H
         const H2 = householder(B.getrow(k).v, k + 1);

         // we need to process columns from k + 1, so make a subset from the start
         const Bvk = B.v.subarray(k * m);
         const Vvk = V.v.subarray(k * n);

         let br = new Float64Array(n - k)
         let vr = new Float64Array(n - k)

         // process first n rows for both B and V
         for (let r = 1; r <= n; r++) {

            // get elements from r-th row
            for (let i = 0; i < n - k; i++) {
               br[i] = Bvk[i * m + r - 1];
               vr[i] = Vvk[i * n + r - 1];
            }

            for (let c = k + 1; c <= n; c++) {
               const hc = H2.getcolref(c - k);
               let sb = 0;
               let sv = 0;
               for (let i = 0; i < H2.nrows; i++) {
                  sb += br[i] * hc[i];
                  sv += vr[i] * hc[i];
               }
               Bvk[(c - k - 1) * m + (r - 1)] = sb;
               Vvk[(c - k - 1) * n + (r - 1)] = sv
            }
         }

         // process rows from (n+1) to m for B only
         for (let r = n + 1; r <= m; r++) {

            // get elements from r-th row
            for (let i = 0; i < n - k; i++) {
               br[i] = Bvk[i * m + r - 1];
            }

            for (let c = k + 1; c <= n; c++) {
               const hc = H2.getcolref(c - k);
               let sb = 0;
               for (let i = 0; i < H2.nrows; i++) {
                  sb += br[i] * hc[i];
               }
               Bvk[(c - k - 1) * m + (r - 1)] = sb;
            }
         }
      }
   }

   const ind = Index.seq(1, n);
   return [B.subset(ind, []), V, Ut.t().subset([], ind)];
}


/**
 * Compute elements of Householder transformation as a matrix.
 *
 * @param {Float64Array} b - array with diagonal values.
 * @param {number} k - position to start with.
 *
 * @returns {Matrix} matrix with transformation values.
 *
 */
export function householder(b, k) {

   // get vector with values
   const h = householderv(b, k);
   const n = h.length;

   // compute matrix as outer product of the vector
   const H = Matrix.zeros(n);
   for (let c = 0; c < n; c++) {
      const hc = H.getcolref(c + 1);
      for (let r = 0; r < n; r++) {
         hc[r] = 1 * (r == c) - 2 * h[r] * h[c];
      }
   }

   return H;
}


/**
 * Compute elements of Householder transformation as a vector.
 *
 * @param {Float64Array} b - array with diagonal values.
 * @param {number} k - position to start with.
 *
 * @returns {Float64Array} vector with transformation values.
 *
 */
export function householderv(b, k) {

   const n = b.length;

   if (k >= n) {
      throw Error("householder: parameter 'k' must be smaller than length of vector 'b'.")
   }

   const hlen = n - k + 1;
   const h = b.slice(k - 1);

   // to avoid computing norm2 twice we will
   // compute it based on the first norm
   const hssq = ssq(h);
   const hn2 = Math.sqrt(hssq);
   const s = Math.sign(h[0]);
   const a = h[0];

   // change first value in the vector
   h[0] = a - s * hn2;

   // recompute the norm
   const hn2a = Math.sqrt(2 * hssq - 2 * s * a * hn2);

   // if norm is zero return vector as is
   if (hn2a < Number.EPSILON) return h

   // normalize vector and return
   for (let i = 0; i < hlen; i++) {
      h[i] /=  hn2a;
   }

   return h;
}





