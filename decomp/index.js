import {nrow, ncol} from '../matrix/index.js';

/**********************************************
 * Functions for decompositions of matrices   *
 **********************************************/

export function svd(X, n) {
   if (!n) n = min([nrow(X) - 1, ncol(X)]);

}