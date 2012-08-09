#ifndef quist_bla_h
#define quist_bla_h

// Basic Linear Algebra

namespace bla {

	/**
	 * Vector dot product.
	 * Dot product of two vectors.
	 */
	template <typename T>
	T dot(T a[], T b[], size_t n) {
		T d = 0;
		for (size_t i = 0; i < n; ++i) {
			d += a[i] * b[i];
		}
		return d;
	}	

	/**
	 * Tridiagonal matrix solver.
	 * Solves tridiagonal systems of linear equations (Thomas algorithm).
	 * @param n number of equations
	 * @param a sub-diagonal, indexed from 1..n-1 (a[0] is ignored)
	 * @param b main diagonal, indexed from 0..n-1; modified
	 * @param c sup-diagonal, indexed from 0..n-2 (a[n-1] is ignored)
	 * @param r right-hand side; modified
	 * @param x output solution
	 */
	template <typename T>
	void tridiag(size_t n, T a[], T b[], T c[], T r[], T x[]) {

		// decomposition, set the coefficients
		for (size_t i = 1; i < n; ++i) {
			T m = a[i] / b[i-1];
			b[i] -= m * c[i-1];
			r[i] -= m * r[i-1];
		}

		x[n-1] = r[n-1] / b[n-1];

		// back substitution
		for (size_t i = n-2; i >= 0; --i) {
			x[i] = ( r[i] - c[i]*x[i+1] ) / b[i];
		}
	}

	/**
	 * Natural cubic spline.
	 * @param n number of data points
	 * @param x values of x variable
	 * @param y values of function evalulated at x
	 * @param b output b coefficients
	 * @param c output c coefficients
	 * @param d output d coefficients
	 */
	template <typename T>
	void ncspline(size_t n, T x[], T y[], T b[], T c[], T d[]) {

		// calculate h_i = x_i+1 - x_i
		T *h = new T[n-1];
		for (size_t i = 0; i < n-1; ++i) {
			h[i] = x[i+1] - x[i];
		}

		// setup diagonals
		T *sub_diag = new T[n], *main_diag = new T[n], *sup_diag = new T[n];
		for (size_t i = 1; i < n-1; ++i) {
			sub_diag[i] = h[i-1];
			main_diag[i] = 2 * (h[i-1] + h[i]);
			sup_diag[i] = h[i];
		}
		sub_diag[n-1] = 0;
		main_diag[0] = main_diag[n-1] = 1;
		sup_diag[0] = 0;

		// setup right-hand side
		T *r = new T[n];
		r[0] = r[n-1] = 0;
		for (size_t i = 1; i < n-1; ++i) {
			r[i] = ( 3 * (y[i+1] - y[i]) / h[i] ) - ( 3 * (y[i] - y[i-1]) / h[i-1] );
		}

		// solve tridiagonal system for c coefficients
		tridiag(n, sub_diag, main_diag, sup_diag, r, c);

		// compute b and d coefficients
		for (size_t i = 0; i < n-1; ++i) {
			b[i] = (y[i+1] - y[i]) / h[i]  -  h[i] * (2*c[i] + c[i+1]) / 3;
			d[i] = (c[i+1] - c[i]) / (3 * h[i]);
		}
		// b[n-1] and c[n-1] are not calculated
		
		// free memory
		delete [] h;
		delete [] sub_diag;
		delete [] main_diag;
		delete [] sup_diag;
		delete [] r;
		
	}

	/**
	 * Spline interpolation.
	 * @param n number of fitted data points
	 * @param x values of x at fitted data points
	 * @param a a coefficients; set to f(x)
	 * @param b b coefficients returned by spline
	 * @param c c coefficients returned by spline
	 * @param d d coefficients returned by spline
	 * @param nn number of interpolants
	 * @param xx values of x at data points to be interpolated
	 * @param yy output interpolated values
	 */
	template <typename T>
	void splint(size_t n, T x[], T a[], T b[], T c[], T d[], size_t nn, T xx[], T yy[]) {

		size_t ii = 0;
		for (size_t i = 0; i < n-1; ++i) {
			// find starting index ii
			while (ii < nn && xx[ii++] < x[i]);
			
			// interpolate until the next spline
			while (ii < nn && xx[ii] < x[i+1]) {
				T t = xx[ii] - x[i];
				yy[ii++] = a[i] + (b[i] + (c[i] + d[i] * t) * t ) * t;
			}

			if (ii >= nn) break;
		}

	}

}

#endif
