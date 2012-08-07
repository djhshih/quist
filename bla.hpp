#ifndef quist_bla_h
#define quist_bla_h

// Basic Linear Algebra

namespace bla {

	// dot product of a and b
	template <typename T>
	T dot(T a[], T b[], size_t n) {
		T d = 0;
		for (size_t i = 0; i < n; ++i) {
			d += a[i] * b[i];
		}
		return d;
	}	

	// solves tridiagonal systems of linear equations (Thomas algorithm)
	template <typename T>
	void tridiag(size_t n, T a[], T b[], T c[], T r[], T x[]) {
		/**
		 * n - number of equations
		 * a - sub-diagonal, indexed from 1..n-1 (a[0] is ignored)
		 * b - main diagonal
		 * c - sup-diagonal, indexed from 0..n (n-1 is ignored)
		 * r - right hand side
		 * x - the solution
		 */

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

}

#endif
