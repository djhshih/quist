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
__global__ void tridiag(size_t n, T* a, T* b, T* c, T* r, T* x) {

	// decomposition, set the coefficients
	for (size_t i = 1; i < n; ++i) {
		T m = a[i] / b[i-1];
		b[i] -= m * c[i-1];
		r[i] -= m * r[i-1];
	}

	x[n-1] = r[n-1] / b[n-1];

	// back substitution
	for (long i = n-2; i >= 0; --i) {
		x[i] = ( r[i] - c[i]*x[i+1] ) / b[i];
	}
}