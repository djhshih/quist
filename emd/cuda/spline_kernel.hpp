
// FIXME
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
//__global__ void ncspline(size_t n, const T* x, const T* y, T* b, T* c, T* d) {
__global__ void ncspline_setup(size_t n, const T* x, const T* y, T* sub_diag, T* main_diag, T* sup_diag, T* r) {
	
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	
	// setup tridiagonal matrix system
	
	if (i == 0 || i == n-1) {
		
		sub_diag[i] = 0;
		main_diag[i] = 1;
		sup_diag[i] = 0;
		r[i] = 0;
		
	} else {
		
		// cache in register values that will be re-used
		// p = previous; c = current; n = next
		T hp = x[i] - x[i-1], hc = x[i+1] - x[i];
		T yp = y[i-1], yc = y[i], yn = y[i+1];
		
		sub_diag[i] = hp;
		main_diag[i] = 2 * (hp + hc);
		sup_diag[i] = hc;
		
		// setup right-hand side
		r[i] = ( 3 * (yn - yc) / hc ) - ( 3 * (yc - yp) / hp );
		
	}
}

template <typename T>
__global__ void ncspline_teardown(size_t n, const T* x, const T* y, const T* c, T* b, T* d) {
	
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < n-1) {
		T hc = x[i+1] - x[i];
		b[i] = (y[i+1] - y[i]) / hc  -  hc * (2*c[i] + c[i+1]) / 3;
		d[i] = (c[i+1] - c[i]) / (3 * hc);
	} else {
		// b[n-1] and d[n-1] are not calculated
		b[i] = 0;
		d[i] = 0;
	}
	
}

// FIXME
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
__global__ void splint(size_t n, const T* x, const T* a, const T* b, const T* c, const T* d, size_t nn, const T* xx, T* yy) {

	size_t ii = 0;
	for (size_t i = 0; i < n-1; ++i) {
		
		// find starting index ii
		while (ii < nn && xx[ii] < x[i]) {
			++i;
		}
		
		// interpolate until the next spline
		while (ii < nn && xx[ii] <= x[i+1]) {
			T t = xx[ii] - x[i];
			yy[ii++] = a[i] + (b[i] + (c[i] + d[i] * t) * t ) * t;
		}

		if (ii >= nn) break;
	}

}
