#ifndef emd_kernel_h
#define emd_kernel_h


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
__device__ void tridiag(size_t n, T a[], T b[], T c[], T r[], T x[]) {

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
__device__ void ncspline(size_t n, const T x[], const T y[], T b[], T c[], T d[]) {

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
__device__ void splint(size_t n, const T x[], const T a[], const T b[], T const c[], const T d[], size_t nn, const T xx[], T yy[]) {

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

/**
	* Spline interpolation.
	* For one-time use.
	* @param n number of fitted data points
	* @param x values of x at fitted data points
	* @param y value of function evaluated at x
	* @param nn number of interpolants
	* @param xx values of x at data points to be interpolated
	* @param yy output interpolated values
	*/
template <typename T>
__device__ void splint(size_t n, const T x[], const T y[], size_t nn, const T xx[], T yy[]) {
	T* b = new T[n];
	T* c = new T[n];
	T* d = new T[n];
	
	ncspline(n, x, y, b, c, d);
	splint(n, x, y, b, c, d, nn, xx, yy);
	
	delete [] b;
	delete [] c;
	delete [] d;
}

template <typename coord_t, typename real_t>
__device__ void minima(size_t n, const coord_t x[], const real_t y[], size_t* kk, coord_t xx[], real_t yy[]) {
		
	// tentatively designate first point as minimum
	xx[0] = x[0];
	yy[0] = y[0];
	
	// identify minima
	size_t k = 1;
	size_t i;
	for (i = 1; i < n-1; ++i) {
		// y_i is less than adjacent values: y_i is a minimum
		if (y[i] <= y[i-1] && y[i] <= y[i+1]) {
			xx[k] = x[i];
			yy[k] = y[i];
			++k;
		}
	}
	
	// tentatively designate last point as minimum
	xx[k] = x[i];
	yy[k] = y[i];
	
	// consider replacing end points with extrapolants
	if (k+1 >= 4) {
		
		real_t slope, t;
		
		slope = (yy[2] - yy[1]) / (xx[2] - xx[1]);
		t = yy[1] - slope * (xx[1] - xx[0]);
		if (t < yy[0]) yy[0] = t;
		
		slope = (yy[k-1] - yy[k-2]) / (xx[k-1] - xx[k-2]);
		t = yy[k-1] + slope * (xx[k] - xx[k-1]);
		if (t < yy[k]) yy[k] = t;
		
	}
	
	*kk = k+1;
}

template <typename coord_t, typename real_t>
__device__ void maxima(size_t n, const coord_t x[], const real_t y[], size_t* kk, coord_t xx[], real_t yy[]) {
	
	// tentatively designate first point as maximum
	xx[0] = x[0];
	yy[0] = y[0];
	
	// identify minima
	size_t k = 1;
	size_t i;
	for (i = 1; i < n-1; ++i) {
		// y_i is greater than adjacent values: y_i is a maximum
		if (y[i] >= y[i-1] && y[i] >= y[i+1]) {
			xx[k] = x[i];
			yy[k] = y[i];
			++k;
		}
	}
	
	// tentatively designate last point as maximum
	xx[k] = x[i];
	yy[k] = y[i];
	
	// consider replacing end points with extrapolants
	if (k+1 >= 4) {
		
		real_t slope, t;
		
		slope = (yy[2] - yy[1]) / (xx[2] - xx[1]);
		t = yy[1] - slope * (xx[1] - xx[0]);
		if (t > yy[0]) yy[0] = t;
		
		slope = (yy[k-1] - yy[k-2]) / (xx[k-1] - xx[k-2]);
		t = yy[k-1] + slope * (xx[k] - xx[k-1]);
		if (t > yy[k]) yy[k] = t;
		
	}
	
	*kk = k+1;
}

/**
	* Copy array from source to destination.
	* Arrays must be pre-allocated.
	* @param dest destination
	* @param src source
	* @param n number of elements
	*/
template <typename T>
__device__ void copy(T dest[], const T src[], size_t n) {
	memcpy((void*)dest, (const void*)src, n * sizeof(T));
}

template <typename T, typename U>
__device__ void emd(size_t N, size_t n, const T x[], const U y[], size_t k, U* modes, size_t max_iter=10) {

	// allocate arrays for storing x and y values of extrema points, 
	//   and upper and lower envelops
	size_t nbytes_x = n * sizeof(T);
	T* min_x = (T*)malloc(nbytes_x);
	T* max_x = (T*)malloc(nbytes_x);
	
	size_t nbytes_y = n * sizeof(U);
	U* min_y = (U*)malloc(nbytes_y);
	U* max_y = (U*)malloc(nbytes_y);
	U* upper = (U*)malloc(nbytes_y);
	U* lower = (U*)malloc(nbytes_y);
	
	U* current = (U*)malloc(nbytes_y);
	
	
	// copy data for computing running signal
	U* running = (U*)malloc(nbytes_y);
	copy(running, y, n);
	
	for (size_t i = 0; i < k-1; ++i) {
		copy(current, running, n);
		
		for (size_t j = 0; j < max_iter; ++j) {
			// find extrema
			size_t n_max, n_min;
			maxima(n, x, current, &n_max, max_x, max_y);
			minima(n, x, current, &n_min, min_x, min_y);
			
			// compute upper and lower envelops (parallelizable)
			splint(n_max, max_x, max_y, n, x, upper);
			splint(n_min, min_x, min_y, n, x, lower);
			
			// substract the mean
			for (size_t ii = 0; ii < n; ++ii) {
				current[ii] -= (upper[ii] + lower[ii]) / 2;
			}
		}
		
		// save current intrinsic mode
		copy(&modes[i*N], current, n);
		
		// update running signal
		for (size_t ii = 0; ii < n; ++ii) {
			running[ii] -= current[ii];
		}
	}
	
	// save the residual
	copy(&modes[(k-1)*n], running, n);
	
	
	free(current);
	free(running);
	
	free(min_x);
	free(max_x);
	free(min_y);
	free(max_y);
	free(upper);
	free(lower);
}

template <typename coord_t, typename real_t>
__global__ void emd_all(size_t wsize, size_t n, const coord_t* x, const real_t* y, size_t k, real_t* modes, size_t max_iter=10) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t start = idx * wsize;
	if (start < n) {
		emd(n, wsize, &x[start], &y[start], k, &modes[start], max_iter);
	}
}

template <typename coord_t, typename real_t>
__global__ void minima_all(size_t wsize, size_t n, const coord_t* x, const real_t* y, size_t* kk, coord_t* xx, real_t* yy) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t start = idx * wsize;
	if (start < n) {
		minima(wsize, &x[start], &y[start], &kk[idx], &xx[start], &yy[start]);
	}
}

#endif