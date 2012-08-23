#ifndef emd_kernel_h
#define emd_kernel_h

#include <curand_kernel.h>
#include <boost/iterator/iterator_concepts.hpp>

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
	
	/*
	// thread 0 of each block allocates shared memory
	__shared__ T* shared;
	const size_t narrays = 3;
	if (threadIdx.x == 0) {
		shared = new T[narrays*n];
	}
	__syncthreads();
	
	T* b = &shared[0];
	T* c = &shared[n];
	T* d = &shared[2*n];
	*/
	
	T* b = new T[n];
	T* c = new T[n];
	T* d = new T[n];
	
	ncspline(n, x, y, b, c, d);
	splint(n, x, y, b, c, d, nn, xx, yy);
	
	delete [] b;
	delete [] c;
	delete [] d;
	
	/*
	delete [] shared;
	*/
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
__device__ void d_emd(size_t N, size_t n, const T x[], const U y[], size_t k, U* modes, size_t max_iter=10) {
	
	__shared__ U* current;
	__shared__ U* running;
	
	// thread 0 of each block allocates shared memory
	if (threadIdx.x == 0) {
		current = new U[n];
		running = new U[n];
	}
	__syncthreads();
	
	U* min_y = new U[n];
	U* max_y = new U[n];
	
	U* upper = new U[n];
	U* lower = new U[n];
	
	T* min_x = new T[n];
	T* max_x = new T[n];
	
	// copy data for computing running signal
	copy(running, y, n);
	
	for (size_t i = 0; i < k-1; ++i) {
		copy(current, running, n);
		
		for (size_t j = 0; j < max_iter; ++j) {
			// find extrema
			size_t n_max, n_min;
			maxima(n, x, current, &n_max, max_x, max_y);
			minima(n, x, current, &n_min, min_x, min_y);
			
			// compute upper and lower envelops
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
	copy(&modes[(k-1)*N], running, n);
	

	// free memory
	delete [] current;
	delete [] running;
	delete [] min_x;
	delete [] max_x;
	delete [] min_y;
	delete [] max_y;
	delete [] upper;
	delete [] lower;
}

/**
	* Down-sample.
	* NB  last datum may never be sampled if population size is not a multiple of the sample size.
	* @param n population size
	* @param ns sample size (ns < n)
	* @param idx output sampled index
	*/
__device__ void downsample(size_t n, size_t ns, size_t idx[], curandState* rstate) {
	
	// sample one datum within each window
	
	if (n % ns == 0) {
		
		// simple down-sampling
		size_t window_size = n / ns;
		size_t wi = 0;
		for (size_t i = 0; i < ns * window_size; i += window_size) {
			size_t j = i + (curand(rstate) % window_size);
			idx[wi++] = j;
		}
	
	} else {
		
		// down-sampling using floating point boundaries
		float window_size = n / (float)ns;
		size_t wi = 0;
		for (float i = 0; (size_t)i < n; i += window_size) {
			// determine current window size
			size_t start = (size_t)i;
			size_t end = (size_t)(i + window_size);
			size_t j = i + (curand(rstate) % (end - start));
			idx[wi++] = j;
		}
		
	}
	
}

// each thread processes a window.
template <typename coord_t, typename real_t>
__global__ void emd_strat(size_t wsize, size_t n, const coord_t* x, const real_t* y, size_t k, real_t* modes, size_t max_iter=10) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t start = idx * wsize;
	
	
	if (start < n) {
		d_emd(n, wsize, &x[start], &y[start], k, &modes[start], max_iter);
	}
}

// each thread processes a down-sampling round.
template <typename T, typename U>
__global__ void dsemd(size_t n, const T* x, const U* y, size_t ns, size_t nr, unsigned int* counts, size_t k, U* ensemble, unsigned long long seed=0, size_t max_iter=10) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	// initialize random seed
	curandState rstate;
  curand_init(seed, idx, 0, &rstate);
	
	size_t* samples = new size_t[ns];
	downsample(n, ns, samples, &rstate);
	
	// allocate for down-sampled data points
	T* x2 = new T[ns];
	U* y2 = new U[ns];
	
	// copy sampled data
	for (size_t j = 0; j < ns; ++j) {
		x2[j] = x[samples[j]];
		y2[j] = y[samples[j]];
	}
	
	U* modes = new U[k*ns];
	d_emd(ns, ns, x2, y2, k, modes, max_iter);
	
	// accumulate sum
	for (size_t i = 0; i < k; ++i) {
		for (size_t j = 0; j < ns; ++j) {
			ensemble[i*n + samples[j]] += modes[i*ns + j];
			__threadfence();
			//atomicAdd(&ensemble[i*n + samples[j]], modes[i*ns + j]);
		}
	}
	
	// keep track of number of times each datum is sampled
	for (size_t j = 0; j < ns; ++j) {
		++(counts[samples[j]]);
		__threadfence();
		//atomicAdd(&counts[samples[j]], 1);
	}
		
	delete [] samples;
	delete [] x2;
	delete [] y2;
	delete [] modes;
	
}

template <typename T, typename U>
__global__ void scale(size_t n, size_t k, T* values, const U* factors) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		for (size_t i = 0; i < k; ++i) {
			values[i*n + idx] /= factors[idx];
		}
	}
}

template <typename coord_t, typename real_t>
__global__ void minima_strat(size_t wsize, size_t n, const coord_t* x, const real_t* y, size_t* kk, coord_t* xx, real_t* yy) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t start = idx * wsize;
	if (start < n) {
		minima(wsize, &x[start], &y[start], &kk[idx], &xx[start], &yy[start]);
	}
}

template <typename coord_t, typename real_t>
__global__ void maxima_strat(size_t wsize, size_t n, const coord_t* x, const real_t* y, size_t* kk, coord_t* xx, real_t* yy) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t start = idx * wsize;
	if (start < n) {
		maxima(wsize, &x[start], &y[start], &kk[idx], &xx[start], &yy[start]);
	}
}

#endif