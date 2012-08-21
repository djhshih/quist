#ifndef emd_kernel_h
#define emd_kernel_h


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

__global__ void emd_all(size_t n) {
	
	size_t k = log2(n) + 1;
	
}

template <typename T, typename U>
__device__ void emd(size_t n, const T x[], const U y[], size_t k, U** modes, size_t max_iter=10) {

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
	
	// copy data for computing running signal
	U* running = copied(y, n);
	U* current = new U[n];
	
	for (size_t i = 0; i < k-1; ++i) {
		copy(current, running, n);
		
		for (size_t j = 0; j < max_iter; ++j) {
			// find extrema
			size_t n_max, n_min;
			maxima(n, x, current, &n_max, max_x, max_y);
			minima(n, x, current, &n_min, min_x, min_y);
			
			// compute upper and lower envelops (parallelizable)
			bla::splint(n_max, max_x, max_y, n, x, upper);
			bla::splint(n_min, min_x, min_y, n, x, lower);
			
			// substract the mean
			for (size_t ii = 0; ii < n; ++ii) {
				current[ii] -= (upper[ii] + lower[ii]) / 2;
			}
		}
		
		// save current intrinsic mode
		modes[i] = copied(current, n);
		
		// update running signal
		for (size_t ii = 0; ii < n; ++ii) {
			running[ii] -= current[ii];
		}
	}
	
	// save the residual
	modes[k-1] = copied(running, n);
	
	
	delete [] current;
	delete [] running;
	
	delete [] min_x;
	delete [] max_x;
	delete [] min_y;
	delete [] max_y;
	delete [] upper;
	delete [] lower;
	
	return modes;
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