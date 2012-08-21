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

template <typename coord_t, typename real_t>
__global__ void minima_all(size_t wsize, size_t n, const coord_t* x, const real_t* y, size_t* kk, coord_t* xx, real_t* yy) {
	size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * wsize;
	if (idx < n) {
		size_t wk;
		minima(wsize, &x[idx], &y[idx], &wk, &xx[idx], &yy[idx]);
		kk[ blockIdx.x * blockDim.x + threadIdx.x ] = wk;
	}
}

#endif