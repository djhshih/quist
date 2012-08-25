#ifndef tridiag_rd_kernel_h
#define tridiag_rd_kernel_h

/**
 * Setup prefix matrices.
 * NB  c[i] must be non-zero for all i
 * NB  B must be pre-initialized to a zero matrix
 * @param B output prefix matrices (inlined in row-major order)
 */
template <typename T>
__global__ void rd_prefix(size_t n, const T* a, const T* b, const T* c, const T* r, T* B) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	
	const size_t B_nelem = 9;
	size_t offset = i*B_nelem;
	
	B[offset] =  - b[i] / c[i];
	B[offset+1] =  - a[i] / c[i];
	B[offset+2] = r[i] / c[i];
	B[offset+3] = B[offset+8] = 1;
	//B[offset+4] = B[offset+5] = B[offset+6] = B[offset+7] = 0;
}

template <typename T>
__global__ void rd_tridiag(size_t n, const T* C, T* x) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	
	const size_t C_nelem = 9;
	size_t offset;
	
	if (i == 0) {
		
		offset = (n-1) * C_nelem;
		x[0] = - C[offset+2] / C[offset];
		
		// ensure X0 is updated for all threads
		__threadfence();
		
	} else {
	
		offset = (i-1) * C_nelem;
		x[i] = C[offset] * x[0] + C[offset+2];
		
	}
}

#endif