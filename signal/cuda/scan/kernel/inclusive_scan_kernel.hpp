// Fermi device has 32 banks for shared memory
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n)  ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))  

template <size_t m, typename T, typename BinaryOp, typename Setter>
__global__ void prescan(const size_t n, const T* x, T* y, BinaryOp binaryOp, Setter copy) {
	extern __shared__ T shared[];
	
	size_t i = threadIdx.x;
	// blockOffset is multiplied by 2, since each thread processes 2 elements
	size_t blockOffset = blockIdx.x * blockDim.x * 2;
	
	// copy 2 elements to shared memory
	size_t ai = i;
	size_t bi = i + (n/2);
	size_t bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	size_t bankOffsetB = CONFLICT_FREE_OFFSET(bi);
	copy(shared[(ai + bankOffsetA)*m], x[(ai + blockOffset)*m]);
	copy(shared[(bi + bankOffsetB)*m], x[(bi + blockOffset)*m]);
	
	size_t offset = 1;

	// up-sweep
	for (size_t d = n/2; d > 0; d /= 2) {
		__syncthreads();
		if (i < d) {
			size_t ai = offset * (2*i+1) - 1;
			size_t bi = offset * (2*i+2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			binaryOp(shared[bi*m], shared[ai*m]);
		}
		offset *= 2;
	}
	
	// replace the last element with identity (to be propagated back to position 0)
	if (i == 0) binaryOp(shared[(n-1 + CONFLICT_FREE_OFFSET(n-1))*m]);
	
	//T* t = new T[m];
	// NB  For undertermined reasons, using dynamically allocated memory causes discrepancy in results
	//     Possible reason: dynamically allocated arrays are not contiguous?
	T t[m];
	
	// traverse down tree and build scan
	for (size_t d = 1; d < n; d *= 2) {
		offset /= 2;
		__syncthreads();
		if (i < d) {
			size_t ai = offset * (2*i+1) - 1;
			size_t bi = offset * (2*i+2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			copy(*t, shared[ai*m]);
			copy(shared[ai*m], shared[bi*m]);
			binaryOp(shared[bi*m], *t, true);
		}
	}
	
	//delete [] t;
	
	__syncthreads();
	
	// write results
	
	// shift results to left to compute *inclusive* scan
	int j = ai + blockOffset - 1;
	
	if (j >= 0) {
		copy(y[j*m], shared[(ai + bankOffsetA)*m]);
		// thread writing to second last element of block also writes result for last element
		++j;
		if ((j+1) % n == 0) {
			copy(y[j*m], shared[(ai + bankOffsetA)*m]);
			binaryOp(y[j*m], x[j*m], true);
		}
	}

	j = bi + blockOffset - 1;
	if (j >= 0) {
		copy(y[j*m], shared[(bi + bankOffsetB)*m]);
		// thread writing to second last element of block also writes result for last element
		++j;
		if ((j+1) % n == 0) {
			copy(y[j*m], shared[(bi + bankOffsetB)*m]);
			binaryOp(y[j*m], x[j*m], true);
		}
	}
}

template <size_t m, typename T, typename Setter>
__global__ void aggregate_block_sum(size_t block_size, const T* y, T* out, Setter copy) {
	copy(out[threadIdx.x*m], y[((threadIdx.x+1) * block_size - 1)*m]);
}

template <size_t m, typename T, typename BinaryOp, typename Setter >
__global__ void add_block_cumsum(size_t n, const T* blocks, T* y, BinaryOp binaryOp, Setter copy) {
	// since inclusive scan was calculated, don't modify elements within first block
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t j = i / (2*blockDim.x);
	if (j > 0) {
		binaryOp(y[i*m], blocks[(j-1)*m]);
	}
}
