// Fermi device has 32 banks for shared memory
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n)  ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))  

// TODO  Use functors

template <typename T>
__global__ void prescan(size_t n, const T* x, T* y) {
	extern __shared__ T shared[];
	
	size_t i = threadIdx.x;
	// blockOffset is multiplied by 2, since each thread processes 2 elements
	size_t blockOffset = blockIdx.x * blockDim.x * 2;
	
	// copy 2 elements to shared memory
	size_t ai = i;
	size_t bi = i + (n/2);
	size_t bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	size_t bankOffsetB = CONFLICT_FREE_OFFSET(bi);
	shared[ai + bankOffsetA] = x[ai + blockOffset];
	shared[bi + bankOffsetB] = x[bi + blockOffset];
	
	size_t offset = 1;

	// up-sweep
	for (size_t d = n/2; d > 0; d /= 2) {
		__syncthreads();
		if (i < d) {
			size_t ai = offset * (2*i+1) - 1;
			size_t bi = offset * (2*i+2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			shared[bi] += shared[ai];
		}
		offset *= 2;
	}
	
	// clear the last element (to be propagated back to position 0)
	if (i == 0) shared[n-1 + CONFLICT_FREE_OFFSET(n-1)] = 0;
	
	// traverse down tree and build scan
	for (size_t d = 1; d < n; d *= 2) {
		offset /= 2;
		__syncthreads();
		if (i < d) {
			size_t ai = offset * (2*i+1) - 1;
			size_t bi = offset * (2*i+2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			T t = shared[ai];
			shared[ai] = shared[bi];
			shared[bi] += t;
		}
	}
	
	__syncthreads();
	
	// write results
	y[ai + blockOffset - 1] = shared[ai + bankOffsetA];
	y[bi + blockOffset - 1] = shared[bi + bankOffsetB];
}

template <typename T>
__global__ void aggregate_block_sum(size_t block_size, const T* x, const T* y, T* out) {
	size_t j = (threadIdx.x+1) * block_size - 1;
	// since exclusive prefix was calculated, need to add x[j]
	out[threadIdx.x] = x[j] + y[j];
}

template <typename T>
__global__ void add_block_cumsum(size_t n, const T* blocks, T* y) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	y[i] += blocks[ i / (2*blockDim.x) ];
}
