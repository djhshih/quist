// Fermi device has 32 banks for shared memory
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n)  ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))  

template <typename T>
struct Adder {
	__device__ __host__ T operator()(T a, T b) const {
		return a + b;
	}
	// identity
	__device__ __host__ T operator()() const {
		return 0;
	}
};

template <typename T>
struct ScalarSetter {
	__device__ __host__ void operator()(T& a, const T& b) const {
		a = b;
	}
};


template <typename T, typename BinaryOp, typename Setter >
__global__ void prescan(size_t n, const T* x, T* y, BinaryOp binaryOp, Setter copy) {
	extern __shared__ T shared[];
	
	size_t i = threadIdx.x;
	// blockOffset is multiplied by 2, since each thread processes 2 elements
	size_t blockOffset = blockIdx.x * blockDim.x * 2;
	
	// copy 2 elements to shared memory
	size_t ai = i;
	size_t bi = i + (n/2);
	size_t bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	size_t bankOffsetB = CONFLICT_FREE_OFFSET(bi);
	copy(shared[ai + bankOffsetA], x[ai + blockOffset]);
	copy(shared[bi + bankOffsetB], x[bi + blockOffset]);
	
	size_t offset = 1;

	// up-sweep
	for (size_t d = n/2; d > 0; d /= 2) {
		__syncthreads();
		if (i < d) {
			size_t ai = offset * (2*i+1) - 1;
			size_t bi = offset * (2*i+2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			copy(shared[bi],  binaryOp(shared[bi], shared[ai]));
		}
		offset *= 2;
	}
	
	// replace the last element with identity (to be propagated back to position 0)
	if (i == 0) copy(shared[n-1 + CONFLICT_FREE_OFFSET(n-1)], binaryOp());
	
	// traverse down tree and build scan
	for (size_t d = 1; d < n; d *= 2) {
		offset /= 2;
		__syncthreads();
		if (i < d) {
			size_t ai = offset * (2*i+1) - 1;
			size_t bi = offset * (2*i+2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			T t;
			copy(t, shared[ai]);
			copy(shared[ai], shared[bi]);
			copy(shared[bi], binaryOp(shared[bi], t));
		}
	}
	
	__syncthreads();
	
	// write results
	
	// shift results to left to compute *inclusive* scan
	int j = ai + blockOffset - 1;
	
	if (j >= 0) {
		copy(y[j], shared[ai + bankOffsetA]);
		// thread writing to second last element of block also writes result for last element
		++j;
		if ((j+1) % n == 0) {
			copy(y[j], binaryOp(shared[ai + bankOffsetA], x[j]));
		}
	}

	j = bi + blockOffset - 1;
	if (j >= 0) {
		copy(y[j], shared[bi + bankOffsetB]);
		// thread writing to second last element of block also writes result for last element
		++j;
		if ((j+1) % n == 0) {
			copy(y[j], binaryOp(shared[bi + bankOffsetB], x[j]));
		}
	}
}

template <typename T>
__global__ void aggregate_block_sum(size_t block_size, const T* y, T* out) {
	out[threadIdx.x] = y[(threadIdx.x+1) * block_size - 1];
}

template <typename T, typename BinaryOp, typename Setter >
__global__ void add_block_cumsum(size_t n, const T* blocks, T* y, BinaryOp binaryOp, Setter copy) {
	// since inclusive scan was calculated, don't modify elements within first block
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t j = i / (2*blockDim.x);
	if (j > 0) {
		copy(y[i], binaryOp(y[i], blocks[j-1]));
	}
}
