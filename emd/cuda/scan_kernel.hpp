// Fermi device has 32 banks for shared memory
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n)  ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))  
#include <boost/concept_check.hpp>


template <typename T>
struct ScalarAdder {
	__device__ __host__ void operator()(T& b, const T& a, bool reversed=false) const {
		b += a;
	}
	// set a to identity
	__device__ __host__ void operator()(T& a) const {
		a = 0;
	}
};

template <typename T>
struct ScalarSetter {
	__device__ __host__ void operator()(T& b, const T& a) const {
		b = a;
	}
};

template <typename T>
struct ArrayAdder {
	ArrayAdder(size_t n) : _n(n) {}
	__device__ __host__ void operator()(T& b, const T& a, bool reversed=false) const {
		const T *pa = &a;
		T *pb = &b;
		for (size_t i = 0; i < _n; ++i) {
			*(pb++) += *(pa++);
		}
	}
	// set to identity
	__device__ __host__ void operator()(T& a) const {
		memset(&a, 0, sizeof(T)*_n);
	}
private:
	size_t _n;
};

template <typename T>
struct ArraySetter {
	ArraySetter(size_t n) : _n(n) {}
	__device__ __host__ void operator()(T& b, const T& a) const {
		memcpy(&b, &a, sizeof(T)*_n);
	}
	private:
		size_t _n;
};

/**
 * Squared matrix mulitplier
 */
template <typename T>
struct MatrixMultipler {
	MatrixMultipler(size_t n) : _n(n) {}
	__device__ __host__ void operator()(T& b, const T& a, bool reversed=false) const {
		const T* pa = &a;
		T* pb = &b;
		T* pc = new T[_n*_n];
		memset(pc, 0, sizeof(T)*_n*_n);
		
		if (reversed) {
			for (size_t i = 0; i < _n; ++i) {
				for (size_t k = 0; k < _n; ++k) {
					for (size_t j = 0; j < _n; ++j) {
						pc[i*_n+j] += pb[i*_n+k]*pa[k*_n+j];
					}
				}
			}
		} else {
			for (size_t i = 0; i < _n; ++i) {
				for (size_t k = 0; k < _n; ++k) {
					for (size_t j = 0; j < _n; ++j) {
						pc[i*_n+j] += pa[i*_n+k]*pb[k*_n+j];
					}
				}
			}
		}
		
		memcpy(pb, pc, sizeof(T)*_n*_n);
		delete [] pc;
	}
	// set to identity
	__device__ __host__ void operator()(T& a) const {
		T* pa = &a;
		memset(pa, 0, sizeof(T)*_n*_n);
		for (size_t i = 0; i < _n; ++i) {
			pa[i*_n + i] = (T)1;
		}
	}
private:
	size_t _n;
};	

template <typename T>
struct MatrixSetter {
	MatrixSetter(size_t n) : _n(n) {}
	__device__ __host__ void operator()(T& b, const T& a) const {
		memcpy(&b, &a, sizeof(T)*_n*_n);
	}
private:
	size_t _n;
};	

/**
 * Mulitpler for 3x3 matrix
 * Prefix
 */
template <typename T>
struct Matrix33Multipler {
	__device__ __host__ void operator()(T& b, const T& a, bool reversed=false) const {
		const T* pa = &a;
		T* pb = &b;
		T pc[9];
		memset(pc, 0, sizeof(T)*9);
		
		if (reversed) {
			for (size_t i = 0; i < 3; ++i) {
				for (size_t k = 0; k < 3; ++k) {
					for (size_t j = 0; j < 3; ++j) {
						pc[i*3+j] += pb[i*3+k]*pa[k*3+j];
					}
				}
			}
		} else {
			for (size_t i = 0; i < 3; ++i) {
				for (size_t k = 0; k < 3; ++k) {
					for (size_t j = 0; j < 3; ++j) {
						pc[i*3+j] += pa[i*3+k]*pb[k*3+j];
					}
				}
			}
		}
		memcpy(pb, pc, sizeof(T)*9);
	}
	// set to identity
	__device__ __host__ void operator()(T& a) const {
		T* pa = &a;
		memset(&pa[1], 0, sizeof(T)*7);
		//pa[1] = pa[2] = pa[3] = pa[5] = pa[6] = pa[7] = 0;
		pa[0] = pa[4] = pa[8] = 1;
	}
};

template <typename T>
struct Matrix33Setter {
	__device__ __host__ void operator()(T& b, const T& a) const {
		memcpy(&b, &a, sizeof(T)*9);
	}
};

/**
 * NB  If the binary operation is not communicative, be sure to provide a prefix binary operator.
 */
template <typename T, typename BinaryOp, typename Setter >
__global__ void prescan(const size_t n, const T* x, T* y, BinaryOp binaryOp, Setter copy, const size_t m=1) {
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
	
	//T* t = new T[9];
	T t[9];
	
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

template <typename T, typename Setter>
__global__ void aggregate_block_sum(size_t block_size, const T* y, T* out, Setter copy, size_t m = 1) {
	copy(out[threadIdx.x*m], y[((threadIdx.x+1) * block_size - 1)*m]);
}

template <typename T, typename BinaryOp, typename Setter >
__global__ void add_block_cumsum(size_t n, const T* blocks, T* y, BinaryOp binaryOp, Setter copy, size_t m = 1) {
	// since inclusive scan was calculated, don't modify elements within first block
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t j = i / (2*blockDim.x);
	if (j > 0) {
		binaryOp(y[i*m], blocks[(j-1)*m]);
	}
}
