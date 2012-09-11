#ifndef _scan_static_matrix_functor_hpp_
#define _scan_static_matrix_functor_hpp_

#include <cstddef>
#include <cstdlib>

/**
 * Mulitpler for static matrix
 * Postfix
 */
template <typename T, size_t n>
struct StaticMatrixMultiplierPostfix {
	__device__ __host__ void operator()(T& b, const T& a, bool reversed=false) const {
		const T* pa = &a;
		T* pb = &b;
		T pc[n*n];
		memset(pc, 0, sizeof(T)*n*n);
		
		if (reversed) {
#pragma unroll
			for (size_t i = 0; i < n; ++i) {
#pragma unroll
				for (size_t k = 0; k < n; ++k) {
#pragma unroll
					for (size_t j = 0; j < n; ++j) {
						pc[i*3+j] += pb[i*n+k]*pa[k*n+j];
					}
				}
			}
		} else {
#pragma unroll
			for (size_t i = 0; i < n; ++i) {
#pragma unroll
				for (size_t k = 0; k < n; ++k) {
#pragma unroll
					for (size_t j = 0; j < n; ++j) {
						pc[i*n+j] += pa[i*n+k]*pb[k*n+j];
					}
				}
			}
		}
		memcpy(pb, pc, sizeof(T)*n*n);
	}
	// set to identity
	__device__ __host__ void operator()(T& a) const {
		T* pa = &a;
		memset(pa, 0, sizeof(T)*n*n);
		for (size_t i = 0; i < n; ++i) {
			pa[i*n+i] = 1;
		}
	}
};

/**
 * Mulitpler for static matrix
 * Postfix
 * Uses Kahan summation for numeric stability
 */
template <typename T, size_t n>
struct StaticMatrixMultiplierPostfixStable {
	__device__ __host__ void operator()(T& b, const T& a, bool reversed=false) const {
		const T* pa = &a;
		T* pb = &b;
		T pc[n*n];
		memset(pc, 0, sizeof(T)*n*n);
		
		if (reversed) {
#pragma unroll
			for (size_t i = 0; i < n; ++i) {
#pragma unroll
				for (size_t j = 0; j < n; ++j) {
					T sum = 0, c = 0;
#pragma unroll
					for (size_t k = 0; k < n; ++k) {
						T y = pb[i*n+k]*pa[k*n+j] - c;
						T t = sum + y;
						c = (t - sum) - y;
						sum = t;
					}
					pc[i*3+j] = sum;
				}
			}
		} else {
#pragma unroll
			for (size_t i = 0; i < n; ++i) {
#pragma unroll
				for (size_t j = 0; j < n; ++j) {
					T sum = 0, c = 0;
#pragma unroll
					for (size_t k = 0; k < n; ++k) {
						T y = pa[i*n+k]*pb[k*n+j] - c;
						T t = sum + y;
						c = (t - sum) - y;
						sum = t;
					}
					pc[i*3+j] = sum;
				}
			}
		}
		memcpy(pb, pc, sizeof(T)*n*n);
	}
	// set to identity
	__device__ __host__ void operator()(T& a) const {
		T* pa = &a;
		memset(pa, 0, sizeof(T)*n*n);
		for (size_t i = 0; i < n; ++i) {
			pa[i*n+i] = 1;
		}
	}
};
    

/**
 * Mulitplier for static matrix
 * Prefix
 */
template <typename T, size_t n>
struct StaticMatrixMultiplierPrefix {
	__device__ __host__ void operator()(T& b, const T& a, bool reversed=false) const {
		const T* pa = &a;
		T* pb = &b;
		T pc[n*n];
		memset(pc, 0, sizeof(T)*n*n);
		
		if (reversed) {
#pragma unroll
			for (size_t i = 0; i < n; ++i) {
#pragma unroll
				for (size_t k = 0; k < n; ++k) {
#pragma unroll
					for (size_t j = 0; j < n; ++j) {
						pc[i*n+j] += pa[i*n+k]*pb[k*n+j];
					}
				}
			}
		} else {
#pragma unroll
			for (size_t i = 0; i < n; ++i) {
#pragma unroll
				for (size_t k = 0; k < n; ++k) {
#pragma unroll
					for (size_t j = 0; j < n; ++j) {
						pc[i*3+j] += pb[i*n+k]*pa[k*n+j];
					}
				}
			}
		}
		memcpy(pb, pc, sizeof(T)*n*n);
	}
	// set to identity
	__device__ __host__ void operator()(T& a) const {
		T* pa = &a;
		memset(pa, 0, sizeof(T)*n*n);
		for (size_t i = 0; i < n; ++i) {
			pa[i*n+i] = 1;
		}
	}
};

/**
 * Mulitpler for static matrix
 * Prefix
 * Uses Kahan summation for numeric stability
 */
template <typename T, size_t n>
struct StaticMatrixMultiplierPrefixStable {
	__device__ __host__ void operator()(T& b, const T& a, bool reversed=false) const {
		const T* pa = &a;
		T* pb = &b;
		T pc[n*n];
		memset(pc, 0, sizeof(T)*n*n);
		
		if (reversed) {
#pragma unroll
			for (size_t i = 0; i < n; ++i) {
#pragma unroll
				for (size_t j = 0; j < n; ++j) {
					T sum = 0, c = 0;
#pragma unroll
					for (size_t k = 0; k < n; ++k) {
						T y = pa[i*n+k]*pb[k*n+j] - c;
						T t = sum + y;
						c = (t - sum) - y;
						sum = t;
					}
					pc[i*3+j] = sum;
				}
			}
		} else {
#pragma unroll
			for (size_t i = 0; i < n; ++i) {
#pragma unroll
				for (size_t j = 0; j < n; ++j) {
					T sum = 0, c = 0;
#pragma unroll
					for (size_t k = 0; k < n; ++k) {
						T y = pb[i*n+k]*pa[k*n+j] - c;
						T t = sum + y;
						c = (t - sum) - y;
						sum = t;
					}
					pc[i*3+j] = sum;
				}
			}
		}
		memcpy(pb, pc, sizeof(T)*n*n);
	}
	// set to identity
	__device__ __host__ void operator()(T& a) const {
		T* pa = &a;
		memset(pa, 0, sizeof(T)*n*n);
		for (size_t i = 0; i < n; ++i) {
			pa[i*n+i] = 1;
		}
	}
};

template <typename T, size_t n>
struct StaticMatrixSetter {
	__device__ __host__ void operator()(T& b, const T& a) const {
		memcpy(&b, &a, sizeof(T)*n*n);
	}
};

#endif