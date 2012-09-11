#ifndef _scan_dynamic_matrix_functor_hpp_
#define _scan_dynamic_matrix_functor_hpp_

#include <cstddef>
#include <cstdlib>

/**
 * Squared matrix mulitplier
 * NB  dynamically allocated arrays seem to cause discrepancy in prescan
 */
template <typename T>
struct MatrixMultiplierPostfix {
	MatrixMultiplierPostfix(size_t n) : _n(n) {}
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

#endif