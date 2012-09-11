#ifndef _scan_array_functor_hpp_
#define _scan_array_functor_hpp_

#include <cstddef>

template <typename T>
struct ArrayAdder {
	ArrayAdder(size_t n) : _n(n) {}
	__device__ __host__ void operator()(T& b, const T& a, bool reversed=false) const {
		const T *pa = &a;
		T *pb = &b;
#pragma unroll
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

#endif