#ifndef _scan_scalar_functor_hpp_
#define _scan_scalar_functor_hpp_

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

#endif