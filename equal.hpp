#ifndef quist_equal_h
#define quist_equal_h

#include <cstdio>

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_arithmetic.hpp>

#define ENABLE_IF_ARITHMETIC typename boost::enable_if<boost::is_arithmetic<T>, T>

template <typename T>
inline ENABLE_IF_ARITHMETIC::type absdiff(T a, T b) {
	return (a > b) ? a - b : b - a;
}

template <typename T>
inline bool eq(T a, T b, ENABLE_IF_ARITHMETIC::type epsilon=std::numeric_limits<T>::epsilon()) {
	// essentially equal
	//return abs(a - b) <= ( (abs(a) > abs(b) ? abs(b) : abs(a)) * epsilon );
	return absdiff(a, b) <= epsilon;
}

template <typename T>
inline bool neq(T a, T b, ENABLE_IF_ARITHMETIC::type epsilon=std::numeric_limits<T>::epsilon()) {
	// essentially equal
	//return abs(a - b) > ( (abs(a) > abs(b) ? abs(b) : abs(a)) * epsilon );
	return absdiff(a, b) > epsilon;
}


template <typename T>
bool check_array_equal(const T res[], const T ans[], size_t n, T epsilon = 1e-5, size_t array_n_peek = 5) {
	bool equal = true;
	size_t i;
	for (i = 0; i < n; ++i) {
		if (neq(res[i], ans[i], epsilon)) {
			equal = false;
			std::printf("%d:\t%f\t%f*\n", i, res[i], ans[i]);
			continue;
		}
		
		if (i < array_n_peek) {
			std::printf("%d:\t%f\t%f\n", i, res[i], ans[i]);
			if (i == array_n_peek - 1) {
				std::printf("...\n");
			}
		}
		
	}
	return equal;
}

template <typename T>
bool check_arrays_equal(const T* const * res, const T* ans, size_t m, size_t n, T epsilon = 1e-5, size_t array_n_peek = 5) {
	bool equal = true;
	for (size_t i = 0; i < m; ++i) {
		for (size_t j = 0; j < n; ++j) {
			size_t k = j+i*n;
			if (neq(res[i][j], ans[k], epsilon)) {
				std::printf("%d,%d:\t%f\t%f*\n", i, j, res[i][j], ans[k]);
				equal = false;
				continue;
			}
			
			if (k < array_n_peek) {
				std::printf("%d,%d:\t%f\t%f\n", i, j, res[i][j], ans[k]);
				if (k == array_n_peek - 1) {
					std::printf("...\n");
				}
			}
		}
	}
	return equal;
}

#endif
