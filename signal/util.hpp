#ifndef _signal_util_hpp_
#define _signal_util_hpp_

namespace signal {
	
	/**
	 * Copy array.
	 * Memory is dynamically allocated to returned array and must be freed.
	 * @param x array to copy
	 * @param n number of elements
	 * @return pointer to a copy of the array
	 */
	template <typename T>
	T* copied(const T x[], size_t n) {
		T* y = new T[n];
		for (size_t i = 0; i < n; ++i) y[i] = x[i];
		return y;
	}
	
	/**
	 * Copy array from source to destination.
	 * Arrays must be pre-allocated.
	 * @param dest destination
	 * @param src source
	 * @param n number of elements
	 */
	template <typename T>
	void copy(T dest[], const T src[], size_t n) {
		for (size_t i = 0; i < n; ++i) dest[i] = src[i];
	}
	
	
	template <typename T>
	void free_arrays(T** x, size_t m) {
		for (size_t i = 0; i < m; ++i) {
			delete [] x[i];
		}
	}
	
	template <typename T>
	T** new_arrays(size_t m, size_t n, T v) {
		T** x = new T*[m];
		for (size_t i = 0; i < m; ++i) {
			x[i] = new T[n];
			for (size_t j = 0; j < n; ++j) {
				x[i][j] = v;
			}
		}
		return x;
	}
	
}

#endif