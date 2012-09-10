#ifndef _signal_dsemd_hpp_
#define _signal_dsemd_hpp_

#include <cstdlib>

#include "util.hpp"
#include "emd.hpp"

namespace signal {

	/**
	 * Down-sample.
	 * NB  last datum may never be sampled if population size is not a multiple of the sample size.
	 * @param n population size
	 * @param ns sample size (ns < n)
	 * @param idx output sampled index
	 */
	void downsample(size_t n, size_t ns, size_t idx[]) {
		
		if (n % ns == 0) {
			
			// simple down-sampling
			size_t window_size = n / ns;
			size_t wi = 0;
			for (size_t i = 0; i < ns * window_size; i += window_size) {
				size_t j = i + (std::rand() % window_size);
				idx[wi++] = j;
			}
		
		} else {
			
			// down-sampling using floating point boundaries
			float window_size = n / (float)ns;
			size_t wi = 0;
			for (float i = 0; (size_t)i < n; i += window_size) {
				// determine current window size
				size_t start = (size_t)i;
				size_t end = (size_t)(i + window_size);
				size_t j = i + (std::rand() % (end - start));
				idx[wi++] = j;
			}
			
		}
		
	}
	
	/**
	 * Empirical Mode Decomposition with down-sampling.
	 * Data are sampled uniformly along the input array.
	 * @param ns number of samples
	 * @param nr number of down-sampling rounds
	 */
	template <typename T, typename U>
	T** dsemd(size_t n, const T x[], const U y[], size_t* kk, size_t ns, size_t nr, size_t max_iter=10) {
		
		size_t k = *kk;
		if (k <= 1) {
			// expected number of intrinsic mode functions is log2(n)
			// plus one to number of output arrays for the residual
			k = math::log2(n) + 1;
			// modify input parameter
			*kk = k;
		}
		
		// allocate memory
		U** ensemble = new_arrays(k, n, (U)0.0);
		size_t* counts = new size_t[n];
		for (size_t j = 0; j < n; ++j) {
			counts[j] = 0;
		}
		
		for (size_t r = 0; r < nr; ++r) {
		
			// allocate for down-sampled data points
			size_t* idx = new size_t[ns];
			T* x2 = new T[ns];
			U* y2 = new U[ns];
		
			downsample(n, ns, idx);
			
			// copy sampled data
			// keep track of number of times each datum is sampled
			for (size_t j = 0; j < ns; ++j) {
				x2[j] = x[idx[j]];
				y2[j] = y[idx[j]];
				++(counts[idx[j]]);
			}
			
			U** modes = emd(ns, x2, y2, kk, max_iter);
			
			// accumulate sum
			for (size_t i = 0; i < k; ++i) {
				for (size_t j = 0; j < ns; ++j) {
					ensemble[i][idx[j]] += modes[i][j];
				}
			}
			
			delete [] idx;
			delete [] x2;
			delete [] y2;
			free_arrays(modes, k);
			
		}
		
		// scale the ensemble values
		for (size_t i = 0; i < k; ++i) {
			for (size_t j = 0; j < n; ++j) {
				ensemble[i][j] /= counts[j];
			}
		}
		
		delete [] counts;
		
		return ensemble;
	}
	
}

#endif