#ifndef _signal_eemd_hpp_
#define _signal_eemd_hpp_

#include <ctime>

#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>

#include <numeric/math.hpp>
#include <numeric/stats.hpp>

#include "emd.hpp"
#include "util.hpp"

namespace signal {

	/**
	 * Ensemble Empirical Mode Decomposition.
	 * @param nsd noise standard deviation, in units of SD of data
	 * @param ne number of ensembles
	 */
	template <typename T, typename U>
	T** eemd(size_t n, const T x[], const U y[], size_t* kk, U nsd, size_t ne, size_t max_iter=10) {
		
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
		
		// calculate SD of data
		U mean, sd;
		stats::summary(y, n, &mean, &sd);
		
		// pseudo random number generator
		typedef boost::mt19937 Prng;
		typedef boost::random::normal_distribution<U> Norm;
		boost::variate_generator<Prng, Norm> rnorm(Prng((unsigned)std::time(NULL)), Norm(0, nsd*sd));
		
		for (size_t r = 0; r < ne; ++r) {
		
			// create new signal with added noise
			U* y2 = new U[n];
			for (size_t i = 0; i < n; ++i) {
				y2[i] = y[i] + rnorm();
			}
				
			U** modes = emd(n, x, y2, kk, max_iter);
			
			// accumulate sum
			for (size_t i = 0; i < k; ++i) {
				for (size_t j = 0; j < n; ++j) {
					ensemble[i][j] += modes[i][j];
				}
			}
			
			delete [] y2;
			free_arrays(modes, k);
		
		}
		
		// scale the ensemble values
		for (size_t i = 0; i < k; ++i) {
			for (size_t j = 0; j < n; ++j) {
				ensemble[i][j] /= ne;
			}
		}
		
		return ensemble;
	}
	
}

#endif