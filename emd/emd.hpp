#ifndef quist_emd_h
#define quist_emd_h

#include <cstdlib>
#include <ctime>
#include <cmath>

#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>

#include <bla/bla.hpp>

// Empirical Mode Decomposition

namespace emd {
	
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
	
	unsigned int log2 (unsigned int val) {
		unsigned int ret = 0;
		while (val != 0) {
			val >>= 1;
			++ret;
		}
		return ret;
	}
	
	template <typename T>
	void summary(const T a[], size_t n, T* mean, T* sd) {
		T m = 0, devsq = 0, t, x;
		for (size_t j = 0; j < n; ++j) {
			x = a[j];	
			t = (x - m);
			m += (x - m) / (j+1);
			devsq += t * (x - m);
		}
		
		if (mean != NULL) *mean = m;
		if (sd != NULL) *sd = std::sqrt( devsq/n );
	}
	
	/**
	 * Local minima.
	 * Find all local minima in data.
	 * @param n number of data points (n >= 4)
	 * @param x x values
	 * @param y y values
	 * @param kk output number of minima
	 * @param xx ouptut x values of minimum points
	 * @param yy output y values of minimum points
	 */
	template <typename T, typename U>
	void minima(size_t n, const T x[], const U y[], size_t* kk, T xx[], U yy[]) {
		
		// tentatively designate first point as minimum
		xx[0] = x[0];
		yy[0] = y[0];
		
		// identify minima
		size_t k = 1;
		size_t i;
		for (i = 1; i < n-1; ++i) {
			// y_i is less than adjacent values: y_i is a minimum
			if (y[i] <= y[i-1] && y[i] <= y[i+1]) {
				xx[k] = x[i];
				yy[k] = y[i];
				++k;
			}
		}
		
		// tentatively designate last point as minimum
		xx[k] = x[i];
		yy[k] = y[i];
		
		// consider replacing end points with extrapolants
		if (k+1 >= 4) {
			
			U slope, t;
			
			slope = (yy[2] - yy[1]) / (xx[2] - xx[1]);
			t = yy[1] - slope * (xx[1] - xx[0]);
			if (t < yy[0]) yy[0] = t;
			
			slope = (yy[k-1] - yy[k-2]) / (xx[k-1] - xx[k-2]);
			t = yy[k-1] + slope * (xx[k] - xx[k-1]);
			if (t < yy[k]) yy[k] = t;
			
		}
		
		*kk = k+1;
	}
	
	/**
	 * Local maxima.
	 * Find all local maxima in data.
	 * @param n number of data points (n >= 4)
	 * @param x x values
	 * @param y y values
	 * @param kk output number of maxima
	 * @param xx ouptut x values of maximum points
	 * @param yy output y values of maximum points
	 */
	template <typename T, typename U>
	void maxima(size_t n, const T x[], const U y[], size_t* kk, T xx[], U yy[]) {
		
		// tentatively designate first point as maximum
		xx[0] = x[0];
		yy[0] = y[0];
		
		// identify minima
		size_t k = 1;
		size_t i;
		for (i = 1; i < n-1; ++i) {
			// y_i is greater than adjacent values: y_i is a maximum
			if (y[i] >= y[i-1] && y[i] >= y[i+1]) {
				xx[k] = x[i];
				yy[k] = y[i];
				++k;
			}
		}
		
		// tentatively designate last point as maximum
		xx[k] = x[i];
		yy[k] = y[i];
		
		// consider replacing end points with extrapolants
		if (k+1 >= 4) {
			
			U slope, t;
			
			slope = (yy[2] - yy[1]) / (xx[2] - xx[1]);
			t = yy[1] - slope * (xx[1] - xx[0]);
			if (t > yy[0]) yy[0] = t;
			
			slope = (yy[k-1] - yy[k-2]) / (xx[k-1] - xx[k-2]);
			t = yy[k-1] + slope * (xx[k] - xx[k-1]);
			if (t > yy[k]) yy[k] = t;
			
		}
		
		*kk = k+1;
	}
	
	/**
	 * Empirical Mode Deomposition.
	 * @param n number of data points (n >= 4)
	 * @param x x values
	 * @param y y values
	 * @param kk expected number of instrinsic mode functions plus residual; 
	 *           if kk <= 1, kk is modified to an estimate
	 * @param max_iter max number of iterations per instrict mode function
	 * @return array of k instrinsic mode functions (array of size n, evaluated at x)
	 */
	template <typename T, typename U>
	T** emd(size_t n, const T x[], const U y[], size_t* kk, size_t max_iter=10) {
	
		size_t k = *kk;
		if (k <= 1) {
			// expected number of intrinsic mode functions is log2(n)
			// plus one to number of output arrays for the residual
			k = log2(n) + 1;
			// modify input parameter
			*kk = k;
		}
		
		// allocate memory
		U** modes = new U*[k];
		
		// allocate arrays for storing x and y values of extrema points, 
		//   and upper and lower envelops
		T* min_x = new T[n];
		T* max_x = new T[n];
		U* min_y = new U[n];
		U* max_y = new U[n];
		U* upper = new U[n];
		U* lower = new U[n];
		
		// copy data for computing running signal
		U* running = copied(y, n);
		U* current = new U[n];
		
		for (size_t i = 0; i < k-1; ++i) {
			copy(current, running, n);
			
			for (size_t j = 0; j < max_iter; ++j) {
				// find extrema
				size_t n_max, n_min;
				maxima(n, x, current, &n_max, max_x, max_y);
				minima(n, x, current, &n_min, min_x, min_y);
				
				// compute upper and lower envelops (parallelizable)
				bla::splint(n_max, max_x, max_y, n, x, upper);
				bla::splint(n_min, min_x, min_y, n, x, lower);
				
				// substract the mean
				for (size_t ii = 0; ii < n; ++ii) {
					current[ii] -= (upper[ii] + lower[ii]) / 2;
				}
			}
			
			// save current intrinsic mode
			modes[i] = copied(current, n);
			
			// update running signal
			for (size_t ii = 0; ii < n; ++ii) {
				running[ii] -= current[ii];
			}
		}
		
		// save the residual
		modes[k-1] = copied(running, n);
		
		
		delete [] current;
		delete [] running;
		
		delete [] min_x;
		delete [] max_x;
		delete [] min_y;
		delete [] max_y;
		delete [] upper;
		delete [] lower;
		
		return modes;
	}

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
			k = log2(n) + 1;
			// modify input parameter
			*kk = k;
		}
		
		// allocate memory
		U** ensemble = new_arrays(k, n, 0.0);
		
		// calculate SD of data
		U mean, sd;
		summary(y, n, &mean, &sd);
		
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
			k = log2(n) + 1;
			// modify input parameter
			*kk = k;
		}
		
		// allocate memory
		U** ensemble = new_arrays(k, n, 0.0);
		size_t* counts = new size_t[n];
		for (size_t j = 0; j < n; ++j) {
			counts[j] = 0;
		}
		
		for (size_t r = 0; r < nr; ++r) {
		
			// allocate for down-sampled data points
			size_t* idx = new size_t[ns];
			T* x2 = new U[ns];
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
	
};

#endif
