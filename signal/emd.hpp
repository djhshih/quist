#ifndef _signal_emd_hpp_
#define _signal_emd_hpp_

#include <numeric/bla.hpp>
#include <numeric/math.hpp>

#include "util.hpp"


// Empirical Mode Decomposition

namespace signal {
	
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
	template <typename U, typename T>
	T** emd(size_t n, const U x[], const T y[], size_t* kk, size_t max_iter=10) {
	
		size_t k = *kk;
		if (k <= 1) {
			// expected number of intrinsic mode functions is log2(n)
			// plus one to number of output arrays for the residual
			k = math::log2(n) + 1;
			// modify input parameter
			*kk = k;
		}
		
		// allocate memory
		T** modes = new T*[k];
		
		// allocate arrays for storing x and y values of extrema points, 
		//   and upper and lower envelops
		U* min_x = new U[n];
		U* max_x = new U[n];
		T* min_y = new T[n];
		T* max_y = new T[n];
		T* upper = new T[n];
		T* lower = new T[n];
		
		// copy data for computing running signal
		T* running = copied(y, n);
		T* current = new T[n];
		
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
	
}

#endif
