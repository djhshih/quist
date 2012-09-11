#ifndef _tridiag_thomas_hpp_
#define _tridiag_thomas_hpp_

#include "kernel/thomas_kernel.hpp"

template <typename real_t>
void thomas(
	size_t N,
	size_t numSystems,
	real_t *d_sub, 
	real_t *d_main, 
	real_t *d_sup, 
	real_t *d_r, 
	real_t *d_x
) {

	// TODO parallelize thomas algorithm for each system
	tridiag <<< 1, 1 >>> (N, d_sub, d_main, d_sup, d_r, d_x);
	
	CUDA_CHECK_ERROR("thomas");
}

#endif