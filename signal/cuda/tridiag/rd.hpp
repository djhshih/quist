#ifndef _tridiag_rd_hpp_
#define _tridiag_rd_hpp_

#include "kernel/rd_kernel.hpp"
#include "../scan/inclusive_scan.hpp"
#include "../scan/functor/static_matrix_functor.hpp"

// FIXME RD appears to be numerically unstable when using float...
// FIXME As N increases (> 128), even using double does not prevent numerical instability
//       Using Kahan summation inside matrix multiplier appears to have improved numerical instability somewhat


template <typename real_t>
void rd(
	size_t N,
	real_t *d_sub, 
	real_t *d_main, 
	real_t *d_sup, 
	real_t *d_r, 
	real_t *d_x
) {

	real_t *d_B, *d_C;
	
	const size_t B_dim = 3;
	const size_t B_nelem = B_dim*B_dim;
	const size_t B_size = N * B_nelem;
	
	const size_t nbytes_B = B_size * sizeof(real_t);
	
	cudaMalloc((void**) &d_B, nbytes_B);
	cudaMalloc((void**) &d_C, nbytes_B);
	
	// clear device arrays
	// TODO Find out why it appears that it is critical to clear d_B before using it...
	cudaMemset(d_B, 0, nbytes_B);
	
	StaticMatrixMultiplierPrefixStable<real_t, B_dim> multiplier;
	StaticMatrixSetter<real_t, B_dim> setter;

	// TODO expose these parameters
	dim3 block_dim = 4;
	dim3 grid_dim = N / block_dim.x + (N % block_dim.x == 0 ? 0 : 1);
	
	// set up B matrices
	rd_prefix <<< grid_dim, block_dim >>> (N, d_sub, d_main, d_sup, d_r, d_B);
	
	// calculator C matrices
	inclusive_scan<B_nelem>(N, d_B, d_C, multiplier, setter);
	
	// solve the tridiag
	rd_tridiag <<< grid_dim, block_dim >>> (N, d_C, d_x);
	
	cudaFree(d_B);
	cudaFree(d_C);
	
	CUDA_CHECK_ERROR("rd");
}

#endif