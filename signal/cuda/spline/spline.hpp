#ifndef _spline_spline_hpp_
#define _spline_spline_hpp_

#include <cstddef>

#include "kernel/spline_kernel.hpp"
#include "../tridiag/rd.hpp"
#include "../tridiag/thomas.hpp"
#include "../cudpp/crpcr.hpp"

template <typename real_t>
void splint(
	size_t N,
	real_t *d_x, 
	real_t *d_y, 
	size_t nn,
	real_t *d_xx, 
	real_t *d_yy
) {
	
	real_t *d_sub, *d_main, *d_sup, *d_r, *d_b, *d_c, *d_d;
	
	size_t nbytes = N * sizeof(real_t);
	
	// allocate array on device
	cudaMalloc((void**) &d_sub, nbytes);
	cudaMalloc((void**) &d_main, nbytes);
	cudaMalloc((void**) &d_sup, nbytes);
	cudaMalloc((void**) &d_r, nbytes);
	cudaMalloc((void**) &d_b, nbytes);
	cudaMalloc((void**) &d_c, nbytes);
	cudaMalloc((void**) &d_d, nbytes);
	
	dim3 block_dim = 4;
	dim3 grid_dim = N / block_dim.x + (N % block_dim.x == 0 ? 0 : 1);

	ncspline_setup <<< grid_dim, block_dim >>> (N, d_x, d_y, d_sub, d_main, d_sup, d_r);
	
	crpcr(N, 1, d_sub, d_main, d_sup, d_r, d_c);
	
	ncspline_teardown <<< grid_dim, block_dim >>> (N, d_x, d_y, d_c, d_b, d_d);
	splint_binary_search <<< grid_dim, block_dim >>> (N, d_x, d_y, d_b, d_c, d_d, nn, d_xx, d_yy);

	cudaFree(d_sub);
	cudaFree(d_main);
	cudaFree(d_sup);
	cudaFree(d_r);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaFree(d_d);
	
	CUDA_CHECK_ERROR("splint");
}

/**
 * Spline interpolation using Recursive Doubling for tridiagonal matrix solver.
 */
template <typename real_t>
void splint_rd_tridag(
	size_t N,
	real_t *d_x, 
	real_t *d_y, 
	size_t nn,
	real_t *d_xx, 
	real_t *d_yy
) {
	
	real_t *d_sub, *d_main, *d_sup, *d_r, *d_b, *d_c, *d_d;
	
	size_t nbytes = N * sizeof(real_t);
	
	// allocate array on device
	cudaMalloc((void**) &d_sub, nbytes);
	cudaMalloc((void**) &d_main, nbytes);
	cudaMalloc((void**) &d_sup, nbytes);
	cudaMalloc((void**) &d_r, nbytes);
	cudaMalloc((void**) &d_b, nbytes);
	cudaMalloc((void**) &d_c, nbytes);
	cudaMalloc((void**) &d_d, nbytes);
	
	// zero the first element of the output (special case for natural cubic spline)
	// need to zero the last element too
	// therefore, might as well zero everything
	cudaMemset(d_c, 0, nbytes);
	
	dim3 block_dim = 4;
	dim3 grid_dim = N / block_dim.x + (N % block_dim.x == 0 ? 0 : 1);

	ncspline_setup <<< grid_dim, block_dim >>> (N, d_x, d_y, d_sub, d_main, d_sup, d_r);
	
	// offset by 1 to skip the first element
	// Assumption for RD: d_sup[i] != 0 for all 0 <= i < n
	// But d_sup[0] = 0 for natural cubic spline. Therefore, skip it.
	// Further reduce N to N-2 to skip the last element as well.
	// prerequisite: d_c[0] and d_c[n-1] are already set to 0
	rd(N-2, d_sub+1, d_main+1, d_sup+1, d_r+1, d_c+1);
	
	ncspline_teardown <<< grid_dim, block_dim >>> (N, d_x, d_y, d_c, d_b, d_d);
	splint_binary_search <<< grid_dim, block_dim >>> (N, d_x, d_y, d_b, d_c, d_d, nn, d_xx, d_yy);

	cudaFree(d_sub);
	cudaFree(d_main);
	cudaFree(d_sup);
	cudaFree(d_r);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaFree(d_d);
	
	CUDA_CHECK_ERROR("splint_rd_tridag");
}

/**
 * Spline interpolation using Thomas algorithm (not parallelizable for single systems) for tridiagonal matrix solver.
 */
template <typename real_t>
void splint_thomas_tridiag(
	size_t N,
	real_t *d_x, 
	real_t *d_y, 
	size_t nn,
	real_t *d_xx, 
	real_t *d_yy
) {
	
	real_t *d_sub, *d_main, *d_sup, *d_r, *d_b, *d_c, *d_d;
	
	size_t nbytes = N * sizeof(real_t);
	
	// allocate array on device
	cudaMalloc((void**) &d_sub, nbytes);
	cudaMalloc((void**) &d_main, nbytes);
	cudaMalloc((void**) &d_sup, nbytes);
	cudaMalloc((void**) &d_r, nbytes);
	cudaMalloc((void**) &d_b, nbytes);
	cudaMalloc((void**) &d_c, nbytes);
	cudaMalloc((void**) &d_d, nbytes);
	
	dim3 block_dim = 4;
	dim3 grid_dim = N / block_dim.x + (N % block_dim.x == 0 ? 0 : 1);

	ncspline_setup <<< grid_dim, block_dim >>> (N, d_x, d_y, d_sub, d_main, d_sup, d_r);
	
	thomas(N, 1, d_sub, d_main, d_sup, d_r, d_c);
	
	ncspline_teardown <<< grid_dim, block_dim >>> (N, d_x, d_y, d_c, d_b, d_d);
	splint_binary_search <<< grid_dim, block_dim >>> (N, d_x, d_y, d_b, d_c, d_d, nn, d_xx, d_yy);

	cudaFree(d_sub);
	cudaFree(d_main);
	cudaFree(d_sup);
	cudaFree(d_r);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaFree(d_d);
	
	CUDA_CHECK_ERROR("splint_thomas_tridiag");
}

/**
 * Spline interpolation using Thomas algorithm (not parallelizable for single systems) for tridiagonal matrix solver.
 * Use linear search during interpolation.
 */
template <typename real_t>
void splint_thomas_tridiag_linear_search(
	size_t N,
	real_t *d_x, 
	real_t *d_y, 
	size_t nn,
	real_t *d_xx, 
	real_t *d_yy
) {
	
	real_t *d_sub, *d_main, *d_sup, *d_r, *d_b, *d_c, *d_d;
	
	size_t nbytes = N * sizeof(real_t);
	
	// allocate array on device
	cudaMalloc((void**) &d_sub, nbytes);
	cudaMalloc((void**) &d_main, nbytes);
	cudaMalloc((void**) &d_sup, nbytes);
	cudaMalloc((void**) &d_r, nbytes);
	cudaMalloc((void**) &d_b, nbytes);
	cudaMalloc((void**) &d_c, nbytes);
	cudaMalloc((void**) &d_d, nbytes);
	
	dim3 block_dim = 4;
	dim3 grid_dim = N / block_dim.x + (N % block_dim.x == 0 ? 0 : 1);

	ncspline_setup <<< grid_dim, block_dim >>> (N, d_x, d_y, d_sub, d_main, d_sup, d_r);
	
	thomas(N, 1, d_sub, d_main, d_sup, d_r, d_c);
	
	ncspline_teardown <<< grid_dim, block_dim >>> (N, d_x, d_y, d_c, d_b, d_d);
	splint_linear_search <<< grid_dim, block_dim >>> (N, d_x, d_y, d_b, d_c, d_d, nn, d_xx, d_yy);

	cudaFree(d_sub);
	cudaFree(d_main);
	cudaFree(d_sup);
	cudaFree(d_r);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaFree(d_d);
	
	CUDA_CHECK_ERROR("splint_thomas_tridiag_linear_search");
}

/**
 * Spline interpolation using Thomas algorithm (not parallelizable for single systems) for tridiagonal matrix solver.
 * Using single threaded search during interpolation.
 */
template <typename real_t>
void splint_thomas_tridiag_single_search(
	size_t N,
	real_t *d_x, 
	real_t *d_y, 
	size_t nn,
	real_t *d_xx, 
	real_t *d_yy
) {
	
	real_t *d_sub, *d_main, *d_sup, *d_r, *d_b, *d_c, *d_d;
	
	size_t nbytes = N * sizeof(real_t);
	
	// allocate array on device
	cudaMalloc((void**) &d_sub, nbytes);
	cudaMalloc((void**) &d_main, nbytes);
	cudaMalloc((void**) &d_sup, nbytes);
	cudaMalloc((void**) &d_r, nbytes);
	cudaMalloc((void**) &d_b, nbytes);
	cudaMalloc((void**) &d_c, nbytes);
	cudaMalloc((void**) &d_d, nbytes);
	
	dim3 block_dim = 4;
	dim3 grid_dim = N / block_dim.x + (N % block_dim.x == 0 ? 0 : 1);

	ncspline_setup <<< grid_dim, block_dim >>> (N, d_x, d_y, d_sub, d_main, d_sup, d_r);
	
	thomas(N, 1, d_sub, d_main, d_sup, d_r, d_c);
	
	ncspline_teardown <<< grid_dim, block_dim >>> (N, d_x, d_y, d_c, d_b, d_d);
	splint_single <<< 1, 1 >>> (N, d_x, d_y, d_b, d_c, d_d, nn, d_xx, d_yy);

	cudaFree(d_sub);
	cudaFree(d_main);
	cudaFree(d_sup);
	cudaFree(d_r);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaFree(d_d);
	
	CUDA_CHECK_ERROR("splint_thomas_tridiag_single_search");
}

#endif