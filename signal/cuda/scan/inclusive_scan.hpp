#ifndef _scan_inclusive_scan_hpp_
#define _scan_inclusive_scan_hpp_

#include <cstddef>

#include <numeric/math.hpp>

#include "../util.hpp"
#include "kernel/inclusive_scan_kernel.hpp"


/**
 * Inclusive scan for functors operating on multiple atomic values per element.
 * 
 * Assumes N < 2^22
 * Since O(N) shared memory is used, the application will likely be memory-bound
 * FIXME To allow bigger, need to iteratively aggregate block sums
 * 
 * FIXME Assumes N >= 2*elemPerBlock ?
 * FIXME Make elemPerBlock tunable to overcome this constraint
 * 
 * @param m number of atomic values per element
 * @param N number of elements
 */
template <size_t m, typename real_t, typename BinaryOp, typename Setter>
void inclusive_scan(
	size_t _N,
	real_t *d_x, 
	real_t *d_y,
	BinaryOp binaryOp,
	Setter copy
) {
	
	real_t *d_block_x, *d_block_y;
	
	const size_t N = math::pow2ceil(_N);
	
	// maximum elements per block: 1024 * 2 (since each thread processes two elements)
	// TODO allow tweaking of this parameter
	const size_t elemPerBlock = 64;
	//const size_t elemPerBlock = 8;
	
	// each thread processes two elements
	dim3 block_dim = elemPerBlock / 2;
	
	// number of blocks
	dim3 grid_dim = N / elemPerBlock + (N % elemPerBlock == 0 ? 0 : 1);
	
	size_t nbytes_block = grid_dim.x * sizeof(real_t) * m;
	
	cudaMalloc((void**) &d_block_x, nbytes_block);
	cudaMalloc((void**) &d_block_y, nbytes_block);

	// elements are divided into blocks
	// each thread processes two elements within a block
	prescan<m> <<< grid_dim, block_dim, elemPerBlock*sizeof(real_t) * m >>> (elemPerBlock, d_x, d_y, binaryOp, copy);
	
	// one block; each thread processes a scan block from above
	aggregate_block_sum<m> <<< 1, grid_dim >>> (elemPerBlock, d_y, d_block_x, copy);
	
	// one block; each thread processes two scan block sums (hence need half the number of scan blocks from previous run)
	prescan<m> <<< 1, grid_dim.x/2, grid_dim.x*sizeof(real_t) * m >>> (grid_dim.x, d_block_x, d_block_y, binaryOp, copy);
	
	// each thread processes one element in original data
	// need twice as many blocks as before, since each thread now processes one element
	add_block_cumsum<m> <<< grid_dim.x*2, block_dim >>> (N, d_block_y, d_y, binaryOp, copy);
	
	cudaFree(d_block_x);
	cudaFree(d_block_y);

	CUDA_CHECK_ERROR("inclusive_scan");
	
}

#endif