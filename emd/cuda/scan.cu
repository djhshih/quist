
// hack to make nvcc work with gcc-4.7
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include <cstdio>

#include "scan_kernel.hpp"

using namespace std;

typedef float real_t;


int main(int argc, char* argv[]) {
	
	real_t *h_x, *h_y;
	//real_t *h_block_x, *h_block_y;
	real_t *d_x, *d_y, *d_block_x, *d_block_y;
	
	// assume N < 2^22
	// since O(N) shared memory is used, the application will likely be memory-bound
	
	const size_t N = 2048;
	
	size_t elemPerBlock = 128;
	
	dim3 block_dim = elemPerBlock / 2;
	dim3 grid_dim = N / elemPerBlock + (N % elemPerBlock == 0 ? 0 : 1);
	
	size_t nbytes = N * sizeof(real_t);
	size_t nbytes_block = grid_dim.x * sizeof(real_t);
	
	// allocate array on host
	h_x = (real_t*)malloc(nbytes);
	h_y = (real_t*)malloc(nbytes);
	//h_block_x = (real_t*)malloc(nbytes_block);
	//h_block_y = (real_t*)malloc(nbytes_block);
	
	
	// allocate array on device
	cudaMalloc((void**) &d_x, nbytes);
	cudaMalloc((void**) &d_y, nbytes);
	cudaMalloc((void**) &d_block_x, nbytes_block);
	cudaMalloc((void**) &d_block_y, nbytes_block);
	
	// initialize host array
	for (size_t i = 0; i < N; ++i) {
		h_x[i] = rand() % 10;
	}
	
	// copy data to device
	cudaMemcpy(d_x, h_x, nbytes, cudaMemcpyHostToDevice);

	// do calculation on device
	prescan <<< grid_dim, block_dim, elemPerBlock*sizeof(real_t) >>> (elemPerBlock, d_x, d_y);
	// each thread processes a scan block from above
	aggregate_block_sum <<< 1, grid_dim >>> (elemPerBlock, d_y, d_block_x);
	prescan <<< 1, grid_dim.x/2, grid_dim.x*sizeof(real_t) >>> (grid_dim.x, d_block_x, d_block_y);
	// need twice as many blocks as before, since each thread now processes one element
	add_block_cumsum <<< grid_dim.x*2, block_dim >>> (N, d_block_y, d_y);

	// retrieve results from device and store it in host array
	cudaMemcpy(h_y, d_y, nbytes, cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_block_x, d_block_x, nbytes_block, cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_block_y, d_block_y, nbytes_block, cudaMemcpyDeviceToHost);
	
	// compute gold standard
	real_t* h_gold = new real_t[N];
	h_gold[0] = h_x[0];
	for (size_t i = 1; i < N; ++i) {
		h_gold[i] = h_gold[i-1] + h_x[i];
	}
	
	// print results
	bool equal = true;
	for (size_t i = 0; i < N; ++i) {
		printf("%d %.0f %.0f %.0f", i, h_x[i], h_y[i], h_gold[i]);
		if (std::abs(h_y[i] - h_gold[i]) > 1e-5) {
			equal = false;
			printf("*\n");
		} else {
			printf("\n");
		}
	}
	if (!equal) printf("Differences detected!\n");
	
	/*
	for (size_t i = 0; i < grid_dim.x; ++i) {
		printf("%d %.0f %.0f\n", i, h_block_x[i], h_block_y[i]);
	}
	*/
	
	cudaError_t code = cudaGetLastError();
	if (code != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(code));
		printf("grid_dim = %d, block_dim = %d\n", grid_dim.x, block_dim.x);
	}
	
	// clean up
	free(h_x);
	free(h_y);
	//free(h_block_x);
	//free(h_block_y);
	
	delete [] h_gold;
	
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_block_x);
	cudaFree(d_block_y);
	
}
