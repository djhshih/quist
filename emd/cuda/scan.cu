
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
	
	const size_t N = 64;
	const size_t m = 9;
	
	size_t elemPerBlock = 4;
	
	dim3 block_dim = elemPerBlock / 2;
	dim3 grid_dim = N / elemPerBlock + (N % elemPerBlock == 0 ? 0 : 1);
	
	size_t nbytes = N * sizeof(real_t) * m;
	size_t nbytes_block = grid_dim.x * sizeof(real_t) * m;
	
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
	for (size_t i = 0; i < N*m; ++i) {
		h_x[i] = (rand() % 3) - 1;
	}
	
	// copy data to device
	cudaMemcpy(d_x, h_x, nbytes, cudaMemcpyHostToDevice);
	
	//ScalarAdder<real_t> adder;
	//ScalarSetter<real_t> setter;
	//ArrayAdder<real_t> adder(m);
	//ArraySetter<real_t> setter(m);
	Matrix33Multipler<real_t> adder;
	Matrix33Setter<real_t> setter;
	//MatrixMultipler<real_t> adder(3);
	//MatrixSetter<real_t> setter(3);
	
	// do calculation on device
	
	// elements are divided into blocks
	// each thread processes two elements within a block
	prescan <<< grid_dim, block_dim, elemPerBlock*sizeof(real_t) * m >>> (elemPerBlock, d_x, d_y, adder, setter, m);
	
	// one block; each thread processes a scan block from above
	aggregate_block_sum <<< 1, grid_dim >>> (elemPerBlock, d_y, d_block_x, setter, m);
	
	// one block; each thread processes two scan block sums (hence need half the number of scan blocks from previous run)
	prescan <<< 1, grid_dim.x/2, grid_dim.x*sizeof(real_t) * m >>> (grid_dim.x, d_block_x, d_block_y, adder, setter, m);
	
	// each thread processes one element in original data
	// need twice as many blocks as before, since each thread now processes one element
	add_block_cumsum <<< grid_dim.x*2, block_dim >>> (N, d_block_y, d_y, adder, setter, m);

	// retrieve results from device and store it in host array
	cudaMemcpy(h_y, d_y, nbytes, cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_block_x, d_block_x, nbytes_block, cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_block_y, d_block_y, nbytes_block, cudaMemcpyDeviceToHost);
	
	// compute gold standard
	real_t* h_gold = new real_t[N*m];
	setter(h_gold[0], h_x[0]);
	for (size_t i = 1; i < N; ++i) {
		setter(h_gold[i*m], h_x[i*m]);
		adder(h_gold[i*m], h_gold[(i-1)*m]);
	}
	
	// print results
	bool equal = true;
	for (size_t i = 0; i < N*m; ++i) {
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
	for (size_t i = 0; i < grid_dim.x*m; ++i) {
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
