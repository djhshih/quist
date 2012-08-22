
// hack to make nvcc work with gcc-4.7
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include <cstdio>
#include <cmath>

#include "emd_kernel.hpp"
#include "../emd.hpp"

using namespace std;


int main(int argc, char* argv[]) {

	float *h_x, *h_y, *h_modes;
	float *d_x, *d_y, *d_modes;
	
	const size_t N = 32 * 32;
	const size_t wsize = 32 * 32 / 4;
	// note: when the window size is too small (i.e. does not contain enough extrema), IMF cannot be extracted
	
	//size_t k = log2((float)N) + 1;
	size_t k = 4;
	
	dim3 block_dim = 1;
	dim3 grid_dim = N / wsize / block_dim.x + (N%block_dim.x == 0 ? 0 : 1);
	
	size_t nbytes = N * sizeof(float);
	size_t nbytes_modes = k * N * sizeof(float);
	
	// allocate array on host
	h_x = (float*)malloc(nbytes);
	h_y = (float*)malloc(nbytes);
	
	h_modes = (float*)malloc(nbytes_modes);

	// allocate array on device
	cudaMalloc((void**) &d_x, nbytes);
	cudaMalloc((void**) &d_y, nbytes);
	cudaMalloc((void**) &d_modes, nbytes_modes);

	// initialize host array
	for (size_t i = 0; i < N; i++) {
		float x = (float)i/M_PI;
		h_x[i] = x;
		h_y[i] = sin(x) + 0.5 * sin(x/10);
	}
	
	// clear device output arrays
	cudaMemset(d_modes, 0, nbytes_modes);
	
	cudaThreadSynchronize();
	
	// copy data to device
	cudaMemcpy(d_x, h_x, nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, nbytes, cudaMemcpyHostToDevice);

	// do calculate on device
	emd_strat <<< grid_dim, block_dim >>> (wsize, N, d_x, d_y, k, d_modes);

	// retrieve results from device and store it in host array
	cudaMemcpy(h_modes, d_modes, nbytes_modes, cudaMemcpyDeviceToHost);
	
	// compute gold standard
	float** gold_modes = emd::emd(N, h_x, h_y, &k);

	// print results
	for (size_t i = 0; i < k; ++i) {
		for (size_t j = 0; j < N; ++j) {
			printf("%d %d %f %f\n", i, j, h_modes[i*N + j], gold_modes[i][j]);
		}
	}
	
	cudaError_t code = cudaGetLastError();
	if (code != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(code));
		printf("grid_dim = %d, block_dim = %d\n", grid_dim.x, block_dim.x);
	}

	// clean up
	free(h_x);
	free(h_y);
	free(h_modes);
	
	emd::free_arrays(gold_modes, k);
	
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_modes);
}
