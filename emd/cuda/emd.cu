
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
	const size_t nlayers = 4;
	// note: when the number of layers is too high (i.e. misses high frequency signal), high frequency IMF 'spill' into subsequent IMF
	// sampling frequency = 1 / nlayers    must be no more than 1/10 of the frequency of the highest frequency IMF
	// Signal frequency does not depend on data size   =>   cannot use more layers for more data!
	
	//size_t k = log2((float)N) + 1;
	size_t k = 4;
	
	dim3 block_dim = 1;
	dim3 grid_dim = nlayers;
	
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
		float x = (float)i*2*M_PI;
		h_x[i] = x;
		h_y[i] = sin(x/40) + 0.8*sin(x/200) + 0.6*sin(x/2000);
	}
	
	// clear device output arrays
	cudaMemset(d_modes, 0, nbytes_modes);
	
	cudaThreadSynchronize();
	
	// copy data to device
	cudaMemcpy(d_x, h_x, nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, nbytes, cudaMemcpyHostToDevice);

	// do calculate on device
	emd_interl <<< grid_dim, block_dim >>> (nlayers, N, d_x, d_y, k, d_modes);

	// retrieve results from device and store it in host array
	cudaMemcpy(h_modes, d_modes, nbytes_modes, cudaMemcpyDeviceToHost);
	
	/*
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
	
	emd::free_arrays(gold_modes, k);
	*/

	// clean up
	free(h_x);
	free(h_y);
	free(h_modes);
	
	
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_modes);
}
