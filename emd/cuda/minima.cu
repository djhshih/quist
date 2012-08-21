#include <cstdio>
#include <cmath>

#include "emd_kernel.hpp"

using namespace std;


int main(int argc, char* argv[]) {

	float *h_x, *h_y, *h_xx, *h_yy, *d_x, *d_y, *d_xx, *d_yy;
	size_t *h_kk, *d_kk;
	
	const size_t N = 512 * 32;
	const size_t wsize = 32;
	
	dim3 block_dim = 4;
	
	size_t nW = N / wsize;
	
	dim3 grid_dim = N / wsize / block_dim.x + (N%block_dim.x == 0 ? 0 : 1);
	
	size_t nbytes = N * sizeof(float);
	size_t nbytes_w = nW * sizeof(size_t);
	
	size_t k = log2((float)N) + 1;

	// allocate array on host
	h_x = (float*)malloc(nbytes);
	h_y = (float*)malloc(nbytes);
	h_xx = (float*)malloc(nbytes);
	h_yy = (float*)malloc(nbytes);
	h_kk = (size_t*)malloc(nbytes_w);

	// allocate array on device
	cudaMalloc((void**) &d_x, nbytes);
	cudaMalloc((void**) &d_y, nbytes);
	cudaMalloc((void**) &d_xx, nbytes);
	cudaMalloc((void**) &d_yy, nbytes);
	cudaMalloc((void**) &d_kk, nbytes_w);

	// initialize host array
	for (size_t i = 0; i < N; i++) {
		h_x[i] = (float)i;
		h_y[i] = sin(i/M_PI);
	}
	
	// clear device output arrays
	cudaMemset(d_xx, 0, nbytes);
	cudaMemset(d_yy, 0, nbytes);
	
	cudaThreadSynchronize();
	
	// copy data to device
	cudaMemcpy(d_x, h_x, nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, nbytes, cudaMemcpyHostToDevice);

	// do calculate on device;
	minima_all <<< grid_dim, block_dim >>> (wsize, N, d_x, d_y, d_kk, d_xx, d_yy);

	// retrieve results from device and store it in host array
	cudaMemcpy(h_xx, d_xx, sizeof(float)*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_yy, d_yy, sizeof(float)*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_kk, d_kk, nbytes_w, cudaMemcpyDeviceToHost);

	// print results
	for (size_t i = 0; i < N; i++) {
		printf("%d %f.0 %f\n", i, h_xx[i], h_yy[i]);
	}
	
	for (size_t i = 0; i < nW; i++) {
		printf("%d %d\n", i, h_kk[i]);
	}
	
	cudaError_t code = cudaGetLastError();
	if (code != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(code));
		printf("grid_dim = %d, block_dim = %d\n", grid_dim.x, block_dim.x);
	}

	// clean up
	free(h_x);
	free(h_y);
	free(h_xx);
	free(h_yy);
	free(h_kk);
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_xx);
	cudaFree(d_yy);
	cudaFree(d_kk);
}
