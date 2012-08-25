
// hack to make nvcc work with gcc-4.7
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include <cstdio>
#include <cmath>
#include <ctime>

#include "tridiag_rd_kernel.hpp"
#include "scan_kernel.hpp"
#include "spline_kernel.hpp"
#include "../../bla/bla.hpp"

using namespace std;

typedef float real_t;


int main(int argc, char* argv[]) {
	
	//std::srand( (unsigned)time(NULL) );
	
	real_t *h_sub, *h_main, *h_sup, *h_r, *h_x;
	
	real_t *d_sub, *d_main, *d_sup, *d_r, *d_x;
	real_t *d_B, *d_C;
	
	const size_t N = 32;
	const size_t B_dim = 3;
	const size_t B_nelem = B_dim*B_dim;
	const size_t B_size = N * B_nelem;
	
	size_t nbytes = N * sizeof(real_t);
	size_t nbytes_B = B_size * sizeof(real_t);
	
	// allocate array on host
	h_sub = (real_t*)malloc(nbytes);
	h_main = (real_t*)malloc(nbytes);
	h_sup = (real_t*)malloc(nbytes);
	h_r = (real_t*)malloc(nbytes);
	h_x = (real_t*)malloc(nbytes);
	
	// allocate array on device
	cudaMalloc((void**) &d_sub, nbytes);
	cudaMalloc((void**) &d_main, nbytes);
	cudaMalloc((void**) &d_sup, nbytes);
	cudaMalloc((void**) &d_r, nbytes);
	cudaMalloc((void**) &d_x, nbytes);
	
	cudaMalloc((void**) &d_B, nbytes_B);
	cudaMalloc((void**) &d_C, nbytes_B);
	
	// initialize host array
	for (size_t i = 0; i < N; ++i) {
		h_sub[i] = std::rand() / 3;
		h_main[i] = std::rand() / 3;
		h_sup[i] = std::rand() / 3;
		h_r[i] = std::rand() / 3;
	}
	
	// clear device arrays
	cudaMemset(d_B, 0, nbytes_B);
	
	// copy data to device
	cudaMemcpy(d_sub, h_sub, nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_main, h_main, nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_sup, h_sup, nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_r, h_r, nbytes, cudaMemcpyHostToDevice);
	
	
	StaticMatrixMultiplierPrefix<real_t, B_dim> multiplier;
	StaticMatrixSetter<real_t, B_dim> setter;

	// do calculation on device
	dim3 block_dim = 4;
	dim3 grid_dim = N / block_dim.x + (N % block_dim.x == 0 ? 0 : 1);
	
	rd_prefix <<< grid_dim, block_dim >>> (N, d_sub, d_main, d_sup, d_r, d_B);
	
	size_t elemPerBlock = 32;
	block_dim = elemPerBlock / 2;
	grid_dim = N / elemPerBlock + (N % elemPerBlock == 0 ? 0 : 1);
	
	prescan<B_nelem> <<< grid_dim, block_dim, elemPerBlock*sizeof(real_t) * B_nelem >>> (elemPerBlock, d_B, d_C, multiplier, setter);
	// TODO  add block sum scan and post-processing
	
	block_dim = 4;
	grid_dim = N / block_dim.x + (N % block_dim.x == 0 ? 0 : 1);
	
	rd_tridiag <<< grid_dim, block_dim >>> (N, d_C, d_x);
	
	// retrieve results from device and store it in host array
	cudaMemcpy(h_x, d_x, nbytes, cudaMemcpyDeviceToHost);
	
	// compute gold standard
	real_t* h_x_gold = new real_t[N];
	bla::tridiag(N, h_sub, h_main, h_sup, h_r, h_x_gold);
	
	// print results
	for (size_t i = 0; i < N; ++i) {
		printf("%d %f %f\n", i, h_x[i], h_x_gold[i]);
	}
	
	cudaError_t code = cudaGetLastError();
	if (code != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(code));
		printf("grid_dim = %d, block_dim = %d\n", grid_dim.x, block_dim.x);
	}
	
	// clean up
	free(h_sub);
	free(h_main);
	free(h_sup);
	free(h_r);
	free(h_x);
	delete [] h_x_gold;
	
	cudaFree(d_sub);
	cudaFree(d_main);
	cudaFree(d_sup);
	cudaFree(d_r);
	cudaFree(d_x);
	cudaFree(d_B);
	cudaFree(d_C);
	
}
