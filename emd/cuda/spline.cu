
// hack to make nvcc work with gcc-4.7
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include <cstdio>
#include <cmath>
#include <ctime>

#include "spline_kernel.hpp"
#include "tridiag_rd_kernel.hpp"
#include "scan_kernel.hpp"
#include "../../bla/bla.hpp"

using namespace std;

typedef float real_t;


int main(int argc, char* argv[]) {
	
	std::srand( (unsigned)time(NULL) );
	
	real_t *h_x, *h_y, *h_xx, *h_yy;
	real_t *h_c, *h_c2;
	
	real_t *d_x, *d_y, *d_xx, *d_yy;
	real_t *d_sub, *d_main, *d_sup, *d_r, *d_b, *d_c, *d_d;
	real_t *d_B, *d_C;
	real_t *d_c2;
	
	const size_t stride = 32;
	const size_t N = 32, nn = (N-1) * stride;
	
	const size_t B_dim = 3;
	const size_t B_nelem = B_dim*B_dim;
	const size_t B_size = N * B_nelem;
	
	dim3 block_dim = 4;
	dim3 grid_dim = N / block_dim.x + (N % block_dim.x == 0 ? 0 : 1);
	
	size_t nbytes = N * sizeof(real_t);
	size_t nbytes_nn = nn * sizeof(real_t);
	size_t nbytes_B = B_size * sizeof(real_t);
	
	// allocate array on host
	h_x = (real_t*)malloc(nbytes);
	h_y = (real_t*)malloc(nbytes);
	h_xx = (real_t*)malloc(nbytes_nn);
	h_yy = (real_t*)malloc(nbytes_nn);
	h_c = (real_t*)malloc(nbytes);
	h_c2 = (real_t*)malloc(nbytes);
	
	// allocate array on device
	cudaMalloc((void**) &d_x, nbytes);
	cudaMalloc((void**) &d_y, nbytes);
	cudaMalloc((void**) &d_sub, nbytes);
	cudaMalloc((void**) &d_main, nbytes);
	cudaMalloc((void**) &d_sup, nbytes);
	cudaMalloc((void**) &d_r, nbytes);
	cudaMalloc((void**) &d_b, nbytes);
	cudaMalloc((void**) &d_c, nbytes);
	cudaMalloc((void**) &d_d, nbytes);
	cudaMalloc((void**) &d_xx, nbytes_nn);
	cudaMalloc((void**) &d_yy, nbytes_nn);
	
	cudaMalloc((void**) &d_B, nbytes_B);
	cudaMalloc((void**) &d_C, nbytes_B);
	
	cudaMalloc((void**) &d_c2, nbytes);
	
	// initialize host array
	for (size_t i = 0; i < N; ++i) {
		h_x[i] = i * stride;
		h_y[i] = (real_t)std::rand() / RAND_MAX;
	}
	for (size_t i = 0; i < nn; ++i) {
		h_xx[i] = i;
	}
	
	// clear device output arrays
	//cudamemset(d_yy, 0, nbytes_nn);
	
	// copy data to device
	cudaMemcpy(d_x, h_x, nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_xx, h_xx, nbytes_nn, cudaMemcpyHostToDevice);
	
	StaticMatrixMultiplierPrefixStable<real_t, B_dim> multiplier;
	StaticMatrixSetter<real_t, B_dim> setter;

	// do calculation on device
	ncspline_setup <<< grid_dim, block_dim >>> (N, d_x, d_y, d_sub, d_main, d_sup, d_r);
	
	tridiag <<< 1, 1 >>> (N, d_sub, d_main, d_sup, d_r, d_c2);
	
	{
	rd_prefix <<< grid_dim, block_dim >>> (N, d_sub, d_main, d_sup, d_r, d_B);
	
	size_t elemPerBlock = 4;
	dim3 block_dim2 = elemPerBlock / 2;
	dim3 grid_dim2 = N / elemPerBlock + (N % elemPerBlock == 0 ? 0 : 1);

	prescan<B_nelem> <<< grid_dim2, block_dim2, elemPerBlock*sizeof(real_t) * B_nelem >>> (elemPerBlock, d_B, d_C, multiplier, setter);
	
	if (grid_dim2.x > 1) {
		
		real_t *d_block_B, *d_block_C;
		
		size_t nbytes_block = grid_dim2.x * sizeof(real_t) * B_nelem;
		cudaMalloc((void**) &d_block_B, nbytes_block);
		cudaMalloc((void**) &d_block_C, nbytes_block);
	
		// one block; each thread processes a scan block from above
		aggregate_block_sum<B_nelem> <<< 1, grid_dim2 >>> (elemPerBlock, d_C, d_block_B, setter);
		
		// one block; each thread processes two scan block sums (hence need half the number of scan blocks from previous run)
		prescan<B_nelem> <<< 1, grid_dim2.x/2, grid_dim2.x*sizeof(real_t) * B_nelem >>> (grid_dim2.x, d_block_B, d_block_C, multiplier, setter);
		
		// each thread processes one element in original data
		// need twice as many blocks as before, since each thread now processes one element
		add_block_cumsum<B_nelem> <<< grid_dim2.x*2, block_dim >>> (N, d_block_C, d_C, multiplier, setter);
		
		cudaFree(d_block_B);
		cudaFree(d_block_C);
	}
	
	rd_tridiag <<< grid_dim, block_dim >>> (N, d_C, d_c);
	}
	
	
	ncspline_teardown <<< grid_dim, block_dim >>> (N, d_x, d_y, d_c, d_b, d_d);
	//splint_single <<< 1, 1 >>> (N, d_x, d_y, d_b, d_c, d_d, nn, d_xx, d_yy);
	//splint_linear_search <<< grid_dim, block_dim >>> (N, d_x, d_y, d_b, d_c, d_d, nn, d_xx, d_yy);
	splint_binary_search <<< grid_dim, block_dim >>> (N, d_x, d_y, d_b, d_c, d_d, nn, d_xx, d_yy);

	// retrieve results from device and store it in host array
	cudaMemcpy(h_yy, d_yy, nbytes_nn, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_c, d_c, nbytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_c2, d_c2, nbytes, cudaMemcpyDeviceToHost);
	
	// compute gold standard
	real_t* h_yy_gold = new real_t[nn];
	bla::splint(N, h_x, h_y, nn, h_xx, h_yy_gold);
	
	// print results
	for (size_t i = 0; i < N; ++i) {
		printf("%d %f %f\n", i, h_c[i], h_c2[i]);
	}
	
	// print results
	for (size_t i = 0; i < nn; ++i) {
		printf("%d %f %f\n", i, h_yy[i], h_yy_gold[i]);
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
	
	free(h_c);
	free(h_c2);
	
	
	delete [] h_yy_gold;
	
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_sub);
	cudaFree(d_main);
	cudaFree(d_sup);
	cudaFree(d_r);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaFree(d_d);
	cudaFree(d_xx);
	cudaFree(d_yy);
	
	cudaFree(d_c2);
	
}
