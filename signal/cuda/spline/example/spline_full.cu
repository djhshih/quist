
// hack to make nvcc work with gcc-4.7
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include <cstdio>
#include <cmath>
#include <ctime>

#include <numeric/bla.hpp>

#include "../kernel/spline_kernel.hpp"
#include "../../tridiag/kernel/rd_kernel.hpp"
#include "../../scan/kernel/inclusive_scan_kernel.hpp"
#include "../../scan/functor/static_matrix_functor.hpp"
#include "../../cudpp/crpcr.hpp"

using namespace std;

typedef double real_t;


int main(int argc, char* argv[]) {
	
	std::srand( (unsigned)time(NULL) );
	
	real_t *h_x, *h_y, *h_xx, *h_yy;
	real_t *h_c, *h_c2;
	real_t *h_sub, *h_main, *h_sup;
	real_t *h_B, *h_C;
	
	real_t *d_x, *d_y, *d_xx, *d_yy;
	real_t *d_sub, *d_main, *d_sup, *d_r, *d_b, *d_c, *d_d;
	real_t *d_B, *d_C;
	real_t *d_c2;
	
	const size_t stride = 32;
	const size_t N = 32, nn = (N-1) * stride;
	
	const size_t B_dim = 3;
	const size_t B_nelem = B_dim*B_dim;
	const size_t B_size = N * B_nelem;
	
	
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
	
	h_sub = (real_t*)malloc(nbytes);
	h_main = (real_t*)malloc(nbytes);
	h_sup = (real_t*)malloc(nbytes);
	
	h_B = (real_t*)malloc(nbytes_B);
	h_C = (real_t*)malloc(nbytes_B);
	
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
	//cudaMemset(d_yy, 0, nbytes_nn);
	
	// zero the first element of the output (special case for natural cubic spline)
	// need to zero the last element too
	// therefore, might as well zero everything
	cudaMemset(d_c, 0, nbytes);
	cudaMemset(d_c2, 0, nbytes);
	cudaMemset(d_C, 0, nbytes_B);
	cudaMemset(d_B, 0, nbytes_B);
	cudaMemset(d_yy, 0, nbytes_nn);
	
	// copy data to device
	cudaMemcpy(d_x, h_x, nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_xx, h_xx, nbytes_nn, cudaMemcpyHostToDevice);
	
	StaticMatrixMultiplierPrefixStable<real_t, B_dim> multiplier;
	StaticMatrixSetter<real_t, B_dim> setter;
	
	dim3 block_dim = 4;
	dim3 grid_dim = N / block_dim.x + (N % block_dim.x == 0 ? 0 : 1);

	// do calculation on device
	ncspline_setup <<< grid_dim, block_dim >>> (N, d_x, d_y, d_sub, d_main, d_sup, d_r);
	
	// offset by 1 to skip the first element
	// Assumption for RD: d_sup[i] != 0 for all 0 <= i < n
	// But d_sup[0] = 0 for natural cubic spline. Therefore, skip it.
	// Further reduce N to N-2 to skip the last element as well.
	rd_prefix <<< grid_dim, block_dim >>> (N-2, d_sub+1, d_main+1, d_sup+1, d_r+1, d_B);
	size_t N2 = N - 2;
	
	size_t elemPerBlock = 8;
	dim3 block_dim2 = elemPerBlock / 2;
	dim3 grid_dim2 = (N2 / elemPerBlock) + (N2 % elemPerBlock == 0 ? 0 : 1);
	
	// NB prescan needs N to be a power of 2

	prescan<B_nelem> <<< grid_dim2, block_dim2, elemPerBlock*sizeof(real_t) * B_nelem >>> (elemPerBlock, d_B, d_C, multiplier, setter);
	
	
	if (grid_dim2.x > 1) {
		
		real_t *d_block_B, *d_block_C;
		
		size_t nbytes_block = grid_dim2.x * sizeof(real_t) * B_nelem;
		cudaMalloc((void**) &d_block_B, nbytes_block);
		cudaMalloc((void**) &d_block_C, nbytes_block);
	
		// one block; each thread processes a scan block from above
		aggregate_block_sum<B_nelem> <<< 1, grid_dim2 >>> (elemPerBlock, d_C, d_block_B, setter);
		
		// one block; each thread processes two scan block sums (hence need half the number of scan blocks from previous run)
		size_t grid_dim3 = (grid_dim2.x / 2) + (grid_dim2.x % 2 == 0 ? 0 : 1);
		prescan<B_nelem> <<< 1, grid_dim3, grid_dim2.x*sizeof(real_t) * B_nelem >>> (grid_dim2.x, d_block_B, d_block_C, multiplier, setter);
		
		// each thread processes one element in original data
		// need twice as many blocks as before, since each thread now processes one element
		add_block_cumsum<B_nelem> <<< grid_dim2.x*2, block_dim >>> (N2, d_block_C, d_C, multiplier, setter);
		
		cudaFree(d_block_B);
		cudaFree(d_block_C);
	}
	
	// offset first element of output, since row i of in tridiagonal matrix was was skipped.
	// same for last element
	// prerequisite: d_c[0] and d_c[n-1] are already set to 0
	//rd_tridiag <<< grid_dim, block_dim >>> (N2, d_C, d_c+1);
	
	crpcr(N, 1, d_sub, d_main, d_sup, d_r, d_c);
	
	
	ncspline_teardown <<< grid_dim, block_dim >>> (N, d_x, d_y, d_c, d_b, d_d);
	//splint_single <<< 1, 1 >>> (N, d_x, d_y, d_b, d_c, d_d, nn, d_xx, d_yy);
	//splint_linear_search <<< grid_dim, block_dim >>> (N, d_x, d_y, d_b, d_c, d_d, nn, d_xx, d_yy);
	splint_binary_search <<< grid_dim, block_dim >>> (N, d_x, d_y, d_b, d_c, d_d, nn, d_xx, d_yy);

	// retrieve results from device and store it in host array
	cudaMemcpy(h_yy, d_yy, nbytes_nn, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_c, d_c, nbytes, cudaMemcpyDeviceToHost);
	
	cudaMemcpy(h_sub, d_sub, nbytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_main, d_main, nbytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_sup, d_sup, nbytes, cudaMemcpyDeviceToHost);
	
	cudaMemcpy(h_B, d_B, nbytes_B, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_C, d_C, nbytes_B, cudaMemcpyDeviceToHost);
	
	// compute gold standard
	
	tridiag <<< 1, 1 >>> (N, d_sub, d_main, d_sup, d_r, d_c2);
	cudaMemcpy(h_c2, d_c2, nbytes, cudaMemcpyDeviceToHost);
	
	real_t* h_yy_gold = new real_t[nn];
	bla::splint(N, h_x, h_y, nn, h_xx, h_yy_gold);
	
	for (size_t i = 0; i < N; ++i) {
		printf("%lu %f %f %f\n", i, h_sub[i], h_main[i], h_sup[i]);
	}
	
	for (size_t i = 0; i < N; ++i) {
		printf("%lu ", i);
		for (size_t j = 0; j < B_nelem; ++j) {
			printf("%f ", h_B[i*B_nelem + j]);
		}
		printf("\n");
	}
	
	for (size_t i = 0; i < N; ++i) {
		printf("%lu ", i);
		for (size_t j = 0; j < B_nelem; ++j) {
			printf("%f ", h_C[i*B_nelem + j]);
		}
		printf("\n");
	}
	
	// print results
	for (size_t i = 0; i < N; ++i) {
		printf("%lu %f %f\n", i, h_c[i], h_c2[i]);
	}
	
	// print results
	for (size_t i = 0; i < nn; ++i) {
		printf("%lu %f %f\n", i, h_yy[i], h_yy_gold[i]);
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
	free(h_sup);
	
	
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
