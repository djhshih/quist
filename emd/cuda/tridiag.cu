
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

// FIXME RD appears to be numerically unstable when using float...
// FIXME As N increases (> 128), even using double does not prevent numerical instability
//       Using Kahan summation inside matrix multiplier appears to have improved numerical instability somewhat
typedef float real_t;

int main(int argc, char* argv[]) {
	
	std::srand( (unsigned)time(NULL) );
	
	real_t *h_sub, *h_main, *h_sup, *h_r, *h_x, *h_x2;
	
	real_t *d_sub, *d_main, *d_sup, *d_r, *d_x, *d_x2;
	real_t *d_B, *d_C;
	
	const size_t N = 1024;
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
	h_x2 = (real_t*)malloc(nbytes);
	
	// allocate array on device
	cudaMalloc((void**) &d_sub, nbytes);
	cudaMalloc((void**) &d_main, nbytes);
	cudaMalloc((void**) &d_sup, nbytes);
	cudaMalloc((void**) &d_r, nbytes);
	cudaMalloc((void**) &d_x, nbytes);
	cudaMalloc((void**) &d_x2, nbytes);
	
	cudaMalloc((void**) &d_B, nbytes_B);
	cudaMalloc((void**) &d_C, nbytes_B);
	
	// initialize host array
	for (size_t i = 0; i < N; ++i) {
		h_sub[i] = (std::rand() % 5) + 5;
		h_main[i] = (std::rand() % 5) + 5;
		h_sup[i] = (std::rand() % 5) + 5;
		h_r[i] = (std::rand() % 5) + 5;
	}
	
	// clear device arrays
	cudaMemset(d_B, 0, nbytes_B);
	
	// copy data to device
	cudaMemcpy(d_sub, h_sub, nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_main, h_main, nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_sup, h_sup, nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_r, h_r, nbytes, cudaMemcpyHostToDevice);
	
	
	StaticMatrixMultiplierPrefixStable<real_t, B_dim> multiplier;
	//StaticMatrixMultiplierPrefix<real_t, B_dim> multiplier;
	StaticMatrixSetter<real_t, B_dim> setter;

	// do calculation on device
	dim3 block_dim = 4;
	dim3 grid_dim = N / block_dim.x + (N % block_dim.x == 0 ? 0 : 1);
	
	rd_prefix <<< grid_dim, block_dim >>> (N, d_sub, d_main, d_sup, d_r, d_B);
	
	tridiag <<< 1, 1 >>> (N, d_sub, d_main, d_sup, d_r, d_x2);
	
	size_t elemPerBlock = 4;
	block_dim = elemPerBlock / 2;
	grid_dim = N / elemPerBlock + (N % elemPerBlock == 0 ? 0 : 1);
	
	prescan<B_nelem> <<< grid_dim, block_dim, elemPerBlock*sizeof(real_t) * B_nelem >>> (elemPerBlock, d_B, d_C, multiplier, setter);
	
	if (grid_dim.x > 1) {
		
		real_t *d_block_B, *d_block_C;
		
		size_t nbytes_block = grid_dim.x * sizeof(real_t) * B_nelem;
		cudaMalloc((void**) &d_block_B, nbytes_block);
		cudaMalloc((void**) &d_block_C, nbytes_block);
	
		// one block; each thread processes a scan block from above
		aggregate_block_sum<B_nelem> <<< 1, grid_dim >>> (elemPerBlock, d_C, d_block_B, setter);
		
		// one block; each thread processes two scan block sums (hence need half the number of scan blocks from previous run)
		prescan<B_nelem> <<< 1, grid_dim.x/2, grid_dim.x*sizeof(real_t) * B_nelem >>> (grid_dim.x, d_block_B, d_block_C, multiplier, setter);
		
		// each thread processes one element in original data
		// need twice as many blocks as before, since each thread now processes one element
		add_block_cumsum<B_nelem> <<< grid_dim.x*2, block_dim >>> (N, d_block_C, d_C, multiplier, setter);
		
		cudaFree(d_block_B);
		cudaFree(d_block_C);
	}
	
	block_dim = 4;
	grid_dim = N / block_dim.x + (N % block_dim.x == 0 ? 0 : 1);
	
	rd_tridiag <<< grid_dim, block_dim >>> (N, d_C, d_x);
	
	// retrieve results from device and store it in host array
	cudaMemcpy(h_x, d_x, nbytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_x2, d_x2, nbytes, cudaMemcpyDeviceToHost);
	
	// compute gold standard
	real_t* h_x_gold = new real_t[N];
	bla::tridiag(N, h_sub, h_main, h_sup, h_r, h_x_gold);
	
	// print results
	bool equal = true;
	for (size_t i = 0; i < N; ++i) {
		printf("%d %f %f %f", i, h_x[i], h_x2[i], h_x_gold[i]);
		if (std::abs(h_x[i] - h_x_gold[i]) > 1e-2) {
			equal = false;
			printf("*\n");
		} else {
			printf("\n");
		}
	}
	if (!equal) printf("Differences detected!\n");
	
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
