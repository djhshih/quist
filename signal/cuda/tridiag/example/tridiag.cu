// hack to make nvcc work with gcc-4.7
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include <cstdio>
#include <cmath>
#include <ctime>

#include <numeric/bla.hpp>

#include "../rd.hpp"
#include "../thomas.hpp"
#include "../../cudpp/crpcr.hpp"

using namespace std;

// FIXME RD appears to be numerically unstable when using float...
// FIXME As N increases (> 128), even using double does not prevent numerical instability
//       Using Kahan summation inside matrix multiplier appears to have improved numerical instability somewhat
typedef float real_t;

int main(int argc, char* argv[]) {
	
	std::srand( (unsigned)time(NULL) );
	
	real_t *h_sub, *h_main, *h_sup, *h_r, *h_x;
	real_t *d_sub, *d_main, *d_sup, *d_r, *d_x;
	
	const size_t N = 1024;
	//const size_t N = 128;
	//const size_t N = 64;
	
	size_t nbytes = N * sizeof(real_t);
	
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
	
	// initialize host array
	for (size_t i = 0; i < N; ++i) {
		h_sub[i] = (std::rand() % 5) + 5;
		h_main[i] = (std::rand() % 5) + 5;
		h_sup[i] = (std::rand() % 5) + 5;
		h_r[i] = (std::rand() % 5) + 5;
	}
	
	// copy data to device
	cudaMemcpy(d_sub, h_sub, nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_main, h_main, nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_sup, h_sup, nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_r, h_r, nbytes, cudaMemcpyHostToDevice);
	
	rd(N, d_sub, d_main, d_sup, d_r, d_x);
	//crpcr(N, 1, d_sub, d_main, d_sup, d_r, d_x);
	//thomas(N, 1, d_sub, d_main, d_sup, d_r, d_x);
	
	// retrieve results from device and store it in host array
	cudaMemcpy(h_x, d_x, nbytes, cudaMemcpyDeviceToHost);
	
	// compute gold standard
	real_t* h_x_gold = new real_t[N];
	bla::tridiag(N, h_sub, h_main, h_sup, h_r, h_x_gold);
	
	// print results
	bool equal = true;
	for (size_t i = 0; i < N; ++i) {
		printf("%lu %f %f", i, h_x[i], h_x_gold[i]);
		if (std::abs(h_x[i] - h_x_gold[i]) > 1e-2) {
			equal = false;
			printf("*\n");
		} else {
			printf("\n");
		}
	}
	if (!equal) printf("Differences detected!\n");
	
	
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
	
	CUDA_CHECK_ERROR("main");
}
