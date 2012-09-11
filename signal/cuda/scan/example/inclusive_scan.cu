// hack to make nvcc work with gcc-4.7
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include <cstdio>
#include <ctime>

#define _DEBUG

#include "../../util.hpp"
#include "../inclusive_scan.hpp"
#include "../functor/static_matrix_functor.hpp"


using namespace std;

typedef float real_t;


int main(int argc, char* argv[]) {
	
	std::srand( (unsigned)time(NULL) );
	
	real_t *h_x, *h_y;
	real_t *d_x, *d_y;
	
	//const size_t N = 512;
	const size_t N = 256;
	const size_t mat_dim = 3;
	const size_t m = mat_dim * mat_dim;
	
	size_t nbytes = N * sizeof(real_t) * m;
	
	// allocate array on host
	h_x = (real_t*)malloc(nbytes);
	h_y = (real_t*)malloc(nbytes);
	
	// initialize host array
	for (size_t i = 0; i < N*m; ++i) {
		h_x[i] = (rand() % 3) - 1;
	}
	
	// allocate array on device
	cudaMalloc((void**) &d_x, nbytes);
	cudaMalloc((void**) &d_y, nbytes);
	
	// copy data to device
	cudaMemcpy(d_x, h_x, nbytes, cudaMemcpyHostToDevice);
	
	StaticMatrixMultiplierPrefixStable<real_t, mat_dim> op;
	StaticMatrixSetter<real_t, mat_dim> setter;
	
	inclusive_scan<m>(N, d_x, d_y, op, setter);
	
	// retrieve results from device and store it in host array
	cudaMemcpy(h_y, d_y, nbytes, cudaMemcpyDeviceToHost);
	
	// compute gold standard
	real_t* h_gold = new real_t[N*m];
	setter(h_gold[0], h_x[0]);
	for (size_t i = 1; i < N; ++i) {
		setter(h_gold[i*m], h_x[i*m]);
		op(h_gold[i*m], h_gold[(i-1)*m]);
	}
	
	// print results
	bool equal = true;
	for (size_t i = 0; i < N*m; ++i) {
		printf("%lu %.0f %.0f %.0f", i, h_x[i], h_y[i], h_gold[i]);
		if (std::abs(h_y[i] - h_gold[i]) > 1e-5) {
			equal = false;
			printf("*\n");
		} else {
			printf("\n");
		}
	}
	if (!equal) printf("Differences detected!\n");
	
	
	CUDA_CHECK_ERROR("main");
	
	// clean up
	
	free(h_x);
	free(h_y);
	
	delete [] h_gold;
	
	cudaFree(d_x);
	cudaFree(d_y);
	
}
