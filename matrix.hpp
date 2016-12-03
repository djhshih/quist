#ifndef quist_matrix_h
#define quist_matrix_h

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cmath>

#include <numeric/bla.hpp>

namespace quist {

	using namespace std; 
	
	namespace score {
		enum Type {
			dot_product, covariance, correlation
		};
	}
	
	template <typename T>
	class Matrix {

	private:
		// number of rows and cols
		size_t m, n;
		// data
		T **rep;
		T *scores, *means, *sds;

		void clearCache() {
			delete [] scores;
			delete [] means;
			delete [] sds;
			scores = means = sds = NULL;
		}

		// calculate means and SDs of each row using Welford's method
		void summarize() {
			if (means != NULL) {
				delete [] means;
			}
			if (sds != NULL) {
				delete [] sds;
			}

			means = new T[m];
			sds = new T[m];

			for (size_t i = 0; i < m; ++i) {
				
				// summarize row
				T m = 0, devsq = 0, t, x;
				for (size_t j = 0; j < n; ++j) {
					x = rep[i][j];	
					t = (x - m);
					m += (x - m) / (j+1);
					devsq += t * (x - m);
				}

				means[i] = m;
				sds[i] = std::sqrt( devsq/n );

			}
		}
		
	public:
		
		Matrix() : rep(NULL), m(0), n(0), scores(NULL), means(NULL), sds(NULL) {}
		
		~Matrix() {
			clear();
		}
		
		void clear() {
			if (rep != NULL) {
				for (size_t i = 0; i < m; ++i) {
					delete [] rep[i];
				}
				rep = NULL;
				m = n = 0;
			}
			clearCache();
		}

		size_t size()  { return m; }
		size_t nrows() { return m; }
		size_t ncols() { return n; }
		
		void read(const string& fileName) {
			fstream file;
			file.open(fileName.c_str(), ios::in);
			if (!file.is_open()) throw runtime_error("Failed to open input file.");
			read(file);
			file.close();
		}
		
		void read(fstream& file) {
			clear();
			string line;
			size_t i = 0;
			istringstream stream;
			
			// get dimensions
			getline(file, line);
			stream.str(line);
			stream >> m >> n;
			
			// allocate array
			rep = new T*[m];
			
			// read in data
			while (true) {
				getline(file, line);
				
				if (file.eof()) break;
				
				stream.clear();
				stream.str(line);
				
				T* row = rep[i++] = new T[n];
				size_t j = 0;
				while (!stream.eof()) {
					stream >> row[j++];
				}
			}
		}
		
		void print() {
			for (size_t i = 0; i < m; ++i) {
				for (size_t j = 0; j < n; ++j) {
					cout << rep[i][j] << ' ';
				}
				cout << endl;
			}
		}

		void center() {
			if (means == NULL) {
				summarize();
			}
			for (size_t i = 0; i < m; ++i) {
				for (size_t j = 0; j < n; ++j) {
					rep[i][j] -= means[i];
				}
			}
		}

		void standardize() {
			if (means == NULL || sds == NULL) {
				summarize();
			}
			for (size_t i = 0; i < m; ++i) {
				for (size_t j = 0; j < n; ++j) {
					rep[i][j] = (rep[i][j] - means[i]) / sds[i];
				}
			}
		}
		
		T* evaluate(size_t window) {
			clearCache();
			scores = new T[m];

			for (size_t i = 0; i < m; ++i) {
				// determine summation window
				long start = i - window/2;
				if (start < 0) start = 0;
				long end = i + window/2 + 1;
				if (end > m) end = m;

				// average score in summation window
				scores[i] = 0;
				for (size_t ii = start; ii < end; ++ii) {
					scores[i] += bla::dot(n, rep[i], rep[ii]);
				}
				// divide by window size (and by n to complete the score calculation)
				scores[i] /= (end - start) * n;
			}

			return scores;
		}
		
	};
	
}

#endif
