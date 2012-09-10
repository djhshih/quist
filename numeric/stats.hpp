#ifndef _stats_hpp_
#define _stats_hpp_

namespace stats {

	template <typename T>
	void summary(const T a[], size_t n, T* mean, T* sd) {
		T m = 0, devsq = 0, t, x;
		for (size_t j = 0; j < n; ++j) {
			x = a[j];	
			t = (x - m);
			m += (x - m) / (j+1);
			devsq += t * (x - m);
		}
		
		if (mean != NULL) *mean = m;
		if (sd != NULL) *sd = std::sqrt( devsq/n );
	}
	
}

#endif