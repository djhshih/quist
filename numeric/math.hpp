#ifndef _math_hpp_
#define _math_hpp_

namespace math {

	inline
	unsigned int log2(unsigned int val) {
		unsigned int ret = 0;
		while (val >>= 1) {
			++ret;
		}
		return ret;
	}
	
	inline
	unsigned int pow2ceil(unsigned int val) {
		unsigned int ret = 1;
		while (ret < val) ret <<= 1;
		return ret;
	}
	
	inline
	unsigned int pow2floor(unsigned int val) {
		return pow2ceil(val) >> 1;
	}
	
}
	
#endif