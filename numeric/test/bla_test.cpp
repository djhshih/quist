#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "Basic Linear Algebra Unit Tests"

#include <string>
#include <stdexcept>
#include <cstdio>

#include <boost/test/unit_test.hpp>

#include "../bla.hpp"
#include "../../equal.hpp"
#include "../../hdf5/data.hpp"

using namespace std;
using namespace hdf5;


BOOST_AUTO_TEST_SUITE (Spline)

BOOST_AUTO_TEST_CASE ( nscpline_test )
{
	char fname[] = "spline_test0.h5";
	Data<> x(fname, "/x/value");
	Data<> y(fname, "/y/value");
	Data<> xx(fname, "/xx/value");
	Data<> ans(fname, "/yy/value");
	
	size_t n = x.size();
	size_t nn = xx.size();
	double* yy = new double[nn];
	
	bla::splint(n, x.data(), y.data(), nn, xx.data(), yy);
	
	BOOST_CHECK_MESSAGE(check_array_equal(yy, ans.data(), nn), "spline results did not match expected");
}

BOOST_AUTO_TEST_CASE ( nscpline_test_float )
{
	char fname[] = "spline_test1.h5";
	Data<float> x(fname, "/x/value");
	Data<float> y(fname, "/y/value");
	Data<float> xx(fname, "/xx/value");
	Data<float> ans(fname, "/yy/value");
	
	size_t n = x.size();
	size_t nn = xx.size();
	float* yy = new float[nn];
	
	bla::splint(n, x.data(), y.data(), nn, xx.data(), yy);
	
	BOOST_CHECK_MESSAGE(check_array_equal(yy, ans.data(), nn), "spline results did not match expected");
}

BOOST_AUTO_TEST_SUITE_END()
