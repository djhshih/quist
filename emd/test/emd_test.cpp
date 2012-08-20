#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "Empirical Mode Decomposition Unit Tests"

#include <string>
#include <stdexcept>
#include <cstdio>

#include <boost/test/unit_test.hpp>

#include "../emd.hpp"
#include "../../equal.hpp"
#include "../../hdf5/data.hpp"

using namespace std;
using namespace hdf5;


BOOST_AUTO_TEST_SUITE (Extrema)

BOOST_AUTO_TEST_CASE ( extrema_test )
{
	char fname[] = "extrema_test.h5";
	Data<> x(fname, "/x/value");
	Data<> y(fname, "/y/value");
	Data<> min_x_ans(fname, "/min_x/value");
	Data<> min_y_ans(fname, "/min_y/value");
	Data<> max_x_ans(fname, "/max_x/value");
	Data<> max_y_ans(fname, "/max_y/value");
	
	size_t n = x.size();
	size_t n_min_ans = min_x_ans.size();
	size_t n_max_ans = max_x_ans.size();
	
	double* min_x = new double[n];
	double* max_x = new double[n];
	double* min_y = new double[n];
	double* max_y = new double[n];
	
	size_t n_min, n_max;
	emd::minima(n, x.data(), y.data(), &n_min, min_x, min_y);
	emd::maxima(n, x.data(), y.data(), &n_max, max_x, max_y);
	
	BOOST_CHECK(n_max_ans == n_max);
	BOOST_CHECK(n_min_ans == n_min);
	
	BOOST_CHECK_MESSAGE(check_array_equal(min_x, min_x_ans.data(), n_min), "Minimum points did not match expected");
	BOOST_CHECK_MESSAGE(check_array_equal(min_y, min_y_ans.data(), n_min), "Minimum points did not match expected");
	
	BOOST_CHECK_MESSAGE(check_array_equal(max_x, max_x_ans.data(), n_max), "Maximum points did not match expected");
	BOOST_CHECK_MESSAGE(check_array_equal(max_y, max_y_ans.data(), n_max), "Maximum points did not match expected");
	
}
BOOST_AUTO_TEST_SUITE_END()


BOOST_AUTO_TEST_SUITE (EMD)

BOOST_AUTO_TEST_CASE ( emd_test )
{
	char fname[] = "emd_test_ncspline.h5";
	Data<> x(fname, "/x/value");
	Data<> y(fname, "/y/value");
	Data<> ans(fname, "/modes/value");
	
	size_t n = x.size();
	size_t nmodes = 4;
	
	double** modes = emd::emd(n, x.data(), y.data(), &nmodes);
	
	BOOST_CHECK_MESSAGE(check_arrays_equal(modes, ans.data(), nmodes, n, 1e-5), "EMD modes did not match expected");
	
	emd::free_arrays(modes, nmodes);
}

BOOST_AUTO_TEST_CASE ( eemd_test )
{
	char fname[] = "emd_test_ncspline.h5";
	Data<> x(fname, "/x/value");
	Data<> y(fname, "/y/value");
	Data<> ans(fname, "/modes/value");
	
	size_t n = x.size();
	size_t nmodes = 4;
	
	double** modes = emd::eemd(n, x.data(), y.data(), &nmodes, 0.01, 20);
	
	// use a lenient difference threshold, since differences are expected
	BOOST_CHECK_MESSAGE(check_arrays_equal(modes, ans.data(), nmodes, n, 1e-2), "EEMD modes did not match expected");
	
	emd::free_arrays(modes, nmodes);
}

BOOST_AUTO_TEST_CASE ( dsemd_test )
{
	char fname[] = "emd_test_ncspline.h5";
	Data<> x(fname, "/x/value");
	Data<> y(fname, "/y/value");
	Data<> ans(fname, "/modes/value");
	
	size_t n = x.size();
	size_t nmodes = 4;
	
	double** modes = emd::dsemd(n, x.data(), y.data(), &nmodes, 100, 1000);
	
	// use a very lenient difference threshold, since differences are expected
	BOOST_CHECK_MESSAGE(check_arrays_equal(modes, ans.data(), nmodes, n, 8e-1), "DS-EMD modes did not match expected");
	
	emd::free_arrays(modes, nmodes);
}

BOOST_AUTO_TEST_SUITE_END()
