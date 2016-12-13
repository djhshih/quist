#include <iostream>
using namespace std;

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <signal/emd.hpp>

#include "matrix.hpp"
using namespace quist;

string progname = "quist";

int main(int argc, char **argv) {

	string inputFileName;
	string outputFileName;
	string outputRawFileName;
	string outputModesFileName;
	
	po::variables_map vm;
	po::options_description opts;
	po::positional_options_description popts;
	
	// specify all options (including positional options)
	opts.add_options()
		("help,h", "print help message")
		("input,i", po::value<string>(&inputFileName)->required(), "input file name")
		("output,o", po::value<string>(&outputFileName)->required(), "output file name")
		("output-raw", po::value<string>(&outputRawFileName), "raw scores output file name")
		("output-modes", po::value<string>(&outputModesFileName), "empirical modes output file name")
		("window,w", po::value<int>(), "summation window size; must be odd [default: 5]")
		("center,c", po::value<bool>(), "center the data [default: false]")
		("standardize,s", po::value<bool>(), "standarize the data [default: false]")
		;

	// identify positional options	
	popts.add("input", 1);
	popts.add("output", 1);
	
	try {
	
		// parse command line arguments
		po::store(
			po::command_line_parser(argc, argv)
				.options(opts).positional(popts).run(),
			vm
		);
		
		if (vm.count("help")) {
			cout << "usage:  " << progname << " [options] <input> <output>" << endl;
			cout << opts << endl;
			return 0;
		}
		
		// throws on error
		po::notify(vm);
	
	} catch(boost::program_options::required_option& e) { 
      std::cerr << "Error: " << e.what() << std::endl;
      return 1; 
    } catch(boost::program_options::error& e) { 
      std::cerr << "Error: " << e.what() << std::endl;
      return 1; 
    } 
	
	if (vm.count("input")) {
		inputFileName = vm["input"].as< string >();
	} else {
		throw invalid_argument("No input file specified.");
	}

	bool center = false, standardize = false;
	if (vm.count("center")) {
		center = vm["center"].as<bool>();
	}
	if (vm.count("standardize")) {
		standardize = vm["standardize"].as<bool>();
	}

	int window = 5;
	if (vm.count("window")) {
		window = vm["window"].as<int>();
	}
	if (window < 3) {
		throw invalid_argument("Window size must be at least 3.");
	}
	if (window % 2 != 1) {
		throw invalid_argument("Window size must be odd.");
	}
	
	// read input data and preprocess
	
	Matrix<double> mat;
	mat.read(inputFileName);

	if (standardize) {
		mat.standardize();
	} else if (center) {
		mat.center();
	}
	
	size_t p = mat.nrows();
	
	double* scores = mat.evaluate(window);
	// `mat` will deallocate scores
		
	// output the detrended scores
	if (!outputRawFileName.empty()) {
		ofstream fout(outputRawFileName.c_str());
		if (!fout.is_open()) {
			string s = "Failed to open output file: ";
			throw runtime_error(s + outputRawFileName);
		}
		for (size_t i = 0; i < p; ++i) {
			fout << scores[i] << endl;
		}
		fout << endl;
	}

	// apply Fisher's z-transformation on the scores
	for (size_t i = 0; i < p; ++i) {
		scores[i] = atanh(scores[i]);
	}

	// detrend the scores
	
	// default x to (1, 2, ..., p)
	long* x = new long[p];
	for (size_t i = 0; i < p; ++i) {
		x[i] = i + 1;
	}
	
	size_t nmodes = 1;
	double** modes = signal::emd(p, x, scores, &nmodes);
	
	// last mode represents the trend
	size_t trend_i = nmodes - 1;
	// subtract the trend from the scores
	for (size_t i = 0; i < p; ++i) {
		scores[i] -= modes[trend_i][i];
	}
	
	// output modes
	if (!outputModesFileName.empty()) {
		ofstream fout(outputModesFileName.c_str());
		if (!fout.is_open()) {
			string s = "Failed to open output file: ";
			throw runtime_error(s + outputModesFileName);
		}
		for (size_t k = 0; k < nmodes; ++k) {
			fout << modes[k][0];
			for (size_t i = 1; i < p; ++i) {
				fout << '\t' << modes[k][i];
			}
			fout << endl;
		}
		fout.close();
	}
	
	// output the detrended scores
	if (!outputFileName.empty()) {
		ofstream fout(outputFileName.c_str());
		if (!fout.is_open()) {
			string s = "Failed to open output file: ";
			throw runtime_error(s + outputFileName);
		}
		for (size_t i = 0; i < p; ++i) {
			fout << scores[i] << endl;
		}
		fout << endl;
	}

	// deallocate
	
	delete [] x;

	for (size_t k = 0; k < nmodes; ++k) {
		delete [] modes[k];
	}
	delete [] modes;
	
	return 0;
}
