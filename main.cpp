#include <iostream>
using namespace std;

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include "matrix.hpp"
using namespace quist;

string progname = "quist";

int main(int argc, char **argv) {
	
	po::variables_map vm;
	po::options_description opts;
	po::positional_options_description popts;
	
	opts.add_options()
		("help,h", "print help message")
		("input,i", po::value<string>(), "input file")
		("center,c", po::value<bool>(), "center the data")
		("standardize,s", po::value<bool>(), "standarize the data")
		("window,w", po::value<int>(), "summation window size")
	;
	popts.add("input", 1);
	
	po::store( po::command_line_parser(argc, argv)
		.options(opts)
		.positional(popts)
		.run(), vm );
	po::notify(vm);
	
	if (vm.count("help")) {
		cout << "usage:  " << progname << " [options] <input file>" << endl;
		cout << opts << endl;
		return 0;
	}
	
	string inputFileName;
	
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

	int window = 0;
	if (vm.count("window")) {
		window = vm["window"].as<int>();
	}
	if (window < 0) {
		throw invalid_argument("Invalid window size");
	}
	
	Matrix<float> mat;
	mat.read(inputFileName);

	if (standardize) {
		mat.standardize();
	} else if (center) {
		mat.center();
	}
	
	float* scores = mat.evaluate(window);
	for (size_t i = 0; i < mat.size(); ++i) {
		cout << scores[i] << endl;
	}
	
	delete [] scores;
	
	return 0;
}

