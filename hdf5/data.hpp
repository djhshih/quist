#ifndef quist_data_h
#define quist_data_h

#include <hdf5.h>
#include <hdf5_hl.h>

namespace hdf5 {

	template <typename T = double>
	class Data {
		
	public:
		
		Data(const char* fname, const char* dataset)
		: _data(NULL) {
			
			herr_t status;
			
			// open file
			hid_t fid = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
			
			_size = getSize(fid, dataset);
			
			// read dataset
			_data = new T[_size];
			T dummy;
			status = readData(fid, dataset, dummy);
			
			// close file
			status = H5Fclose(fid);
		}
		
		~Data() {
			delete [] _data;
		}
		
		const T* data() const {
			return _data;
		}
		
		size_t size() const {
			return _size;
		}
		
	private:
		
		herr_t readData(hid_t fid, const char* dataset, float) {
			return H5LTread_dataset(fid, dataset, H5T_NATIVE_FLOAT, _data);
		}
		
		herr_t readData(hid_t fid, const char* dataset, double) {
			return H5LTread_dataset(fid, dataset, H5T_NATIVE_DOUBLE, _data);
		}
		
		herr_t readData(hid_t fid, const char* dataset, int) {
			return H5LTread_dataset(fid, dataset, H5T_NATIVE_INT, _data);
		}
		
		herr_t readData(hid_t fid, const char* dataset, short) {
			return H5LTread_dataset(fid, dataset, H5T_NATIVE_SHORT, _data);
		}
		
		herr_t readData(hid_t fid, const char* dataset, long) {
			return H5LTread_dataset(fid, dataset, H5T_NATIVE_LONG, _data);
		}
		
		size_t getSize(hid_t fid, const char* dataset) {
			
			// get dimensionality
			int rank;
			herr_t status = H5LTget_dataset_ndims(fid, dataset, &rank);
			
			hsize_t* dims = new hsize_t[rank];
			
			// get dataset size
			status = H5LTget_dataset_info(fid, dataset, dims, NULL, NULL);
			size_t size = 1;
			for (size_t i = 0; i < rank; i++) {
				size *= (size_t)(dims[i]);
			}
			
			delete [] dims;
			
			return size;
		}
		
		T* _data;
		size_t _size;
		
	};

}

#endif
