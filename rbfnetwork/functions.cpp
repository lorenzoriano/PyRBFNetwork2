#include <boost/python.hpp>
#include <boost/random.hpp>
#include <arrayobject.h>
#include <cmath>

namespace bp = boost::python;

void single_row_output(double* res_row, int input_index) {


}

typedef boost::mt19937 base_generator_type;
static base_generator_type __static__generator(static_cast<unsigned int>(std::time(0)));

#define GET_ELEM(mat, i, j) ((double*) PyArray_GETPTR2(mat, i, j))
#define GAUSS(x, sigma) (std::exp(-x/(sigma*sigma)))

void first_layer_output(PyObject* input_matrix, PyObject* kernels_matrix,
		PyObject* res_matrix, int num_kernels, int num_inputs, int input_size,
		double sigma) {

	for (int input_index = 0; input_index<num_inputs; input_index++) {
		*GET_ELEM(res_matrix, input_index, 0) = 1.0; //bias
		for (int kernel_index = 0; kernel_index<num_kernels; kernel_index++) {

			double sum = 0;
			for (int i=0; i<input_size; i++) {
				double tmp = *GET_ELEM(input_matrix, input_index, i) -
							 *GET_ELEM(kernels_matrix, kernel_index, i);
				sum += tmp*tmp;
			}
			*GET_ELEM(res_matrix, input_index, kernel_index+1/*bias*/) = GAUSS(sum, sigma);
		}
	}

}

void sample_inputs(unsigned int n, PyObject* out, PyObject* kernels,
		unsigned int kernel_size, unsigned int input_size, double sigma) {

    //This is to select the kernels
    boost::uniform_smallint<> uni_dist(0,kernel_size - 1);
    boost::variate_generator<base_generator_type&, boost::uniform_smallint<> > rand_kernel(__static__generator, uni_dist);

    double variance = sigma*sigma;
    for (unsigned int i=0; i<n; i++) {
        unsigned int kernel_to_use = rand_kernel();
        for (unsigned int j=0; j<input_size; j++) {
            //Each kernel has the same spherical covariance matrix, so it's pretty straightforward
            //to generate random numbers from a multi-dimensional kernel
            double mean = *GET_ELEM(kernels,kernel_to_use, j);
            boost::normal_distribution<> nd(mean, variance);
            boost::variate_generator<base_generator_type&, boost::normal_distribution<> > rand_nor(__static__generator, nd);
            *GET_ELEM(out,i,j) = rand_nor();
        }
    }
}



BOOST_PYTHON_MODULE(libfunctions) {
	import_array();
	bp::def("first_layer_output", first_layer_output);
	bp::def("sample_inputs", sample_inputs);

}
