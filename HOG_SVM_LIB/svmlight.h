/** 
 * @file:   svmlight.h
 * @author: Jan Hendriks (dahoc3150 [at] yahoo.com)
 * @date:   Created on 11. Mai 2011
 * @brief:  Wrapper interface for SVMlight, 
 * @see http://www.cs.cornell.edu/people/tj/svm_light/ for SVMlight details and terms of use
 * 
 */

#ifndef SVMLIGHT_H
#define	SVMLIGHT_H

#include <stdio.h>
#include <vector>
#include <string>
// svmlight related
// namespace required for avoiding collisions of declarations (e.g. LINEAR being declared in flann, svmlight and libsvm)
namespace svmlight {
    extern "C" {
        #include "svm_common.h"
        #include "svm_learn.h"
    }
}

using namespace svmlight;

class SVMlight {
private:
    DOC** docs; // training examples
    long totwords, totdoc, i; // support vector stuff
    double* target;
    double* alpha_in;
    KERNEL_CACHE* kernel_cache;
    MODEL* model; // SVM model
	long svnum;
	double vcdim;
    SVMlight();

    virtual ~SVMlight();

	
public:
    LEARN_PARM* learn_parm;
    KERNEL_PARM* kernel_parm;

    static SVMlight* getInstance();
	
    void saveModelToFile(const std::string _modelFileName, const std::string _identifier = "svmlight") ;
    void loadModelFromFile(const std::string _modelFileName, const std::string _identifier = "svmlight") ;

    // read in a problem (in svmlight format)
    void read_problem(char *filename) ;
	void read_problem(char *filename, const int *map, long lengthMap, double classLabel);
	void setParameters(std::string _alpha_file, long _type, double _svmC, long _kernel_type, 
		long _remove_inconsistent, long _verbosity, double  _rbf_gamma);
	void setParameters(long _kernel_type, double _svmC, double  _rbf_gamma);
	long getKernelType(){
		return kernel_parm->kernel_type;
	}
	double getSvmC(){
		return learn_parm->svm_c;
	}
	double getRbfGamma(){
		return kernel_parm->rbf_gamma;
	}
	long getSvNum(){
		return this->svnum;
	}
	double getVcDim(){
		return this->vcdim;
	}
	// Calls the actual machine learning algorithm
	void train();

    /**
     * Generates a single detecting feature vector (v1 | b) from the trained support vectors, for use e.g. with the HOG algorithm
     * vec1 = sum_1_n (alpha_y*x_i). (vec1 is a 1 x n column vector. n = feature vector length )
     * @param singleDetectorVector resulting single detector vector for use in openCV HOG
     * @param singleDetectorVectorIndices
     */
    void getSingleDetectingVector(std::vector<float>& singleDetectorVector, std::vector<unsigned int>& singleDetectorVectorIndices) ;
	double classify(const std::vector<float>& featureVectorSample);
	bool getAccuracy(double &xa_error, double &xa_recall, double &xa_precision);
	void release();
};

#endif	/* SVMLIGHT_H */
