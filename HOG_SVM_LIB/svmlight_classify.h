#ifndef _SVMLIGHT_CLASSIFY_H_
#define _SVMLIGHT_CLASSIFY_H_
#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>
// svmlight related
// namespace required for avoiding collisions of declarations (e.g. LINEAR being declared in flann, svmlight and libsvm)

extern "C" {
	#include "svm_common.h"
	#include "svm_learn.h"
}

class SvmLightClassify{
private:
	MODEL* model; // SVM model
public:
	SvmLightClassify();
	~SvmLightClassify();
	void loadModelFromFile(const std::string& _modelFileName);
	double classify(const std::vector<float>& featureVectorSample);
	void getSVMDecriptor(std::vector<float>& svmDecriptor);
	long getSVMKernelType();
};


#endif //_SVMLIGHT_CLASSIFY_H_