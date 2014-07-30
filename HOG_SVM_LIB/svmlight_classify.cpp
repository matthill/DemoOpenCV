#ifdef DETECT_MEM_LEAK
#include <vld.h>
#endif
#include "svmlight_classify.h"

SvmLightClassify::SvmLightClassify(){
	model = NULL;
}
SvmLightClassify::~SvmLightClassify(){
	if (model){
		
		free_model(model, 1);
		model = NULL;
	}
}
void SvmLightClassify::loadModelFromFile(const std::string& _modelFileName) {
	this->model = read_model(const_cast<char*>(_modelFileName.c_str()));
	/*if (model->kernel_parm.kernel_type == LINEAR) { // linear kernel 
		// compute weight vector 
		add_weight_vector_to_linear_model(model);
	}*/
	

}
void SvmLightClassify::getSVMDecriptor(std::vector<float>& svmDecriptor){
	svmDecriptor.clear();
	for (long i = 0; i < model->totwords; i++) {
		svmDecriptor.push_back(model->lin_weights[i + 1]);
	}
	svmDecriptor.push_back(-model->b);
}
double SvmLightClassify::classify(const std::vector<float>& featureVectorSample){
	DOC *doc;   /* test example */
	_WORD *words;
	double dist;
	long max_words_doc;
	char *comment = "";
	max_words_doc = long(featureVectorSample.size());
	max_words_doc += 2;
	words = (_WORD *)my_malloc(sizeof(_WORD)*(max_words_doc + 10));
	for (size_t i = 0; i < featureVectorSample.size(); i++){
		words[i].wnum = i + 1;
		words[i].weight = (FVAL)featureVectorSample[i];
	}
	words[featureVectorSample.size()].wnum = 0;
	if (model->kernel_parm.kernel_type == LINEAR) {/* For linear kernel,     */
		for (long j = 0; (words[j]).wnum != 0; j++) {     /* check if feature numbers   */
			if ((words[j]).wnum>model->totwords)   /* are not larger than in     */
				(words[j]).wnum = 0;                  /* model. Remove feature if   */
		}                                       /* necessary.                 */
	}
	doc = create_example(-1, 0, 0, 0.0, create_svector(words, comment, 1.0));
	if (model->kernel_parm.kernel_type == LINEAR) {   /* linear kernel */
		dist = classify_example_linear(model, doc);
	}
	else {                                           /* non-linear kernel */
		dist = classify_example(model, doc);
	}
	//std::cout << "distance: " << dist << std::endl;
	free(words);
	free_example(doc, 1);
	return dist;
}

long SvmLightClassify::getSVMKernelType() {
	return this->model->kernel_parm.kernel_type;
}