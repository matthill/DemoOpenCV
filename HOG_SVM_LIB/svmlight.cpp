/** 
 * @file:   svmlight.h
 * @author: Jan Hendriks (dahoc3150 [at] yahoo.com)
 * @date:   Created on 11. Mai 2011
 * @brief:  Wrapper interface for SVMlight, 
 * @see http://www.cs.cornell.edu/people/tj/svm_light/ for SVMlight details and terms of use
 * 
 */
#ifdef DETECT_MEM_LEAK
#include <vld.h>
#endif
#include "SVMlight.h"


SVMlight::SVMlight() {
	// Init variables
	alpha_in = NULL;
	//kernel_cache=kernel_cache_init(totdoc, 40);
	kernel_cache = NULL; // Cache not needed with linear kernel
	model = (MODEL *) my_malloc(sizeof (MODEL));
	learn_parm = new LEARN_PARM;
	kernel_parm = new KERNEL_PARM;
	// Init parameters
	verbosity = 1; // Show some messages -v 1
	//learn_parm->alphafile[0] = ' '; // NULL; // Important, otherwise files with strange/invalid names appear in the working directory
	std::string tmp = "alphas.txt";
	for (size_t i = 0; i < tmp.size(); i++){
		learn_parm->alphafile[i] = tmp[i]; // Important, otherwise files with strange/invalid names appear in the working directory
	}
	learn_parm->alphafile[tmp.size()] = '\0';

	learn_parm->biased_hyperplane = 1;
	learn_parm->sharedslack = 0; // 1
	learn_parm->skip_final_opt_check = 0;
	learn_parm->svm_maxqpsize = 10;
	learn_parm->svm_newvarsinqp = 0;
	learn_parm->svm_iter_to_shrink = 100; // 2 is for linear;
	learn_parm->kernel_cache_size = 100;
	learn_parm->maxiter = 100000;
	learn_parm->svm_costratio = 1.0;
	learn_parm->svm_costratio_unlab = 1.0;
	learn_parm->svm_unlabbound = 1E-5;
	learn_parm->eps = 0.1;
	learn_parm->transduction_posratio = -1.0;
	learn_parm->epsilon_crit = 0.001;
	learn_parm->epsilon_a = 1E-15;
	learn_parm->compute_loo = 0;
	learn_parm->rho = 1.0;
	learn_parm->xa_depth = 0;
	// The HOG paper uses a soft classifier (C = 0.01), set to 0.0 to get the default calculation
	learn_parm->svm_c = 0.5; // -c 0.01
	learn_parm->type = CLASSIFICATION;
	learn_parm->remove_inconsistent = 0; // -i 0 - Important
	kernel_parm->rbf_gamma = 1.0;
	kernel_parm->coef_lin = 1;
	kernel_parm->coef_const = 1;
	kernel_parm->kernel_type = LINEAR; // -t 0
	kernel_parm->poly_degree = 3;
}

SVMlight::~SVMlight() {
	// Cleanup area
	// Free the memory used for the cache
	
	if (kernel_cache)
	{
		kernel_cache_cleanup(kernel_cache);
		kernel_cache = NULL;
	}
	if (alpha_in){
		free(alpha_in);
		alpha_in = NULL;
	}
	if (model){
		free_model(model, 0);
		model = NULL;
	}
	if (docs)
	{
		for (i = 0; i < totdoc; i++)
		{
			free_example(docs[i], 0);
		}

		free(docs);
		docs = NULL;
	}
	if (target){
		free(target);
		target = NULL;
	}
	if (kernel_parm){
		delete[] kernel_parm;
		kernel_parm = NULL;
	}
	if (learn_parm){
		delete[] learn_parm;
		learn_parm = NULL;
	}
	releaseSvm_hideo();
}

	


void SVMlight::saveModelToFile(const std::string _modelFileName, const std::string _identifier) {
	if (this->model->kernel_parm.kernel_type == LINEAR) {
		write_linear_model(const_cast<char*>(_modelFileName.c_str()), model);
	}
	else
	{
		write_model(const_cast<char*>(_modelFileName.c_str()), model);
	}
}

void SVMlight::loadModelFromFile(const std::string _modelFileName, const std::string _identifier) {
	this->model = read_model(const_cast<char*>(_modelFileName.c_str()));
}

// read in a problem (in svmlight format)
void SVMlight::read_problem(char *filename) {
	// Reads and parses the specified file
	read_documents(filename, &docs, &target, &totwords, &totdoc);
	
	if (kernel_parm->kernel_type == LINEAR){
		printf("KERNEL CACHE = NULL");
		kernel_cache = NULL;
	}
	else
	{
		printf("INIT KERNEL CACHE\n");
		kernel_cache = kernel_cache_init(totdoc, learn_parm->kernel_cache_size);
	}
}

// read in a problem (in svmlight format)
void SVMlight::read_problem(char *filename, const int *map, long lengthMap, double classLabel) {

	// Reads and parses the specified file
	read_documentsBaseMap(filename, &docs, &target, &totwords, &totdoc, map, lengthMap, classLabel);

	if (kernel_parm->kernel_type == LINEAR){
		printf("KERNEL CACHE = NULL");
		kernel_cache = NULL;
	}
	else
	{
		printf("INIT KERNEL CACHE\n");
		kernel_cache = kernel_cache_init(totdoc, learn_parm->kernel_cache_size);
	}
}

void SVMlight::setParameters(std::string _alpha_file, long _type, double _svmC, long _kernel_type, 
	long _remove_inconsistent, long _verbosity, double  _rbf_gamma)
{
	verbosity = _verbosity;
	
	for (size_t i = 0; i < _alpha_file.size(); i++){
		learn_parm->alphafile[i] = _alpha_file[i]; // Important, otherwise files with strange/invalid names appear in the working directory
	}
	learn_parm->alphafile[_alpha_file.size()] = '\0';

	
	// The HOG paper uses a soft classifier (C = 0.01), set to 0.0 to get the default calculation
	learn_parm->svm_c = _svmC; // -c 0.01
	learn_parm->type = _type;
	learn_parm->remove_inconsistent = _remove_inconsistent; // -i 0 - Important
	kernel_parm->rbf_gamma = _rbf_gamma;
	kernel_parm->kernel_type = _kernel_type; // -t 0
}

void SVMlight::setParameters(long _kernel_type, double _svmC, double  _rbf_gamma){
	learn_parm->svm_c = _svmC; // -c 0.01
	kernel_parm->kernel_type = _kernel_type;
	kernel_parm->rbf_gamma = _rbf_gamma;
}
// Calls the actual machine learning algorithm
void SVMlight::train() {
	svm_learn_classification_details(docs, target, totdoc, totwords, learn_parm, kernel_parm, kernel_cache, model, alpha_in, &this->svnum, &this->vcdim);
	if (model->kernel_parm.kernel_type == LINEAR) { /* linear kernel */
		/* compute weight vector */
		add_weight_vector_to_linear_model(model);
	}
}

/**
 * Generates a single detecting feature vector (v1 | b) from the trained support vectors, for use e.g. with the HOG algorithm
 * vec1 = sum_1_n (alpha_y*x_i). (vec1 is a 1 x n column vector. n = feature vector length )
 * @param singleDetectorVector resulting single detector vector for use in openCV HOG
 * @param singleDetectorVectorIndices
 */
void SVMlight::getSingleDetectingVector(std::vector<float>& singleDetectorVector, std::vector<unsigned int>& singleDetectorVectorIndices) {
	// Now we use the trained svm to retrieve the single detector vector
	DOC** supveclist = model->supvec;
	printf("Calculating single descriptor vector out of support vectors (may take some time)\n");
	// Retrieve single detecting vector (v1 | b) from returned ones by calculating vec1 = sum_1_n (alpha_y*x_i). (vec1 is a n x1 column vector. n = feature vector length )
	singleDetectorVector.clear();
	singleDetectorVector.resize(model->totwords + 1, 0.);
	printf("Resulting vector size %lu\n", singleDetectorVector.size());
	
	// Walk over every support vector
	for (long ssv = 1; ssv < model->sv_num; ++ssv) { // Don't know what's inside model->supvec[0] ?!
		// Get a single support vector
		DOC* singleSupportVector = supveclist[ssv]; // Get next support vector
		SVECTOR* singleSupportVectorValues = singleSupportVector->fvec;
		svmlight::_WORD singleSupportVectorComponent;
		// Walk through components of the support vector and populate our detector vector
		for (unsigned long singleFeature = 0; singleFeature < model->totwords; ++singleFeature) {
			singleSupportVectorComponent = singleSupportVectorValues->words[singleFeature];
			singleDetectorVector.at(singleSupportVectorComponent.wnum) += (singleSupportVectorComponent.weight * model->alpha[ssv]);
		}
	}

	// This is a threshold value which is also recorded in the lear code in lib/windetect.cpp at line 1297 as linearbias and in the original paper as constant epsilon, but no comment on how it is generated
	singleDetectorVector.at(model->totwords) = -model->b; /** @NOTE the minus sign! */
}
void SVMlight::release()
{
	// Cleanup area
	// Free the memory used for the cache
	if (kernel_cache)
	{
		kernel_cache_cleanup(kernel_cache);
		kernel_cache = NULL;
	}
	if (alpha_in){
		free(alpha_in);
		alpha_in = NULL;
	}
	//free_model(model, 0);
	if (model->supvec) {
		free(model->supvec);
		model->supvec = NULL;
	}
	if (model->alpha) {
		free(model->alpha);
		model->alpha = NULL;
	}
	if (model->index){
		free(model->index);
		model->index = NULL;
	}
	if (model->lin_weights){
		free(model->lin_weights);
		model->lin_weights = NULL;
	}
	if (docs)
	{
		for (i = 0; i < totdoc; i++)
		{
			free_example(docs[i], 1);
		}

		free(docs);
		docs = NULL;
	}
	if (target){
		free(target);
		target = NULL;
	}
}
bool SVMlight::getAccuracy(double &xa_error, double &xa_recall, double &xa_precision)
{
	if (model)
	{
		xa_error = model->xa_error;
		xa_recall = model->xa_recall;
		xa_precision = model->xa_precision;
		return true;
	}

	return false;
}
double SVMlight::classify(const std::vector<float>& featureVectorSample){
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
/// Singleton
SVMlight* SVMlight::getInstance() {
	static SVMlight theInstance;
	return &theInstance;
}

