#ifndef _SVM_HOG_TRAIN_
#define _SVM_HOG_TRAIN_
#include "config.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/ml/ml.hpp>
#include "svmlight.h"
#include "FTS_Hog.h"
struct _Accuracy{
	long    kernel_type;
	double	svmC;
	double  rbf_gamma;
	long    poly_degree;
	double  train_error, train_recall, train_precision;
	double  test_error, test_recall, test_precision;
};
struct FeatureHog{
	std::string file_name;
	int label;
	std::vector<float> featureVector;
};

class SVM_HOG_EXPORT SvmHogTrain {
private:
	FTS_HOGDescriptor hog;
	cv::Size padding, winStride;
	std::vector<std::string> listDir;
	std::vector<std::string> listFileToTrain;
	std::vector<std::vector<float> > listDescriptorVector;
	std::vector<_Accuracy> listAcc;

	// eval
	std::vector<std::vector<FeatureHog>> listOfFeatureOfClass;
	std::vector<std::string> listLabels;
	int numOfSeparation;

public:
	SvmHogTrain();
	~SvmHogTrain();
	void setHogParameters(cv::Size _winSize, cv::Size _blockSize, cv::Size _blockStride, cv::Size _cellSize, cv::Size _padding, cv::Size _winStride);
	void setSvmParameters(std::string _alpha_file, long _type, double _svmC, long _kernel_type, 
	long _remove_inconsistent, long _verbosity, double  _rbf_gamma);
	void setSvmParameters(long _kernel_type, double _svmC, double  _rbf_gamma);
	void setListLabels(std::vector<std::string>& _listLabels);
	void clearListDir();
	void clearListFileToTrain();
	void addDirToList(const std::string& dir);
	void addListFileToTrain(const std::string& file_name);
	void calculateFeaturesFromInput(const std::string& imageFilename, std::vector<float>& featureVector);
	void saveDescriptorVectorToFile(std::vector<float>& descriptorVector, std::vector<unsigned int>& _vectorIndices, std::string fileName);
	bool extractFeatureToTrain();
	void trainBinary(const std::string &sInHOHFeatureFile);
	void trainBinaryBaseMap(const std::string &sInHOHFeatureFile, const int *map, long lengthMap, double classLabel);

	void trainMultiClass(const std::string &file_input, const std::vector<long> &kernel_type, const std::vector<double> & vSvmC, const std::vector<double> &vGamma);
	//void trainDataAll();

	void trainAllData(const std::string& file_input, const std::string& strPathOutput, const std::vector<std::string>& listLabel, std::vector<long> &kernel_type, const std::vector<double> & vSvmC, const std::vector<double> &vGamma);

	void saveListDescriptorVectorToFile(const std::string &file_name);
	//void saveAccuracy(const std::string & file_name);

	//validation
	void swapRandomListImage(std::vector<std::string>& listImage);
	bool extractAllFeature(bool isRandom); // extract feature for each label
	void saveMultiClassFeatureToFile(std::string sMulClassFeatureFile); // save all features into file with multi-class labels
	void crossValidation(const std::string& file_input);
	void crossValidation(const std::string& file_input, std::fstream& File);

	void validation(const std::string& file_input, const std::vector<FeatureHog>& listOfFeatureTrain, const std::vector<FeatureHog>& listOfFeatureTest, std::fstream& File);

	void saveListFeatureToFile(const std::vector< std::vector<float> >& listFeatureVectorPositive,
		const std::vector< std::vector<float> >& listFeatureVectorNegative, const std::string& file_name);
	bool detectTest(const cv::HOGDescriptor& hog_test, cv::Mat& imageData);
	void evaluate(const std::vector<std::vector<float>>& vValPosFiles, const std::vector<std::vector<float> >& vValNegFiles, int &tp, int &tn, int &fp, int &fn);
	void saveAccuracyValidation(const std::string & file_name);

	//learning curve
	void separateData(float trainingPercent, std::vector<FeatureHog>& listOfFeatureTrain, std::vector<FeatureHog>& listOfFeatureTest);
	void saveMultiClassFeatureToFile(const std::vector<FeatureHog>& listOfFeatureTrain, const std::string& sMulClassFeatureFile); // save all features into file with multi-class labels
	void learningCurveAnalysis(const std::string& file_dataInput, const std::vector<FeatureHog>& listOfFeatureTrain, const std::vector<FeatureHog>& listOfFeatureTest, int sampleStep, int classLabel);
};

#endif//_SVM_HOG_TRAIN_