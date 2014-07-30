#ifndef HOG_SVM_CLASSIFIERS_H
#define HOG_SVM_CLASSIFIERS_H
#include "config.h"
#include <vector>
#include "svmlight_classify.h"
//#include <opencv2/ml/ml.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include "FTS_Hog.h"
#define USE_SVM_CLASSIFY

enum CLASSIFIER_SET{ CUSTOM = -1, DIGIT_ONLY = 0, LETTER_ONLY = 1, DIGIT_LETTER = 2};

struct SVM_HOG_EXPORT character{
	std::string strLabel;
	double confidence;
};
struct SVM_HOG_EXPORT DetectionObject{
	DetectionObject(int x, int y, int w, int h, double _score, const std::string& _label){
		boundingBox = cv::Rect(x, y, w, h);
		score = _score;
		label = _label;
	}
	DetectionObject(){}
	cv::Rect boundingBox;
	double score;
	std::string label;
};
class SVM_HOG_EXPORT HogSvmClassifiers{
private:
#ifdef USE_SVM_CLASSIFY
	FTS_HOGDescriptor hog;
	cv::Size winStride, padding;
	std::vector<SvmLightClassify*> listModelSvms;
	std::vector<std::string> listStrLabel;
	std::vector<int> listModelIdx;
	std::map<std::string, int> classIdxMap;
#else
	std::vector<cv::HOGDescriptor> listHog;
	void str2Size(const std::string& str, cv::Size& pt);
#endif //USE_SVM_CLASSIFY
public:
#ifdef USE_SVM_CLASSIFY
	HogSvmClassifiers(){}
	HogSvmClassifiers(const HogSvmClassifiers& hsc, cv::Size _winStride, cv::Size _padding);
	~HogSvmClassifiers();
	cv::Size getHogWinSize(){ return this->hog.winSize; }
	bool loadModelFormPath(const std::string& strPath);
	void getHogParameters(cv::HOGDescriptor& _hog, cv::Size& _winStride, cv::Size& _padding);
	void setHogParameters(cv::Size _winSize, cv::Size _blockSize, cv::Size _blockStride, cv::Size _cellSize, cv::Size _winStride, cv::Size _padding);
	void setHogParameters(const cv::HOGDescriptor& hog, cv::Size _winStride, cv::Size _padding);
	void calculateFeaturesFromInput(const cv::Mat& imgTest, std::vector<float>& featureVector);
	void loadModelFromFile(const std::string& _modelFileName);
	double classify(int indexModel, const std::vector<float>& featureVectorSample);
	void setClassifierIdx(CLASSIFIER_SET classifierSetType, const std::vector<std::string>& customClassLabels);
	void customMultiClassify(const cv::Mat& imgTest, std::vector<character>& listSortChars);
	void singleClassify(const cv::Mat& imgTest, const std::string& posClassName, const std::string& negClassName, character& resChar);
	void singleClassify(const std::vector<float>& featureVector, const std::string& posClassName, const std::string& negClassName, character& resChar);
	void multiClassify(const cv::Mat& imgTest, std::vector<character>& listSortChars, bool bAppend = false);
	void multiClassify(const std::vector<float>& featureVector, std::vector<character>& listSortChars, bool bAppend);
	//void multiclassClassify(const cv::Mat& imgTest, const std::vector<std::string>& listedChars, std::vector<character>& listSortChars);
	//void multiclassClassify(const cv::Mat& imgTest, const std::vector<int>& modelIdx, std::vector<character>& listSortChars);
	//detection
	void detectSingleScale(cv::Mat& image, double scale, std::vector<DetectionObject>& objects);
	void detectSingleScale(cv::Mat& image, int stripCount, cv::Size processingRectSize,
		int stripSize, int yStep, double factor, std::vector<DetectionObject>& candidates);
	void detectMultiScaleBasedBoost(const cv::Mat& image, std::vector<DetectionObject>& objects,
		double scaleFactor, int minNeighbors,
		int flags, cv::Size minObjectSize, cv::Size maxObjectSize);
	void detectMultiScale(const cv::Mat& image, std::vector<DetectionObject>& objects,
		double scaleFactor, int minNeighbors, double hitThreshold,
		cv::Size minObjectSize, cv::Size maxObjectSize);
#else
	HogSvmClassifiers(){}
	void loadVectorDescriptorsFromFile(const std::string &file_name);
	void detectBinary(const cv::Mat& imageData, size_t i, double& weight);
	void classifiers(const cv::Mat& img, std::vector<character>& listSortChars);
#endif //USE_SVM_CLASSIFY
};
#endif //HOG_SVM_CLASSIFIERS_H