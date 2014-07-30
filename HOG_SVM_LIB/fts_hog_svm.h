#ifndef _FTS_HOG_SVM_HH_
#define _FTS_HOG_SVM_HH_
#include <vector>
#include "svmlight_classify.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "Hog_svm_classifiers.h"
#include "config.h"
using namespace cv;

//struct SVM_HOG_EXPORT character{
//	std::string strLabel;
//	double confidence;
//};

struct FTS_HOGSVMDetectionROI
{
	// scale(size) of the bounding box
	double scale;
	// set of requrested locations to be evaluated
	vector<cv::Point> locations;
	// vector that will contain confidence values for each location
	vector<double> confidences;
};
struct SVM_HOG_EXPORT FTS_SVM_HOGDescriptor
{
public:
	enum { L2Hys = 0 };
	enum { DEFAULT_NLEVELS = 64 };
	enum CLASSIFIER_SET{ CUSTOM = -1, DIGIT_ONLY = 0, LETTER_ONLY = 1, DIGIT_LETTER = 2 };
	FTS_SVM_HOGDescriptor() : winSize(64, 128), blockSize(16, 16), blockStride(8, 8),
		cellSize(8, 8), nbins(9), derivAperture(1), winSigma(-1),
		histogramNormType(FTS_SVM_HOGDescriptor::L2Hys), L2HysThreshold(0.2), gammaCorrection(true),
		nlevels(FTS_SVM_HOGDescriptor::DEFAULT_NLEVELS)
	{}

	FTS_SVM_HOGDescriptor(Size _winSize, Size _blockSize, Size _blockStride,
		Size _cellSize, int _nbins, int _derivAperture = 1, double _winSigma = -1,
		int _histogramNormType = FTS_SVM_HOGDescriptor::L2Hys,
		double _L2HysThreshold = 0.2, bool _gammaCorrection = false,
		int _nlevels = FTS_SVM_HOGDescriptor::DEFAULT_NLEVELS)
		: winSize(_winSize), blockSize(_blockSize), blockStride(_blockStride), cellSize(_cellSize),
		nbins(_nbins), derivAperture(_derivAperture), winSigma(_winSigma),
		histogramNormType(_histogramNormType), L2HysThreshold(_L2HysThreshold),
		gammaCorrection(_gammaCorrection), nlevels(_nlevels)
	{}

	FTS_SVM_HOGDescriptor(const String& filename)
	{
		load(filename);
	}

	FTS_SVM_HOGDescriptor(const FTS_SVM_HOGDescriptor& d)
	{
		d.copyTo(*this);
	}

	virtual ~FTS_SVM_HOGDescriptor() {}

	size_t getDescriptorSize() const;
	bool checkDetectorSize() const;
	double getWinSigma() const;

	virtual void setSVMDetector(InputArray _svmdetector);

	virtual bool read(FileNode& fn);
	virtual void write(FileStorage& fs, const String& objname) const;

	virtual bool load(const String& filename, const String& objname = String());
	virtual void save(const String& filename, const String& objname = String()) const;
	virtual void copyTo(FTS_SVM_HOGDescriptor& c) const;

	virtual void compute(const Mat& img,
		CV_OUT vector<float>& descriptors,
		Size winStride = Size(), Size padding = Size(),
		const vector<Point>& locations = vector<Point>()) const;
	//with found weights output
	virtual void detect(const Mat& img, CV_OUT vector<Point>& foundLocations,
		CV_OUT vector<double>& weights,
		double hitThreshold = 0, Size winStride = Size(),
		Size padding = Size(),
		const vector<Point>& searchLocations = vector<Point>(),
		std::string& className = std::string()) const;
	//without found weights output
	virtual void detect(const Mat& img, CV_OUT vector<Point>& foundLocations,
		double hitThreshold = 0, Size winStride = Size(),
		Size padding = Size(),
		const vector<Point>& searchLocations = vector<Point>()) const;
	//with result weights output
	virtual void detectMultiScale(const Mat& img, CV_OUT vector<Rect>& foundLocations,
		CV_OUT vector<double>& foundWeights,
		cv::Size minObjectSize, cv::Size maxObjectSize,
		double hitThreshold = 0,
		double scale = 1.05,
		double finalThreshold = 2.0, bool useMeanshiftGrouping = false) const;
	//without found weights output
	virtual void detectMultiScale(const Mat& img, CV_OUT vector<Rect>& foundLocations,
		cv::Size minObjectSize, cv::Size maxObjectSize,
		double hitThreshold = 0, double scale = 1.05,
		double finalThreshold = 2.0, bool useMeanshiftGrouping = false) const;
	virtual bool loadModelFormPath(const std::string& strPath);
	virtual void computeGradient(const Mat& img, CV_OUT Mat& grad, CV_OUT Mat& angleOfs,
		Size paddingTL = Size(), Size paddingBR = Size()) const;

	static vector<float> getDefaultPeopleDetector();
	static vector<float> getDaimlerPeopleDetector();

	Size winSize;
	Size blockSize;
	Size blockStride;
	Size cellSize;
	int nbins;
	int derivAperture;
	double winSigma;
	int histogramNormType;
	double L2HysThreshold;
	bool gammaCorrection;
	vector<float> svmDetector;
	int nlevels;

	// HOANG: Additional properties
	cv::Size winStride, padding;
	std::vector<SvmLightClassify*> listModelSvms;
	std::vector<std::string> listStrLabel;
	std::vector<int> listModelIdx;
	std::map<std::string, int> classIdxMap;
	int indexModel;

	void getHogParameters(cv::HOGDescriptor& _hog, cv::Size& _winStride, cv::Size& _padding);
	void setHogParameters(cv::Size _winSize, cv::Size _blockSize, cv::Size _blockStride, cv::Size _cellSize, cv::Size _winStride, cv::Size _padding);
	void setHogParameters(const cv::HOGDescriptor& hog, cv::Size _winStride, cv::Size _padding);
	void loadModelFromFile(const std::string& _modelFileName);
	void calculateFeaturesFromInput(const cv::Mat& imgTest, std::vector<float>& featureVector);
	double classify(int indexModel, const std::vector<float>& featureVectorSample);
	void setClassifierIdx(CLASSIFIER_SET classifierSetType, const std::vector<std::string>& customClassLabels);
	void customMultiClassify(const cv::Mat& imgTest, std::vector<character>& listSortChars);
	void singleClassify(const cv::Mat& imgTest, const std::string& posClassName, const std::string& negClassName, character& resChar);
	void singleClassify(const std::vector<float>& featureVector, const std::string& posClassName, const std::string& negClassName, character& resChar);
	void multiClassify(const cv::Mat& imgTest, std::vector<character>& listSortChars, bool bAppend = false);
	void multiClassify(const std::vector<float>& featureVector, std::vector<character>& listSortChars, bool bAppend);

	// evaluate specified ROI and return confidence value for each location
	void detectROI(const cv::Mat& img, const vector<cv::Point> &locations,
		CV_OUT std::vector<cv::Point>& foundLocations, CV_OUT std::vector<double>& confidences,
		double hitThreshold = 0, cv::Size winStride = Size(),
		cv::Size padding = Size()) const;

	// evaluate specified ROI and return confidence value for each location in multiple scales
	void detectMultiScaleROI(const cv::Mat& img,
		CV_OUT std::vector<cv::Rect>& foundLocations,
		std::vector<FTS_HOGSVMDetectionROI>& locations,
		double hitThreshold = 0,
		int groupThreshold = 0) const;

	// read/parse Dalal's alt model file
	void readALTModel(std::string modelfile);
	void groupRectangles(vector<cv::Rect>& rectList, vector<double>& weights, int groupThreshold, double eps) const;
};

#endif //_FTS_HOG_HH_
