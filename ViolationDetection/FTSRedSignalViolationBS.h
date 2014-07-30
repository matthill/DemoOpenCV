#ifndef _FTS_LV_REDSIGNALVIOLATIONBS_
#define _FTS_LV_REDSIGNALVIOLATIONBS_


#include "FTSVideoAlgorithm.h"
#include "CVLight.h"
#include "CVLine.h"

//#define MEASURE_TIME_REDLIGHT
//#define TUNNING_RED_LIGHT
class FTSRedSignalViolationBS:
	public FTSVideoAlgorithm {

protected:
	std::string strRoiImg;
	std::string strStopRoiImage;
	float fScaleRatio;
	float fVarThreshold;
	float fMaxLearningRate;

	// Traffic light
	Light lightRed, lightGreen, lightYellow;
	cv::Rect rectTrafficLight;
	Line_<double> inLine, outLine;

	int iHistory;
	int iTrainingFrame;
	int iInterval;
	int iContourMinArea;
	int iContourMaxArea;
	// 1  Prefer direction
	// 2  Double-line
	int detectionMode;

	bool bHardThreshold;
	//tracker params
	float _dt;
	float _Accel_noise_mag;
	double _dist_thres;
	double _cos_thres;
	int _maximum_allowed_skipped_frames;
	int _max_trace_length;
	double _very_large_cost;
public:
	static std::string const className() { return "FTSRedSignalViolationBS"; }
	virtual cv::AlgorithmInfo* info() const;
	virtual void read(const FileNode& fn);
	virtual void write(FileStorage& fs) const;
	virtual void operator() (FTSCamera camInfo);


	FTSRedSignalViolationBS();
	~FTSRedSignalViolationBS();
	void transformLightIntoNewAxis(Light& light, cv::Point transformMat);
	bool checkRedSignal(bool isRed, const cv::Mat& redTL, Light& red_light, const cv::Mat& yelTL, Light& yel_light, const cv::Mat& grnTL, Light& grn_light);
};

#endif