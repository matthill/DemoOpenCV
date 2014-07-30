#ifndef _FTS_LV_PLATESCANNER_H_
#define _FTS_LV_PLATESCANNER_H_

#include "FTSVideoAlgorithm.h"
#include "CVLine.h"

//#define TUNNING_PLATE_SCANNER
//#define MEASURE_TIME_PLATE_SCANNER // calc time 
class FTSPlateScanner: public FTSVideoAlgorithm {

private:
	cv::Size subPixWinSize;
	cv::TermCriteria termcrit;

protected:
	std::string strRoiImg;
	std::string strRoiVehicleImg;

	float fVarThreshold;
	float fMaxLearningRate;
	float fScaleRatio;

	int iHistory;
	int iTrainingFrame;
	int iInterval;
	int iContourMinArea;
	int iContourMaxArea;

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
	static std::string const className() { return "FTSPlateScanner"; }
	virtual cv::AlgorithmInfo* info() const;
	virtual void read(const cv::FileNode& fn);
	virtual void write(cv::FileStorage& fs) const;

	virtual void operator() (FTSCamera camInfo);

	FTSPlateScanner();
	~FTSPlateScanner();
};

#endif