#ifndef _FTS_LV_LANEVIOLATIONREGION_H_
#define _FTS_LV_LANEVIOLATIONREGION_H_

#include "FTSVideoAlgorithm.h"
#include "CVLine.h"

//#define TUNING_LANE_VIOLATION
//#define MEASURE_TIME_LANEVIOLATIONREGION // calc time 
class FTSLaneViolationRegion: public FTSVideoAlgorithm {

private:
	cv::Size subPixWinSize;
	cv::TermCriteria termcrit;

protected:
	std::string strRoiImg;
	std::string strRoiCarImg;
	std::string strRoiBikeImg;

	float fVarThreshold;
	float fMaxLearningRate;
	float fScaleRatio;

	int iHistory;
	int iTrainingFrame;
	int iInterval;
	int iBikeMinSize;
	int iBikeMaxSize;
	int iCarMinSize;
	int iCarMaxSize;
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
	static std::string const className() { return "FTSLaneViolationRegion"; }
	virtual cv::AlgorithmInfo* info() const;
	virtual void read(const cv::FileNode& fn);
	virtual void write(cv::FileStorage& fs) const;

	virtual void operator() (FTSCamera camInfo);

	FTSLaneViolationRegion();
	~FTSLaneViolationRegion();
};

#endif