#ifndef _FTS_LV_LANEVIOLATION_H_
#define _FTS_LV_LANEVIOLATION_H_

#include "FTSVideoAlgorithm.h"
#include "CVLine.h"

//#define USE_LANE_MAP
#ifndef USE_LANE_MAP
#define USE_COUNTING_LINE
#endif //USE_LANE_MAP

class FTSLaneViolation: public FTSVideoAlgorithm {

private:
	cv::Size subPixWinSize;
	cv::TermCriteria termcrit;

protected:
	std::string strRoiImg;

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
	
	//tracker params
	float _dt;
	float _Accel_noise_mag;
	double _dist_thres;
	double _cos_thres;
	int _maximum_allowed_skipped_frames;
	int _max_trace_length;
	double _very_large_cost;

	bool bHardThreshold;
	Line_<double> lineCar, lineBike, lineTruck;
public:
	static std::string const className() { return "FTSLaneViolation"; }
	virtual cv::AlgorithmInfo* info() const;
	virtual void read(const cv::FileNode& fn);
	virtual void write(cv::FileStorage& fs) const;

	virtual void operator() (FTSCamera camInfo, std::queue<ViolationEvent>& taskQueue);

	FTSLaneViolation();
	~FTSLaneViolation();
};

#endif