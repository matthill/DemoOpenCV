#ifndef _FTS_LV_VEHICLECOUNTING_
#define _FTS_LV_VEHICLECOUNTING_


#include "FTSVideoAlgorithm.h"

struct EqPredicate {
	int thresholdX;
	int thresholdY;
	EqPredicate(int thx, int thy) {
		this->thresholdX = thx;
		this->thresholdY = thy;
	}
	EqPredicate() {}
	bool operator()(const cv::Point2d &p1, const cv::Point2d &p2) {
		if (std::abs(p1.x - p2.x) < this->thresholdX && std::abs(p1.y - p2.y) < this->thresholdY) {
			return true;
		}
		return false;
	}
};

class FTSVehicleCounting:
	public FTSVideoAlgorithm {

private:
	cv::Size subPixWinSize;
	cv::TermCriteria termcrit;
	//void vehicleTracking(cv::Mat imgGrayFrame, cv::Mat imgPrevGrayFrame, std::vector<cv::Rect> violatingVehicleBbs, std::vector<cv::Point2f> violatingVehicleCenters, std::vector<cv::Point2f> prevTrackedPs) {
	//	
	//	std::vector<cv::Point2f> trackedPs;

	//	int newTrackNum = 0;

	//	if (!violatingVehicleBbs.empty()) {
	//		// If there are NO tracks in previous frame
	//		if (prevTrackedPs.empty()) {
	//			trackedPs = violatingVehicleCenters;
	//			// Replace centers by corner points around them
	//			cornerSubPix(imgGrayFrame, trackedPs, this->subPixWinSize, cv::Size(-1, -1), termcrit);
	//			newTrackNum = int(trackedPs.size());
	//		} else // If there are tracks in previous frame
	//		{
	//			newTrackNum = trackObjectsOpticalFlow(imgPrevGrayFrame, imgGrayFrame, prevTrackedPs, trackedPs, violatingVehicleCenters, violatingVehicleBbs);
	//		}
	//		//no violation - clear all tracking info
	//	} else {
	//		trackedPs.clear();
	//		prevTrackedPs.clear();
	//	}

	//}

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

	bool bNight;

	std::vector<cv::Point2d> countingLine;

	std::vector<cv::Point2d> vProjQuad;
	std::vector<cv::Point2d> vProjRect;
public:
	virtual cv::AlgorithmInfo* info() const;

	virtual void operator() (FTSCamera camInfo);

	FTSVehicleCounting();
	~FTSVehicleCounting();
};

#endif