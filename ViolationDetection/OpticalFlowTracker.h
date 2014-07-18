#ifndef _FTS_LV_OPTICAL_FLOW_TRACKER_
#define _FTS_LV_OPTICAL_FLOW_TRACKER_

#include "opencv2/opencv.hpp"
#include "ConnectedComponent.h"

class OpticalFlowTracker {
public:
	cv::Mat imgPreGray, imgCurrentGray;
	std::vector<cv::Point2d> prevTrackedPs, trackedPs;
	std::vector<cv::Rect> newBbsDetected;
	cv::Size subPixWinSize;
	cv::TermCriteria termcrit;
	OpticalFlowTracker();
	void trackObjectsOpticalFlow(std::vector<cv::Point2d>& detectedPs, std::vector<cv::Rect>& objBbs);
	void Update(cv::Mat& img, ListConnectComponent& ccListCC);
	std::vector<cv::Rect>& getNewBoudingBox();
};

#endif//_FTS_LV_OPTICAL_FLOW_TRACKER_