#include "OpticalFlowTracker.h"

OpticalFlowTracker::OpticalFlowTracker() {
	subPixWinSize = cv::Size(30, 30);
	termcrit.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;
	termcrit.maxCount = 5;
	termcrit.epsilon = 0.3;
}
void OpticalFlowTracker::trackObjectsOpticalFlow(std::vector<cv::Point2d>& detectedPs, std::vector<cv::Rect>& objBbs) {
	std::vector<cv::Point2d> predPs;
	std::vector<uchar> status;
	std::vector<float> err;
	cv::Size winSize(31, 31);
	cv::TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 5, 0.3);
	//std::cout << "Size1 " << prevTrackedPs.size() << "; Size2 " << predPs.size() << endl;
	std::vector<Point2f> tmpPrevTracks, tmpPredPs;
	cv::Mat(prevTrackedPs).copyTo(tmpPrevTracks);
	calcOpticalFlowPyrLK(imgPreGray, imgCurrentGray, tmpPrevTracks, tmpPredPs, status, err, winSize, 3, termcrit, 0, 0.001);
	cv::Mat(tmpPredPs).convertTo(predPs, cv::Mat(predPs).type());


	int ind = 0;
	std::vector<cv::Point2d>::iterator ip = detectedPs.begin();
	std::vector<cv::Rect>::iterator it = objBbs.begin();
	for (; it != objBbs.end() || ip != detectedPs.end(); ) {
		bool isNewTrack = true;
		for (int j = 0; j < predPs.size(); j++) {
			if (it->contains(predPs[j])) {
				isNewTrack = false;
				trackedPs.push_back(predPs[j]);
				break;
			}
		}
		if (isNewTrack) {
			trackedPs.push_back(detectedPs[ind]);
			newBbsDetected.push_back(objBbs[ind]);
			ind++;
			it++;
			ip++;
		} else {
			it = objBbs.erase(it);
			ip = detectedPs.erase(ip);
		}
	}

}
void OpticalFlowTracker::Update(cv::Mat& img, ListConnectComponent& ccListCC) {
	
	if (imgPreGray.empty()) {
		img.copyTo(imgPreGray);
	
		return;
	}
	newBbsDetected.clear();
	img.copyTo(imgCurrentGray);
	if (ccListCC.getSize() == 0) {

		if (prevTrackedPs.empty()) {
			//continue;
		} else {
			trackedPs.clear();
			prevTrackedPs.clear();
		}
	} else // If violations are detected
	{
		// If there are NO tracks in previous frame
		if (prevTrackedPs.empty()) {
			ccListCC.cloneListCenter(trackedPs);
			ccListCC.cloneListBoundingBox(newBbsDetected);
			// Replace centers by corner points around them
			std::vector<Point2f> tmpPs;
			//cv::Mat(trackedPs).convertTo(tmpPs, cv::Mat(tmpPs).type());
			cv::Mat(trackedPs).copyTo(tmpPs);
			cornerSubPix(imgPreGray, tmpPs, subPixWinSize, Size(-1, -1), termcrit);
			cv::Mat(tmpPs).convertTo(trackedPs, cv::Mat(trackedPs).type());

		} else // If there are tracks in previous frame
		{
			std::vector<cv::Point2d> objCenters;
			std::vector<cv::Rect> objBbs;
			std::vector<int> objAreas;
			ccListCC.cloneListCenter(objCenters);
			ccListCC.cloneListBoundingBox(objBbs);
			ccListCC.cloneListArea(objAreas);
			trackObjectsOpticalFlow(objCenters, objBbs);
			objCenters.clear();
			objBbs.clear();
			objAreas.clear();
		}
	}
	prevTrackedPs = trackedPs;
	imgCurrentGray.copyTo(imgPreGray);
}
std::vector<cv::Rect>& OpticalFlowTracker::getNewBoudingBox() {
	return newBbsDetected;
}
