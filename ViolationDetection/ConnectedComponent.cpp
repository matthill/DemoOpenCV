#include "ConnectedComponent.h"
#include "CVUtil.h"

#if _MSC_VER <= 1600
#include "util.h"
#endif

size_t ListConnectComponent::getSize() {
	return listCenters.size();
}
bool ListConnectComponent::isEmpty() {
	return listCenters.empty();
}
void ListConnectComponent::push_back(const cv::Point2d &center, const int &area, const cv::Rect& boundingBox) {
	listCenters.push_back(center);
	listAreas.push_back(area);
	listBbs.push_back(boundingBox);
}
void ListConnectComponent::clear() {
	listCenters.clear();
	listAreas.clear();
	listBbs.clear();
}

std::vector<cv::Point2d>& ListConnectComponent::getListCenter() {
	return listCenters;
}
void ListConnectComponent::cloneListCenter(std::vector<cv::Point2d>& _listCenters) {
	_listCenters = listCenters;
}
std::vector<int>& ListConnectComponent::getListArea() {
	return listAreas;
}
void ListConnectComponent::cloneListArea(std::vector<int>& _listAreas) {
	_listAreas = listAreas;
}

std::vector<cv::Rect>& ListConnectComponent::getListBoundingBox() {
	return listBbs;
}
void ListConnectComponent::cloneListBoundingBox(std::vector<cv::Rect>& _listBbs) {
	_listBbs = listBbs;
}
void ListConnectComponent::extractConnectedComponentsFormContours(const std::vector< std::vector<cv::Point> > &contours) {
	clear();
	for (size_t i = 0; i < contours.size(); i++) {
		cv::Rect bb = cv::boundingRect(contours[i]);
		cv::Point c;
		c.x = bb.x + int(std::round(bb.width / 2));
		c.y = bb.y + int(std::round(bb.height / 2));
		int area = int(contourArea(contours[i]));

		push_back(c, area, bb);
	}
}
int ListConnectComponent::removeJunkDetections(int maxWidth, int maxHeight, double dMaxAspectRatio) {

	std::vector<cv::Rect>::iterator itBbs = listBbs.begin();
	std::vector<cv::Point2d>::iterator itPts = listCenters.begin();
	std::vector<int>::iterator itAr = listAreas.begin();
	int iNumRemovedBbs = 0;

	for (; itBbs != listBbs.end() && itPts != listCenters.end() && itAr != listAreas.end();) {
		if (itBbs->width > maxWidth || itBbs->height > maxHeight || itBbs->height / itBbs->width > dMaxAspectRatio) {
			itBbs = listBbs.erase(itBbs);
			itPts = listCenters.erase(itPts);
			itAr = listAreas.erase(itAr);
			iNumRemovedBbs++;
		} else {
			itBbs++;
			itPts++;
			itAr++;
		}
	}
	return iNumRemovedBbs;
}
void ListConnectComponent::detectLaneViolationVehicles(const cv::Mat &laneMap, int iMinSize, int iMaxSize, ListConnectComponent &ccLaneViolating, bool isBike) {
	float k = 1.0 / 2.0;
	for (int i = 0; i < listAreas.size(); i++) {
		int conArea = listAreas[i];

		if (laneMap.at<uchar>(listCenters[i]) > 0) {
			if (conArea > iMinSize && conArea <= iMaxSize && conArea > int(k*listBbs[i].area())) {
				if (isBike){
					float ratio = float(listBbs[i].height) / listBbs[i].width;
					if (ratio >= 0.9){
						ccLaneViolating.push_back(listCenters[i], listAreas[i], listBbs[i]);
					}
				}
				else{
					ccLaneViolating.push_back(listCenters[i], listAreas[i], listBbs[i]);
				}
			}
		}
	}
}
void ListConnectComponent::detectLaneViolationVehicles(const cv::Mat &laneMap, int iMinSize, int iMaxSize, ListConnectComponent &ccLaneViolating){
	detectLaneViolationVehicles(laneMap, iMinSize, iMaxSize, ccLaneViolating, false);
}
void ListConnectComponent::detectLaneViolationVehicles(const Line_<double>& lineLane, CTracker &tracker, int iMinSize, int iMaxSize, ListConnectComponent &ccLaneViolation) {
	for (size_t i = 0; i < tracker.tracks.size(); i++) {

		// Igore caught tracks

		size_t traceLength = tracker.tracks[i]->trace.size();

		if (traceLength < 3) {
			continue;
		}
		bool isIntersect = checkSegmentsIntersection(lineLane.start, lineLane.end, tracker.tracks[i]->trace[0], tracker.tracks[i]->trace[traceLength - 1]);
		if (true) {
			int vehicleID = tracker.assignment[i];
			if (vehicleID > -1) {
				int conArea = listAreas[vehicleID];
				ccLaneViolation.push_back(listCenters[vehicleID], listAreas[vehicleID], listBbs[vehicleID]);
			}
		}
	}
}
