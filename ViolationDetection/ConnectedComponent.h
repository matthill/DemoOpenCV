#ifndef _FTS_LV_CC_
#define _FTS_LV_CC_


#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "CVLine.h"
#include "Ctracker.h"

struct ConnectedComponent {
	cv::Point2d center;
	int area;
	cv::Rect boundingBox;
};

class ListConnectComponent {
public:
	//std::vector<ConnectedComponent> listCC;
	cv::Point2d translation;
	float fScaleRatio;
	std::vector<cv::Point2d> listCenters;
	std::vector<int> listAreas;
	std::vector<cv::Rect> listBbs;
	ListConnectComponent() {}
	void setTransformationParameter(cv::Point _translation, float _fScaleRatio);
	size_t getSize();
	bool isEmpty();
	void ListConnectComponent::push_back(const cv::Point2d &center, const int &area, const cv::Rect& boundingBox);
	void clear();
	//ConnectedComponent operator [] (int i);
	std::vector<cv::Point2d>& getListCenter();
	void  cloneListCenter(std::vector<cv::Point2d>& _listCenters);
	std::vector<int>& getListArea();
	void cloneListArea(std::vector<int>& _listAreas);
	std::vector<cv::Rect>& getListBoundingBox();
	void cloneListBoundingBox(std::vector<cv::Rect>& _listBbs);
	void extractConnectedComponentsFormContours(const std::vector< std::vector<cv::Point> > &contours);
	int removeJunkDetections(int maxWidth, int maxHeight, double dMaxAspectRatio);
	void detectLaneViolationVehicles(const cv::Mat& laneMap, int iMinSize, int iMaxSize, ListConnectComponent &ccLaneViolating, bool isBike);
	void detectLaneViolationVehicles(const cv::Mat &laneMap, int iMinSize, int iMaxSize, ListConnectComponent &ccLaneViolating);
	void detectLaneViolationVehicles(const Line_<double>& lineLane, CTracker &tracker, int iMinSize, int iMaxSize, ListConnectComponent &ccLaneViolation);

	void getOriginalObject(int index, cv::Rect& object);
	void getListOriginalObjects(std::vector<cv::Rect>& objects);
};
#endif