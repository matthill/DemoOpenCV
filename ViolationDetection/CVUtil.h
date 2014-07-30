#ifndef _FTS_LV_CVUTIL_
#define _FTS_LV_CVUTIL_


#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "Ctracker.h"
#include "CVLine.h"
#include "ConnectedComponent.h"



void dilation(cv::Mat& img, int dilation_elem, int dilation_size);
void erosion(cv::Mat& img, int erosion_elem, int erosion_size);
//int intersectROI(cv::Mat img, cv::Mat roi_img, cv::Mat& res_img);
void overlayMap(const cv::Mat& img, cv::Mat& ol_img, cv::Mat map_img);
float meanQueue(const std::deque<float>& qu);
std::string getDensityStatus(float occupied_space, int& density_rate);
std::vector<cv::Rect> getContoursBoundingBoxes(const std::vector<std::vector<cv::Point>>& contours);
void examineContours(std::vector< std::vector<cv::Point> >& contours, int minSize, int maxSize);
void maxMat(const cv::Mat& A, const cv::Mat& B, cv::Mat& C);
float calculateLearningRate(float* list_values, float density, int density_rate, float max_learning_rate);

void detectLaneViolationVehicles(const cv::Mat& laneMap, int iMinSize, int iMaxSize, const std::vector<cv::Rect>& objBbs, const std::vector<cv::Point2d>& objCenters, std::vector<int> objAreas, std::vector<cv::Rect>& laneViolationBbs, std::vector<cv::Point2d>& laneViolatingCenters, std::vector<int>& laneViolatingAreas);

void detectLaneViolationVehicles(Line_<double> lineLane, CTracker &tracker, int iMinSize, int iMaxSize,
								 std::vector<cv::Rect> objBbs, std::vector<cv::Point2d> objCenters, std::vector<int> objAreas,
								 std::vector<cv::Rect>& laneViolationBbs, std::vector<cv::Point2d>& laneViolatingCenters, std::vector<int>& laneViolatingAreas);
void markViolations(cv::Mat& img, const std::vector<cv::Rect>& violatingCcs, cv::Scalar color);

float calOccupiedSpace(cv::Mat& fore, unsigned int roi_area, int threshold);
unsigned int countPixels(const cv::Mat& mat, int threshold);

void overlayContourAreas(cv::Mat& img, const cv::Mat& laneMap, const std::vector<cv::Point2d>& objCenters, const std::vector<int>& objAreas);
void overlayContourAreas(cv::Mat& img, Line_<double> line, const CTracker &tracker, const std::vector<int>& objAreas);
void extractConnectedComponents(const std::vector< std::vector<cv::Point> >& contours, std::vector<cv::Rect>& objRects, std::vector<cv::Point2d>& objCenters, std::vector<int>& objAreas);
int trackObjectsOpticalFlow(const cv::Mat& prevGray, const cv::Mat& gray, const std::vector<cv::Point2d>& prevTrackedPs, std::vector<cv::Point2d>& trackedPs, std::vector<cv::Point2d>& detectedPs, std::vector<cv::Rect>& objBbs);

void detectObjsInROI(const cv::Mat& mRoiImg, const std::vector<cv::Rect>& objBbs, const std::vector<cv::Point2d>& objCenters, const std::vector<int>& objAreas, std::vector<cv::Rect>& violatingVehicleBbs, std::vector<cv::Point2d>& violatingVehicleCenters, std::vector<int>& violatingVehicleAreas);
void detectObjsInROI(const cv::Mat& mRoiImg, const std::vector<cv::Rect>& objBbs, const std::vector<cv::Point2d>& objCenters, std::vector<cv::Rect>& violatingVehicleBbs, std::vector<cv::Point2d>& violatingVehicleCenters);
void detectObjsInROI(const cv::Mat& mRoiRSVImg, const CTracker& tracker, std::vector<cv::Rect>& objBbs, std::vector<cv::Rect>& violatingVehicleBbs, std::vector<cv::Point2d>& violatingVehicleCenters);

int intersectROI(const cv::Mat& img, const cv::Mat& roi_img, cv::Mat& res_img);
void drawTracks(cv::Mat& imgFrame, const CTracker& tracker);
int removeJunkDetections(std::vector<cv::Rect>& bbs, std::vector<cv::Point2d>& points, std::vector<int>& areas, int maxWidth, int maxHeight, double dMaxAspectRatio);

int countNewVehicles(const cv::Mat& imgPrevGrayFrame, const cv::Mat& imgGrayFrame, std::vector<cv::Point2d>& vehicleLaneViolatingCenters, std::vector<cv::Rect>& vehicleLaneViolationBbs, std::vector<cv::Point2d>& prevTrackedVehiclePs, std::vector<cv::Point2d>& trackedVehiclePs, cv::Size subPixWinSize, cv::TermCriteria termcrit);
bool checkRectsOverlap(cv::Rect rect1, cv::Rect rect2);
void unionRects(cv::Rect rect1, cv::Rect rect2, cv::Rect& unionRect);
void intersectRects(cv::Rect rect1, cv::Rect rect2, cv::Rect& intersectRect);

void checkCrossCountingLines(CTracker &tracker, const std::vector<cv::Point2d>& inNormLine, const std::vector<cv::Point2d>& outNormLine);
bool checkCrossALine(CTrack *track, std::vector<cv::Point2d> line);
bool checkCrossALine(CTrack *track, Line_<double> line);
double pointToLine(cv::Point2d p, cv::Point2d norm, cv::Point2d p0);
double pointToLine(cv::Point2d p, const std::vector<cv::Point2d>& line);
void translateListOfPoints(std::vector<cv::Point2d> &points, cv::Point2d v);
void perspectiveTransform2d(const std::vector<cv::Point2d>& objCenters, std::vector<cv::Point2d> &projObjCenters, const cv::Mat& homoTransform);

void detectBasedMovingDirection(CTracker &suspectsTracker, std::vector<int> &indices, Line_<double> outLine);
void detectCrossingDoubleLines(CTracker &tracker, std::vector<int> &indices, Line_<double> inLine, Line_<double>  outLine);
void detectCrossingSingleLines(CTracker &tracker, std::vector<int> &indices, Line_<double>  outLine);
bool intersectSegments(cv::Point2d o1, cv::Point2d p1, cv::Point2d o2, cv::Point2d p2, double &t1, double &t2);
bool checkSegmentsIntersection(cv::Point2d o1, cv::Point2d p1, cv::Point2d o2, cv::Point2d p2);
//bool intersectSegments(cv::Point2d o1, cv::Point2d p1, cv::Point2d o2, cv::Point2d p2);
cv::Rect getMaxContoursFromROIImage(const cv::Mat& img);
std::string getCurrentTimeInVideoAsString(float fFrameRate, int indexFrame);
std::string getCurrentTimeInVideoAsString(int timeAsSecond);
#endif