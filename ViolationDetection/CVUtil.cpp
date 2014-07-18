#ifdef _MSC_VER
#define snprintf _snprintf_s
#endif


#include "util.h"
#include "CVUtil.h"
#include <string>

const int DENSITY_LEVEL[5] = {0, 25, 65, 80, 100};
const std::string DENSITY_TYPE[] = {"Free", "Uncrowded", "Crowded", "Too crowded"};


void drawTracks(cv::Mat& imgFrame, const CTracker& tracker) {
	cv::Scalar Colors[] = {cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 255), cv::Scalar(255, 127, 255), cv::Scalar(127, 0, 255), cv::Scalar(127, 0, 127)};
	std::cout << tracker.tracks.size() << endl;

	for (int i = 0; i<tracker.tracks.size(); i++) {
		if (tracker.tracks[i]->trace.size()>1) {
			for (int j = 0; j < tracker.tracks[i]->trace.size() - 1; j++) {
				std::cout << "Trace " << tracker.tracks[i]->trace[j] << std::endl;
				line(imgFrame, tracker.tracks[i]->trace[j], tracker.tracks[i]->trace[j + 1], Colors[tracker.tracks[i]->track_id % 9], 2, CV_AA);
			}
		}
	}
}

void dilation(cv::Mat& img, int dilation_elem, int dilation_size) {
	cv::Mat element = cv::getStructuringElement(dilation_elem,
												cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
												cv::Point(dilation_size, dilation_size));

	/// Apply the dilation operation
	dilate(img, img, element);

}

void erosion(cv::Mat& img, int erosion_elem, int erosion_size) {
	cv::Mat element = getStructuringElement(erosion_elem,
											cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
											cv::Point(erosion_size, erosion_size));

	/// Apply the erosion operation
	erode(img, img, element);

}

void overlayMap(const cv::Mat& img, cv::Mat& ol_img, const cv::Mat& map_img) {
	unsigned int nRows = img.rows, nCols = img.cols;
	if (img.channels() == 3) {
		char color[] = {1, 0, 0};
		std::vector<cv::Mat> channels(3);
		for (unsigned i = 0; i < 3; i++) {
			if (color[i]) {
				channels[i] = 255 * map_img;
			} else {
				channels[i] = cv::Mat::zeros(nRows, nCols, CV_8U);
			}
		}
		cv::Mat tmp_img;
		cv::merge(channels, tmp_img);
		//Mat result;
		maxMat(img, tmp_img, ol_img);
	}
}

std::vector<cv::Rect> getContoursBoundingBoxes(const std::vector< std::vector<cv::Point> >& contours) {
	std::vector<cv::Rect> lBoundingBoxes(contours.size());
	for (int i = 0; i < contours.size(); i++) {
		lBoundingBoxes.push_back(cv::boundingRect(contours[i]));
	}
	return lBoundingBoxes;
}

void maxMat(const cv::Mat& A, const cv::Mat& B, cv::Mat&  C) {
	for (int i = 0; i < A.rows; i++) {
		const uchar* data1 = A.ptr<uchar>(i);
		const uchar* data2 = B.ptr<uchar>(i);
		uchar* data3 = C.ptr<uchar>(i);
		for (int j = 0; j < A.channels()*A.cols; j++) {
			/*uchar v1;
			v1 = *data1;
			uchar v2;
			v2 = *data2;*/
			*data3 = std::max(*data1, *data2);
			data3++;
			data1++;
			data2++;
		}
	}
}

void detectObjsInROI(const cv::Mat& mRoiRSVImg, const CTracker& tracker, std::vector<cv::Rect>& objBbs, std::vector<cv::Rect>& violatingVehicleBbs, std::vector<cv::Point2d>& violatingVehicleCenters) {
	for (int i = 0; i < tracker.tracks.size(); i++) {
		size_t iTraceLength = tracker.tracks[i]->trace.size();
		Point2d pTmplastTrace = tracker.tracks[i]->trace[iTraceLength - 1];
		Point2d pLastTrace;
		cv::Mat(pTmplastTrace).convertTo(cv::Mat(pLastTrace), cv::Mat(pTmplastTrace).type());
		if (mRoiRSVImg.at<uchar>(pLastTrace) > 0) {
			for (std::vector<cv::Rect>::iterator it = objBbs.begin(); it != objBbs.end();) {
				if ((*it).contains(pLastTrace)) {
					it = objBbs.erase(it);
					violatingVehicleBbs.push_back(*it);
				} else {
					it++;
				}
			}
		}
	}
}

void detectObjsInROI(const cv::Mat& mRoiImg, const std::vector<cv::Rect>& objBbs, const std::vector<cv::Point2d>& objCenters, std::vector<cv::Rect>& violatingVehicleBbs, std::vector<cv::Point2d>& violatingVehicleCenters) {
	for (int i = 0; i < objBbs.size(); i++) {
		//int conArea = objAreas[i];

		//std::cout << "Value: " << laneMap.at<uchar>(objCenters[i]) << "\n";
		if (mRoiImg.at<uchar>(objCenters[i]) > 0) {
			//logFile << objCenters[i].x << " " << objCenters[i].y << " " << objBbs[i].width << " " << objBbs[i].height << " " << objAreas[i] << "\n";
			violatingVehicleBbs.push_back(objBbs[i]);
			violatingVehicleCenters.push_back(objCenters[i]);
			//violatingVehicleAreas.push_back(objAreas[i]);
		}
	}
}

void detectObjsInROI(const cv::Mat& mRoiImg, const std::vector<cv::Rect>& objBbs, const std::vector<cv::Point2d>& objCenters, const std::vector<int>& objAreas, std::vector<cv::Rect>& violatingVehicleBbs, std::vector<cv::Point2d>& violatingVehicleCenters, std::vector<int>& violatingVehicleAreas) {
	for (int i = 0; i < objBbs.size(); i++) {
		int conArea = objAreas[i];

		//std::cout << "Value: " << laneMap.at<uchar>(objCenters[i]) << "\n";
		if (mRoiImg.at<uchar>(objCenters[i]) > 0) {
			//logFile << objCenters[i].x << " " << objCenters[i].y << " " << objBbs[i].width << " " << objBbs[i].height << " " << objAreas[i] << "\n";
			violatingVehicleBbs.push_back(objBbs[i]);
			violatingVehicleCenters.push_back(objCenters[i]);
			violatingVehicleAreas.push_back(objAreas[i]);
		}
	}
}

float meanQueue(const std::deque<float>& qu) {
	float sum_value = 0;
	for (int i = 0; i < qu.size(); i++) {
		sum_value += qu[i];
	}
	return sum_value / qu.size();
}

std::string getDensityStatus(float occupied_space, int& density_rate) {
	std::string density;
	if (occupied_space < DENSITY_LEVEL[1]) {
		density_rate = 0;
	} else if (occupied_space < DENSITY_LEVEL[2]) {
		density_rate = 1;
	} else if (occupied_space < DENSITY_LEVEL[3]) {
		density_rate = 2;
	} else {
		density_rate = 3;
	}

	return DENSITY_TYPE[density_rate];
}

void examineContours(std::vector< std::vector<cv::Point> >& contours, int minSize, int maxSize) {
	//printf("Remove small CCs\n");
	//bool large_contours_found = false;

	//ostringstream convert;
	for (std::vector< std::vector<cv::Point> >::iterator it = contours.begin(); it != contours.end();) {
		float contour_area = (float) contourArea(*it, 0);
		if (contour_area < minSize /*|| circle_area/contour_area > 2.3*/) {
			it = contours.erase(it);
		} else if (contourArea(*it, 0) > maxSize) {
			//large_contours_found = true;
			it++;
		} else {
			it++;
		}
	}

	//return large_contours_found;
}

float calculateLearningRate(float* list_values, float density, int density_rate, float max_learning_rate) {
	float lp_den = float(DENSITY_LEVEL[density_rate]);
	float up_den = float(DENSITY_LEVEL[density_rate + 1]);
	float lp = list_values[density_rate];
	float up = list_values[density_rate + 1];
	float y = lp + (up - lp)*(density - lp_den) / (up_den - lp_den);
	return max_learning_rate * exp(-y*y / 2) / sqrt(2 * PI);
}

void detectLaneViolationVehicles(const cv::Mat& laneMap, int iMinSize, int iMaxSize, const std::vector<cv::Rect>& objBbs, const std::vector<cv::Point2d>& objCenters, std::vector<int> objAreas, std::vector<cv::Rect>& laneViolationBbs, std::vector<cv::Point2d>& laneViolatingCenters, std::vector<int>& laneViolatingAreas) {
	for (int i = 0; i < objBbs.size(); i++) {
		int conArea = objAreas[i];

		if (laneMap.at<uchar>(objCenters[i]) > 0) {
			if (conArea > iMinSize && conArea <= iMaxSize) {
				laneViolationBbs.push_back(objBbs[i]);
				laneViolatingCenters.push_back(objCenters[i]);
				laneViolatingAreas.push_back(objAreas[i]);
			}
		}
	}
}

void detectLaneViolationVehicles(Line_<double> lineLane, CTracker &tracker, int iMinSize, int iMaxSize,
								 std::vector<cv::Rect> objBbs, std::vector<cv::Point2d> objCenters, std::vector<int> objAreas,
								 std::vector<cv::Rect>& laneViolationBbs, std::vector<cv::Point2d>& laneViolatingCenters, std::vector<int>& laneViolatingAreas) {
	for (size_t i = 0; i < tracker.tracks.size(); i++) {
		// Igore caught tracks

		size_t traceLength = tracker.tracks[i]->trace.size();

		if (traceLength < 3) {
			continue;
		}
		bool isIntersect = checkSegmentsIntersection(lineLane.start, lineLane.end, tracker.tracks[i]->trace[0], tracker.tracks[i]->trace[traceLength - 1]);
		if (isIntersect) {
			int vehicleID = tracker.assignment[i];
			if (vehicleID > -1) {
				int conArea = objAreas[vehicleID];
				if (conArea > iMinSize && conArea <= iMaxSize) {
					laneViolationBbs.push_back(objBbs[vehicleID]);
					laneViolatingCenters.push_back(objCenters[vehicleID]);
					laneViolatingAreas.push_back(objAreas[vehicleID]);
					cout << "Detect intersection area " << conArea << " minSize=" << iMinSize << " maxSize=" << iMaxSize << endl;
				}
			}
		}
	}
}

void markViolations(cv::Mat& img, const std::vector<cv::Rect>& violatingBbs, cv::Scalar color) {
	for (int i = 0; i < violatingBbs.size(); i++) {
		rectangle(img, violatingBbs[i], color, 2);
	}
}

float calOccupiedSpace(cv::Mat& fore, unsigned int roi_area, int threshold) {
	unsigned int N = fore.rows * fore.cols;

	unsigned int num_fore_pxl = countPixels(fore, 0);
	//countNonZero
	return (float) 100 * num_fore_pxl / roi_area;

}

unsigned int countPixels(const cv::Mat& mat, int threshold) {
	const uchar* p;
	unsigned int num_pxl = 0, nRows = mat.rows, nCols = mat.cols;
	if (mat.isContinuous()) {

		nCols *= nRows;
		nRows = 1;
	}
	for (unsigned int i = 0; i < nRows; ++i) {
		p = mat.ptr<uchar>(i);
		for (unsigned int j = 0; j < nCols; ++j) {
			if (p[j] > threshold)
				num_pxl++;
		}
	}
	return num_pxl;
}

void overlayContourAreas(cv::Mat& img, const cv::Mat& laneMap, const std::vector<cv::Point2d>& objCenters, const std::vector<int>& objAreas) {
	for (int i = 0; i < objCenters.size(); i++) {
		if (laneMap.at<uchar>(objCenters[i]) > 0) {

			std::string text;
			std::stringstream convert;
			convert << objAreas[i];
			text = convert.str();
			putText(img, text, objCenters[i], CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
		}
	}
}

void overlayContourAreas(cv::Mat& img, Line_<double> line, const CTracker &tracker, const std::vector<int> &objAreas) {
	for (int i = 0; i < tracker.tracks.size(); i++) {
		size_t n = tracker.tracks[i]->trace.size();
		if (tracker.tracks[i]->trace.size() > 2) {
			int ind = tracker.assignment[i];
			if (ind > -1 && checkSegmentsIntersection(line.start, line.end, tracker.tracks[i]->trace[0], tracker.tracks[i]->trace[n - 1])) {
				if (objAreas.size() == 0) continue;
				std::string text;
				std::stringstream convert;
				convert << objAreas[ind];
				text = convert.str();
				putText(img, text, tracker.tracks[i]->trace[n - 1], CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
			}
		}
	}
}

void extractConnectedComponents(const std::vector< std::vector<cv::Point> >& contours, std::vector<cv::Rect>& objRects, std::vector<cv::Point2d>& objCenters, std::vector<int>& objAreas) {
	for (int i = 0; i < contours.size(); i++) {
		cv::Rect bb = cv::boundingRect(contours[i]);
		cv::Point c;
		c.x = bb.x + int(std::round(bb.width / 2));
		c.y = bb.y + int(std::round(bb.height / 2));
		int area = int(contourArea(contours[i]));

		objRects.push_back(bb);
		objCenters.push_back(c);
		objAreas.push_back(area);
	}
}

int trackObjectsOpticalFlow(const cv::Mat& prevGray, const cv::Mat& gray, const std::vector<cv::Point2d>& prevTrackedPs, std::vector<cv::Point2d>& trackedPs, std::vector<cv::Point2d>& detectedPs, std::vector<cv::Rect>& objBbs) {
	std::vector<cv::Point2d> predPs;
	std::vector<uchar> status;
	std::vector<float> err;
	cv::Size winSize(31, 31);
	cv::TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 5, 0.3);
	//std::cout << "Size1 " << prevTrackedPs.size() << "; Size2 " << predPs.size() << endl;
	std::vector<Point2f> tmpPrevTracks, tmpPredPs;
	cv::Mat(prevTrackedPs).copyTo(tmpPrevTracks);
	calcOpticalFlowPyrLK(prevGray, gray, tmpPrevTracks, tmpPredPs, status, err, winSize, 3, termcrit, 0, 0.001);
	cv::Mat(tmpPredPs).convertTo(predPs, cv::Mat(predPs).type());

	int newTrackNum = 0;
	int ind = 0;
	std::vector<cv::Point2d>::iterator ip = detectedPs.begin();
	std::vector<cv::Rect>::iterator it = objBbs.begin();
	for (; it != objBbs.end() || ip != detectedPs.end(); ind++) {
		bool isNewTrack = true;
		for (int j = 0; j < predPs.size(); j++) {
			if (it->contains(predPs[j])) {
				isNewTrack = false;
				trackedPs.push_back(predPs[j]);
				break;
			}
		}
		if (isNewTrack) {
			newTrackNum++;
			trackedPs.push_back(detectedPs[ind]);
			ind++;
			it++;
			ip++;
		} else {
			it = objBbs.erase(it);
			ip = detectedPs.erase(ip);
		}

	}
	return newTrackNum;
}

int intersectROI(const cv::Mat& img, const cv::Mat& roi_img, cv::Mat& res_img) {
	res_img = img.mul(roi_img);
	return 1;
}

int removeJunkDetections(std::vector<cv::Rect>& bbs, std::vector<cv::Point2d>& points, std::vector<int>& areas, int maxWidth, int maxHeight, double dMaxAspectRatio) {
	std::vector<cv::Rect>::iterator itBbs = bbs.begin();
	std::vector<cv::Point2d>::iterator itPts = points.begin();
	std::vector<int>::iterator itAr = areas.begin();
	int iNumRemovedBbs = 0;

	for (; itBbs != bbs.end() && itPts != points.end() && itAr != areas.end();) {
		if (itBbs->width > maxWidth || itBbs->height > maxHeight || itBbs->height / itBbs->width > dMaxAspectRatio) {
			itBbs = bbs.erase(itBbs);
			itPts = points.erase(itPts);
			itAr = areas.erase(itAr);
			iNumRemovedBbs++;
		} else {
			itBbs++;
			itPts++;
			itAr++;
		}
	}
	return iNumRemovedBbs;
}
int countNewVehicles(const cv::Mat& imgPrevGrayFrame, const cv::Mat& imgGrayFrame, std::vector<cv::Point2d>&  vehicleLaneViolatingCenters, std::vector<cv::Rect>& vehicleLaneViolationBbs, std::vector<cv::Point2d>& prevTrackedVehiclePs, std::vector<cv::Point2d>& trackedVehiclePs, cv::Size subPixWinSize, cv::TermCriteria termcrit) {
	int newTrackNum = 0;
	if (vehicleLaneViolatingCenters.empty()) {
		if (prevTrackedVehiclePs.empty()) {
			//continue;
		} else {
			trackedVehiclePs.clear();
			prevTrackedVehiclePs.clear();
		}
	} else // If violations are detected
	{
		// If there are NO tracks in previous frame
		if (prevTrackedVehiclePs.empty()) {
			trackedVehiclePs = vehicleLaneViolatingCenters;
			// Replace centers by corner points around them
			std::vector<Point2f> tmpPs;
			//cv::Mat(trackedPs).convertTo(tmpPs, cv::Mat(tmpPs).type());
			cv::Mat(trackedVehiclePs).copyTo(tmpPs);
			cornerSubPix(imgGrayFrame, tmpPs, subPixWinSize, Size(-1, -1), termcrit);
			cv::Mat(tmpPs).convertTo(trackedVehiclePs, cv::Mat(trackedVehiclePs).type());
			newTrackNum = int(trackedVehiclePs.size());
		} else // If there are tracks in previous frame
		{
			newTrackNum = trackObjectsOpticalFlow(imgPrevGrayFrame, imgGrayFrame, prevTrackedVehiclePs, trackedVehiclePs, vehicleLaneViolatingCenters, vehicleLaneViolationBbs);
		}
	}
	return newTrackNum;
}

bool checkRectsOverlap(cv::Rect rect1, cv::Rect rect2) {
	int minx1 = rect1.x;
	int maxx1 = rect1.x + rect1.width;
	int miny1 = rect1.y;
	int maxy1 = rect1.y + rect1.height;

	int minx2 = rect2.x;
	int maxx2 = rect2.x + rect2.width;
	int miny2 = rect2.y;
	int maxy2 = rect2.y + rect2.height;

	return ((maxx2 - minx1)*(minx2 - maxx1) < 0 && (maxy2 - miny1)*(miny2 - maxy1) < 0);
}

void intersectRects(cv::Rect rect1, cv::Rect rect2, cv::Rect& intersectRect) {
	int minx1 = rect1.x;
	int maxx1 = rect1.x + rect1.width;
	int miny1 = rect1.y;
	int maxy1 = rect1.y + rect1.height;

	int minx2 = rect2.x;
	int maxx2 = rect2.x + rect2.width;
	int miny2 = rect2.y;
	int maxy2 = rect2.y + rect2.height;

	intersectRect.x = max(minx1, minx2);
	intersectRect.y = max(miny1, miny2);

	int maxx, maxy;

	maxx = min(maxx1, maxx2);
	maxy = min(maxy1, maxy2);

	intersectRect.width = maxx - intersectRect.x + 1;
	intersectRect.height = maxy - intersectRect.y + 1;
}

void unionRects(cv::Rect rect1, cv::Rect rect2, cv::Rect& unionRect) {

	int minx1 = rect1.x;
	int maxx1 = rect1.x + rect1.width;
	int miny1 = rect1.y;
	int maxy1 = rect1.y + rect1.height;

	int minx2 = rect2.x;
	int maxx2 = rect2.x + rect2.width;
	int miny2 = rect2.y;
	int maxy2 = rect2.y + rect2.height;

	unionRect.x = std::min(minx1, minx2);
	unionRect.y = std::min(miny1, miny2);

	int maxx, maxy;

	maxx = max(maxx1, maxx2);
	maxy = max(maxy1, maxy2);

	unionRect.width = maxx - unionRect.x + 1;
	unionRect.height = maxy - unionRect.y + 1;
}

void checkCrossCountingLines(CTracker &tracker, const std::vector<cv::Point2d>& inNormLine, const std::vector<cv::Point2d>& outNormLine) {
	for (int i = 0; i < tracker.tracks.size(); i++) {
		bool isPos = false;
		bool isNeg = false;
		if (!tracker.tracks[i]->isIn) {
			tracker.tracks[i]->isIn = checkCrossALine(tracker.tracks[i], inNormLine);

			/*for (int j = 0; j < tracker.tracks[i]->trace.size(); j++)
			{
			cv::Point2d p = tracker.tracks[i]->trace[j];
			if (!isPos && pointToLine(p, inNormLine) > 0)
			{
			isPos = true;
			}
			else if (!isNeg && pointToLine(p, inNormLine) < 0)
			{
			isNeg = true;
			}
			if (isPos && isNeg)
			{
			tracker.tracks[i]->isIn = true;
			break;
			}
			}*/
		}
		if (!tracker.tracks[i]->isOut) {
			tracker.tracks[i]->isOut = checkCrossALine(tracker.tracks[i], outNormLine);
		}
	}
}

void detectCrossingDoubleLines(CTracker &tracker, std::vector<int> &indices, Line_<double> inLine, Line_<double>  outLine) {
	for (size_t i = 0; i < tracker.tracks.size(); i++) {
		// Igore caught tracks

		size_t traceLength = tracker.tracks[i]->trace.size();

		if (tracker.tracks[i]->isCaught || traceLength < 3) {
			continue;
		}
		bool bIn = false;
		bool bOut = false;

		//bOut = checkCrossALine(tracker.tracks[i], outLine);
		bOut = checkSegmentsIntersection(outLine.start, outLine.end, tracker.tracks[i]->trace[0], tracker.tracks[i]->trace[traceLength - 1]);
		if (bOut) {
			double t1, t2;
			intersectSegments(inLine.start, inLine.end, tracker.tracks[i]->trace[0], tracker.tracks[i]->trace[traceLength - 1], t1, t2);
			if (t1 >= 0 && t1 <= 1 && t2 <= 0) {
				bIn = true;
			}
		}

		if (bIn && bOut) {
			indices.push_back(i);
		}
		//if (!tracker.tracks[i]->isIn) {
		//	//tracker.tracks[i]->isIn = checkCrossALine(tracker.tracks[i], inLine);
		//	tracker.tracks[i]->isIn = intersectSegments(inLine.start, inLine.end, tracker.tracks[i]->trace[0], tracker.tracks[i]->trace[traceLength - 1]);
		//}
		//if (!tracker.tracks[i]->isOut) {
		//	tracker.tracks[i]->isOut = checkCrossALine(tracker.tracks[i], outLine);
		//}
		//if (tracker.tracks[i]->isIn && tracker.tracks[i]->isOut) {
		//	indices.push_back(i);
		//	//tracker.tracks[i]->isCaught = true;
		//}		
	}
}

bool checkCrossALine(CTrack *track, Line_<double> line) {
	std::vector<cv::Point2d> points;
	points.push_back(cv::Point2d(line.start.x, line.start.y));
	points.push_back(cv::Point2d(line.end.x, line.end.y));
	return checkCrossALine(track, points);
}

bool checkCrossALine(CTrack *track, std::vector<cv::Point2d> line) {
	bool isPos = false;
	bool isNeg = false;
	bool isCross = false;
	cv::Point2d firstTrace = track->trace[0];
	cv::Point2d lastTrace = track->trace[track->trace.size() - 1];
	double s1 = pointToLine(firstTrace, line);
	double s2 = pointToLine(lastTrace, line);
	if (s1*s2 < 0) {
		isCross = true;
	}

	/*for (int j = 0; j < track->trace.size (); j++) {
		cv::Point2d p = track->trace[j];
		double sign = pointToLine(p, line);
		if (!isPos && sign > 0) {
		isPos = true;
		}
		else if (!isNeg && sign < 0) {
		isNeg = true;
		}
		if (isPos && isNeg) {
		isCross = true;
		break;
		}
		}*/
	return isCross;
}

double pointToLine(cv::Point2d p, cv::Point2d norm, cv::Point2d p0) {
	return norm.x*(p.x - p0.x) + norm.y*(p.y - p0.y);
}

double pointToLine(cv::Point2d p, const std::vector<cv::Point2d>& line) {
	cv::Point2d norm(line[1].y - line[0].y, -(line[1].x - line[0].x));
	return pointToLine(p, norm, line[0]);
}

void translateListOfPoints(std::vector<cv::Point2d> &points, cv::Point2d v) {
	for (size_t i = 0; i < points.size(); i++) {
		points[i].x += v.x;
		points[i].y += v.y;
	}
}

void groupHeadlightsInProjectedCoordinate(std::vector<cv::Point2d> projObjCenters, std::vector<cv::Point2d> &vehicleCenters, int thX, int thY) {
	for (size_t i = 0; i < projObjCenters.size(); i++) {

	}
}

void perspectiveTransform2d(const std::vector<cv::Point2d>& objCenters, std::vector<cv::Point2d> &projObjCenters, const cv::Mat& homoTransform) {
	std::vector<cv::Point2f> tmpObjCenters, tmpProjObjCenters;
	cv::Mat(objCenters).copyTo(tmpObjCenters);
	cv::Mat(projObjCenters).copyTo(tmpProjObjCenters);
	cv::perspectiveTransform(tmpObjCenters, tmpProjObjCenters, homoTransform);
	//cv::transform(tmpObjCenters, tmpProjObjCenters, homoMat);
	cv::Mat(tmpProjObjCenters).convertTo(projObjCenters, cv::Mat(projObjCenters).type());
}

void detectBasedMovingDirection(CTracker &suspectsTracker, std::vector<int> &indices, Line_<double> outLine) {
	for (size_t i = 0; i < suspectsTracker.tracks.size(); i++) {
		if (suspectsTracker.tracks[i]->trace.size() > 2 && !suspectsTracker.tracks[i]->isCaught) {
			size_t traceLength = suspectsTracker.tracks[i]->trace.size();
			bool bCross = checkCrossALine(suspectsTracker.tracks[i], outLine);
			bool bDetected;
			// Prefer-direction mode

			cv::Vec4f line;
			// Check moving direction
			bool bOrtho;
			std::vector<cv::Point2f> vTmp;
			cv::Mat(suspectsTracker.tracks[i]->trace).copyTo(vTmp);
			cv::fitLine(vTmp, line, CV_DIST_L2, 0.0, 0.01, 0.01);
			cv::Point2d movingDirection(line[2], line[3]);
			cv::Point2d lastestDirection(suspectsTracker.tracks[i]->trace[traceLength - 1] - suspectsTracker.tracks[i]->trace[traceLength - 2]);
			if (movingDirection.dot(lastestDirection) < 0) {
				movingDirection = -movingDirection;
			}
			//cv::Point2d movingDirection(suspectsTracker.tracks[i]->trace[traceLength - 1] - suspectsTracker.tracks[i]->trace[0]);
			cv::Point2d stdVector(0, -1);
			double mag = std::sqrt(movingDirection.x*movingDirection.x + movingDirection.y*movingDirection.y);
			movingDirection = movingDirection*(1 / mag);
			if (bCross) {
				//std::cout << ">>>>>> Cross!" << std::endl << " Inner product: " << movingDirection.dot(stdVector) << std::endl;
			}

			if (movingDirection.dot(stdVector) > 0) {
				bOrtho = true;
			} else {
				bOrtho = false;
			}
			bDetected = !suspectsTracker.tracks[i]->isCaught && bCross && bOrtho;

			if (bDetected) {
				indices.push_back(i);
				//suspectsTracker.tracks[i]->isCaught = true;
			}
		}
	}
}

bool checkSegmentsIntersection(cv::Point2d o1, cv::Point2d p1, cv::Point2d o2, cv::Point2d p2) {
	double t1, t2;
	bool bIntersection = false;
	if (intersectSegments(o1, p1, o2, p2, t1, t2)) {
		if (t1 >= 0 && t1 <= 1 && t2 >= 0 && t2 <= 1) {
			bIntersection = true;
		} else {
			bIntersection = false;
		}
	}
	return bIntersection;
}

bool intersectSegments(cv::Point2d o1, cv::Point2d p1, cv::Point2d o2, cv::Point2d p2, double &t1, double &t2) {
	Point2d x = o2 - o1;
	Point2d d1 = p1 - o1;
	Point2d d2 = p2 - o2;

	double cross = d1.x*d2.y - d1.y*d2.x;
	//std::cout << "Dot: " << cross << std::endl;
	if (std::fabs(cross) < /*EPS*/1e-2)
		return false;
	if (cross != 0) {
		t1 = (x.x * d2.y - x.y * d2.x) / cross;
		t2 = (x.x * d1.y - x.y * d1.x) / cross;
		//std::cout << "  t: " << t1 << std::endl;	
		return true;
	}

	return false;
}

cv::Rect getMaxContoursFromROIImage(const cv::Mat& img) {
	std::vector< std::vector<cv::Point> > vContours;
	cv::Rect rectRange;
	cv::findContours(img, vContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	unsigned int iRoiArea = cv::countNonZero(img);
	int rectMax = 0;
	size_t indexMax = 0;
	for (size_t k = 0; k < vContours.size(); ++k) {
		rectRange = boundingRect(vContours[k]);
		if (rectMax < rectRange.width * rectRange.height) {
			rectMax = rectRange.width * rectRange.height;
			indexMax = k;
		}

	}
	rectRange = boundingRect(vContours[indexMax]);
	return rectRange;
}
std::string getCurrentTimeInVideoAsString(float fFrameRate, int indexFrame) {
	/*int timeAsSecond = int(indexFrame / fFrameRate);
	int hour = timeAsSecond / 3600;
	int min = (timeAsSecond / 60) % 60;
	int second = timeAsSecond % 60;
	char buff[100];
	snprintf(buff, 100, "%02d\%02d\%02d", hour, min, second);
	return buff;*/
	char buff[100];
	snprintf(buff, 100, "%d", indexFrame);
	return buff;
}

std::string getCurrentTimeInVideoAsString(int timeAsSecond) {
	int hour = timeAsSecond / 3600;
	int min = (timeAsSecond / 60) % 60;
	int second = timeAsSecond % 60;
	char buff[100];
	snprintf(buff, 100, "%02d\%02d\%02d", hour, min, second);
	return buff;
}
//bool intersectSegments(cv::Point2d o1, cv::Point2d p1, cv::Point2d o2, cv::Point2d p2)
//{
//	cv::Point2d r;
//	return intersectSegments(o1, p1, o2, p2, r);
//}