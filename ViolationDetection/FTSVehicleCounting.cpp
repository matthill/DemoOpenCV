#include <regex>

#include "FTSVehicleCounting.h"
#include "util.h"
#include "CVUtil.h"
#include "ConnectedComponent.h"
#include "FTSANPR.h"
#include "ViolationEvent.h"

void FTSVehicleCounting::operator() (FTSCamera camInfo) {
	//open video stream
	int numOf4WheelVehicles = 0;
	int numOfBikes = 0;
	cv::VideoCapture video(camInfo.strVideoSrc);
	if (!video.isOpened())
		return;

	cv::Mat img, imgFrame, imgBack, imgFore, imgRoiWhole, imgRoiCar, imgRoiBike;
	std::vector< std::vector<cv::Point> > vContours;
	cv::Rect rectRange;

	//get video info
	int iFrameWidth = (int) (video.get(CV_CAP_PROP_FRAME_WIDTH) * this->fScaleRatio);
	int iFrameHeight = (int) (video.get(CV_CAP_PROP_FRAME_HEIGHT) * this->fScaleRatio);
	float fFrameRate = (float) video.get(CV_CAP_PROP_FPS);


	ListConnectComponent ccObjs;
	std::vector<cv::Point2d> projObjCenters;


	ListConnectComponent ccBikes, ccCars;

	cv::Size subPixWinSize(30, 30);
	cv::TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 5, 0.3);
	cv::Mat projectedImg;

	//Read ROI img
	img = cv::imread(this->strRoiImg, CV_LOAD_IMAGE_GRAYSCALE);
	if (img.empty()) {
		if (bDebug)
			cout << "ROI Image's missing" << endl;
		return;
	}
	cv::resize(img, img, cv::Size(iFrameWidth, iFrameHeight));
	rectRange = getMaxContoursFromROIImage(img);
	imgRoiWhole = img(rectRange);
	unsigned int iRoiArea = cv::countNonZero(imgRoiWhole);

	// Settings for night-mode
	// Translate vertices of the quad. which is actually a rectangle in real world
	int newWidth, newHeight;
	// Counting line in projected coordinate
	int projY0;
	// Homography matrix for transforming points in original coordinate to ground-directed coordinate
	cv::Mat homoTransform;
	if (bNight) {
		for (size_t i = 0; i < this->vProjQuad.size(); i++) {
			vProjQuad[i].x *= this->fScaleRatio;
			vProjQuad[i].y *= this->fScaleRatio;
		}
		for (size_t i = 0; i < this->countingLine.size(); i++) {
			this->countingLine[i].x *= this->fScaleRatio;
			this->countingLine[i].y *= this->fScaleRatio;
		}
		translateListOfPoints(this->vProjQuad, cv::Point2d(-rectRange.x, -rectRange.y));
		translateListOfPoints(this->countingLine, cv::Point2d(-rectRange.x, -rectRange.y));
		projY0 = int(countingLine[0].y + countingLine[1].y) / 2;

		newWidth = this->vProjRect[2].x - this->vProjRect[0].x;
		newHeight = this->vProjRect[2].y - this->vProjRect[0].y;
		homoTransform = cv::findHomography(this->vProjQuad, this->vProjRect, CV_RANSAC);
		//projectedImg = cv::Mat::zeros(newHeight, newWidth, CV_8UC3);

	}
	std::cout << vProjQuad << std::endl;
	//Read Car ROI img
	//img = cv::imread(this->strRoiCarImg, CV_LOAD_IMAGE_GRAYSCALE);
	//std::cout << "Video name: " << this->strRoiCarImg << std::endl;
	//cv::resize(img, img, cv::Size(iFrameWidth, iFrameHeight));
	//imgRoiCar = img(rectRange);

	////Read Bike ROI img
	//img = cv::imread(this->strRoiBikeImg, CV_LOAD_IMAGE_GRAYSCALE);
	//cv::resize(img, img, cv::Size(iFrameWidth, iFrameHeight));
	//imgRoiBike = img(rectRange);

	//duong.tb add log file

	std::cmatch base_name_res;
	std::tr1::regex base_name_pat("([^\\\\]+)$");
	std::regex_search(camInfo.strVideoSrc.c_str(), base_name_res, base_name_pat);
	string fileVideoName = base_name_res[1];
	ofstream out_log_stream;
	out_log_stream.open("FTSVehicleCounting_" + fileVideoName + ".log");

	cv::Mat prev_fore = cv::Mat::zeros(iFrameHeight, iFrameWidth, CV_8UC1);
	cv::Mat imgGrayFrame = cv::Mat::zeros(iFrameHeight, iFrameWidth, CV_8UC1);
	cv::Mat imgPrevGrayFrame = cv::Mat::zeros(iFrameHeight, iFrameWidth, CV_8UC1);
	cv::Mat imgBinFrame = cv::Mat::zeros(iFrameHeight, iFrameWidth, CV_8UC1);

	cv::Mat imgOverlay = cv::Mat::zeros(iFrameHeight, iFrameWidth, CV_8UC3);

	//cv::Mat bikeLaneMap = cv::Mat::zeros(iFrameHeight, iFrameWidth, CV_8U);
	//cv::Mat carLaneMap = cv::Mat::zeros(iFrameHeight, iFrameWidth, CV_8U);

	cv::BackgroundSubtractorMOG2 bg(this->iHistory, this->fVarThreshold, true);
	bg.set("nmixtures", 3);



	std::vector<cv::Point2d> vehicleCenters, prevTrackedBikePs, trackedBikePs, prevTrackedCarPs, trackedCarPs;

	float hue_thresh = 0.2f;
	std::deque<float> density_queue;
	unsigned int num_group_frame = (int) (fFrameRate * this->iInterval);

	float fLearningRate = this->fMaxLearningRate;
	float list_values[5] = {0, 1, 2, 3, 4};

	// Tracker settings
	CTracker tracker(0.2f, 0.5f, 80.0, -1.0, 10, 10, 1000000);
	CTracker projTracker(0.2f, 0.5f, 30.0, -0.5, 10, 10, 1000000);
	//CTracker carTracker(0.2, 0.5, 120.0, 0, 10, 10, 1000000);

	// Counting line in original coordinate
	int y_0 = int(std::round(2.0*iFrameHeight / 5.0));
	int index = 0;
	int iMaxWidth = iFrameWidth*0.85;
	int iMaxHeight = iFrameHeight*0.85;
	double dAspectRatio = 4.5;
	//int y_0 = (int)std::round((double)2 * rectRange.height / 5);
	//cout << "y_0 " << y_0 << endl;

	// Image in transformed coordinate

	//cv::Mat homoMat = (cv::Mat_<double>(3, 3) << 0.0000, 0.0001, -0.2445, 0.0015, -0.0010, -0.9696, 0.0000, 0.0000, -0.0010);

	std::vector<std::vector<cv::Point2d> > centerGroups;

	while (cv::waitKey(1) != 27) {
		//read image frame 
		video >> img;
		if (img.empty())
			break;
		cv::resize(img, imgFrame, cv::Size(iFrameWidth, iFrameHeight));

		imgFrame = imgFrame(rectRange);

		cvtColor(imgFrame, imgGrayFrame, CV_BGR2GRAY);

		// Night mode
		if (bNight) {
			projectedImg = cv::Mat::zeros(newHeight, newWidth, CV_8UC3);
			// Draw counting line
			cv::line(projectedImg, cv::Point(this->vProjRect[0].x, projY0), cv::Point(this->vProjRect[1].x, projY0), cv::Scalar(0, 0, 255), 2, CV_AA);
			// Threshold to extract head lights
			cv::threshold(imgGrayFrame, imgBinFrame, 250, 255, THRESH_BINARY);
			// Remove head lights which are out of ROI
			imgFore = imgBinFrame.mul(imgRoiWhole);
			// Find contours of head lights
			cv::findContours(imgFore, vContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
			// Remove small head lights
			examineContours(vContours, this->iContourMinArea, this->iContourMaxArea);
			// Extract info of head lights
			//extractConnectedComponents(vContours, ccObjs);
			ccObjs.extractConnectedComponentsFormContours(vContours);

			// If head lights exist
			if (ccObjs.getSize() > 0) {
				// Transform detected headlights into ground-directed coordinate
				perspectiveTransform2d(ccObjs.getListCenter(), projObjCenters, homoTransform);
				//groupHeadlightsInProjectedCoordinate(projObjCenters, vehicleCenters);

				// Labels of head lights
				std::vector<int> labels;

				// Group head lights of the same auto
				cv::partition(projObjCenters, labels, EqPredicate(160, 60));

				// Find the number of groups 
				int iMaxLabel = *std::max_element(labels.begin(), labels.end()) + 1;

				centerGroups.reserve(iMaxLabel);
				vehicleCenters.reserve(iMaxLabel);
				if (bDebug) {
					std::cout << "Num of labels: " << labels.size() << "  " << iMaxLabel << std::endl;
				}
				// Push groups of head lights into vector centerGroups
				for (size_t i = 0; i < iMaxLabel; i++) {
					std::vector<cv::Point2d> ps;
					for (size_t j = 0; j < labels.size(); j++) {
						if (labels[j] == i) {
							ps.push_back(projObjCenters[j]);
							if (bDebug) {
								std::cout << "    label " << j << ": " << labels[j] << std::endl;
							}
						}
						//std::cout << " Labels: " << labels[j] << std::endl;
					}
					centerGroups.push_back(ps);
				}

				// Find representatives of head light groups
				for (size_t i = 0; i < iMaxLabel; i++) {
					if (centerGroups[i].size() == 1) {
						vehicleCenters.push_back(centerGroups[i][0]);
					} else {
						cv::Point2d p(0, 0);
						for (size_t j = 0; j < centerGroups[i].size(); j++) {
							p = p + centerGroups[i][j];
						}
						p.x = p.x / centerGroups[i].size();
						p.y = p.y / centerGroups[i].size();
						vehicleCenters.push_back(p);
					}
				}

				// Track projected centers
				projTracker.Update(vehicleCenters);
				if (bDebug) {
					std::cout << "Num of transform centers: " << projObjCenters.size() << std::endl;
					for (size_t i = 0; i < projObjCenters.size(); i++) {
						std::cout << "   " << projObjCenters[i] << std::endl;
					}
					// Draw projected object centers
					for (size_t i = 0; i < projObjCenters.size(); i++) {
						cv::circle(projectedImg, projObjCenters[i], 3, cv::Scalar(255, 0, 255), 2);
					}
				}
			}

			if (bDebug) {
				// Draw tracks
				projTracker.drawTrackToImage(projectedImg);
			}
			// Count vehicle based on projected coordinate
			for (int i = 0; i<projTracker.tracks.size(); i++) {


				size_t iTraceLength = projTracker.tracks[i]->trace.size();
				// If a track is long enough
				if (iTraceLength > 5 && !projTracker.tracks[i]->isCount) {
					// A track crosses counting line in projected coordinate
					if ((projTracker.tracks[i]->trace[0].y - projY0) * (projTracker.tracks[i]->trace[iTraceLength - 1].y - projY0) <= 0) // if intersect
					{
						projTracker.tracks[i]->isCount = true;
						int vehicleInd = projTracker.assignment[i];
						if (bDebug) {
							std::cout << "Vehicle index: " << vehicleInd << std::endl;
						}
						// Avoid prediction track (track without corresponding detection in current frame)
						if (vehicleInd >= 0) {
							std::vector<cv::Point> tmp;
							cv::Mat(centerGroups[vehicleInd]).copyTo(tmp);
							cv::Rect bb = boundingRect(tmp);
							if (centerGroups[vehicleInd].size() > 1 && bb.width > 20) {
								numOf4WheelVehicles++;
							} else {
								numOfBikes++;
							}
						}
					}
				}
			}

			//cv::imshow("MovingObj", imgFore*255);
			// Show projected image
			cv::imshow("Transformed Image", projectedImg);

			// Show tracks in original coordinate
			if (bDebug) {
				if (!ccObjs.isEmpty()) {
					tracker.Update(ccObjs.getListCenter());
					tracker.drawTrackToImage(imgFrame);
				} else {
					tracker.~CTracker();
				}
			}
		}
		// Not Night-mode
		else {
			bg(imgFrame, imgFore, fLearningRate);
			bg.getBackgroundImage(imgBack);

			line(imgFrame, cv::Point2d(0, y_0), cv::Point2d(imgFrame.cols, y_0), Scalar(0, 255, 0), 2, CV_AA);
			//process frame after training
			if (index > this->iTrainingFrame) {

				//START - Using bg subtraction to dectect violation and put to vehicleCenters object
				dilation(imgFore, cv::MORPH_RECT, 1);
				erosion(imgFore, cv::MORPH_RECT, 1);

				imgFore = imgFore.mul(imgRoiWhole);

				cv::findContours(imgFore, vContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

				examineContours(vContours, this->iContourMinArea, this->iContourMaxArea);
				//extractConnectedComponents(vContours, ccObjs);
				ccObjs.extractConnectedComponentsFormContours(vContours);
				//bikeLaneMap = imgFore.mul(imgRoiBike);
				//carLaneMap = imgFore.mul(imgRoiCar);

				/*
				int newTrackNum = 0;
				detectLaneViolationVehicles(carLaneMap, this->iCarMinSize, this->iCarMaxSize, objBbs, objCenters, objAreas, carBbs, carCenters, carAreas);

				if (bDebug) {
				//std::cout << "Num of approaching cars: " << carCenters.size() << std::endl;
				//std::cout << "Num of tracks: " << carTracker.tracks.size() << std::endl;
				}


				newTrackNum = countNewVehicles(imgPrevGrayFrame, imgGrayFrame, carCenters, carBbs, prevTrackedCarPs, trackedCarPs, subPixWinSize, termcrit);

				//int iTooLargeSize = int(std::round(iFrameHeight*iFrameWidth*0.75));
				int iMaxWidth = iFrameWidth*0.85;
				int iMaxHeight = iFrameHeight*0.85;
				int iNumInvalid = 0;
				iNumInvalid = removeTooLargeDetections(carBbs, carCenters, iMaxWidth, iMaxHeight);

				numOf4WheelVehicles = numOf4WheelVehicles + newTrackNum - iNumInvalid;

				detectLaneViolationVehicles(bikeLaneMap, this->iBikeMinSize, this->iBikeMaxSize, objBbs, objCenters, objAreas, bikeBbs, bikeCenters, bikeAreas);//duong
				//newTrackNum = countNewVehicles(imgPrevGrayFrame, imgGrayFrame, bikeCenters, bikeBbs, prevTrackedBikePs, trackedBikePs, subPixWinSize, termcrit);
				*/
				//numOfBikes += newTrackNum;
				//duong end
				//int y_0 = 250;


				//line(imgFrame, cv::Point2d( 0, y_0), cv::Point2d(imgFrame.cols, y_0), Scalar(0, 255, 0), 2, CV_AA);
				//detectVehicles( this->iBikeMinSize, this->iBikeMaxSize, objBbs, objCenters, objAreas, bikeBbs, bikeCenters, bikeAreas);
				if (!ccObjs.isEmpty()) {
					tracker.Update(ccObjs.getListCenter());
					if (bDebug) {
						tracker.drawTrackToImage(imgFrame);
					}
					for (int i = 0; i<tracker.tracks.size(); i++) {

						size_t iTraceLength = tracker.tracks[i]->trace.size();
						if (iTraceLength > 10 && !tracker.tracks[i]->isCount) {
							int newTrackNumBike = 0;
							int newTrackNumCar = 0;
							if ((tracker.tracks[i]->trace[0].y - y_0) * (tracker.tracks[i]->trace[iTraceLength - 1].y - y_0) <= 0) // if intersect
							{
								int vehicleID = tracker.assignment[i];
								if (vehicleID > -1) {
									if (ccObjs.getListArea()[vehicleID] > this->iBikeMinSize && ccObjs.getListArea()[vehicleID] < this->iBikeMaxSize) {
										tracker.tracks[i]->isCount = true;
										newTrackNumBike++;
										ccBikes.push_back(ccObjs.getListCenter()[vehicleID], ccObjs.getListArea()[vehicleID], ccObjs.getListBoundingBox()[vehicleID]);
									} else if (ccObjs.getListArea()[vehicleID] > this->iCarMinSize && ccObjs.getListArea()[vehicleID] < this->iCarMaxSize) {
										tracker.tracks[i]->isCount = true;
										newTrackNumCar++;
										ccCars.push_back(ccObjs.getListCenter()[vehicleID], ccObjs.getListArea()[vehicleID], ccObjs.getListBoundingBox()[vehicleID]);
									}
								}

							}

							int iNumInvalidCar = 0;
							iNumInvalidCar = ccCars.removeJunkDetections(iMaxWidth, iMaxHeight, dAspectRatio);
							int iNumInvalidBike = ccBikes.removeJunkDetections(iMaxWidth, iMaxHeight, dAspectRatio);
							numOfBikes += (newTrackNumBike - iNumInvalidBike);
							numOf4WheelVehicles += (newTrackNumCar - iNumInvalidCar);
						}
					}
				}
				//numOfBikes += newTrackNum;

				if (bDebug) {
					for (size_t i = 0; i < trackedCarPs.size(); i++) {
						circle(imgFrame, trackedCarPs[i], 3, Scalar(0, 255, 0), 1, CV_AA);
					}
				}
				cv::imshow("Background" + camInfo.strCameraId, imgBack);

				float fOccupiedSpace = (float) (cv::countNonZero(imgFore) * 100) / iRoiArea;
				density_queue.push_back(fOccupiedSpace);
				if (density_queue.size() > num_group_frame) {
					density_queue.pop_front();
				}
				float mean_density = meanQueue(density_queue);
				int density_rate;
				std::string density = getDensityStatus(fOccupiedSpace, density_rate);

				fLearningRate = calculateLearningRate(list_values, mean_density, density_rate, this->fMaxLearningRate);
				// END - Recalculate learning rate base on occupied ratio
				char status[100];
				sprintf(status, "%d. Status: %s (%.1f %%) ", index, density.c_str(), mean_density);
				std::string density_status(status);
				putText(imgOverlay, density_status, cv::Point(20, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2, 8, false);
			}
		}



		//debug case - display image
		if (bDebug) {
			//overlayMap(imgFrame, imgOverlay, imgFore);
			imgFrame.copyTo(imgOverlay);

			//mark violation
			for (size_t i = 0; i < ccCars.getSize(); i++) {
				rectangle(imgOverlay, ccCars.getListBoundingBox()[i], cv::Scalar(255, 255, 0), 2);
			}

			for (size_t i = 0; i < ccBikes.getSize(); i++) {
				rectangle(imgOverlay, ccBikes.getListBoundingBox()[i], cv::Scalar(255, 0, 255), 2);
			}

			//cv::drawContours(imgOverlay, vContours, -1, cv::Scalar(255, 0, 0), 1);
			//overlayContourAreas(imgOverlay, imgRoiBike, objCenters, objAreas);
			//overlayContourAreas(imgOverlay, imgRoiCar, ccObjs.getListArea());			

			char countBike[50];
			sprintf(countBike, "Bike: %d", numOfBikes);
			char countCar[50];
			sprintf(countCar, "Car: %d", numOf4WheelVehicles);

			//std::cout << this->camInfo.strCameraId << ": " << status << std::endl;


			putText(imgOverlay, std::string(countBike), cv::Point(20, 40), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 255), 2, 8, false);

			putText(imgOverlay, std::string(countCar), cv::Point(20, 60), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 2, 8, false);
			cv::line(imgOverlay, cv::Point(this->countingLine[0].x, this->countingLine[0].y), cv::Point(this->countingLine[1].x, this->countingLine[1].y), 2, CV_AA);
			/*cv::findContours(imgRoiWhole, vContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
			cv::drawContours(imgOverlay, vContours, -1, cv::Scalar(0, 0, 255), 1);
			cv::findContours(imgRoiCar, vContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
			cv::drawContours(imgOverlay, vContours, -1, cv::Scalar(0, 0, 255), 1);
			cv::findContours(imgRoiBike, vContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
			cv::drawContours(imgOverlay, vContours, -1, cv::Scalar(0, 0, 255), 1);*/

			cv::imshow("Overlay" + camInfo.strCameraId, imgOverlay);


			//clear		
			for (size_t i = 0; i < centerGroups.size(); i++) {
				centerGroups[i].clear();
			}
			centerGroups.clear();
			projObjCenters.clear();
			vehicleCenters.clear();


			ccObjs.clear();
			ccCars.clear();
			ccBikes.clear();

		}

		imgGrayFrame.copyTo(imgPrevGrayFrame);
		prevTrackedCarPs = trackedCarPs;
		prevTrackedBikePs = trackedBikePs;

		//duong.tb add log file
		std::string str_FTSVehicleCounting_log = "";
		char ch_buffer[1000];
		_itoa(index, ch_buffer, 10);
		str_FTSVehicleCounting_log += ch_buffer;
		str_FTSVehicleCounting_log += " ";
		_itoa(numOf4WheelVehicles, ch_buffer, 10);
		str_FTSVehicleCounting_log += ch_buffer;
		str_FTSVehicleCounting_log += " ";
		_itoa(numOfBikes, ch_buffer, 10);
		str_FTSVehicleCounting_log += ch_buffer;
		str_FTSVehicleCounting_log += "\n";
		out_log_stream << str_FTSVehicleCounting_log.c_str();

		index++;		
		if(index == INT_MAX) //trungnt1 add to avoid overflow
			index = 0;
	}
	out_log_stream.close();
}

FTSVehicleCounting::FTSVehicleCounting() {

	// Following init value will be override in config file - only for development
	this->strRoiImg = "D:\\Data\\CauPhuMy\\phumy2-20140513-1101_ROI_1.png";
	this->strRoiCarImg = "D:\\Data\\CauPhuMy\\phumy2-20140513-1101_ROI_1.png";
	this->strRoiBikeImg = "D:\\Data\\CauPhuMy\\phumy2-20140513-1101_ROI_1.png";

	this->fVarThreshold = 50;
	this->fMaxLearningRate = .005f;
	this->fScaleRatio = .5;

	this->iHistory = 2;
	this->iTrainingFrame = 1000;
	this->iInterval = 2;

	this->bNight = false;
	if (bNight) {
		// top-left
		this->vProjQuad.push_back(cv::Point2d(932, 235));
		// top-right
		this->vProjQuad.push_back(cv::Point2d(1167, 223));
		// bottom-right
		this->vProjQuad.push_back(cv::Point2d(1278, 911));
		// bottom-left
		this->vProjQuad.push_back(cv::Point2d(290, 937));

		int factor = 30;
		this->vProjRect.push_back(cv::Point2d(10, 0));
		this->vProjRect.push_back(cv::Point2d(7.5 * factor, 0));
		this->vProjRect.push_back(cv::Point2d(7.5 * factor, 15 * factor));
		this->vProjRect.push_back(cv::Point2d(10, 15 * factor));

		this->iContourMinArea = 10;
		this->iContourMaxArea = 100;
		this->iBikeMinSize = 10;
		this->iBikeMaxSize = 100;
		this->iCarMinSize = 10;
		this->iCarMaxSize = 300;
	} else {
		this->iContourMinArea = 800;
		this->iContourMaxArea = 1000000;
		this->iBikeMinSize = 800;
		this->iBikeMaxSize = 10000;
		this->iCarMinSize = 10000;
		this->iCarMaxSize = 3000000;
	}

	countingLine.push_back(cv::Point2d(325, 835));
	countingLine.push_back(cv::Point2d(1360, 835));
}

FTSVehicleCounting::~FTSVehicleCounting() {}
