#pragma once
#include "Kalman.h"
#include "HungarianAlg.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
using namespace cv;
using namespace std;

class CTrack
{
public:
	vector<Point2d> trace;
	bool isCount;
	bool isIn;
	bool isOut;
	bool isCaught;
	static size_t NextTrackID;
	size_t track_id;
	size_t skipped_frames;
	Point2d prediction;

	bool isHavePlate;
	cv::Mat imgPlate;
	TKalmanFilter* KF;
	CTrack(Point2f p, float dt, float Accel_noise_mag);
	~CTrack();
};


class CTracker
{
public:

	float dt;

	float Accel_noise_mag;
	double very_large_cost;
	double dist_thres;
	double cos_thres;
	int maximum_allowed_skipped_frames;
	int max_trace_length;

	std::vector<int> assignment;

	vector<CTrack*> tracks;
	void Update(vector<Point2d>& detections);
	void drawTrackToImage(cv::Mat &img);
	CTracker(float _dt, float _Accel_noise_mag, double _dist_thres = 60, double _cos_thres = -0.5, int _maximum_allowed_skipped_frames = 10, int _max_trace_length = 10, double _very_large_cost = 1000000);
	//CTracker(float _dt, float _Accel_noise_mag, double _dist_thres = 60, double _cos_thres = -, int _maximum_allowed_skipped_frames, int _max_trace_length, double _very_large_cost = 1000000);
	void updateSkipedSkippedFrames();
	void setPlateForTrackers(const cv::Mat& originalImage, std::vector<cv::Rect> &listPlateObjs);
	~CTracker(void);
};

