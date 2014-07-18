#pragma once
#include "Kalman.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
using namespace cv;
using namespace std;
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
TKalmanFilter::TKalmanFilter(Point2f pt, float dt, float Accel_noise_mag)
{
	//приращение времени (чем меньше эта величина, тем "инертнее" цель)
	deltatime = dt; //0.2

	// ”скорение мы не знаем, поэтому относим его к шуму процесса.
	// «ато мы можем предполагать, какие величины ускорени¤ может выдать отслеживаемый объект. 
	// Ўум процесса. (стандартное отклонение величины ускорени¤: м/с^2)
	// показывает, насколько сильно объект может ускоритьс¤.
	//float Accel_noise_mag = 0.5; 

	//4 переменных состо¤ни¤, 2 переменных измерени¤
	kalman = new KalmanFilter(4, 2, 0);
	// ћатрица перехода
	kalman->transitionMatrix = (Mat_<float>(4, 4) << 1, 0, deltatime, 0, 0, 1, 0, deltatime, 0, 0, 1, 0, 0, 0, 0, 1);

	// init... 
	LastResult = pt;
	kalman->statePre.at<float>(0) = pt.x; // x
	kalman->statePre.at<float>(1) = pt.y; // y

	kalman->statePre.at<float>(2) = 0;
	kalman->statePre.at<float>(3) = 0;

	kalman->statePost.at<float>(0) = pt.x;
	kalman->statePost.at<float>(1) = pt.y;

	setIdentity(kalman->measurementMatrix);

	kalman->processNoiseCov = (Mat_<float>(4, 4) <<
		pow(deltatime, 4.0) / 4.0, 0, pow(deltatime, 3.0) / 2.0, 0,
		0, pow(deltatime, 4.0) / 4.0, 0, pow(deltatime, 3.0) / 2.0,
		pow(deltatime, 3.0) / 2.0, 0, pow(deltatime, 2.0), 0,
		0, pow(deltatime, 3.0) / 2.0, 0, pow(deltatime, 2.0));


	kalman->processNoiseCov *= Accel_noise_mag;

	setIdentity(kalman->measurementNoiseCov, Scalar::all(0.1));

	setIdentity(kalman->errorCovPost, Scalar::all(.1));

}
//---------------------------------------------------------------------------
TKalmanFilter::~TKalmanFilter()
{
	delete kalman;
}

//---------------------------------------------------------------------------
Point2f TKalmanFilter::GetPrediction()
{
	Mat prediction = kalman->predict();
	LastResult = Point2f(prediction.at<float>(0), prediction.at<float>(1));
	return LastResult;
}
//---------------------------------------------------------------------------
Point2f TKalmanFilter::Update(Point2f p, bool DataCorrect)
{
	Mat measurement(2, 1, CV_32FC1);
	if (!DataCorrect)
	{
		measurement.at<float>(0) = LastResult.x;  //уточн¤ем использу¤ предсказание
		measurement.at<float>(1) = LastResult.y;
	}
	else
	{
		measurement.at<float>(0) = p.x;  //уточн¤ем, использу¤ данные измерений
		measurement.at<float>(1) = p.y;
	}
	//  оррекци¤
	Mat estimated = kalman->correct(measurement);
	LastResult.x = estimated.at<float>(0);  //уточн¤ем, использу¤ данные измерений
	LastResult.y = estimated.at<float>(1);
	return LastResult;
}
//---------------------------------------------------------------------------