#ifndef _FTS_LV_UTIL_
#define _FTS_LV_UTIL_


#ifdef _WIN32
#include <direct.h>
#endif

#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <cmath>
#include <ctime>
#include <cstdio>

#include <opencv2/opencv.hpp>


#if _MSC_VER <= 1600
namespace std
{
	static inline double round(double val) {
		return floor(val + 0.5);
	}
}
#endif

const float PI = 3.1416f;

bool hasEnding(const std::string& fullString, const std::string& ending);

void mkdir(const std::string& strDir);
bool isDirExist(const std::string& strDir);
bool isFileExist(const std::string& strFile);

void writeParamFile(cv::Ptr<cv::Algorithm> algorithm, const std::string& strParamFile);
void readParamFile(cv::Ptr<cv::Algorithm> algorithm, const std::string& strParamFile);

std::string createDirectory(std::string strParent, std::string strSub, std::string strDeviceID, std::string strDate, std::string strType);

std::string getDateTimeString(time_t timer, const std::string& strFormat);
bool stringCompare(const std::string& left, const std::string& right);

std::string formatString(const char* fmt, ...);

#endif