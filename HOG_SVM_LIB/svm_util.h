
#ifndef SVM_UTIL_H
#define SVM_UTIL_H
#include "config.h"
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/core/operations.hpp>
#include "Hog_svm_classifiers.h"
enum GROUP_TYPE{ GROUP_MAX = 0, GROUP_AVG };
class FTS_SimilarRects
{
public:
	FTS_SimilarRects(double _eps) : eps(_eps) {}
	inline bool operator()(const DetectionObject& obj1, const DetectionObject& obj2) const
	{
		cv::Rect r1 = obj1.boundingBox;
		cv::Rect r2 = obj2.boundingBox;
		double delta = eps*(std::min(r1.width, r2.width) + std::min(r1.height, r2.height))*0.5;
		return std::abs(r1.x - r2.x) <= delta &&
			std::abs(r1.y - r2.y) <= delta &&
			std::abs(r1.x + r1.width - r2.x - r2.width) <= delta &&
			std::abs(r1.y + r1.height - r2.y - r2.height) <= delta;
	}
	double eps;
};
std::string toLowerCase(const std::string& in);
void SVM_HOG_EXPORT getFilesInDirectory(const std::string& dirName, std::vector<std::string>& fileNames, const std::vector<std::string>& validExtensions);
static void storeCursor(void) {
	printf("\033[s");
}

void clearListOfString(std::vector<std::string> &vs);
void clearListOfVectorFloat(std::vector< std::vector<float> > &vs);
bool isFileExist(const std::string& strFile);
bool isDirExist(const std::string& strDir);
void groupRectangles(std::vector<DetectionObject>& objList, int groupThreshold, double eps, GROUP_TYPE type);
static void resetCursor(void) {
	printf("\033[u");
}

//++28.06 trung add to build success on vs2010
#if _MSC_VER  <= 1600
namespace std
{
	std::string to_string(int i);
	std::string to_string(size_t i);
	std::string to_string(float i);
}
#endif
//--

#endif //SVM_UTIL_H