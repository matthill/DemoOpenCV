#ifndef _FTS_CVLIGHT_
#define _FTS_CVLIGHT_

#include <opencv2/opencv.hpp>

class Light {
public:
	cv::Rect rect;
	// type:
	//	0: colors (green/red)
	//	1: on/off
	//int type;
	int threshold;
	Light();

	Light(int miny, int maxy, int minx, int maxx, int th, int type_);

	bool isOn(const cv::Mat& mat);

	void write(cv::FileStorage& fs) const;

	void read(const cv::FileNode& node);
};


//These write and read functions must exist as per the inline functions in operations.hpp
static void write(cv::FileStorage& fs, const std::string&, const Light& x) {
	x.write(fs);
}
static void read(const cv::FileNode& node, Light& x, const Light& default_value = Light()) {
	if (node.empty())
		x = default_value;
	else
		x.read(node);
}


#endif