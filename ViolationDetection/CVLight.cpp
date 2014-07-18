#include "CVLight.h"
Light::Light() {}

Light::Light(int miny, int maxy, int minx, int maxx, int th, int type_) {
	this->rect = cv::Rect(minx, miny, maxx - minx, maxy - miny);
	//this->type = type_;
	this->threshold = th;
}

bool Light::isOn(const cv::Mat& mat) {
	cv::Mat light_area = mat(rect);
	int mean = int(cv::mean(light_area)[0]);

	return mean > this->threshold;
}

void Light::write(cv::FileStorage& fs) const {
	fs << "{" << "x" << this->rect.x
		<< "y" << this->rect.y
		<< "width" << this->rect.width
		<< "height" << this->rect.height
		<< "threshold" << this->threshold << "}";
}

void Light::read(const cv::FileNode& node) {
	this->rect = cv::Rect((int) node["x"], (int) node["y"], (int) node["width"], (int) node["height"]);
	this->threshold = (int) node["threshold"];
}


