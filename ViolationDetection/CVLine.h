#ifndef _FTS_CVLINE_H_
#define _FTS_CVLINE_H_

#include <opencv2/opencv.hpp>

template<typename _Tp> class Line_ {
public:
	cv::Point_<_Tp> start;
	cv::Point_<_Tp> end;

	Line_() {};
	Line_(cv::Point_<_Tp> _start, cv::Point_<_Tp> _end) {
		start = _start;
		end = _end;
	}
	Line_(_Tp x1, _Tp y1, _Tp x2, _Tp y2) {
		start = cv::Point_<_Tp>(x1, y1);
		end = cv::Point_<_Tp>(x2, y2);
	}

	Line_& operator = (const Line_& l) {
		this->start = l.start;
		this->end = l.end;

		return *this;
	};

	// checks whether the rectangle contains the point
	bool cross(const Line_<_Tp>& line) const {

		_Tp s1_x = this->end.x - this->start.x;
		_Tp s1_y = this->end.y - this->start.y;
		_Tp s2_x = line.end.x - line.start.x;
		_Tp s2_y = line.end.y - line.start.y;

		_Tp s = (-s1_y * (this->start.x - line.start.x) + s1_x * (this->start.y - line.start.y)) / (-s2_x * s1_y + s1_x * s2_y);
		_Tp t = (s2_x * (this->start.y - line.start.y) - s2_y * (this->start.x - line.start.y)) / (-s2_x * s1_y + s1_x * s2_y);

		// Collision detected
		if (s >= 0 && s <= 1 && t >= 0 && t <= 1) {
			//get intersec point - no need now
			//if (i_x != NULL)
			//	*i_x = p0_x + (t * s1_x);
			//if (i_y != NULL)
			//	*i_y = p0_y + (t * s1_y);
			return true;
		}

		return false; // No collision
	};

	void write(cv::FileStorage& fs) const {
		fs << "{" << "x1" << this->start.x
			<< "y1" << this->start.y
			<< "x2" << this->end.x
			<< "y2" << this->end.y << "}";
	}

	void read(const cv::FileNode& node) {
		this->start = cv::Point_<_Tp>((_Tp) node["x1"], (_Tp) node["y1"]);
		this->end = cv::Point_<_Tp>((_Tp) node["x2"], (_Tp) node["y2"]);
	}

	~Line_() {};
};

//These write and read functions must exist as per the inline functions in operations.hpp
static void write(cv::FileStorage& fs, const std::string&, const Line_<double>& x) {
	x.write(fs);
}
static void read(const cv::FileNode& node, Line_<double>& x, const Line_<double>& default_value = Line_<double>()) {
	if (node.empty())
		x = default_value;
	else
		x.read(node);
}

#endif