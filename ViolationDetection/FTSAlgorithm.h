#ifndef _FTS_LV_ALGORITHM_
#define _FTS_LV_ALGORITHM_

#if _MSC_VER > 1600
#include <thread>
#else
#include <boost/thread.hpp>
#endif

#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include "ViolationEvent.h"

class FTSAlgorithm: public cv::Algorithm {

protected:
	bool bDebug;
	boost::log::sources::severity_channel_logger< severity_level, std::string > lg;

public:
	static std::string const className() { return "FTSAlgorithm"; }
	virtual cv::AlgorithmInfo* info() const;
	FTSAlgorithm(bool _debug = true) {
		this->bDebug = _debug;
	}

	~FTSAlgorithm() {};
};



#endif