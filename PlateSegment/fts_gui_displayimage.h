/*
 * fts_gui_displayimage.h
 *
 *  Created on: May 9, 2014
 *      Author: sensen
 */

#ifndef FTS_GUI_DISPLAYIMAGE_H_
#define FTS_GUI_DISPLAYIMAGE_H_

#include "fts_base_externals.h"

class FTS_GUI_DisplayImage {
public:
	FTS_GUI_DisplayImage();
	virtual ~FTS_GUI_DisplayImage();


	static void ShowAndScaleBy2( const std::string& sWindowName,
			  const cv::Mat& oSrc,
			  double fx, double fy,
			  const unsigned int nPositionX,
			  const unsigned int nPositionY );

	static void ShowGroupScaleBy2( const std::string& sWindowName,
			const float rScale,
							const std::vector< cv::Mat >& oGroupx2,
							const unsigned int nCols );

	static const double SCALE_X;
	static const double SCALE_Y;
};

#endif /* FTS_GUI_DISPLAYIMAGE_H_ */
