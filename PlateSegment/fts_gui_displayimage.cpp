/*
 * fts_gui_displayimage.cpp
 *
 *  Created on: May 9, 2014
 *      Author: sensen
 */

#include "fts_gui_displayimage.h"

const double FTS_GUI_DisplayImage::SCALE_X( 1.0 );
const double FTS_GUI_DisplayImage::SCALE_Y( 1.0 );

FTS_GUI_DisplayImage::FTS_GUI_DisplayImage() {
	// TODO Auto-generated constructor stub

}

FTS_GUI_DisplayImage::~FTS_GUI_DisplayImage() {
	// TODO Auto-generated destructor stub
}


void FTS_GUI_DisplayImage::ShowAndScaleBy2( const std::string& sWindowName,
					  const cv::Mat& oSrc,
					  double fx, double fy,
					  const unsigned int nPositionX,
					  const unsigned int nPositionY )
{
	cv::Mat oSrcx2;
	cv::resize( oSrc, oSrcx2, cv::Size(), fx, fy, CV_INTER_LINEAR );
	cv::imshow( sWindowName, oSrcx2 );

	cvMoveWindow( sWindowName.c_str(), nPositionX, nPositionY );
}

void FTS_GUI_DisplayImage::ShowGroupScaleBy2( const std::string& sWindowName,
						const float rScale,
						const std::vector< cv::Mat >& oGroup,
						const unsigned int nCols )
{
	unsigned int nPadded = 15;
	std::vector< cv::Mat > oGroupx2( oGroup.size() );
	for( unsigned int i = 0; i < oGroup.size(); i++ )
	{
		cv::resize( oGroup[i], oGroupx2[i], cv::Size(), rScale, rScale, CV_INTER_LINEAR );
	}

	unsigned int nWidth  = ( oGroupx2[0].cols + nPadded ) * nCols;
	unsigned int numRows = ceil((float) oGroupx2.size() / (float) nCols);
	unsigned int nHeight = ( oGroupx2[0].rows + nPadded ) * numRows;

	cv::Mat oDst = Mat::zeros( cv::Size( nWidth, nHeight ), CV_8UC3 );

	for( unsigned int i = 0; i < nCols * numRows; i++ )
	{
		if( i < oGroupx2.size() )
		{
			cv::Mat oROIMat = oDst( Rect( ( i % nCols ) * ( oGroupx2[i].cols + nPadded ),
										  floor( (float) i / nCols ) * ( oGroupx2[i].rows + nPadded ),
										  oGroupx2[i].cols,
										  oGroupx2[i].rows ) );

			if( oGroupx2[i].type() != CV_8UC3 )
			{
				cv::Mat oColor;
				cv::cvtColor( oGroupx2[i], oColor, CV_GRAY2BGR );
				oColor.copyTo( oROIMat );
			}
			else
			{
				oGroupx2[i].copyTo( oROIMat );
			}
		}
		else
		{
			Mat black = Mat::zeros( oGroupx2[0].size(), CV_8UC3);
			cv::Mat oROIMat = oDst( Rect( ( i % nCols ) * ( oGroupx2[i].cols + nPadded ),
					  floor( (float) i / nCols ) * ( oGroupx2[i].rows + nPadded ),
					  oGroupx2[i].cols,
					  oGroupx2[i].rows ) );
			black.copyTo( oROIMat );
		}
	}

	cv::imshow( sWindowName, oDst );
}

