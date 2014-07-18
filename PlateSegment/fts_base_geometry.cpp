#include "fts_base_geometry.h"
//
//#include "opencv2/core/core.hpp"
//using namespace cv;

FTS_BASE_Geometry::FTS_BASE_Geometry()
{
}

FTS_BASE_Geometry::~FTS_BASE_Geometry()
{
}

int FTS_BASE_Geometry::isInsideRectangle( CvRect oRect, CvPoint2D32f oPoint )
{
	if ( oPoint.x > oRect.x && oPoint.x < oRect.x + oRect.width &&
         oPoint.y > oRect.y && oPoint.y < oRect.y + oRect.height)
    {
        return INSIDE;
    }
    else
    {
    	return OUTSIDE;
    }
}

void FTS_BASE_Geometry::MajorMinorAxes(
        CvPoint2D32f* poPoints,
        unsigned int nNumOfPoints,
        float& rMajorAxisLength,
        float& rMinorAxisLength )
{
	// Duc 05 July 2012
	// Exception Handling when nNumOfPoints = 1
	// otherwise, exception thrown and crash
	// terminate called after throwing an instance of 'cv::Exception'
	// what():  /home/sensen/Downloads/OpenCV-2.3.1/modules/core/src/lapack.cpp:1676:
	// error: (-215) w.type() == type && (w.size() == cv::Size(nm,1) ||
	// w.size() == cv::Size(1, nm) || w.size() == cv::Size(nm, nm) || w.size()
	// == cv::Size(n, m)) in function cvSVD
	if( nNumOfPoints <= 1 )
	{
		rMajorAxisLength = 0;
		rMinorAxisLength = 1;
		return;
	}

#ifndef WIN32
    FTS_BASE_CV_MAT_ON_STACK( A, 2, nNumOfPoints, CV_32FC1 );
#else	
	CvMat A;
	cvInitMatHeader( &A, 2, nNumOfPoints, CV_32FC1 );
    unsigned char* AData = new unsigned char[ 2 * A.step ];
    cvInitMatHeader( &A, 2, nNumOfPoints, CV_32FC1, AData, A.step);
#endif

#ifndef WIN32
    FTS_BASE_CV_MAT_ON_STACK( W, 2,            2, CV_32FC1 );
#else	
	CvMat W;
	cvInitMatHeader( &W, 2, 2, CV_32FC1 );
    unsigned char* WData = new unsigned char[ 2 * W.step ];
    cvInitMatHeader( &W, 2, 2, CV_32FC1, WData, W.step);
#endif

    // Find mean in each dimension
    // ------------------------------------------------------------------------
    float rSumX = 0;
    float rSumY = 0;
    for( unsigned int i = 0; i < nNumOfPoints; ++i )
    {
        rSumX += poPoints[i].x;
        rSumY += poPoints[i].y;
    }

    float rMeanX = rSumX / (float)nNumOfPoints;
    float rMeanY = rSumY / (float)nNumOfPoints;

    float* prA = A.data.fl;
    for( unsigned int i = 0; i < nNumOfPoints; ++i )
    {
        prA[ i                ] = poPoints[ i ].x - rMeanX;
        prA[ i + nNumOfPoints ] = poPoints[ i ].y - rMeanY;
    }

    // Find major and minor axes
    // ------------------------------------------------------------------------
    cvSVD( &A, &W, 0, 0 );

    rMajorAxisLength = cvmGet( &W, 0, 0 );
    rMinorAxisLength = cvmGet( &W, 1, 1 );
#ifdef WIN32
	delete AData;
	delete WData;
#endif	
}

FTS_BASE_RotationSkewAngles::FTS_BASE_RotationSkewAngles()
	: m_rRotationAngleDegrees( 0.0f )
	, m_rSkewAngleDegrees( 0.0f )
{
	// Nothing
}

FTS_BASE_RotationSkewAngles::~FTS_BASE_RotationSkewAngles()
{
}
