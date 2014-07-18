
#include "fts_anpr_rotate.h"
#include "fts_base_util.h"

//#define DBUG_L4

FTS_ANPR_Rotate::FTS_ANPR_Rotate()
{
    m_poStorage = cvCreateMemStorage( 0 );
}

FTS_ANPR_Rotate::~FTS_ANPR_Rotate()
{
    cvReleaseMemStorage( &m_poStorage );
}

void FTS_ANPR_Rotate::rotate( const cv::Mat& oInput, cv::Mat& oOutput, bool bBlackChar )
{
    double rAngle = 0;
    if (  !findPlateAngle( oInput, rAngle, bBlackChar )  )
    {
    	oOutput = oInput.clone();
        return;
    }

    // Source image
    IplImage oSrcIpl = oInput;

    // Destination image
    oOutput.create( oInput.size(), oInput.type() );
    IplImage oDstIpl = oOutput;

    if( rAngle >= 45 )
    {
        FTS_BASE_Util::Rotate( &oSrcIpl, &oDstIpl, (90 - rAngle), true );
    }
    else
    {
    	FTS_BASE_Util::Rotate( &oSrcIpl, &oDstIpl, (360 - rAngle), true );
    }
}

bool FTS_ANPR_Rotate::findPlateAngle( const cv::Mat& oSrc, double& rAngle, bool bBlackChar )
{
    IplConvKernel* poMorphKernel;

    m_oTempBuffer1.create( oSrc.size(), oSrc.type() );
    IplImage oBinIpl = m_oTempBuffer1;

    IplImage oSrcIpl = oSrc;

    //TODO  get this from the detection directory
/*#ifndef WIN32	
    FTS_CV_SAFE_CALL( cvAdaptiveThreshold( &oSrcIpl, 
                                          &oBinIpl,  
                                          255,       
                                          CV_ADAPTIVE_THRESH_MEAN_C, 
                                          bBlackChar?CV_THRESH_BINARY:CV_THRESH_BINARY_INV, 
                                          21, 
                                          5 ) );
#else										  
	FTS_CV_SAFE_CALL( cvAdaptiveThreshold, &oSrcIpl, 
                                          &oBinIpl,  
                                          255,       
                                          CV_ADAPTIVE_THRESH_MEAN_C, 
                                          bBlackChar?CV_THRESH_BINARY:CV_THRESH_BINARY_INV, 
                                          3, 
                                          5 );
#endif*/										  

	cvThreshold(&oSrcIpl, &oBinIpl, 0, 255, (bBlackChar?CV_THRESH_BINARY:CV_THRESH_BINARY_INV) | CV_THRESH_OTSU);	

    // Variable declaration
    CvRect oBiggestCluster = cvRect(0,0,1,1);

    m_oTempBuffer2.create( oSrc.rows+2, oSrc.cols+2, oSrc.type() );
    IplImage oOtsuExtendedIpl = m_oTempBuffer2;

#ifndef WIN32
	FTS_CV_SAFE_CALL( cvZero( &oOtsuExtendedIpl ) );
    FTS_CV_SAFE_CALL( cvSetImageROI( &oOtsuExtendedIpl, cvRect( 1, 1, oBinIpl.width, oBinIpl.height ) ) );

    FTS_CV_SAFE_CALL( cvCopy( &oBinIpl, &oOtsuExtendedIpl ) );
    FTS_CV_SAFE_CALL( cvResetImageROI( &oOtsuExtendedIpl ) );
#else
    FTS_CV_SAFE_CALL( cvZero, &oOtsuExtendedIpl );
    FTS_CV_SAFE_CALL( cvSetImageROI, &oOtsuExtendedIpl, cvRect( 1, 1, oBinIpl.width, oBinIpl.height ) );

    FTS_CV_SAFE_CALL( cvCopy, &oBinIpl, &oOtsuExtendedIpl );
    FTS_CV_SAFE_CALL( cvResetImageROI, &oOtsuExtendedIpl );
#endif

#ifdef DBUG_L4
    cvNamedWindow( "Binary extended", 1 );
    cvShowImage( "Binary extended", &oOtsuExtendedIpl );
    //cvWaitKey(0);
#endif


//    Declare oContour pointer
    CvSeq* poContour = 0;
    CvSeq* poBiggestContour = 0;

//    Declare a bounding box
    CvRect oBox;

//    Declare a CvContourScanner
    CvContourScanner oContourScanner;

//    Initializes contour scanning process
    oContourScanner = cvStartFindContours( &oOtsuExtendedIpl,
                                           m_poStorage,
                                           sizeof(CvContour),
                                           CV_RETR_EXTERNAL,//CV_RETR_LIST,
                                           CV_CHAIN_APPROX_SIMPLE,
                                           cvPoint(0,0) );

    unsigned int nImgArea = oSrc.cols
                          * oSrc.rows;

    double rAreaFactor = 1.0
                       / nImgArea;

    while(  (poContour = cvFindNextContour( oContourScanner ))  )
    {
        // Finding bounding boxs that meet the ratio tests

        double rClusterSize = fabs( cvContourArea( poContour, CV_WHOLE_SEQ ) );
        double rContourCoverage = rClusterSize * rAreaFactor;

        if ( rContourCoverage > 0.40 ) //! if contour size is larger than 40% of the image then it's likely to be the plate
        {
            oBox = cvBoundingRect( poContour, 0 );

            if ( oBox.width > oBiggestCluster.width )
            {
                poBiggestContour = poContour;
                oBiggestCluster = oBox;
            }
        }

    } // while(  oContour = cvFindNextContour( oContourScanner )  )

    bool bReturn = false;

    if (  oBiggestCluster.width > (int)ceil( 0.6 * oBinIpl.width )  )
    {
    	CvBox2D o2DRect;
    	if( oSrc.cols < 100 )	// No morphology on small images
    	{
    		o2DRect = cvMinAreaRect2( poBiggestContour, NULL );
    	}
    	else
    	{
			cvZero( &oOtsuExtendedIpl );
			cvDrawContours( &oOtsuExtendedIpl, poBiggestContour, CV_RGB(255,255,255), CV_RGB(255,255,255), 0, -1, 8, cvPoint(0,0) );

			// Free the contour scanner since poBiggestContour will not be used again.
			cvEndFindContours( &oContourScanner );
			cvClearMemStorage( m_poStorage );

			//poMorphKernel = cvCreateStructuringElementEx( 5, 1, 2, 0, CV_SHAPE_RECT );
			poMorphKernel = cvCreateStructuringElementEx( 5, 1, 2, 0, CV_SHAPE_RECT );
			cvMorphologyEx( &oOtsuExtendedIpl, &oOtsuExtendedIpl, NULL, poMorphKernel, CV_MOP_CLOSE,  1 );
			cvReleaseStructuringElement( &poMorphKernel );

			// fill the contour
			oContourScanner = cvStartFindContours( &oOtsuExtendedIpl,
												   m_poStorage,
												   sizeof(CvContour),
												   CV_RETR_EXTERNAL,//CV_RETR_LIST,
												   CV_CHAIN_APPROX_SIMPLE,
												   cvPoint(0,0) );

			// There should only be 1 component now
			poContour = cvFindNextContour( oContourScanner );

			if ( poContour == 0 )
			{
				cvEndFindContours( &oContourScanner );
				return false;
			}
			// Fill the contour
			cvZero( &oOtsuExtendedIpl );
			cvDrawContours( &oOtsuExtendedIpl, poContour, CV_RGB(255,255,255), CV_RGB(255,255,255), 0, -1, 8, cvPoint(0,0) );

			// Free the contour scanner since poBiggestContour will not be used again.
			cvEndFindContours( &oContourScanner );
			cvClearMemStorage( m_poStorage );

			poMorphKernel = cvCreateStructuringElementEx( 19, 13, 9, 6, CV_SHAPE_RECT );
			cvMorphologyEx( &oOtsuExtendedIpl, &oOtsuExtendedIpl, NULL, poMorphKernel, CV_MOP_OPEN,  1 );
			cvReleaseStructuringElement( &poMorphKernel );

	#ifdef DBUG_L4
			cvNamedWindow( "After Morph", 1 );
			cvShowImage  ( "After Morph", &oOtsuExtendedIpl );
	#endif
			// Get the contour of the morphologically processed image
			// ------------------------------------------------

			// Initializes contour scanning process
			oContourScanner = cvStartFindContours( &oOtsuExtendedIpl,
												   m_poStorage,
												   sizeof(CvContour),
												   CV_RETR_EXTERNAL,//CV_RETR_LIST,
												   CV_CHAIN_APPROX_SIMPLE,
												   cvPoint(0,0) );

			poContour = cvFindNextContour( oContourScanner );

			if ( poContour == 0 )
			{
				cvEndFindContours( &oContourScanner );
				return false;
			}

			o2DRect = cvMinAreaRect2( poContour, NULL );
    	}
//    	for(int i = 0 ; i < ( poBiggestContour ? poBiggestContour->total : 0 ) ; i++)
//		{
//    		CvRect *r = (CvRect*)cvGetSeqElem(poBiggestContour, i);
//		}
//    	RotatedRect o2DRect = minAreaRect( Mat(poBiggestContour) );
//
//    	poBiggestContour->

#ifdef DBUG_L4
        IplImage* poHullIpl = cvCreateImage( cvGetSize( &oOtsuExtendedIpl ), 8, 3 );
        cvSetImageROI( poHullIpl, cvRect( 1, 1, oBinIpl.width, oBinIpl.height ) );
        cvCvtColor( &oBinIpl, poHullIpl, CV_GRAY2BGR );
        cvResetImageROI( poHullIpl );

        CvPoint2D32f poPoints2D32F[ 4 ];
        cvBoxPoints( o2DRect, poPoints2D32F );

        CvPoint poPoints[ 4 ];
        poPoints[0] = cvPointFrom32f( poPoints2D32F[ 0 ] );
        poPoints[1] = cvPointFrom32f( poPoints2D32F[ 1 ] );
        poPoints[2] = cvPointFrom32f( poPoints2D32F[ 2 ] );
        poPoints[3] = cvPointFrom32f( poPoints2D32F[ 3 ] );

        int nVertexCount = 4;
        CvPoint* poPointsT = poPoints;

        cvPolyLine( poHullIpl, &poPointsT, &nVertexCount, 1, 1, CV_RGB( 0, 255, 0 ) , 1, 8, 0 );

        cvNamedWindow( "Contour Hull", 1 );
        cvShowImage  ( "Contour Hull", poHullIpl );
        //cvWaitKey( 0 );
		cvReleaseImage( &poHullIpl );
#endif

        if ( fabs(o2DRect.angle) > 1 && fabs(90 - o2DRect.angle) > 1 )
        {
            rAngle = o2DRect.angle;
            rAngle *= (-1.0f);
            bReturn = true;

#ifdef DBUG_L4
            printf( "CC width = %d,   Angle = %f\n", oBiggestCluster.width, rAngle );
#endif
        }
    }

    cvEndFindContours( &oContourScanner );
    cvClearMemStorage( m_poStorage );

    return bReturn;
}









