/*
 *
 *
 * FTS_ANPR_util.h
 *
 *
 */
#include "fts_anpr_util.h"
#include "fts_base_util.h"

#include "opencv/cv.h"
#include "opencv2/core/core.hpp"
#include "fts_base_binarizewolf.h"
using namespace cv;

#include "dirent.h"
#include <algorithm>
#include <string>
//void FTS_ANPR_Util::RemoveSmallConnectedComponents( cv::Mat& oBinary,
//                                                   cv::Mat& oTemp,
//                                                   cv::Mat* poDst,
//                                                   CvMemStorage* poStorage,
//                                                   double rAreaThresh,
//                                                   CvScalar oFillColor,
//                                                   bool bFillContourBoundingBox )
//{
//    // cvStartFindContours() modifies the input image, so we need a copy.
//    oTemp.copy( oBinary );
//    if ( poDst )
//    {
//        poDst->copy( oBinary );
//    }
//    else
//    {
//        poDst = &oBinary;
//    }
//
//    IplImage oTempIpl;   oTemp.wrapWithII( oTempIpl );
//    IplImage oDstIpl;   poDst->wrapWithII( oDstIpl  );
//
//    CvSeq* poContour;
//    CvContourScanner oContourScanner;
//
//    oContourScanner = cvStartFindContours( &oTempIpl,
//                                           poStorage,
//                                           sizeof(CvContour),
//                                           CV_RETR_EXTERNAL, //CV_RETR_LIST,
//                                           CV_CHAIN_APPROX_SIMPLE,
//                                           cvPoint(0,0) );
//
//    // Contour scanning process
//    while (  (poContour = cvFindNextContour( oContourScanner ))  )
//    {
//        double rArea = fabs( cvContourArea( poContour ) );
//
//        if ( rArea >= rAreaThresh )
//        {
//            continue;
//        }
//
//        if ( bFillContourBoundingBox )
//        {
//            // Remove area by drawing the bounding box instead of just the area. Ask Subhash why.
//            cv::Rect oBox = cvBoundingRect( poContour, 0 );
//            cvRectangle( &oDstIpl,
//                         cvPoint( oBox.x, oBox.y ),
//                         cvPoint( oBox.x + oBox.width  - 1,
//                                  oBox.y + oBox.height - 1 ),
//                         oFillColor,
//                         CV_FILLED
//                         );
//        }
//        else
//        {
//            SN_EXIT( "bFillContourBoundingBox == false is not supported yet" );
//        }
//
//    } // while more contours
//
//    cvEndFindContours( &oContourScanner );
//
//}


bool FTS_ANPR_Util::RobustFitLinePDF( const std::vector<CvPoint2D32f>& oPoints,
                                     float rInlierPDFThresh,
                                     std::vector<int>& oInlierFlags )
{
    unsigned int nNumPoints = oPoints.size();

    if ( nNumPoints <= 3 )
    {
        return false;
    }

    //
    // Find two points with minimum sum of absolute y difference between previous and next char
    // ie. y coord               =  5  6  9 11 20 21 20
    //                                  --------------------------------
    //     abs y diff with prev  =     1  3  2  9  1  1
    //     abs y diff with next  =     3  2  9  1  1
    //                              --------------------------------
    //     sum y diffs           =     4  5 11 10  2
    //     find 2 minimums       =     4           2
    // Fit line to this two points     ^           ^
    // ------------------------------------------------------------------------------------------------

    const unsigned int nNumDiffs = nNumPoints - 2;

#ifndef WIN32
    FTS_BASE_STACK_ARRAY( float, nNumDiffs, oSumYDiffToPrevAndNextChar );
#else
	vector<float> oSumYDiffToPrevAndNextChar(nNumDiffs);
#endif

    for ( unsigned int i = 0; i < nNumDiffs; ++i )
    {
        oSumYDiffToPrevAndNextChar.at( i ) = fabs( oPoints.at( i+1 ).y - oPoints.at( i   ).y )
                                           + fabs( oPoints.at( i+1 ).y - oPoints.at( i+2 ).y );

    }

    const unsigned int nNumFitPoints = 2;

    FTS_BASE_STACK_ARRAY( CvPoint2D32f, nNumFitPoints, oFitPoints );

    float rMaxDiff = *std::max_element( oSumYDiffToPrevAndNextChar.begin(),
                                        oSumYDiffToPrevAndNextChar.end  () );


    // Get the nNumFitPoints smallest differences
    for( unsigned int i = 0; i < nNumFitPoints; ++i )
    {
        int nMinIdx = std::distance(                   oSumYDiffToPrevAndNextChar.begin(),
                                     std::min_element( oSumYDiffToPrevAndNextChar.begin(),
                                                       oSumYDiffToPrevAndNextChar.end  ()
                                                       )
                                     );

        oFitPoints.at( i ) = oPoints.at( nMinIdx );

        // Mask out the minimum;
        oSumYDiffToPrevAndNextChar.at( nMinIdx ) = rMaxDiff * 2.0f;
    }

    //
    // Now use two chosen points to find the best-fit line
    // ------------------------------------------------------------------------------------------------

    float prLine[4];
    cvFitLine2D( oFitPoints.begin(), nNumFitPoints, CV_DIST_L2, 0, 0.0, 0.0, prLine);

    CvPoint2D32f oLineDir;
    oLineDir.x = prLine[ 0 ];    // the normalized vector that is collinear to the line
    oLineDir.y = prLine[ 1 ];

#ifndef WIN32
    FTS_BASE_STACK_ARRAY( float, nNumPoints, oYProjection );
#else
	vector<float> oYProjection(nNumPoints);
#endif

    //
    // Calculate Y-projection on the new axises
    // For now just use the Y-centre
    // ------------------------------------------------------------------------------------------------
    for( unsigned int i = 0; i < nNumPoints; ++i )
    {
        oYProjection.at( i ) = oLineDir.x * oPoints.at( i ).y
                             - oLineDir.y * oPoints.at( i ).x;
    }

    //
    // Find mean and variance of the project onto the y-axis
    // ------------------------------------------------------------------------------------------------
    float rMean = std::accumulate( oYProjection.begin(),
                                   oYProjection.end  (),
                                   0.0f  );

    rMean /= nNumPoints;


    float rSum = 0;
    for( unsigned int i = 0; i < nNumPoints; ++i )
    {
        rSum +=   ( oYProjection.at( i ) - rMean )
                * ( oYProjection.at( i ) - rMean );
    }

    float rSigma = std::max( 1.5f, (float) sqrt( rSum / nNumPoints ) );

#ifndef WIN32
    FTS_BASE_STACK_ARRAY( float, nNumPoints, oPDF );
#else
	vector<float> oPDF(nNumPoints);
#endif

    //
    // Calculate pdf f(x) = 1/(sigma*sqrt(2*PI)).e^( -0.5 * (x-mean)^2 / sigma^2 )
    // But since we will scale the maximum pdf to 1, we can drop the scalling constant.
    // ------------------------------------------------------------------------------------------------
    float rSigma2 = rSigma * rSigma;
    for( unsigned int i = 0; i < nNumPoints; ++i )
    {
        float z = oYProjection.at( i ) - rMean;

        oPDF.at( i ) = exp( -0.5f * z * z / rSigma2 );
    }

    //
    // Normalize PDF such that maximum PDF value is 1
    // ------------------------------------------------------------------------------------------------
    float rMaxDPF = *std::max_element( oPDF.begin(), oPDF.end() );

    for( unsigned int i = 0; i < nNumPoints; ++i )
    {
        oPDF.at( i ) /= rMaxDPF;

        #ifdef DBUG_L3
        SN_P( oPDF.at( i ) );
        #endif
    }

    //
    // Find outliers
    // ------------------------------------------------------------------------------------------------
    oInlierFlags.resize( nNumPoints );
    for( unsigned int i = 0; i < nNumPoints; ++i )
    {
        oInlierFlags.at( i ) =  ( oPDF.at( i ) > rInlierPDFThresh );
    }

    return true;

}


void FTS_ANPR_Util::RemoveHorizontalLongLines( cv::Mat& oImage,
                                               cv::Rect oROI,
                                               unsigned int nLongLineLengthThreshold )
{
	assert( oImage.type() == CV_8UC1 );

    for( int y = 0; y < oROI.height; ++y )
    {
//        unsigned char* pcRow = oImage.pixel( oROI.x,  oROI.y + y );
    	unsigned char* pcRow = (unsigned char*)oImage.ptr<unsigned char>( y );

        int x = 0;
        while( x < oROI.width )
        {
            // Move till white pixel
            // ----------------------------------------------------------------
            while(  x < oROI.width  &&  pcRow[ oROI.x + x ] == 0  ) ++x;
            int nFirst = x;

            // Move till black pixel
            // ----------------------------------------------------------------
            while(  x < oROI.width  &&  pcRow[ oROI.x +  x ] != 0  ) ++x;
            int nLast = x;

            // Remove line if longer than threshold
            // ----------------------------------------------------------------
            if( nLast - nFirst > (int)nLongLineLengthThreshold )
            {
                // This small loop could be faster that memcpy, no function call
                // overhead.
                for( int i = nFirst; i < nLast; ++i )
                {
                    pcRow[ oROI.x + i ] = 0;
                }
            }

        } // while

    } // for each row
}


void FTS_ANPR_Util::RemoveVerticalLongLines( cv::Mat& oImage,
                                             cv::Rect oROI,
                                             unsigned int nLongLineLengthThreshold )
{
    assert( oImage.type() == CV_8UC1 );

    FTS_BASE_Util::CropRect( oROI, oImage.cols, oImage.rows );

    for( int x = 0; x < oROI.width; ++x )
    {
    	unsigned char cRowCol = oImage.at<unsigned char>( oROI.x + x,  oROI.y );

        int y = 0;
        while( y < oROI.height )
        {
            // Move till white pixel
            // ----------------------------------------------------------------
            while(  y < oROI.height  &&  cRowCol == 0  )
            {
                ++y;
                cRowCol = oImage.at<unsigned char>( oROI.x + x,  oROI.y + y );
            }
            int nFirst = y;

            // Move till black pixel
            // ----------------------------------------------------------------
            while(  y < oROI.height  &&  cRowCol != 0  )
            {
                ++y;
                cRowCol = oImage.at<unsigned char>( oROI.x + x,  oROI.y + y );
            }
            int nLast = y;

            // Remove line if longer than threshold
            // ----------------------------------------------------------------
            if( nLast - nFirst > (int) nLongLineLengthThreshold )
            {
                for( int i = nFirst; i < nLast; i++ )
                {
                	oImage.at<unsigned char>( oROI.x + x,  oROI.y + i ) = 0;
                }
            }

        } // while

    } // for each row

}

int FTS_ANPR_Util::RemoveLeftRightEdge(
        IplImage* poBinSrcImg,
        IplImage* poBinDstImg,
        int iEdgeLengthThreshold,
        float fEdgeHeightRatio )
{
    /**
     * Declare data
     */
    IplImage* poBinCopyImg = cvCloneImage(poBinSrcImg);
    int iRow, iCol;
    int iCurRow;
    int iImgWidth = poBinCopyImg->width;
    int iImgHeight = poBinCopyImg->height;
#ifndef WIN32
    int ppiPixelVal[iImgHeight][iImgWidth];
#else
	Mat_<int> ppiPixelVal(iImgHeight, iImgWidth);
#endif

#ifndef WIN32
    line_segment_struct ppoVerticalLineSegment[iImgWidth][iImgHeight];
#else
	Mat_<line_segment_struct> ppoVerticalLineSegment(iImgWidth, iImgHeight);
#endif
    for (iCol=0; iCol<iImgWidth; iCol++)
    {
        memset(ppoVerticalLineSegment[iCol], 0, sizeof(line_segment_struct) * iImgHeight);
    }

    /**
     * Do vertical scan
     */
    /* Convert poBinCopyImg image into 2D array */
    for(iRow = 0; iRow < iImgHeight; iRow++)
    {
        char *pcPtr = &poBinCopyImg->imageData[poBinCopyImg->widthStep*iRow];

        for(iCol = 0; iCol < iImgWidth; iCol++)
        {
            if( (unsigned char)pcPtr[iCol] >= FTS_BASE_Util::BINARYMAXVALUE)
            {
                ppiPixelVal[iRow][iCol] = 1;
            }
            else
            {
                ppiPixelVal[iRow][iCol] = 0;
            }
        }
    }

    /* Do scan */
    iCurRow = 0;
    int iLabel = 0;
    for (iCol=0; iCol<iImgWidth; iCol++)
    {
        iRow=0;
        while (1)
        {
            /* Scan until detecting 1 */
            while (ppiPixelVal[iCurRow][iCol] == 0)
            {
                iCurRow++;
                if (iCurRow==iImgHeight) break;
            }

            if (iCurRow==iImgHeight)
            {
                break;
            }

            /* Mark the iStart position */
            ppoVerticalLineSegment[iCol][iRow].iLabel = 1;
            ppoVerticalLineSegment[iCol][iRow].iStart = iCurRow;

            /* Look for the iEnd position */
            while(ppiPixelVal[iCurRow][iCol] == 1)
            {
                iCurRow++;
                if (iCurRow==iImgHeight) break;
            }

            ppoVerticalLineSegment[iCol][iRow].iEnd = iCurRow-1;
			
            iRow++;
            iLabel++;
            if (iCurRow==iImgHeight)
            {
                break;
            }
        }

        iCurRow=0;
    }


    /**
     * Remove left and right edge noises
     */
    for (iCol = 0; iCol < iImgWidth; iCol++)
    {
        if ( iCol < iEdgeLengthThreshold || iCol > (iImgWidth - iEdgeLengthThreshold) )
        {
            for (iRow = 0; iRow < iImgHeight; iRow++)
            {
                if (ppoVerticalLineSegment[iCol][iRow].iLabel == 0) break;
                //printf("At column %d Length = %d\n", iCol, ppoVerticalLineSegment[iCol][iRow].iEnd - ppoVerticalLineSegment[iCol][iRow].iStart);
                if (ppoVerticalLineSegment[iCol][iRow].iEnd - ppoVerticalLineSegment[iCol][iRow].iStart > iImgHeight*fEdgeHeightRatio)
                {
                    //printf("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT\n");
                    int iSegPos;
                    for(iSegPos = ppoVerticalLineSegment[iCol][iRow].iStart; iSegPos<=ppoVerticalLineSegment[iCol][iRow].iEnd; iSegPos++)
                    {
                        poBinCopyImg->imageData[iCol + poBinCopyImg->widthStep*iSegPos] = 0;
                    }
                }
            }
        }
    }

    /**
     * Store the result to destination image
     */
    #ifdef DBUG_L3
        SN_P("cvCopy(poBinCopyImg, poBinDstImg, NULL)");
    #endif
    cvCopy(poBinCopyImg, poBinDstImg, NULL);

    /**
     * Release images
     */
    cvReleaseImage(&poBinCopyImg);

    return 1;
}

int FTS_ANPR_Util::RemoveHorizontalLongLineOld(
                              IplImage* poBinSrcImg,
                              IplImage* poBinDstImg,
                              int iEdgeLengthThreshold,
                              int iLongLineThreshold )
{
    /**
     * Declare data
     */
    IplImage* poBinCopyImg = cvCloneImage(poBinSrcImg);
    int iRow, iCol;
    int iCurCol;
    int iImgWidth = poBinCopyImg->width;
    int iImgHeight = poBinCopyImg->height;
#ifndef WIN32
    int ppiPixelVal[iImgHeight][iImgWidth];
#else
	Mat_<int> ppiPixelVal(iImgHeight,iImgWidth);
#endif

#ifndef WIN32
    line_segment_struct ppoVerticalLineSegment[iImgHeight][iImgWidth];
#else
	Mat_<line_segment_struct> ppoVerticalLineSegment(iImgHeight,iImgWidth);
#endif
    for (iRow=0; iRow<iImgHeight; iRow++)
    {
        memset(ppoVerticalLineSegment[iRow], 0, sizeof(line_segment_struct) * iImgWidth);
    }

    /**
     * Do horizontal scan
     */
    /* Convert poBinCopyImg image into 2D array */
    for(iRow = 0; iRow < iImgHeight; iRow++)
    {
        char *pcPtr = &poBinCopyImg->imageData[poBinCopyImg->widthStep*iRow];
        for(iCol = 0; iCol < iImgWidth; iCol++)
        {
            if( (unsigned char)pcPtr[iCol] >= FTS_BASE_Util::BINARYMAXVALUE)
            {
                ppiPixelVal[iRow][iCol] = 1;
            }
            else
            {
                ppiPixelVal[iRow][iCol] = 0;
            }
        }
    }

    /* Do scan */
    iCurCol = 0;
    int iLabel = 0;
    for (iRow=0; iRow<iImgHeight; iRow++)
    {
        iCol=0;
        while (1)
        {
            /* Scan until detecting 1 */
            while (ppiPixelVal[iRow][iCurCol] == 0)
            {
                iCurCol++;
                if (iCurCol==iImgWidth) break;
            }

            if (iCurCol==iImgWidth)
            {
                break;
            }

            /* Mark the iStart position */
            ppoVerticalLineSegment[iRow][iCol].iLabel = 1;
            ppoVerticalLineSegment[iRow][iCol].iStart = iCurCol;

            #ifdef DBUG_L3
            //printf("Starting Col = %d\n", ppoVerticalLineSegment[iRow][iCol].iStart);
            #endif
            /* Look for the iEnd position */
            while(ppiPixelVal[iRow][iCurCol] == 1)
            {
                //printf("I shouldnt see this line, iCurCol = %d\n", iCurCol);
                iCurCol++;
                if (iCurCol==iImgWidth) break;
            }

            ppoVerticalLineSegment[iRow][iCol].iEnd = iCurCol-1;
            #ifdef DBUG_L3
            //printf("End Col = %d\n", ppoVerticalLineSegment[iRow][iCol].iEnd);
            #endif
			
            iCol++;
            iLabel++;
            if (iCurCol==iImgWidth)
            {
                break;
            }
        }

        iCurCol=0;
    }

    /**
     * Remove long line noises
     */
    for (iRow = 0; iRow < iImgHeight; iRow++)
    {
        if ( iRow < iEdgeLengthThreshold || iRow > (iImgHeight - iEdgeLengthThreshold) )
        {
            char *pcPtr = &poBinCopyImg->imageData[poBinCopyImg->widthStep*iRow];
            for (iCol = 0; iCol < iImgWidth; iCol++)
            {
                if (ppoVerticalLineSegment[iRow][iCol].iLabel == 0) break;
//              #ifdef DBUG_L3
//              printf("At column %d Length = %d\n", iCol, ppoVerticalLineSegment[iCol][iRow].iEnd - ppoVerticalLineSegment[iCol][iRow].iStart);
//              #endif
                if (ppoVerticalLineSegment[iRow][iCol].iEnd - ppoVerticalLineSegment[iRow][iCol].iStart > iLongLineThreshold)
                {
                    //printf("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT\n");
                    int iSegPos;
                    for(iSegPos = ppoVerticalLineSegment[iRow][iCol].iStart; iSegPos<=ppoVerticalLineSegment[iRow][iCol].iEnd; iSegPos++)
                    {
                        pcPtr[iSegPos] = 0;
                    }
                }
            }
        }
    }

    /**
     * Store the result to destination image
     */
    #ifdef DBUG_L3
        SN_P("cvCopy(poBinCopyImg, poBinDstImg, NULL))");
    #endif
    cvCopy(poBinCopyImg, poBinDstImg, NULL);

    /**
     * Release images
     */
    cvReleaseImage(&poBinCopyImg);

    return 1;
}

// This function find the horizontal linear cut to separate two layers of a dual plate
// Return the row index if found or -1 if error
int FTS_ANPR_Util::HorzLinearCut(
        IplImage* poGreyImg,
        bool bBlackChar,
        int nStartRow,
        int nEndRow )
{
    if ( nEndRow <= nStartRow )
        return -1;

    int nRow, nCol;
    //int nSumRow[nEndRow - nStartRow];
	int* nSumRow = new int[nEndRow - nStartRow];
    int nCount = 0;

    for ( nRow = nStartRow; nRow < nEndRow; nRow++, nCount++ )
    {
        nSumRow[nCount] = 0;
        for ( nCol = 0; nCol < poGreyImg->width; nCol++ )
            nSumRow[nCount] += ((uchar*)((poGreyImg)->imageData + (poGreyImg)->widthStep*nRow))[nCol];
    }

	int ret;
    if( bBlackChar )
    {
        ret = nStartRow + FTS_BASE_MaxElementIndex( nSumRow, nSumRow + nEndRow - nStartRow );
    }
    else
    {
        ret = nStartRow + FTS_BASE_MinElementIndex( nSumRow, nSumRow + nEndRow - nStartRow );
    }

	delete nSumRow;
	return ret;
}

double FTS_ANPR_Util::ComputeSkew(const cv::Mat& src, bool bDisplayRes)
{
	//// Load in grayscale.
	//cv::Mat src = cv::imread(filename, 0);
	cv::Size size = src.size();   
	//cv::bitwise_not(src, src);

	std::vector<cv::Vec4i> lines;
	cv::HoughLinesP(src, lines, 1, CV_PI/180, 100, size.width / 2.f, 20);

	cv::Mat disp_lines(size, CV_8UC1, cv::Scalar(0, 0, 0));
    double angle = 0.;
    unsigned nb_lines = lines.size();
    for (unsigned i = 0; i < nb_lines; ++i)
    {
        cv::line(disp_lines, cv::Point(lines[i][0], lines[i][1]),
                 cv::Point(lines[i][2], lines[i][3]), cv::Scalar(255, 0 ,0));
        angle += atan2((double)lines[i][3] - lines[i][1],
                       (double)lines[i][2] - lines[i][0]);
    }
    angle /= nb_lines; // mean angle, in radians.

	if(bDisplayRes)
	{
		//std::cout << "File " << filename << ": " << angle * 180 / CV_PI << std::endl;		
		cv::imshow("Skew angle", disp_lines);
		//cv::waitKey(0);
		//cv::destroyWindow("Skew angle");
	}
	return angle * 180 / CV_PI;
}

void FTS_ANPR_Util::Deskew(cv::Mat& src, double angle, cv::Mat& cropped)
{
	//cv::Mat img = cv::imread(filename, 0);
	//cv::bitwise_not(img, img);
 
	std::vector<cv::Point> points;
	cv::Mat_<uchar>::iterator it = src.begin<uchar>();
	cv::Mat_<uchar>::iterator end = src.end<uchar>();
	for (; it != end; ++it)
	{
		if (*it)
			points.push_back(it.pos());
	}

	cv::RotatedRect box = cv::minAreaRect(cv::Mat(points));

	cv::Mat rot_mat = cv::getRotationMatrix2D(box.center, angle, 1);

	cv::Mat rotated;
	cv::warpAffine(src, rotated, rot_mat, src.size(), cv::INTER_CUBIC);

	cv::Size box_size = box.size;
	if (box.angle < -45.)
		std::swap(box_size.width, box_size.height);
	//cv::Mat cropped;
	cv::getRectSubPix(rotated, box_size, box.center, cropped);

	return;
}

//double FTS_ANPR_Util::CalcOtsuThreshold(const cv::Mat& src)
//{
//	int nbHistLevels = 256;
//	// calculate histogram
//	int* histData = (int*)calloc(nbHistLevels, sizeof(int));
//	IplImage* img=cvCloneImage(&(IplImage)src);
//	int ptr = 0;
//	while (ptr < img->imageSize) {
//		int h = img->imageData[ptr];
//		histData[h]++;
//		ptr ++;
//	}
//
//	// total number of pixels
//	int total = img->imageSize;
//
//	float sum = 0;
//	int t;
//	for (t=0; t < nbHistLevels; t++)
//		sum += t * histData[t];
//
//	float sumB = 0;
//	int wB = 0;
//	int wF = 0;
//	float varMax = 0;
//	int threshold = 0;
//
//	for (t=0; t < nbHistLevels; t++) {
//		wB += histData[t];               // Weight Background
//		if (wB == 0)
//			continue;
//
//		wF = total - wB;                 // Weight Foreground
//		if (wF == 0)
//			break;
//
//		sumB += (float) (t * histData[t]);
//
//		float mB = sumB / wB;            // Mean Background
//		float mF = (sum - sumB) / wF;    // Mean Foreground
//
//		// Calculate Between Class Variance
//		float varBetween = (float)wB * (float)wF * (mB - mF) * (mB - mF);
//
//		// Check if new maximum found
//		if (varBetween > varMax) {
//			varMax = varBetween;
//			threshold = t;
//		}
//	}
//	printf("threshold=%d\n", threshold);
//
//	return threshold;
//}

double FTS_ANPR_Util::SumMatRows(cv::Mat mat, std::vector<int>& hist) {
	uchar *p;
	double maxHist = 0;
	for (int i = 0; i < mat.rows; i++)
	{
		p = mat.ptr<uchar>(i);
		int sum = 0;
		for (int j = 0; j < mat.cols; j++)
		{
			if (p[j] > 0) {
				sum += 1;
			}
			
		}
		if (sum > maxHist) {
			maxHist = sum;
		}
		hist.push_back(sum);
	}
	return maxHist;
}

//void FTS_ANPR_Util::LPMiddleCut(const cv::Mat&  imgGrayFullLp, cv::Mat &imgUpperLp, cv::Mat &imgLowerLp) {
bool FTS_ANPR_Util::LPMiddleCut(const cv::Mat&  imgGrayFullLp, cv::Mat &imgUpperLp, cv::Mat &imgLowerLp, bool bPreferCenter) 
{
	int numThres = 3;
	double step = 10;
	int width = imgGrayFullLp.size().width;
	int height = imgGrayFullLp.size().height;
		
	cv::Mat imgRowCombination = cv::Mat::zeros(imgGrayFullLp.size().height, imgGrayFullLp.size().width*numThres, CV_8UC1);

	vector<double> thresholds;
	cv::Mat tmp = imgGrayFullLp.clone();

	int extend = imgGrayFullLp.size().height / 20;

	//double t0 = CalcOtsuThreshold_(tmp);
	double t0 = 127.0;
	//std::cout << "Otsu: " << t0 << std::endl;
	//thresholds.push_back(t0);
	cv::Mat imgBin;
	int ind = 0;
	int win = std::min(15, std::min(height, width));
	int k = 0;

	// Thresholding
	cv::Rect roi(0, 0, imgGrayFullLp.size().width, imgGrayFullLp.size().height);
	imgBin = imgRowCombination(roi);
	threshold(imgGrayFullLp, imgBin, 127, 255, THRESH_OTSU);

	roi = cv::Rect(width, 0, imgGrayFullLp.size().width, imgGrayFullLp.size().height);
	//imgBin = imgRowCombination(roi);
	FTS_BASE_BinarizeWolf(imgGrayFullLp, imgRowCombination(roi), WOLFJOLION, win, win, 0.05);

	roi = cv::Rect(2*width, 0, imgGrayFullLp.size().width, imgGrayFullLp.size().height);
	//imgBin = imgRowCombination(roi);
	FTS_BASE_BinarizeWolf(imgGrayFullLp, imgRowCombination(roi), SAUVOLA, win, win, 0.3);

	imgRowCombination = 255 - imgRowCombination;
	//for (int i = -std::floor(numThres / 2); i <= std::floor(numThres / 2); i++)
	//{
	//	double th = t0 + step*i;
	//	//std::cout << "   Threshold: " << th << std::endl;		
	//	cv::Rect roi(width*ind, 0, imgGrayFullLp.size().width, imgGrayFullLp.size().height);
	//	imgBin = imgRowCombination(roi);
	//	threshold(imgGrayFullLp, imgBin, th, 255, THRESH_BINARY_INV);
	//	ind++;
	//}

	std::vector<int> hist;
	//cv::reduce(imgRowCombination, hist, 1, CV_REDUCE_SUM, -1);
	double maxHist;
	maxHist = SumMatRows(imgRowCombination, hist);
	/*hist.erase(hist.begin(), hist.begin() + hist.size() / 4);
	hist.erase(hist.end() - hist.size() / 4, hist.end());*/
	//
	int meanDist = hist.size() / 2;

	for (int i = 0; i < hist.size(); i++)
	{
		int minDistToEdge = std::min(i + 1, int(hist.size() - i + 1));
		int dist = std::abs(i - meanDist);
		if (minDistToEdge > hist.size() / 3) 
		{
			hist[i] += dist;
		}
		else
		{
			hist[i] = maxHist + meanDist;
		}
	}
		
	std::vector<int> sortedIdx;
	int chosenInd = -1;
	if (bPreferCenter)
	{
		// Find 3 smallest valleys to be candidates of the middle cut
		// Choose the middle candidate to avoid top and bottom margins
		std::vector<int> valleyIdx;
		std::vector<int> valley;
		for (int i = 1; i < hist.size()-1; i++)
		{
			if (hist[i] >= hist[i - 1] && hist[i] < hist[i + 1]) {
				valley.push_back(hist[i]);
				valleyIdx.push_back(i);
			}			
		}
		
		cv::sortIdx(valley, sortedIdx, CV_SORT_ASCENDING);

		int minDistToCenter = 10000;
		for (int i = 0; i < valley.size(); i++)
		{
			int y = valleyIdx[sortedIdx[i]];
			int dist = std::abs(y - meanDist);

			//int minDistToEdge = std::min(y, int(hist.size() - y));
			if (dist < minDistToCenter)
			{
				minDistToCenter = dist;
				chosenInd = valleyIdx[sortedIdx[i]];
			}
		}
	}
	else
	{
		cv::sortIdx(hist, sortedIdx, CV_SORT_ASCENDING);
		chosenInd = sortedIdx[0];
	}

	//int numCands = 3;
	if (chosenInd == -1)
	{
		//imgGrayFullLp.copyTo(imgUpperLp);
		//imgGrayFullLp.copyTo(imgLowerLp);
		return false; //plate image source is one row
	}
	/*double minV, maxV;
	Point minLoc, maxLoc;
	cv::minMaxLoc(hist, &minV, &maxV, &minLoc, &maxLoc);*/

	double midY = chosenInd;

	int upperMinY = 0;
	int upperMaxY = std::min(int(midY) + extend, height - 1);

	int lowerMinY = std::max(int(midY) - extend, 0);
	int lowerMaxY = height - 1;

	int upperHeight = upperMaxY - upperMinY + 1;
	int lowerHeight = lowerMaxY - lowerMinY + 1;
#ifdef DBUG_L4
	std::cout << "upperMinY = " << upperMinY << " upperMaxY = " << upperMaxY << " Height = " << upperHeight << std::endl;
	std::cout << "lowerMinY = " << lowerMinY << " lowerMaxY = " << lowerMaxY << " Height = " << lowerHeight << std::endl;
#endif
	cv::Rect upperRect(0, upperMinY, imgGrayFullLp.size().width, upperHeight);
	cv::Rect lowerRect(0, lowerMinY, imgGrayFullLp.size().width, lowerHeight);

	/*cv::Mat imgROIUpper(imgUpperLp, upperRect);
	cv::Mat imgROILower(imgLowerLp, lowerRect);*/

	imgUpperLp.create(upperHeight, imgGrayFullLp.size().width, imgGrayFullLp.type());
	imgLowerLp.create(lowerHeight, imgGrayFullLp.size().width, imgGrayFullLp.type());

	imgGrayFullLp(upperRect).copyTo(imgUpperLp);
	imgGrayFullLp(lowerRect).copyTo(imgLowerLp);
	/*imgGrayFullLp(upperRect).copyTo(imgROIUpper);
	imgGrayFullLp(lowerRect).copyTo(imgROILower);*/
		
#ifdef DBUG_L4
	cv::Mat histImage = cv::Mat::zeros(hist.size(), maxHist, CV_8UC3);

	for (int i = 0; i < hist.size(); i++)
	{
		line(histImage, cv::Point(0,i), Point(hist[i],i),Scalar(255, 0, 0), 1, 8, 0);
	}
	line(histImage, cv::Point(0, chosenInd), Point(hist[chosenInd], chosenInd), Scalar(0, 0, 255), 1, 8, 0);
	/*for (int i = 0; i < valley.size(); i++)
	{
		int j = valleyIdx[sortedIdx[i]];
		line(histImage, cv::Point(0, j), Point(hist[j], j), Scalar(0, 0, 255), 1, 8, 0);
	}*/

	imshow("Histogram", histImage);
#endif
	//tmp.release();
	/*imgBin.release();
	imgROIUpper.release();
	imgROILower.release();
	hist.clear();
	imgRowCombination.release();*/
		
	/*imgUpperLp.create(upperHeight, imgGrayFullLp.size().width, imgGrayFullLp.type());
	imgLowerLp.create(lowerHeight, imgGrayFullLp.size().width, imgGrayFullLp.type());*/
	/*imgUpperLp = cv::Mat::zeros(upperHeight, imgGrayFullLp.size().width, CV_8U);
	imgLowerLp = cv::Mat::zeros(lowerHeight, imgGrayFullLp.size().width, CV_8U);*/
	/*imgGrayFullLp(upperRect).copyTo(imgUpperLp);
	imgGrayFullLp(lowerRect).copyTo(imgLowerLp);*/
	/*imgGrayFullLp(cv::Rect(0, 0, imgUpperLp.size().width, imgUpperLp.size().height)).copyTo(imgUpperLp);
	imgGrayFullLp(cv::Rect(0, midY - 5, imgUpperLp.size().width, imgUpperLp.size().height)).copyTo(imgLowerLp);*/

	return true; //can cut input plate image to 2 rows
}

std::string FTS_ANPR_Util::toLowerCase(const std::string& in) {
	std::string t;
	for (std::string::const_iterator i = in.begin(); i != in.end(); ++i) {
		t += tolower(*i);
	}
	return t;
}

void FTS_ANPR_Util::getFilesInDirectory(const std::string& dirName, std::vector<std::string>& fileNames, const std::vector<std::string>& validExtensions){
	printf("Opening directory %s\n", dirName.c_str());
	struct dirent* ep;
	size_t extensionLocation;
	DIR* dp = opendir(dirName.c_str());
	if (dp != NULL) {
		while ((ep = readdir(dp))) {
			// Ignore (sub-)directories like . , .. , .svn, etc.
			if (ep->d_type & DT_DIR) {
				continue;
			}
			extensionLocation = std::string(ep->d_name).find_last_of("."); // Assume the last point marks beginning of extension like file.ext
			// Check if extension is matching the wanted ones
			std::string tempExt = toLowerCase(std::string(ep->d_name).substr(extensionLocation + 1));
			if (find(validExtensions.begin(), validExtensions.end(), tempExt) != validExtensions.end()) {
//				printf("Found matching data file '%s'\n", ep->d_name);
				fileNames.push_back((std::string)dirName + ep->d_name);
			}
			else {
				printf("Found file does not match required file type, skipping: '%s'\n", ep->d_name);
			}
		}
		(void)closedir(dp);
	}
	else {
		printf("Error opening directory '%s'!\n", dirName.c_str());
	}
	return;
}

int FTS_ANPR_Util::findMedianBlobWidthOfWoHInRange( const vector<FTS_IP_SimpleBlobDetector::SimpleBlob> oBlobs,
													const float rMinWoH,
													const float rMaxWoH,
													const int nMinMedian )
{
	vector<FTS_IP_SimpleBlobDetector::SimpleBlob> oAcceptedBlobs;
	for( size_t i = 0; i < oBlobs.size(); i++ )
	{
		float rWoH = (float)oBlobs[i].oBB.width / (float)oBlobs[i].oBB.height;
		if( rWoH >= rMinWoH && rWoH <= rMaxWoH )
		{
			oAcceptedBlobs.push_back( oBlobs[i] );
		}
	}

	if( !oAcceptedBlobs.empty() )
	{
		return findMedianBlobWidth( oAcceptedBlobs, nMinMedian );
	}

	return findMedianBlobWidth( oBlobs, nMinMedian );
}

int FTS_ANPR_Util::findMedianBBWidthOfWoHInRange( const vector<Rect> oBlobs,
												  const float rMinWoH,
												  const float rMaxWoH,
												  const int nMinMedian )
{
	vector< Rect > oAcceptedBlobs;
	for( size_t i = 0; i < oBlobs.size(); i++ )
	{
		float rWoH = (float)oBlobs[i].width / (float)oBlobs[i].height;
		if( rWoH >= rMinWoH && rWoH <= rMaxWoH )
		{
			oAcceptedBlobs.push_back( oBlobs[i] );
		}
	}

	if( !oAcceptedBlobs.empty() )
	{
		return findMedianBBWidth( oAcceptedBlobs, nMinMedian );
	}

	return findMedianBBWidth( oBlobs, nMinMedian );
}

int FTS_ANPR_Util::findMedianBlobWidth( const vector<FTS_IP_SimpleBlobDetector::SimpleBlob> oBlobs, const int nMinMedian )
{
	FTS_BASE_STACK_ARRAY( int, oBlobs.size(), oWidths );
	fillBlobWidthArray( oWidths, oBlobs );
	return max( nMinMedian, FTS_BASE_MedianBiasHigh( oWidths ) );
}

int FTS_ANPR_Util::findMedianBBWidth( const vector<Rect> oBlobs, const int nMinMedian )
{
	FTS_BASE_STACK_ARRAY( int, oBlobs.size(), oWidths );
	fillBBWidthArray( oWidths, oBlobs );
	return max( nMinMedian, FTS_BASE_MedianBiasHigh( oWidths ) );
}

int FTS_ANPR_Util::findMedianBlobHeight( const vector<FTS_IP_SimpleBlobDetector::SimpleBlob> oBlobs, const int nMinMedian )
{
	FTS_BASE_STACK_ARRAY( int, oBlobs.size(), oHeights );
	fillBlobHeightArray( oHeights, oBlobs );
	return max( nMinMedian, FTS_BASE_MedianBiasHigh( oHeights ) );
}

int FTS_ANPR_Util::findMedianBBHeight( const vector<Rect> oBlobs, const int nMinMedian )
{
	FTS_BASE_STACK_ARRAY( int, oBlobs.size(), oHeights );
	fillBBHeightArray( oHeights, oBlobs );
	return max( nMinMedian, FTS_BASE_MedianBiasHigh( oHeights ) );
}

void FTS_ANPR_Util::fillBlobWidthArray( FTS_BASE_StackArray<int>& oArray,
						 	 	 	    const vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& blobs )
{
    assert( oArray.size() == blobs.size() );

    for( unsigned int i = 0; i < blobs.size(); i++ )
	{
    	oArray.at(i) = blobs[i].oBB.width;
	}
}

void FTS_ANPR_Util::fillBBWidthArray( FTS_BASE_StackArray<int>& oArray,
						  	  	  	  const vector<Rect>& blobs )
{
    assert( oArray.size() == blobs.size() );

    for( unsigned int i = 0; i < blobs.size(); i++ )
	{
    	oArray.at(i) = blobs[i].width;
	}
}

void FTS_ANPR_Util::fillBlobHeightArray( FTS_BASE_StackArray<int>& oArray,
						  	  	  	     const vector<FTS_IP_SimpleBlobDetector::SimpleBlob>& blobs )
{
    assert( oArray.size() == blobs.size() );

    for( unsigned int i = 0; i < blobs.size(); i++ )
	{
    	oArray.at(i) = blobs[i].oBB.height;
	}
}

void FTS_ANPR_Util::fillBBHeightArray( FTS_BASE_StackArray<int>& oArray,
						  	  	  	   const vector<Rect>& blobs )
{
    assert( oArray.size() == blobs.size() );

    for( unsigned int i = 0; i < blobs.size(); i++ )
	{
    	oArray.at(i) = blobs[i].height;
	}
}
