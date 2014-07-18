/*
 *
 * sn_nlpr_seg.h
 *
 */

#include "fts_anpr_seg.h"


FTS_ANPR_SegChar::FTS_ANPR_SegChar() :
    m_nTag(0), m_oCharRect(cvRect(-1, -1, -1, -1))
{
    // Nothing
}

FTS_ANPR_SegChar::~FTS_ANPR_SegChar()
{
    // Nothing
}

bool FTS_ANPR_SegChar::clone(const FTS_ANPR_SegChar& r)
{
    m_bClean = r.m_bClean;
    m_nTag = r.m_nTag;
    m_oCharRect = r.m_oCharRect;

    m_oCharBin  = r.m_oCharBin.clone();
    m_oCharGray = r.m_oCharGray.clone();

    return true;
}

void FTS_ANPR_SegChar::crop( const cv::Rect oROI )
{
    m_oCharBin  = m_oCharBin( cv::Rect(oROI) ).clone();;
    m_oCharGray = m_oCharGray( cv::Rect(oROI) ).clone();;

    m_oCharRect.x      += oROI.x;
    m_oCharRect.y      += oROI.y;
    m_oCharRect.width   = oROI.width;
    m_oCharRect.height  = oROI.width;
}






/*
 * Class FTS_ANPR_SegResult: contain Plate Character Segment Result
 *
*/
FTS_ANPR_SegResult::FTS_ANPR_SegResult()
    : m_oPlateSize( cvSize(0,0) )
	, m_bIsTheBestHypo( false )
{
    // Nothing
}

FTS_ANPR_SegResult::FTS_ANPR_SegResult(cv::Size oPlateSize)
    : m_oPlateSize(oPlateSize)
	, m_bIsTheBestHypo( false )
{
    // Nothing
}

FTS_ANPR_SegResult::~FTS_ANPR_SegResult()
{
    clear();
}

void FTS_ANPR_SegResult::clone( const FTS_ANPR_SegResult& oSrc )
{
    m_oPlateSize = oSrc.m_oPlateSize;


    std::list< FTS_ANPR_SegChar* >::const_iterator i  = oSrc.m_oChars.begin();
    std::list< FTS_ANPR_SegChar* >::const_iterator iE = oSrc.m_oChars.end();

    for( ; i != iE; ++i )
    {
    	FTS_ANPR_SegChar* poSegChar = new FTS_ANPR_SegChar();

        poSegChar->clone( *(*i) );

        m_oChars.push_back( poSegChar );
    }

    m_bIsTheBestHypo = oSrc.m_bIsTheBestHypo;
}

void FTS_ANPR_SegResult::clear()
{
    std::for_each( m_oChars.begin(), m_oChars.end(),
                   boost::checked_deleter<FTS_ANPR_SegChar>() );

    m_oChars.clear();
}

void FTS_ANPR_SegResult::getCharTopArray( FTS_BASE_StackArray<int>& oArray ) const
{
    assert( oArray.size() == m_oChars.size() );

    std::list<FTS_ANPR_SegChar*>::const_iterator i  = m_oChars.begin();
    std::list<FTS_ANPR_SegChar*>::const_iterator iE = m_oChars.end();

    for( unsigned int n = 0; i != iE; ++i, ++n )
    {
        oArray.at( n ) = (*i)->m_oCharRect.y;
    }
}

// Return the lower y values of the chars, i.e. (y + height - 1)
void FTS_ANPR_SegResult::getCharBottomArray( FTS_BASE_StackArray<int>& oArray ) const
{
    assert( oArray.size() == m_oChars.size() );

    std::list<FTS_ANPR_SegChar*>::const_iterator i  = m_oChars.begin();
    std::list<FTS_ANPR_SegChar*>::const_iterator iE = m_oChars.end();

    for( unsigned int n = 0; i != iE; ++i, ++n )
    {
        oArray.at( n ) =   (*i)->m_oCharRect.y
                         + (*i)->m_oCharRect.height
                         - 1;
    }
}

void FTS_ANPR_SegResult::getWidthArray( FTS_BASE_StackArray<int>& oArray ) const
{
    assert( oArray.size() == m_oChars.size() );

    std::list<FTS_ANPR_SegChar*>::const_iterator i  = m_oChars.begin();
    std::list<FTS_ANPR_SegChar*>::const_iterator iE = m_oChars.end();

    for( unsigned int n = 0; i != iE; ++i, ++n )
    {
        oArray.at( n ) = (*i)->getWidth();
    }
}


void FTS_ANPR_SegResult::getHeightArray( FTS_BASE_StackArray<int>& oArray ) const
{
    assert( oArray.size() == m_oChars.size() );

    std::list<FTS_ANPR_SegChar*>::const_iterator i  = m_oChars.begin();
    std::list<FTS_ANPR_SegChar*>::const_iterator iE = m_oChars.end();

    for (unsigned int n = 0; i != iE; ++i, ++n)
    {
        oArray.at(n) = (*i)->getHeight(); // quits on boundary check failure
    }
}

void FTS_ANPR_SegResult::getAreaArray( FTS_BASE_StackArray<int>& oArray ) const
{
    assert( oArray.size() == m_oChars.size() );

    std::list<FTS_ANPR_SegChar*>::const_iterator i  = m_oChars.begin();
    std::list<FTS_ANPR_SegChar*>::const_iterator iE = m_oChars.end();
    for (unsigned int n = 0; i != iE; ++i, ++n)
    {
        oArray.at(n) = (*i)->getArea(); // quits on boundary check failure
    }
}

void FTS_ANPR_SegResult::getHoWArray(FTS_BASE_StackArray<float>& oArray) const
{
    assert( oArray.size() == m_oChars.size() );

    std::list<FTS_ANPR_SegChar*>::const_iterator i  = m_oChars.begin();
    std::list<FTS_ANPR_SegChar*>::const_iterator iE = m_oChars.end();
    for (unsigned int n = 0; i != iE; ++i, ++n)
    {
        oArray.at(n) = (*i)->getHoW(); // quits on boundary check failure
    }
}

void FTS_ANPR_SegResult::getCharCentroidArray( FTS_BASE_StackArray<CvPoint2D32f>& oArray ) const
{
    assert( oArray.size() == m_oChars.size() );

    std::list<FTS_ANPR_SegChar*>::const_iterator i  = m_oChars.begin();
    std::list<FTS_ANPR_SegChar*>::const_iterator iE = m_oChars.end();
    for( unsigned int n = 0; i != iE; ++i, ++n )
    {
        oArray.at(n) = (*i)->getCentroid(); // quits on boundary check failure
    }
}

int FTS_ANPR_SegResult::medianWidth()
{
    FTS_BASE_STACK_ARRAY( int, m_oChars.size(), oWidths );
	//FTS_BASE_StackArray<int> oWidths(m_oChars.size());

    getWidthArray( oWidths );

    return FTS_BASE_Median( oWidths );
}

int FTS_ANPR_SegResult::medianHeight()
{
    FTS_BASE_STACK_ARRAY( int, m_oChars.size(), oHeights );

    getHeightArray( oHeights );

    return FTS_BASE_Median( oHeights );
}

int FTS_ANPR_SegResult::medianArea()
{
    FTS_BASE_STACK_ARRAY( int, m_oChars.size(), oAreas );

    getAreaArray( oAreas );

    return FTS_BASE_Median( oAreas );
}

float FTS_ANPR_SegResult::medianHoW()
{
    FTS_BASE_STACK_ARRAY( float, m_oChars.size(), oHoWs );

    getHoWArray( oHoWs );

    return FTS_BASE_Median( oHoWs );
}

bool FTS_ANPR_SegResult::overlapsWith( FTS_ANPR_SegResult* poAnotherSegResult )
{
    int nCount = 0;

    std::list< FTS_ANPR_SegChar* >::const_iterator i  = m_oChars.begin();
    std::list< FTS_ANPR_SegChar* >::const_iterator iE = m_oChars.end();
    for( ; i != iE; ++i )
    {
        cv::Rect r1 = (*i)->m_oCharRect;

        std::list< FTS_ANPR_SegChar* >::const_iterator ii  = poAnotherSegResult->m_oChars.begin();
        std::list< FTS_ANPR_SegChar* >::const_iterator iiE = poAnotherSegResult->m_oChars.end();
        for( ; ii != iiE; ++ii )
        {
            cv::Rect r2 = (*ii)->m_oCharRect;

            if ( r1.x == r2.x && r1.y == r2.y && r1.width  == r2.width && r1.height == r2.height )
                nCount++;

            if ( nCount >=2 )
                return true;
        }
    }

    return false;
}

// Returns:  minor axis length / major axis length, [0, 1]
float FTS_ANPR_SegResult::minorMajorAxesRatio()
{
    FTS_BASE_STACK_ARRAY( CvPoint2D32f, m_oChars.size(), oArray );

    getCharCentroidArray( oArray );

    float rMajorAxisLength = 0;
    float rMinorAxisLength = 0;

    FTS_BASE_Geometry::MajorMinorAxes(
            oArray.begin(),
            oArray.size(),
            rMajorAxisLength,
            rMinorAxisLength );

    // if rMinorAxisLength == 0 then result is zero.
    return   rMinorAxisLength
           / rMajorAxisLength;
}



















FTS_ANPR_Seg::FTS_ANPR_Seg()
	: m_nMinCharHeightToPlateHeightRatio( 0.25f )
    , m_nSmallCompAreaThresh( 50 ) // 0 to 10
    , m_nLargeCompAreaThresh( 10800 ) // 0 to 20000
    , m_nMaxNumCharCandidates( 30 ) // 0 to 100
    , m_rMaxCharHeightOverWidthRatio( 10.0f ) // 0 to 20
    , m_rMinCharHeightOverWidthRatio( 0.6f ) // 0 to 20

    , m_nIsSameSizeAndNoOverlapMaxWidthDiff( 2 )
    , m_nIsSameSizeAndNoOverlapMaxHeightDiff( 3 )
{
	m_poStorage = cvCreateMemStorage( 0 );

	// We need this to use opencv's partitioning algorithms.
	// Should be OK using the same storage, we don't really have any persistent content
	// in there. So m_poStorage is cleared pretty often, meaning there won't be much fragmentation.
	m_poSegCharSeq = cvCreateSeq( 0, sizeof(CvSeq), sizeof(FTS_ANPR_SegChar*), m_poStorage );
}

FTS_ANPR_Seg::~FTS_ANPR_Seg()
{
	cvReleaseMemStorage(&m_poStorage);
}

bool FTS_ANPR_Seg::testArea( const cv::Rect& oBox )
{
    unsigned int nBoxArea =   oBox.height
                            * oBox.width;

    return (    nBoxArea >= m_nSmallCompAreaThresh
             && nBoxArea <= m_nLargeCompAreaThresh );
}


bool FTS_ANPR_Seg::testHeightOverWidth( const cv::Rect& oBox )
{
    if( oBox.width == 0 )
    {
        return false;
    }

    float rHoW =   (float) oBox.height
                 / (float) oBox.width;

    return (    rHoW >= m_rMinCharHeightOverWidthRatio
             && rHoW <= m_rMaxCharHeightOverWidthRatio );
}


bool FTS_ANPR_Seg::testHeight( int nCharHeight, int nPlateHeight )
{
    float r =   (float) nCharHeight
              / (float) nPlateHeight;

    return ( r >= m_nMinCharHeightToPlateHeightRatio );
}


void FTS_ANPR_Seg::extractCharByCCAnalysis( const cv::Mat& oBin,
                                            FTS_ANPR_SegResult& oSegResult )
{
    // Padd the input image first
    // ------------------------------------------------------------------------
	m_oPadded.create( oBin.rows + 2,
					  oBin.cols  + 2,
					  CV_8UC1 );
	cv::copyMakeBorder( oBin, m_oPadded, 1, 1, 1, 1, cv::BORDER_CONSTANT );

    IplImage iiBin    = oBin;
    IplImage iiPadded = m_oPadded;

    cvCopyMakeBorder( &iiBin,
                      &iiPadded,
                      cvPoint( 1, 1 ),
                      IPL_BORDER_CONSTANT,
                      cvScalarAll( 0 )  ); // pad with black border


    // Initializes contour scanning process
    // ------------------------------------------------------------------------
    CvSeq* poContour = 0;
    CvContourScanner oContourScanner;

    oContourScanner = cvStartFindContours( &iiPadded,
                                           m_poStorage,
                                           sizeof( CvContour ),
                                           CV_RETR_EXTERNAL, //CV_RETR_LIST,
                                           CV_CHAIN_APPROX_SIMPLE,
                                           cvPoint( 0, 0 )  );

    // Contour scanning process
    // ------------------------------------------------------------------------
    while(  ( poContour = cvFindNextContour( oContourScanner ) )  )
    {
        // Finding bounding boxes that meet the ratio tests
        // --------------------------------------------------------------------
        CvRect oBox = cvBoundingRect( poContour, 0 );

        if(    !testArea( oBox )
            || !testHeightOverWidth( oBox )
            || !testHeight( oBox.height, iiBin.height )  )
        {
            continue;
        }

        std::list< FTS_ANPR_SegChar*>& oChars = oSegResult.m_oChars;

        // Make sure not too many candidates
        // --------------------------------------------------------------------
        if( oChars.size() >= m_nMaxNumCharCandidates )
        {
            break; // exit the while loop
        }

        // Store the character candidate to the segmentation structure
        // --------------------------------------------------------------------
        oChars.push_back( new FTS_ANPR_SegChar );

        FTS_ANPR_SegChar& oSegChar = *( oChars.back() ); // fill in the empty object

        oSegChar.m_oCharRect = oBox;

        // Offset the bounding box from coordinates in padded image, into coordinates of input image.
        --oSegChar.m_oCharRect.x;
        --oSegChar.m_oCharRect.y;

//        oSegChar.m_oCharBin.resize(oBox.width, oBox.height, SN_PIX_FMT_GREY );
        oSegChar.m_oCharBin = cv::Mat::zeros( cv::Size( oSegChar.m_oCharRect.width, oSegChar.m_oCharRect.height ), CV_8UC1 );

        IplImage iiSegCharBin = oSegChar.m_oCharBin;
//        cvZero( &iiSegCharBin );
//        printf("width = %d, height = %d\n", oSegChar.m_oCharRect.width, oSegChar.m_oCharRect.height );

        // Draw the outer contour and fill all holes. No internal holes
        // after this.
        cvDrawContours( &iiSegCharBin,
                        poContour,
                        CV_RGB( 255, 255, 255 ),
                        CV_RGB( 255, 255, 255 ),
                        1,
                        CV_FILLED,
                        8,
                        cvPoint( -oBox.x, -oBox.y ) // offset contour to smaller image
                        );

        // Recover all the holes in the original image
        cvSetImageROI( &iiBin, oSegChar.m_oCharRect );
        cvAnd( &iiBin, &iiSegCharBin, &iiSegCharBin, 0 );

//        cv::namedWindow( "CCCCCCCCCCCCCCCCCCCCCCC" );
//        cv::imshow( "CCCCCCCCCCCCCCCCCCCCCCC", oSegChar.m_oCharBin );
//        cv::waitKey();
    }

    cvResetImageROI( &iiBin );
    cvEndFindContours( &oContourScanner );


    // Sort the segments using x-coordinate
    // --------------------------------------------------------------------
    oSegResult.m_oChars.sort( &FTS_ANPR_SegChar::LessInX );
}
//void FTS_ANPR_Seg::extractCharByCCAnalysis( const cv::Mat& oBin,
//								   	   	    FTS_ANPR_SegResult& oSegResult )
//{
////	cv::namedWindow( "Connected Components" );
//
//    // Padd the input image first
//    // ------------------------------------------------------------------------
//    m_oPadded.create( oBin.rows + 2,
//    				  oBin.cols  + 2,
//					  CV_8UC1 );
//    cv::copyMakeBorder( oBin, m_oPadded, 1, 1, 1, 1, cv::BORDER_CONSTANT );
//
//	std::vector< std::vector< cv::Point > > contours;
//	std::vector< cv::Vec4i > hierarchy;
//
//	// Find contours
//	cv::findContours( m_oPadded,
//				      contours,
//				      hierarchy,
//				      CV_RETR_CCOMP,
//				      CV_CHAIN_APPROX_SIMPLE,
//				      cv::Point(0, 0) );
//
//	// Approximate contours to polygons + get bounding rects and circles
//	std::vector<std::vector<cv::Point> > contours_poly( contours.size() );
//	std::vector<cv::Rect> boundRect( contours.size() );
//	std::vector<cv::Point2f>center( contours.size() );
//	std::vector<float>radius( contours.size() );
//
//
////	for( unsigned int i = 0; i < contours.size(); i++ )
////	{
////		cv::approxPolyDP( cv::Mat(contours[i]), contours_poly[i], 3, true );
////		boundRect[i] = cv::boundingRect( cv::Mat(contours_poly[i]) );
////		cv::minEnclosingCircle( (cv::Mat)contours_poly[i], center[i], radius[i] );
////	}
////
////
////	  /// Draw polygonal contour + bonding rects + circles
////	cv::Mat drawing = cv::Mat::zeros( m_oPadded.size(), CV_8UC3 );
////	for( unsigned int i = 0; i< contours.size(); i++ )
////	{
////		// Test size
////		if(    !testArea( boundRect[i] )
////			|| !testHeightOverWidth( boundRect[i] )
////			|| !testHeight( boundRect[i].height, oBin.rows )  )
////		{
////			continue;
////		}
////
////		cv::Scalar color( (rand()&255), (rand()&255), (rand()&255) );
////		drawContours( drawing, contours, i, color, CV_FILLED, 8, std::vector<cv::Vec4i>(), 0, cv::Point() );
////
////		cv::Mat m_oCharBin = cv::Mat::zeros( cv::Size( boundRect[i].width, boundRect[i].height ), CV_8UC1 );
////		cv::drawContours( m_oCharBin,
////						  contours,
////						  i,
////						  CV_RGB( 255, 255, 255 ),
////						  CV_FILLED,
////						  8,
////						  hierarchy,
////						  0,
////						  cv::Point( -(boundRect[i].x), -(boundRect[i].y) ) );
////
////		cv::imshow( "CCCCCCCCCCCCCCCCCCCCCCC", m_oCharBin );
////				cv::waitKey();
////	}
////
////	  /// Show in a window
////	  cv::namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
////	  cv::imshow( "Contours", drawing );
//
//
//	cv::Mat drawing = cv::Mat::zeros( m_oPadded.size(), CV_8UC1 );
//	char ch[50];
//	for( unsigned int i = 0; i < contours.size(); i++ )
//	{
//		// Get measurements
//		cv::approxPolyDP( cv::Mat(contours[i]), contours_poly[i], 3, true );
//		boundRect[i] = cv::boundingRect( cv::Mat(contours_poly[i]) );
//		cv::minEnclosingCircle( (cv::Mat)contours_poly[i], center[i], radius[i] );
//
//		// Test size
//		if(    !testArea( boundRect[i] )
//			|| !testHeightOverWidth( boundRect[i] )
//			|| !testHeight( boundRect[i].height, oBin.rows )  )
//		{
//			continue;
//		}
//
//		std::list< FTS_ANPR_SegChar*>& oChars = oSegResult.m_oChars;
//
//		// Make sure not too many candidates
//		// --------------------------------------------------------------------
//		if( oChars.size() >= m_nMaxNumCharCandidates )
//		{
//			break; // exit the while loop
//		}
//
//		// Store the character candidate to the segmentation structure
//		// --------------------------------------------------------------------
//		oChars.push_back( new FTS_ANPR_SegChar );
//
//		FTS_ANPR_SegChar& oSegChar = *( oChars.back() ); // fill in the empty object
//		oSegChar.m_oCharRect = boundRect[i];
//
//		// Offset the bounding box from coordinates in padded image, into coordinates of input image.
//		--oSegChar.m_oCharRect.x;
//		--oSegChar.m_oCharRect.y;
//		printf( "w = %d, h = %d\n", oSegChar.m_oCharRect.width, oSegChar.m_oCharRect.height );
//
//		cv::Rect oRect( boundRect[i].x, boundRect[i].y, boundRect[i].width, boundRect[i].height+2 );
//		cv::drawContours( drawing, contours, i, CV_RGB( 255, 255, 255 ), CV_FILLED, 8, std::vector<cv::Vec4i>(), 0, cv::Point() );
//		oSegChar.m_oCharBin = drawing( oRect ).clone();
//
////		oSegChar.m_oCharBin = cv::Mat::zeros( cv::Size( oSegChar.m_oCharRect.width, oSegChar.m_oCharRect.height ), CV_8UC1 );
////		cv::drawContours( oSegChar.m_oCharBin,
////					      contours,
////					      i,
////					      CV_RGB( 255, 255, 255 ),
////					      CV_FILLED,
////					      8,
////					      hierarchy,
////					      1,
////					      cv::Point( -(boundRect[i].x), -(boundRect[i].y) ) );
//
////		sprintf( ch, "/home/sensen/Desktop/fpt/%d.png", i++);
////		cv::imwrite( ch, oSegChar.m_oCharBin );
////
//		cv::namedWindow( "Character", 0 );
//		cv::imshow( "Character", oSegChar.m_oCharBin );
//		cv::waitKey();
//
//	}
//	cv::namedWindow( "Contours", 0 );
//	cv::namedWindow( "Contours1", 1 );
//	cv::imshow( "Contours", drawing );
//	cv::imshow( "Contours1", drawing );
//
//    // Sort the segments using x-coordinate
//    // --------------------------------------------------------------------
//    oSegResult.m_oChars.sort( &FTS_ANPR_SegChar::LessInX );
//}


int FTS_ANPR_Seg::isSameSizeAndNoOverlap( const void* poSegChar1,
                                         const void* poSegChar2,
                                         void* poSeg )
{
    int nRet = 1;

    const FTS_ANPR_SegChar& o1 = **(const FTS_ANPR_SegChar**) poSegChar1;
    const FTS_ANPR_SegChar& o2 = **(const FTS_ANPR_SegChar**) poSegChar2;

    const FTS_ANPR_Seg& oSeg = *(const FTS_ANPR_Seg*) poSeg;

    int nX1 = o1.m_oCharRect.x;
    int nX2 = o2.m_oCharRect.x;

    int nW1 = o1.m_oCharRect.width;
    int nW2 = o2.m_oCharRect.width;

    int w = abs( nW1 - nW2 );
    int h = abs(   (int) o1.m_oCharRect.height
                 - (int) o2.m_oCharRect.height  );


    // If chars not the same size, return false
    if(    w > (int) oSeg.m_nIsSameSizeAndNoOverlapMaxWidthDiff
        || h > (int) oSeg.m_nIsSameSizeAndNoOverlapMaxHeightDiff )
    {
        nRet = 0;
    }

    // Gets here if chars are same size
    // If chars overlap, then return false
    if( nX1 < nX2 )
    {
        if( nX1 + nW1 > nX2 )
        {
            nRet = 0;
        }
    }
    else
    {
        if( nX2 + nW2 > nX1 )
        {
            nRet = 0;
        }
    }

    // Chars are same size AND do not overlap.
    return nRet;
}

int FTS_ANPR_Seg::isSameHeight( const void* poSegChar1,
							   const void* poSegChar2,
							   void* poSeg )
{
    int nRet = 1;

    const FTS_ANPR_SegChar& o1 = **(const FTS_ANPR_SegChar**) poSegChar1;
    const FTS_ANPR_SegChar& o2 = **(const FTS_ANPR_SegChar**) poSegChar2;

    const FTS_ANPR_Seg& oSeg = *(const FTS_ANPR_Seg*) poSeg;
    int h = abs(   (int) o1.m_oCharRect.height
                 - (int) o2.m_oCharRect.height  );


    // If chars not the same height, return false
    if( h > (int) oSeg.m_nIsSameSizeAndNoOverlapMaxHeightDiff )
    {
        nRet = 0;
    }

    // Chars are same size AND do not overlap.
    return nRet;
}

// Returns the number of clean char
int FTS_ANPR_Seg::findCleanChar( FTS_ANPR_SegResult& oSegResult, CvCmpFunc is_equal )
{
    cvClearSeq( m_poSegCharSeq );

    // Fill an opencv sequence with chars
    // --------------------------------------------------------------------
    std::list<FTS_ANPR_SegChar*>::iterator i  = oSegResult.m_oChars.begin();
    std::list<FTS_ANPR_SegChar*>::iterator iE = oSegResult.m_oChars.end();
    for( ; i != iE; ++i )
    {
        FTS_ANPR_SegChar* poSegChar = *i;

        cvSeqPush( m_poSegCharSeq, &poSegChar );
    }

    // Partition into equivalent classes such that all chars in the same class
    // are the same size and non-overlapping.
    // --------------------------------------------------------------------
    CvSeq* poLabels = 0;
    int nClassCount = cvSeqPartition( m_poSegCharSeq,
                                      0,
                                      &poLabels,
                                      is_equal,// FTS_ANPR_Seg::isSameSizeAndNoOverlap,
                                      this );

    int nCleanCharCount = 0;

    if( nClassCount < (int) oSegResult.m_oChars.size() ) // means there's at least one class with at least 2 chars in it
    {
        // Find the biggest partition. This partition contains all the chars that are
        // the same size and DO NOT overlap with each other.
        // --------------------------------------------------------------------

        std::vector<int>& oCount = m_oIntVector; // Num chars in each class

        oCount.clear();
        oCount.assign( nClassCount, 0 );

        for( int n = 0; n < poLabels->total; ++n )
        {
            int nClassIdx = *(int*) cvGetSeqElem( poLabels, n );
            ++oCount.at( nClassIdx );
        }

        int nBiggestClassIdx = std::distance(                   oCount.begin(),
                                              std::max_element( oCount.begin(), oCount.end() )
                                              );

        // Mark all chars in the biggest class as clean chars.
        // --------------------------------------------------------------------
        i  = oSegResult.m_oChars.begin();
        iE = oSegResult.m_oChars.end();
        for( unsigned int nIdx = 0; i != iE; ++i, ++nIdx )
        {
            int nClassIdx = *(int*) cvGetSeqElem(poLabels, nIdx);
            if (nClassIdx == nBiggestClassIdx)
            {
                (*i)->m_bClean = true;
                ++nCleanCharCount;
            }
            else
            {
                (*i)->m_bClean = false;
            }
        }

    } // if at least one class with more than one char


    // Release the memory storage of label sequence
    // --------------------------------------------------------------------
    cvClearSeq( poLabels );
    cvClearSeq( m_poSegCharSeq );

    cvClearMemStorage( m_poStorage ); // clear, don't deallocate

    return nCleanCharCount;
}

void FTS_ANPR_Seg::maskCleanChar( FTS_ANPR_SegResult& oSegResult,
                                  cv::Mat& oDst,
                                  const cv::Scalar& oMaskValue )
{
    IplImage iiDst = oDst;

    std::list<FTS_ANPR_SegChar*>::iterator i  = oSegResult.m_oChars.begin();
    std::list<FTS_ANPR_SegChar*>::iterator iE = oSegResult.m_oChars.end();
    for(; i != iE; ++i )
    {
        if( (*i)->m_bClean == false )
        {
            continue;
        }

        FTS_ANPR_SegChar& oSC = *(*i);

        CvRect oRect;

        // And now:
        oRect = oSC.m_oCharRect;

        //TODO: I don't really know why the -2
        oRect.width = std::max( 1,
                                oRect.width - 2 );

        cvSetImageROI( &iiDst, oRect );
        cvSet( &iiDst, oMaskValue );
    }

    cvResetImageROI( &iiDst );
}

