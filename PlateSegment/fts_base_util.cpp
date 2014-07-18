
#include "fts_base_util.h"
#include <boost/regex.hpp>

using namespace std;

const unsigned int FTS_BASE_Util::MIN_DENSITY_DEFAULT 		= 0;
const unsigned int FTS_BASE_Util::MAX_DENSITY_DEFAULT 		= INT_MAX;
const unsigned int FTS_BASE_Util::MIN_AREA_DEFAULT 			= 0;
const unsigned int FTS_BASE_Util::MAX_AREA_DEFAULT 			= INT_MAX;
const float 	   FTS_BASE_Util::MIN_HOW_RATIO_DEFAULT		= 0.0;
const float 	   FTS_BASE_Util::MAX_HOW_RATIO_DEFAULT		= 1.0;
const float 	   FTS_BASE_Util::MIN_DENSITY_RATIO_DEFAULT = 0.0;
const float 	   FTS_BASE_Util::MAX_DENSITY_RATIO_DEFAULT = 1.0;

FTS_BASE_Util::FTS_BASE_Util()
{
}

FTS_BASE_Util::~FTS_BASE_Util()
{
}

long long getCurrentTimeInMS() 
{
#ifndef WIN32
	struct timeval tv;

	gettimeofday(&tv, NULL);

	uint64 ret = tv.tv_usec;
	/* Convert from micro seconds (10^-6) to milliseconds (10^-3) */
	ret /= 1000;

	/* Adds the seconds (10^0) after converting them to milliseconds (10^-3) */
	ret += (tv.tv_sec * 1000);

	return ret;
#else
    static LARGE_INTEGER s_frequency;
    static BOOL s_use_qpc = QueryPerformanceFrequency(&s_frequency);
    if (s_use_qpc) 
	{
        LARGE_INTEGER now;
        QueryPerformanceCounter(&now);
        return (1000LL * now.QuadPart) / s_frequency.QuadPart;
    } 
	else 
	{
        return GetTickCount();
    }
#endif
}

void FTS_BASE_Util::Rotate(
        const IplImage* piiSrc,
              IplImage* piiDst,
        float rAngleDegrees,
        bool bFillOutlierModeReplicateBorder,
        CvScalar oFillOutlierWith )
{
    // Setup affine matrix for rotation
    // ------------------------------------------------------------------------
    CvPoint2D32f oCenter = cvPoint2D32f( piiSrc->width  * 0.5f,
                                         piiSrc->height * 0.5f );

#ifndef WIN32
    FTS_BASE_CV_MAT_ON_STACK( oMap, 2, 3, CV_32FC1 );
#else
	CvMat oMap;
	cvInitMatHeader( &oMap, 2, 3, CV_32FC1 );
    unsigned char* oMapData = new unsigned char[ 2 * oMap.step ];
    cvInitMatHeader( &oMap, 2, 3, CV_32FC1, oMapData, oMap.step);
#endif	

    if( bFillOutlierModeReplicateBorder )
    {
        cv2DRotationMatrix( cvPoint2D32f(0,0), -rAngleDegrees, 1.0, &oMap );

        // Set translation
        cvmSet( &oMap, 0, 2, oCenter.x );
        cvmSet( &oMap, 1, 2, oCenter.y );

        cvGetQuadrangleSubPix( piiSrc, piiDst, &oMap );
    }
    else
    {
        cv2DRotationMatrix( oCenter, rAngleDegrees, 1.0, &oMap );

        cvWarpAffine(
                piiSrc,
                piiDst,
                &oMap,
                CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS,
                oFillOutlierWith
                );
    }
}

bool FTS_BASE_Util::parseTycoFilenameByRegex( const std::string& sFilename,
												const std::string& sRegex,
												std::string& sDate,
												std::string& sTime,
												std::string& sCamId,
												std::string& sZoneId,
												std::string& sFpm,
												std::string& sFileExt )
{
	boost::smatch what;
	const boost::regex oFilenameRegex( sRegex );

	if( !boost::regex_match( sFilename, what, oFilenameRegex ) )
	{
//		FTS_ERR( "File name format is not matched with regex." );
		return false;
	}

	// There are 7 matches, the first one is the full file path
	sDate 	 = static_cast<const string&>( what[1] ).c_str();
	sTime 	 = static_cast<const string&>( what[2] ).c_str();
	sCamId   = static_cast<const string&>( what[3] ).c_str();
	sZoneId  = static_cast<const string&>( what[4] ).c_str();
	sFpm     = static_cast<const string&>( what[5] ).c_str();
	sFileExt = static_cast<const string&>( what[6] ).c_str();

	return true;
}


int FTS_BASE_Util::OtsuAlgorithm( FTS_BASE_StackArray<double>& oHist )
{
    const unsigned char RETURN_INVALID = -1;

    // Find min, max value
    // ----------------------------------------------------------------------
    unsigned int nMinValue = 0;
    for( ; nMinValue < oHist.size(); ++nMinValue )
    {
        if( oHist.at(nMinValue) != 0 )
        {
            break;
        }
    }

    if( nMinValue == oHist.size() )
    {
        return RETURN_INVALID;
    }

    int nMaxValue = oHist.size() - 1;
    for( ; nMaxValue >= 0; --nMaxValue )
    {
        if( oHist.at(nMaxValue) != 0 )
        {
            break;
        }
    }

    // Normalise histogram
    // ----------------------------------------------------------------------
    double rPixelCount = std::accumulate( oHist.begin(), oHist.end(), 0.0 );

    FTS_BASE_STACK_ARRAY( float, 256, oNormHist );

    // Find mean
    // ----------------------------------------------------------------------
    float rMean = 0;
    for( unsigned int i = 0; i < oHist.size(); ++i )
    {
        double t = oHist.at( i ) / rPixelCount;

        oNormHist.at( i ) = t;

        rMean += i * t;
    }

    float rDetT = 0;
    for( unsigned int i = 0; i < oHist.size(); ++i )
    {
        rDetT += oNormHist.at( i ) * FTS_BASE_SQ( (float)i - rMean );
    }

    int nReturn = RETURN_INVALID;

    double w0 = 0, w1 = 0, u0 = 0, u1 = 0, rDetB = 0, n = 0, nMaxN = 0;
    for( int i = nMinValue; i <= nMaxValue; ++i )
    {
        w0 += oNormHist.at( i );

        w1 = 1 - w0;

        u0 = 0;

        for( int j = 0; j <= i; ++j )
        {
            u0 += j * oNormHist.at( j );
        }

        if( w0 != 0 )
        {
            u0 /= w0;
        }

        if( w1 != 0 )
        {
            u1 = (rMean - w0 * u0 ) / w1;
        }

        rDetB = w0 * w1 * FTS_BASE_SQ( u0 - u1 );

        if( rDetT != 0 )
        {
            n = rDetB / rDetT;
        }
        else
        {
            return RETURN_INVALID;
        }

        if( nMaxN < n )
        {
            nMaxN = n;
            nReturn = i;
        }
    }

    return nReturn;
}

void FTS_BASE_Util::Tokenize( const string& sInput, const char* pcDelim, vector< string >& svTokens )
{
    svTokens.clear();
    unsigned int nDelimStart = sInput.find_first_not_of( pcDelim );
    unsigned int nDelimEnd;
    string sSub;

    if ( nDelimStart == string::npos )
    {
        return;
    }

    while( true )
    {
        nDelimEnd = sInput.find( pcDelim, nDelimStart );
        if ( nDelimEnd == string::npos )
        {
            //! save remainder of the string before returning
            if ( nDelimStart != 0 )
            {
                sSub = sInput.substr( nDelimStart );
                if ( sSub.length() > 0 )
                {
                    svTokens.push_back( sSub );
                }
            }
            return;
        }

//        FTS_ERR( "Found!" );
        sSub = sInput.substr( nDelimStart, nDelimEnd - nDelimStart );

        svTokens.push_back( sSub );
        nDelimStart = nDelimEnd + strlen( pcDelim );
    }
}

void FTS_BASE_Util::CreateDirectory( const std::string& sDir )
{
    bool bDummy;
    CreateDirectory( sDir, bDummy );
}

void FTS_BASE_Util::CreateDirectory( const std::string& sDir, bool& bExists )
{
    bExists = false;

    struct stat buffer;
    bool bDirectoryExists = ( stat( sDir.c_str(), &buffer ) == 0 );

    if( bDirectoryExists )
    {
        bExists = true;
        return;
    }

    const std::string sCommand = "mkdir -p -m 0777 " + sDir;
    int ret = system( sCommand.c_str() );
    if( ret != 0 )
    {
    	FTS_ERR( "System call fails" );
    }
}

cv::Rect FTS_BASE_Util::enlargeRect(
        const cv::Rect oRect,
        float rEnlargeFactor,
        unsigned int nClipWidth,
        unsigned int nClipHeight )
{

    cv::Rect oEnlarged;

    oEnlarged.width  = oRect.width  * rEnlargeFactor;
    oEnlarged.height = oRect.height * rEnlargeFactor;
    oEnlarged.x      = oRect.x  -  ( oEnlarged.width  - oRect.width  ) / 2;
    oEnlarged.y      = oRect.y  -  ( oEnlarged.height - oRect.height ) / 2;

    return FTS_BASE_Clip( oEnlarged, nClipWidth, nClipHeight );
}


void FTS_BASE_Util::findReplace(
		std::string& str,
		const std::string& oldStr,
		const std::string& newStr)
{
	size_t pos = 0;
	while((pos = str.find(oldStr, pos)) != std::string::npos)
	{
	 str.replace(pos, oldStr.length(), newStr);
	 pos += newStr.length();
	}
}


bool FTS_BASE_Util::Exists (const std::string& name)
{
	struct stat buffer;
	return (stat (name.c_str(), &buffer) == 0);
}

void FTS_BASE_Util::CropRect( cv::Rect& oRect, unsigned int nWidth, unsigned int nHeight )
{
    int nLeft   = oRect.x;
    int nRight  = oRect.x + oRect.width;
    int nTop    = oRect.y;
    int nBottom = oRect.y + oRect.height;

    nLeft = ( nLeft >= 0 )  ?  nLeft
                            :  0;

    nTop  = ( nTop  >= 0 )  ?  nTop
                            :  0;

    nRight  = ( nRight  <= (int)nWidth  )  ?  nRight
                                           :  (int)nWidth;

    nBottom = ( nBottom <= (int)nHeight )  ?  nBottom
                                           :  (int)nHeight;

    oRect.x = nLeft;
    oRect.y = nTop;

    oRect.width  = nRight  - nLeft;
    oRect.height = nBottom - nTop;
}


void FTS_BASE_Util::removeNoises( const cv::Mat& oBin,
										cv::Mat& oDst,
								  const unsigned int nMinDensity,
								  const unsigned int nMaxDensity,
								  const unsigned int nMinArea,
								  const unsigned int nMaxArea,
								  const float rMinHoWRatio,
								  const float rMaxHoWRatio,
								  const float rMinDensityRatio,
								  const float rMaxDensityRatio )
{
	FTS_ASSERT( oBin.type() == CV_8UC1 );

    // Pad the input image first
	cv::Mat oPadded;
	oPadded.create( oBin.rows + 2,
				    oBin.cols  + 2,
				    oBin.type() );
	oDst = cv::Mat::zeros( oBin.size(), oBin.type() );

	cv::copyMakeBorder( oBin, oPadded, 1, 1, 1, 1, cv::BORDER_CONSTANT );

    IplImage iiBin    = oBin;
    IplImage iiPadded = oPadded;
    IplImage iiDst 	  = oDst;

    cvCopyMakeBorder( &iiBin,
                      &iiPadded,
                      cvPoint( 1, 1 ),
                      IPL_BORDER_CONSTANT,
                      cvScalarAll( 0 )  ); // pad with black border


    // Initializes contour scanning process
    // ------------------------------------------------------------------------
    CvSeq* poContour = 0;
    CvContourScanner oContourScanner;

    CvMemStorage* poStorage = cvCreateMemStorage( 0 );
    oContourScanner = cvStartFindContours( &iiPadded,
                                           poStorage,
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

        // Sanity check
        if( oBox.width == 0 )
        {
            continue;
        }

        unsigned int nDensity = cvContourArea( poContour );
        unsigned int nArea    = oBox.width * oBox.height;
        float rHoW		      = (float) oBox.height / (float) oBox.width;
        float rDensityRatio	  = (float) nDensity / (float) nArea;

        if(    nDensity < nMinDensity
        	|| nDensity > nMaxDensity

        	|| nArea < nMinArea
			|| nArea > nMaxArea

        	|| rHoW  < rMinHoWRatio
        	|| rHoW  > rMaxHoWRatio

        	|| rDensityRatio  < rMinDensityRatio
			|| rDensityRatio  > rMaxDensityRatio)	// noises ==> ignore
        {
        	continue;
        }

//        printf( "area = %d\n", nDensity );
        // Draw the outer contour and fill all holes. No internal holes after this.
        cvDrawContours( &iiDst,
                        poContour,
                        CV_RGB( 255, 255, 255 ),
                        CV_RGB( 255, 255, 255 ),
                        1,
                        CV_FILLED,
                        8,
                        cvPoint( -1, -1 ) // offset contour to smaller image
                        );
    }

    cvEndFindContours( &oContourScanner );

    cvReleaseMemStorage( &poStorage );
}


double FTS_BASE_Util::distanceBetweenPoints(cv::Point p1, cv::Point p2)
{
  float asquared = (p2.x - p1.x)*(p2.x - p1.x);
  float bsquared = (p2.y - p1.y)*(p2.y - p1.y);

  return sqrt(asquared + bsquared);
}

float FTS_BASE_Util::angleBetweenPoints(cv::Point p1, cv::Point p2)
{
  int deltaY = p2.y - p1.y;
  int deltaX = p2.x - p1.x;

  return atan2((float) deltaY, (float) deltaX) * (180 / CV_PI);
}

bool FTS_BASE_Util::IsRectIntersect( const int X1, const int Y1, const int W1, const int H1,
					  const int X2, const int Y2, const int W2, const int H2 )
{
	if(    X1 + W1 < X2
	    || X2 + W2 < X1
	    || Y1 + H1 < Y2
	    || Y2 + H2 < Y1 )
	{
	    return false;
	}

	return true;
}

bool FTS_BASE_Util::IsRectIntersect(cv::Rect rect1, cv::Rect rect2) {
	if (rect1.x + rect1.width <= rect2.x  ||
		rect1.y + rect1.height <= rect2.y ||
		rect2.x + rect2.width <= rect1.x  ||
		rect2.y + rect2.height <= rect1.y) {
		return false;
	}

	return true;	
}

void FTS_BASE_Util::IntersectRects(cv::Rect rect1, cv::Rect rect2, cv::Rect& intersectRect) {
	int minx1 = rect1.x;
	int maxx1 = rect1.x + rect1.width;
	int miny1 = rect1.y;
	int maxy1 = rect1.y + rect1.height;

	int minx2 = rect2.x;
	int maxx2 = rect2.x + rect2.width;
	int miny2 = rect2.y;
	int maxy2 = rect2.y + rect2.height;

	intersectRect.x = max(minx1, minx2);
	intersectRect.y = max(miny1, miny2);

	int maxx, maxy;

	maxx = min(maxx1, maxx2);
	maxy = min(maxy1, maxy2);

	intersectRect.width = maxx - intersectRect.x + 1;
	intersectRect.height = maxy - intersectRect.y + 1;
}

void FTS_BASE_Util::UnionRects(cv::Rect rect1, cv::Rect rect2, cv::Rect& unionRect) {

	int minx1 = rect1.x;
	int maxx1 = rect1.x + rect1.width;
	int miny1 = rect1.y;
	int maxy1 = rect1.y + rect1.height;

	int minx2 = rect2.x;
	int maxx2 = rect2.x + rect2.width;
	int miny2 = rect2.y;
	int maxy2 = rect2.y + rect2.height;

	unionRect.x = std::min(minx1, minx2);
	unionRect.y = std::min(miny1, miny2);

	int maxx, maxy;

	maxx = max(maxx1, maxx2);
	maxy = max(maxy1, maxy2);

	unionRect.width = maxx - unionRect.x + 1;
	unionRect.height = maxy - unionRect.y + 1;
}

static bool compareSlices(cv::Rect rect1, cv::Rect rect2)
{
	return rect1.x < rect2.x;
}

std::vector<cv::Rect> FTS_BASE_Util::CheckAndMergeOverlapRects(std::vector<cv::Rect>& src) 
{
	std::vector<cv::Rect> res;
	res.clear();

	sort(src.begin(), src.end(), compareSlices);

    //BOOST_AUTO(first_slice, rectangles.begin());
	vector<bool> stat(src.size());
	for(size_t i = 0 ; i < src.size(); i++)
	{
		stat[i] = false;
	}

	//vector<cv::Rect>::iterator first_slice, second_slice;
	cv::Rect first_slice, second_slice;

    // Iterate through adjacent slices, merging any that overlap.
    for (size_t i = 0 ; i < src.size(); i++)
    {
        // Skip over any slices that have been merged.
        if (/*first_slice->size().height == 0*/stat[i])
        {
            continue;
        }

		first_slice = src[i];

        //BOOST_AUTO(second_slice, first_slice + 1);
		//second_slice = first_slice + 1;

        // Iterate through all adjacent slices that share a y-coordinate
        // until we either run out of slices, or cannot merge a slice.
        for (size_t j = i+1; j < src.size(); j++)
        {
			second_slice = src[j];

			if(IsRectIntersect(first_slice, second_slice))
			{
				// Set the width of the first slice to be equivalent to the
				// rightmost point of the two rectangles.
				first_slice.x = std::min(first_slice.x, second_slice.x);
				first_slice.y = std::min(first_slice.y, second_slice.y);
				first_slice.width = (std::max)(first_slice.x + first_slice.width, second_slice.x + second_slice.width) - first_slice.x;
				first_slice.height = (std::max)(first_slice.y + first_slice.height, second_slice.y + second_slice.height) - first_slice.y;

				// Mark the second slice as having been merged.
				stat[j] = true;
			}
        }

		res.push_back(first_slice);
    }

    // Snip out any rectangles that have been merged (have 0 height).
    /*src.erase(remove_if(rectangles.begin()
					  , rectangles.end()
					  , has_empty_height)
				, rectangles.end());*/

    return res;
}

void FTS_BASE_Util::extendRect(cv::Rect& rect, int padding) {
	rect.x -= padding;
	rect.y -= padding;
	rect.width += (2 * padding);
	rect.height += (2 * padding);
}
