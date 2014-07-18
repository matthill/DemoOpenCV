
#include "fts_anpr_cropper.h"

#define CVA_ROI_ROW( type, img, roi, r ) ( ( (type)( (img)->imageData + (img)->widthStep * (((roi).y)+(r)) )  ) + ((roi).x)*((img)->nChannels) )

const unsigned int FTS_ANPR_Cropper::UNINITIALISED = UINT_MAX;

const float FTS_ANPR_Cropper::s_rScaleHorLargeSmoothHalfWindow     = 75.0f/85.0f; // 70 works well too
const float FTS_ANPR_Cropper::s_rScaleHorSmallSmoothHalfWindow     = 30.0f/85.0f; //15
const float FTS_ANPR_Cropper::s_rScaleHorStepHalfWindow            = 25.0f/85.0f;
const float FTS_ANPR_Cropper::s_rScaleBoundaryWindowMaxWidth       = 50.0f/85.0f;
const float FTS_ANPR_Cropper::s_rScaleVerSmallSmoothHalfWindow     =  5.0f/85.0f;
const float FTS_ANPR_Cropper::s_rScaleEdgeTestPlateWidthMin        = 50.0f/85.0f;
const float FTS_ANPR_Cropper::s_rScaleEdgeTestPlateHeightMin       = 15.0f/85.0f;

FTS_ANPR_Cropper::FTS_ANPR_Cropper()
	: m_rSumVerThreshFactor        ( 1.7f )
	, m_rHorSmallSmoothThreshFrac  ( 0.5f )
	, m_rVerBoundaryWindowThresh   ( 0.3f )
	, m_rNonPlateTestThresh1       ( 1.2f )
	, m_rNonPlateTestThresh2       ( 0.3f )
	, m_nMaxPlateHalfWidth         ( 85 )
	, m_nHorLargeSmoothHalfWindow  ( 75 ) // 70 works well too
	, m_nHorSmallSmoothHalfWindow  ( 30 ) //15
	, m_nHorStepHalfWindow         ( 25 )
	, m_nBoundaryWindowMaxWidth    ( 50 )
	, m_nVerSmallSmoothHalfWindow  ( 5 )
//    m_nMaxHalfPlateSize             = 100;
	, m_nEdgeTestPlateWidthMin     ( 50 )
	, m_nEdgeTestPlateHeightMin    ( 15 )
	, m_nSumVerArrSize			   ( 0 )
	, m_nSumHorArrSize			   ( 0 )
{


//    uninitParameters();
}

// Set all parameters to their default values
void FTS_ANPR_Cropper::uninitParameters()
{
    // New, default -1 means use s_rScale... to calculate the values
    m_nMaxPlateHalfWidth            = 85; // this parameter scales all others if other params are UNINITIALISED
    m_nHorLargeSmoothHalfWindow     = UNINITIALISED;
    m_nHorSmallSmoothHalfWindow     = UNINITIALISED;
    m_nHorStepHalfWindow            = UNINITIALISED;
    m_nBoundaryWindowMaxWidth       = UNINITIALISED;
    m_nVerSmallSmoothHalfWindow     = UNINITIALISED;
    m_nEdgeTestPlateWidthMin        = UNINITIALISED;
    m_nEdgeTestPlateHeightMin       = UNINITIALISED;
}

#ifdef WIN32
static inline double round(double val)
{   
    return floor(val + 0.5);
}
#endif

// Compute the parameter if it's is uninitialised
void FTS_ANPR_Cropper::computeDefaultParamter( unsigned int nParameter, float rScale )
{
    if( nParameter == UNINITIALISED )
    {
        float f = round( (float)m_nMaxPlateHalfWidth * rScale );

        nParameter = (unsigned int) f;
    }
}

// Compute default values for all parameters that have default values
void FTS_ANPR_Cropper::computeDefaultParamters()
{
    computeDefaultParamter(      m_nHorLargeSmoothHalfWindow,
                            s_rScaleHorLargeSmoothHalfWindow );

    computeDefaultParamter(      m_nHorSmallSmoothHalfWindow,
                            s_rScaleHorSmallSmoothHalfWindow );

    computeDefaultParamter(      m_nHorStepHalfWindow,
                            s_rScaleHorStepHalfWindow );

    computeDefaultParamter(      m_nBoundaryWindowMaxWidth,
                            s_rScaleBoundaryWindowMaxWidth );

    computeDefaultParamter(      m_nVerSmallSmoothHalfWindow,
                            s_rScaleVerSmallSmoothHalfWindow );

    computeDefaultParamter(      m_nEdgeTestPlateWidthMin,
                            s_rScaleEdgeTestPlateWidthMin );

    computeDefaultParamter(      m_nEdgeTestPlateHeightMin,
                            s_rScaleEdgeTestPlateHeightMin );
}



FTS_ANPR_Cropper::~FTS_ANPR_Cropper()
{
    // Nothing
}

bool FTS_ANPR_Cropper::processDetection( const cv::Mat& oSrc, CvRect& oCropRect )
{
	unsigned int nImageWidth  = oSrc.cols;
	unsigned int nImageHeight = oSrc.rows;

	if ( m_nSumVerArrSize < nImageWidth )
	{
		 m_nSumVerArrSize = nImageWidth;

		 m_oSumVer            .resize( m_nSumVerArrSize );
		 m_oSumVerRunSum      .resize( m_nSumVerArrSize );
		 m_oSumVerSmoothLarge .resize( m_nSumVerArrSize );
		 m_oSumVerSmoothSmall .resize( m_nSumVerArrSize );
		 m_oSumVerStepFiltered.resize( m_nSumVerArrSize );
	}

	if ( m_nSumHorArrSize < nImageHeight )
	{
		 m_nSumHorArrSize = nImageHeight;

		 m_oSumHor            .resize( m_nSumHorArrSize );
		 m_oSumHorRunSum      .resize( m_nSumHorArrSize );
		 m_oSumHorSmooth      .resize( m_nSumHorArrSize );
	}

	return crop( oSrc, oCropRect );
}


bool FTS_ANPR_Cropper::processDetection( IplImage* poSrc, CvRect& oCropRect )
{
	unsigned int nImageWidth  = poSrc->width;
	unsigned int nImageHeight = poSrc->height;
	if ( m_nSumVerArrSize < nImageWidth )
	{
		 m_nSumVerArrSize = nImageWidth;

		 m_oSumVer            .resize( m_nSumVerArrSize );
		 m_oSumVerRunSum      .resize( m_nSumVerArrSize );
		 m_oSumVerSmoothLarge .resize( m_nSumVerArrSize );
		 m_oSumVerSmoothSmall .resize( m_nSumVerArrSize );
		 m_oSumVerStepFiltered.resize( m_nSumVerArrSize );
	}

	if ( m_nSumHorArrSize < nImageHeight )
	{
		 m_nSumHorArrSize = nImageHeight;

		 m_oSumHor            .resize( m_nSumHorArrSize );
		 m_oSumHorRunSum      .resize( m_nSumHorArrSize );
		 m_oSumHorSmooth      .resize( m_nSumHorArrSize );
	}

    return crop( poSrc, oCropRect );
}

bool FTS_ANPR_Cropper::crop( const cv::Mat& oSrc, CvRect& oCropRect )
{
	IplImage oII = oSrc;
	return crop( &oII, oCropRect );
}

// Returns true if there is a plate in the image
bool FTS_ANPR_Cropper::crop( IplImage* poSrc, CvRect& oCropRect )
{
    oCropRect = cvRect( 0,0,0,0 );

   // -) Find vertical edge image using sobel filter
    IplImage* poEdgeIpl16SC1 = cvCreateImage( cvSize(poSrc->width, poSrc->height),IPL_DEPTH_16S,1 );
    cvSobel( poSrc, poEdgeIpl16SC1, 1, 0, 3 );
    cvAbs( poEdgeIpl16SC1, poEdgeIpl16SC1 );

    unsigned int nLeft;
    unsigned int nRight;
    double       rSumVerMean;

    leftRightCrop( *poEdgeIpl16SC1,
                   nLeft,
                   nRight,
                   rSumVerMean );

    // For top bottom cropping, limit our attention to to only the left right cropping boundaries.
    // ----------------------------------------------------------------
    cvSetImageROI(  poEdgeIpl16SC1, cvRect( nLeft, 0, nRight-nLeft+1, poEdgeIpl16SC1->height )  );


    unsigned int nTop;
    unsigned int nBottom;

    topBottomCrop( *poEdgeIpl16SC1,
                   nTop,
                   nBottom );

    // -) Classify as plate/non-plate
    int nStrongEdgeCount = 0;
    for ( unsigned int x = nLeft; x <= nRight; ++x )
    {
        if (  m_oSumVer.at( x )  >  m_rNonPlateTestThresh1 * rSumVerMean  )
        {
            ++nStrongEdgeCount;
        }
    }

    // If too few strong edges, then not a plate
    if ( nStrongEdgeCount < ( nRight - nLeft + 1 ) * m_rNonPlateTestThresh2 )
    {
//    	std::cout << "Not a plate: too few strong edges" << std::endl;
        return false;
    }

    // Geometric tests
    if ( nRight  - nLeft + 1 < m_nEdgeTestPlateWidthMin
    ||   nBottom - nTop  + 1 < m_nEdgeTestPlateHeightMin )
    {
//    	std::cout << "Not a plate: Failed Geometric tests" << std::endl;
        return false;
    }

    oCropRect.x      = std::min( (int)nLeft,                  (int)poSrc->width  - 2 );
    oCropRect.width  = std::max( (int)nRight -(int)nLeft + 1, (int)1                        );
    oCropRect.y      = std::min( (int)nTop,                   (int)poSrc->height - 2 );
    oCropRect.height = std::max( (int)nBottom-(int)nTop  + 1, (int)1                        );

    return true;
}




void FTS_ANPR_Cropper::leftRightCrop( IplImage& oEdge16SC1,
                                    unsigned int& nLeftOut,
                                    unsigned int& nRightOut,
                                    double&       rSumVerMeanOut )
{
    unsigned int nSrcW = oEdge16SC1.width;

    // -) Calulcate projection of edge image onto the x-axis
    // ----------------------------------------------------------------
    sumVer16SC1( &oEdge16SC1, &m_oSumVer.front() );

#ifdef DBUG_L3
    SN_P( m_oSumVer.size() );
#endif

    rSumVerMeanOut = mean( &m_oSumVer.front(), nSrcW );
    clipHi( &m_oSumVer.front(),
            &m_oSumVer.front(),
            nSrcW,
            rSumVerMeanOut * m_rSumVerThreshFactor );

    // -) Compute the run sum to enable efficient smoothing
    // ----------------------------------------------------------------
    runSum( &m_oSumVer.front(),
            &m_oSumVerRunSum.front(),
            nSrcW );

    // -) First round smoothing using large window, rough location of plate.
    // ----------------------------------------------------------------
    smooth( &m_oSumVerRunSum.front(),
            &m_oSumVerSmoothLarge.front(),
            nSrcW,
            m_nHorLargeSmoothHalfWindow );

    int nSumVerSmoothLargeMaxPos  = max_element( m_oSumVerSmoothLarge.begin(),
                                                 m_oSumVerSmoothLarge.begin() + nSrcW )
                                  -            ( m_oSumVerSmoothLarge.begin() );


    int  nFirstCropLeft = nSumVerSmoothLargeMaxPos - m_nMaxPlateHalfWidth;
    if ( nFirstCropLeft < 0 )
    {
         nFirstCropLeft = 0;
    }

    int  nFirstCropRight = nSumVerSmoothLargeMaxPos + m_nMaxPlateHalfWidth;
    if ( nFirstCropRight >= (int)nSrcW )
    {
         nFirstCropRight =  (int)nSrcW - 1;
    }

    // -) Second round smoothing with smaller window, refine location of plate boundaries.
    // ----------------------------------------------------------------
    smooth( &m_oSumVerRunSum.front(),
            &m_oSumVerSmoothSmall.front(),
            nSrcW,
            m_nHorSmallSmoothHalfWindow );

    double rSumVerSmoothSmallMax = *max_element( m_oSumVerSmoothSmall.begin() + nFirstCropLeft,
                                                 m_oSumVerSmoothSmall.begin() + nFirstCropRight + 1 );

    double rSumVerSmoothSmallMin = *min_element( m_oSumVerSmoothSmall.begin() + nFirstCropLeft,
                                                 m_oSumVerSmoothSmall.begin() + nFirstCropRight + 1 );

    double rSmoothSmallThr = m_rHorSmallSmoothThreshFrac
                           * ( rSumVerSmoothSmallMax - rSumVerSmoothSmallMin )
                           +                           rSumVerSmoothSmallMin;


    // -) Step filtering to localise the maximal plate edge response
    // ----------------------------------------------------------------
    stepFilter( &m_oSumVerRunSum.front(),
                &m_oSumVerStepFiltered.front(),
                nSrcW,
                m_nHorStepHalfWindow );

    // search from nLeft first crop boundary to the nRight for > 0.65%
    // ----------------------------------------------------------------------------
    int nLeftLo, nLeftHi, nLeft;

    // Search right
    nLeft = searchRightTillAboveThreshold( &m_oSumVerSmoothSmall.front(),
                                           nSrcW,
                                           nFirstCropLeft,
                                           rSmoothSmallThr );

    nLeftLo = nLeft  -  m_nBoundaryWindowMaxWidth / 2;
    if ( nLeftLo < 0 )
    {
         nLeftLo = 0;
    }

    nLeftHi = nLeft  +  m_nBoundaryWindowMaxWidth / 2;
    if ( nLeftHi >= (int)nSrcW )
    {
         nLeftHi =  (int)nSrcW - 1;
    }

    nLeft = min_element( m_oSumVerStepFiltered.begin() + nLeftLo,
                         m_oSumVerStepFiltered.begin() + nLeftHi + 1 )
          -            ( m_oSumVerStepFiltered.begin() );


    // search from nLeft first crop boundary to the nRight for > 0.65%
    // ----------------------------------------------------------------------------
    int nRightLo, nRightHi, nRight;

    // Search left
    nRight = searchLeftTillAboveThreshold( &m_oSumVerSmoothSmall.front(),
                                           nSrcW,
                                           nFirstCropRight,
                                           rSmoothSmallThr );


    nRightLo = nRight  +  m_nBoundaryWindowMaxWidth / 2;
    if ( nRightLo >= (int)nSrcW )
    {
         nRightLo =  (int)nSrcW - 1;
    }

    nRightHi = nRight  -  m_nBoundaryWindowMaxWidth / 2;
    if ( nRightHi < 0 )
    {
         nRightHi = 0;
    }

    nRight  = max_element( m_oSumVerStepFiltered.begin() + nRightHi,
                           m_oSumVerStepFiltered.begin() + nRightLo + 1 )
            -            ( m_oSumVerStepFiltered.begin() );

    nLeftOut  = ( unsigned int ) nLeft;
    nRightOut = ( unsigned int ) nRight;
}


// Supports ROI in input image
void FTS_ANPR_Cropper::topBottomCrop( IplImage& oEdge16SC1,
                                    unsigned int& nTopOut,
                                    unsigned int& nBottomOut )
{
    unsigned int nSrcH = oEdge16SC1.height;

    // Essentially the same process as the left right cropping
    // ----------------------------------------------------------------------------
    sumHor16SC1( &oEdge16SC1, &m_oSumHor.front() );

    runSum( &m_oSumHor.front(),
            &m_oSumHorRunSum.front(),
            nSrcH );

    smooth( &m_oSumHorRunSum.front(),
            &m_oSumHorSmooth.front(),
            nSrcH,
            m_nVerSmallSmoothHalfWindow );


    int    nSumHorSmoothMaxPos  = max_element( m_oSumHorSmooth.begin(),
                                               m_oSumHorSmooth.begin() + nSrcH )
                                -            ( m_oSumHorSmooth.begin() );

    double rSumHorSmoothMax = m_oSumHorSmooth.at( nSumHorSmoothMaxPos );

    double rSumHorSmoothMin = *min_element( m_oSumHorSmooth.begin(),
                                            m_oSumHorSmooth.begin() + nSrcH );

    double rThrLo = m_rVerBoundaryWindowThresh
                  * ( rSumHorSmoothMax - rSumHorSmoothMin )
                  + rSumHorSmoothMin;


    // Searching left
    int nTop    = searchLeftTillBelowThreshold ( &m_oSumHorSmooth.front(), nSrcH, nSumHorSmoothMaxPos, rThrLo );
    // Searching right
    int nBottom = searchRightTillBelowThreshold( &m_oSumHorSmooth.front(), nSrcH, nSumHorSmoothMaxPos, rThrLo );

    if ( nTop <= -1 )
    {
         nTop =  0;
    }

    if ( nBottom <= -1 )
    {
         nBottom =  nSrcH - 1;
    }

    nTopOut    = nTop;
    nBottomOut = nBottom;
}

int FTS_ANPR_Cropper::searchLeftTillBelowThreshold( double* prSrc, unsigned int nSrcSize, unsigned int nStaPos, double rThreshold )
{
    for ( int x = nStaPos; x >= 0; --x )
    {
        if ( prSrc[ x ] < rThreshold )
        {
            return x;
        }
    }
    return -1;
}

int FTS_ANPR_Cropper::searchLeftTillAboveThreshold( double* prSrc, unsigned int nSrcSize, unsigned int nStaPos, double rThreshold )
{
    for ( int x = nStaPos; x >= 0; --x )
    {
        if ( prSrc[ x ] > rThreshold )
        {
            return x;
        }
    }
    return -1;
}

int FTS_ANPR_Cropper::searchRightTillBelowThreshold( double* prSrc, unsigned int nSrcSize, unsigned int nStaPos, double rThreshold )
{
    for ( unsigned int x = nStaPos; x < nSrcSize; ++x )
    {
        if ( prSrc[ x ] < rThreshold )
        {
            return x;
        }
    }
    return -1;
}

int FTS_ANPR_Cropper::searchRightTillAboveThreshold( double* prSrc, unsigned int nSrcSize, unsigned int nStaPos, double rThreshold )
{
    for ( unsigned int x = nStaPos; x < nSrcSize; ++x )
    {
        if ( prSrc[ x ] > rThreshold )
        {
            return x;
        }
    }
    return -1;
}



// Function: stepFilter
// Purpose: Apply a symetric step filter.
//          dst[i] is the result of smoothing with [-1, -1, ..., -1, -1, 1, 1, ... 1, 1]
//                                                                    ^
//                                                                    |
//          anchor of the filter is the last -1   ---------------------
//          Borders are padded with zeros.
// Return:
// Param:
//  runSum:
//      -- the "running sum"/"intergral image" of the original image to be filtered with the
//          step function.
//  dst:
//      -- dst
//  len:
//      -- dst len == sunSum len
//  halfStep:
//      -- width of half of the step filter. Entire filter width = halfStep*2
void FTS_ANPR_Cropper::stepFilter( double* prRunSum, double* prDst, int nLen, int nHalfStep )
{
    int i, l, r;
    double lSum, rSum;
    double total = prRunSum[ nLen-1 ];
    for ( i = 0; i < nLen; ++i )
    {
        l = i - nHalfStep;
        r = i + nHalfStep;
        if ( l < 0 )
        {
            lSum = 0;
        }
        else
        {
            lSum = prRunSum[ l ];
        }
        if ( r >= nLen )
        {
            rSum = total;
        }
        else
        {
            rSum = prRunSum[ r ];
        }
        prDst[ i ] = 2*prRunSum[ i ] - lSum - rSum;
    }
}


// Function: smooth
// Purpose: Smooth image with a rectangular filter.
// Return:
// Param:
//  runSum:
//      -- the "running sum"/"intergral image" of the original image to be smoothed
//  dst:
//      -- dst
//  len:
//      -- dst len == sunSum len
//  halfWind:
//      -- entire filter width = halfWind*2+1. Filter anchor is at the centre
void FTS_ANPR_Cropper::smooth( double* prRunSum, double* prDst, int nLen, int nHalfWind )
{
    int i, l, r;
    float lSum, rSum;
    float total = prRunSum[ nLen-1 ];
    for ( i = 0; i < nLen; ++i )
    {
        l = i - nHalfWind - 1;
        r = i + nHalfWind;
        if ( l < 0 )
        {
            lSum = 0;
        }
        else
        {
            lSum = prRunSum[ l ];
        }
        if ( r >= nLen )
        {
            rSum = total;
        }
        else
        {
            rSum = prRunSum[ r ];
        }
        prDst[ i ] = rSum - lSum;
    }
}

// Function: runSum
// Purpose: Running sum of src in dst. dst[i] is sum of pixel values in src upto and including i.
void FTS_ANPR_Cropper::runSum( double* prSrc, double* prDst, int nLen )
{
    double sum = 0;
    double* end = prSrc + nLen;
    for (; prSrc < end; ++prSrc, ++prDst)
    {
        sum += *prSrc;
        *prDst = sum;
    }
}

// Function: clipHi
// Purpose: if (src[i] > thresh) dst[i] = thresh; else dst[i] = src[i];
void FTS_ANPR_Cropper::clipHi( double* prSrc, double* prDst, int nLen, double rThresh )
{
    double* end = prSrc + nLen;
    for ( ; prSrc < end; ++prSrc, ++prDst )
    {
        if ( *prSrc > rThresh )
        {
            *prDst = rThresh;
        }
        else *prDst = *prSrc;
    }
}

// Function: mean
// Purpose: Find mean of array
// Return: mean
double FTS_ANPR_Cropper::mean( double* prSrc, int nLen )
{
    double sum = 0;
    double* end = prSrc + nLen;
    for (; prSrc < end; ++prSrc)
    {
        sum += *prSrc;
    }
    return sum/( double )nLen;
}


// Function: sumVer16SC1
// Purpose: Sum each colum of image
// Return:
// Param:
//  src:
//      -- must be 16SC1
//  sum:
//      -- length = src->width;
void FTS_ANPR_Cropper::sumVer16SC1( IplImage* poSrc, double* rSum )
{
    CvRect roi = cvGetImageROI( poSrc );
    int r, c;
    short* row;

    for ( c = 0; c < roi.width; ++c )
    {
        rSum[c] = 0;
    }

        //?
    for (r = 1; r < roi.height-1; ++r)
    {

        row = CVA_ROI_ROW(short*, poSrc, roi, r);
        for (c = 0; c < roi.width; ++c)
        {
            rSum[c] += (double)row[c];
        }
    }

}

// Function: sumVer16SC1
// Purpose: Sum each row of image
// Return:
// Param:
//  src:
//      -- must be 16SC1
//  sum:
//      -- length = src->height;
void FTS_ANPR_Cropper::sumHor16SC1( IplImage* poSrc, double* rSum )
{
    CvRect roi = cvGetImageROI( poSrc );
    int r, c;
    short* row;

    rSum[ 0 ] = 0;
    rSum[ roi.height-1 ] = 0;

        //?
    for (r = 1; r < roi.height-1; ++r)
    {
        row = CVA_ROI_ROW(short*, poSrc, roi, r);
        double s = 0;
        for (c = 0; c < roi.width; ++c)
        {
            s += (double)row[c];
        }
        rSum[r] = s;
    }

}


















