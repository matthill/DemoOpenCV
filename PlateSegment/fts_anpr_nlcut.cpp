
#include "fts_anpr_nlcut.h"


FTS_ANPR_NLCut::FTS_ANPR_NLCut()
	: m_nNonLinearPentaly( 0 )
	, m_nNCutStepMargin( 5 )
    , m_nNCutMinNumSegments( 5 )
    , m_nNCutMaxNumSegments( 20 )
    , m_oCost()
    , m_oScore()
    , m_oBackPtr()
{
    // Nothing
}


FTS_ANPR_NLCut::~FTS_ANPR_NLCut()
{
    // Nothing
}


//! Vertical non-linear cut
void FTS_ANPR_NLCut::cut( const cv::Mat& oImage,
				unsigned int nXStart,
				unsigned int nXEnd,
				bool bForward,
				bool bBlackChar,
				const std::vector<int>& oCutCoordsPrev,
					  std::vector<int>& oCutCoords )
{
    fillCostMatrix( oImage, nXStart, nXEnd, bForward, bBlackChar, oCutCoordsPrev, m_oCost );

    fillScoreMatrix( m_oCost, m_oScore, m_oBackPtr );

    traceBack( m_oScore, m_oBackPtr, nXStart, oCutCoords );
}


void FTS_ANPR_NLCut::fillCostMatrix( const cv::Mat& oImage,
                                           unsigned int nXStart,
                                           unsigned int nXEnd,
                                           bool bForward,
                                           bool bBlackChar,
                                           const std::vector<int>& oCutCoordsPrev,
                                           cv::Mat& oCost )
{
    unsigned int h = oImage.rows;
    unsigned int w = nXEnd - nXStart;


    // Fill in the cost matrix
    // ------------------------------------------------------------------------
    oCost.create( h, w, CV_32FC1 );

    for( unsigned int y = 0; y < h; ++y )
    {
        const unsigned char* pcImageRow = (unsigned char*)oImage.ptr<unsigned char>( y );

        float* prCostRow = (float*)oCost.ptr<float>( y );

        for( unsigned int x = 0; x < w; ++x )
        {
            unsigned char cPixel = pcImageRow[ x + nXStart ];

            if( bBlackChar )
            {
                cPixel = 255 - cPixel;
            }

            if( bForward )
            {
                if( (int)(x + nXStart) <= oCutCoordsPrev[y] )
                {
                    prCostRow[ x ] = 10000; //FLT_MAX;
                }
                else
                {
                    prCostRow[ x ] = cPixel;
                }
            }
            else
            {
                if( (int)(x + nXStart) >= oCutCoordsPrev[y] )
                {
                    prCostRow[ x ] = 10000; //FLT_MAX;
                }
                else
                {
                    prCostRow[ x ] = cPixel;
                }
            }
        } // for x
    } // for y

    assert( oCost.type() == CV_32FC1 );
}


void FTS_ANPR_NLCut::fillScoreMatrix( const cv::Mat& oCost,
							  cv::Mat& oScore,
							  cv::Mat& oBackPtr )
{
    assert( oCost.type() == CV_32FC1 );

    unsigned int w = oCost.cols;
    unsigned int h = oCost.rows;
//    printf( "row = %d, col = %d\n", oCost.rows, oCost.cols );

    oScore  .create( oCost.rows, oCost.cols, CV_32FC1 );
    oBackPtr.create( oCost.rows, oCost.cols, CV_32SC1 );

    // Copy the first row of cost matrix into the score matrix, this is
    // the base condition for the dynamic programming recursion
    // ------------------------------------------------------------------------
    const float* prCostRow    = (float*)oCost   .ptr<float>( 0 );
          float* prScoreRow   = (float*)oScore  .ptr<float>( 0 );
          int*   pnBackPtrRow = (int*)  oBackPtr.ptr<int>( 0 );

    for( unsigned int x = 0; x < w; ++x )
    {
        prScoreRow  [ x ] = prCostRow[ x ];
        pnBackPtrRow[ x ] = x;
    }


    // Resursion to calculate all oScore[iRow,iColumn]:
    //
    // oScore[iRow,iColumn] = oCost[iRow][iCol] + iMinScoreVal;
    // oBackPtr[iRow][iColumn] = iMinIndex;
    // ------------------------------------------------------------------------
    for( unsigned int y = 1; y < h; ++y )
    {
        const float* prScoreRowPrev = (float*)oScore  .ptr<float>( y - 1 );
                     prScoreRow     = (float*)oScore  .ptr<float>( y );
                     prCostRow      = (float*)oCost   .ptr<float>( y );
                     pnBackPtrRow   = (int*)  oBackPtr.ptr<int>( y );

        for( unsigned int x = 0; x < w; ++x )
        {
            // Check and fix the boundary
            int nMostLeft = x - 1;
            if( nMostLeft < 0 )
            {
                nMostLeft = 0;
            }

            int nMostRight = x + 1;
            if( nMostRight >= (int)w )
            {
                nMostRight = (int)( w - 1 );
            }

            // Find the min value and min index
            // ----------------------------------------------------------------
            float rPenalty = abs( nMostLeft - (int)x ) * m_nNonLinearPentaly;
            float rMinScore = prScoreRowPrev[ nMostLeft ] + rPenalty;
            int   nMinIndex = nMostLeft;
            for( int i = nMostLeft+1; i <= nMostRight; ++i )
            {
                // Penalising horizontal move
                rPenalty = abs( i - (int)x ) * m_nNonLinearPentaly;

                float rScore = prScoreRowPrev[ i ] + rPenalty;
                if( rMinScore > rScore )
                {
                    rMinScore = rScore;
                    nMinIndex = i;
                }
            }

            // Calculate oScore[iRow][iColumn], and the back tracking pointer
            // ----------------------------------------------------------------

            prScoreRow  [ x ] = rMinScore + prCostRow[ x ];
            pnBackPtrRow[ x ] = nMinIndex;

        } // for x
    } // for y
}


void FTS_ANPR_NLCut::traceBack( const cv::Mat& oScore,
						const cv::Mat& oBackPtr,
						unsigned int nXStart,
						std::vector<int>& oCutCoords )
{
    unsigned int h = oScore.rows;

    // Find the min value and min index of the last row from oScoreMatrix
    const float* prScoreLastRow = (float*)oScore.ptr<float>( h-1 );

    const float* prMinLastRowScore = std::min_element( prScoreLastRow,
                                                       prScoreLastRow + oScore.cols );

    int nMinLastRowIndex = prMinLastRowScore - prScoreLastRow;

    oCutCoords.resize( h );

    // The last node of the shortest path
    oCutCoords.at( h - 1 ) = nMinLastRowIndex + nXStart;

    // Tracing backward
    int nPrevIndex = nMinLastRowIndex;
    for( unsigned int y = h - 1; y > 0; --y )
    {
        nPrevIndex = oBackPtr.at<int>( y, nPrevIndex );
        oCutCoords.at( y - 1 ) = nPrevIndex + nXStart;
    }
}

void FTS_ANPR_NLCut::segmentUsingNCut( cv::Mat& oGray,
							bool bBlackChar,
							std::vector< std::vector<int> >& oNCutCoords )
{
    std::vector<int> m_oNCutInitCoord;

    // Depend on the character size, horizontal step window will vary
    unsigned int nMinNumSegments = m_nNCutMinNumSegments;

    int nStep = cvRound(   (float) oGray.cols
                         / (float) nMinNumSegments
                         - m_nNCutStepMargin );

    //! HAZARD if we have a partial crop of the plate, there might not be this many segments, therefore iStep will be a negative value
    while( nStep <= 0 ) //! Attempt to recover by dividing by smaller numbers until iStep is positive
    {
        --nMinNumSegments;
        if( nMinNumSegments == 0 )
        {
            nStep = 1;
            break;
        }
        nStep = cvRound(   (float) oGray.cols
                         / (float) nMinNumSegments
                         - m_nNCutStepMargin
                         );
    }

    m_oNCutInitCoord.clear();
    m_oNCutInitCoord.resize( oGray.rows, 0 );

    oNCutCoords.clear();
    oNCutCoords.push_back( std::vector<int>() ); // create new empty vector, SN_LPR_NonlinearCut::nlCut() will fill it in

    std::vector<int>* poCoord = &oNCutCoords.at(0);

    // First cut arbitrarily the whole plate
    // --------------------------------------------------------------------
    cut( oGray,
                         0,                    // nXStart
                         oGray.cols, // nXEnd
                         true,                 // bForward
                         bBlackChar,
                         m_oNCutInitCoord,     // oCutCoordsPrev
                         *poCoord
                         );

    // Get vertical mid point of cut
    // --------------------------------------------------------------------
    int nMidHeight = cvRound(  (float)oGray.rows / 2.0f  );

    int nEnd = poCoord->at(nMidHeight) - m_nNCutStepMargin;

    int nMinInd = std::distance(                   poCoord->begin(),
                                 std::min_element( poCoord->begin(), poCoord->end() )
                                 );

    int nMin = poCoord->at( nMinInd );

    int nStart;

    // Now cut backward
    // --------------------------------------------------------------------
    while(    nMin - nStep > 0
           && nEnd - nStep > 0
           && oNCutCoords.size() < m_nNCutMaxNumSegments )
    {
        // printf("iEnd = %d, iStart = %d, iStep = %d\n",iEnd,iStart,iStep);

        nStart = std::max( 0,
                           nEnd - nStep );

        oNCutCoords.push_back( std::vector<int>() ); // create new empty vector, SN_LPR_NonlinearCut::nlCut() will fill it in

        cut( oGray,
                             nStart, // nXStart
                             nEnd,   // nXEnd
                             false,  // bForward
                             bBlackChar,
                             oNCutCoords.at( oNCutCoords.size() - 2 ),  // oCutCoordsPrev
                             oNCutCoords.back()     // oCutCoordsPrev
                             );

        poCoord = &oNCutCoords.back();

        nEnd = poCoord->at(nMidHeight) - m_nNCutStepMargin;

        nMinInd = std::distance(                   poCoord->begin(),
                                 std::min_element( poCoord->begin(), poCoord->end() )
                                 );

        nMin = poCoord->at( nMinInd );
    }

#if DEBUG_OLD
    printf("0 - m_pnCutCoords[%d] = %d\n",m_nNumOfSeg-1,m_pnCutCoords[m_nNumOfSeg-1][cvRound(iHeight/2)]);
#endif

    // Because we cut backward, now reverse the array before cut forward
    // --------------------------------------------------------------------
    std::reverse( oNCutCoords.begin(),
                  oNCutCoords.end()
                  );

#if DEBUG_OLD
    printf( "The middle cut is segment %d\n", m_nNumOfSeg );
#endif

    // Save the start and the max position
    // --------------------------------------------------------------------
    poCoord = &oNCutCoords.back();

    nStart = poCoord->at( nMidHeight )  +  m_nNCutStepMargin;

    int nMaxIdx = 0;

#if DEBUG_OLD
    printf( "1 - m_pnCutCoords[%d] = %d\n",m_nNumOfSeg-1,m_pnCutCoords[m_nNumOfSeg-1][cvRound(iHeight/2)] );
#endif

    nMaxIdx = std::distance(                   poCoord->begin(),
                             std::max_element( poCoord->begin(), poCoord->end() )
                             );

    int nMax = poCoord->at( nMaxIdx );

    // Now let's cut forward
    // --------------------------------------------------------------------
    while(    nMax   + nStep < oGray.cols
           && nStart + nStep < oGray.cols
           && oNCutCoords.size() < m_nNCutMaxNumSegments )
    {
        nEnd = std::min( nStart + nStep,  oGray.cols );

        oNCutCoords.push_back( std::vector<int>() ); // create new empty vector, SN_LPR_NonlinearCut::nlCut() will fill it in

        cut( oGray,
			 nStart, // nXStart
			 nEnd,   // nXEnd
			 true,  // bForward
			 bBlackChar,
			 oNCutCoords.at( oNCutCoords.size() - 2 ),  // oCutCoordsPrev
			 oNCutCoords.back()     // oCutCoordsPrev
			 );

        poCoord = &oNCutCoords.back();

        nStart = poCoord->at( nMidHeight )  +  m_nNCutStepMargin;

        nMaxIdx = std::distance(                   poCoord->begin(),
                                 std::max_element( poCoord->begin(), poCoord->end() )
                                 );

        nMax = poCoord->at( nMaxIdx );
    }
}

void FTS_ANPR_NLCut::drawNCut( cv::Mat& oGrayOrBGR,
					  const std::vector< std::vector<int> >&  oNCutCoords,
					  CvScalar oColor,
					  bool b4Connected )
{
    for( unsigned int i = 0; i < oNCutCoords.size(); ++i )
    {
        const std::vector<int>& oCoord = oNCutCoords.at( i );

        for( unsigned int j = 1; j < oCoord.size(); ++j )
        {

          cv::line( oGrayOrBGR,
				  	cv::Point( oCoord.at( j - 1 ),  j - 1 ),
				  	cv::Point( oCoord.at( j     ),  j     ),
				  	oColor, 1 );
        }
    }
}
