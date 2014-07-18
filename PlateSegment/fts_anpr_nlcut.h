
#ifndef _FTS_ANPR_NLCUT_H_
#define _FTS_ANPR_NLCUT_H_

#include "fts_base_externals.h"

/*!
 *
 * Nonlinear cut on images
 *
 */
class FTS_ANPR_NLCut
{

public:

    // Constructors / Destructor / Assignment Operator / Copy Constructor
    explicit FTS_ANPR_NLCut();
    virtual ~FTS_ANPR_NLCut();


public:

    // Interface
    virtual void cut( const cv::Mat& oImage,
                      unsigned int nXStart,
                      unsigned int nXEnd,
                      bool bForward,
                      bool bBlackChar,
                      const std::vector<int>& oCutCoordsPrev,
                            std::vector<int>& oCutCoords );

    virtual void fillCostMatrix( const cv::Mat& oImage,
                                 unsigned int nXStart,
                                 unsigned int nXEnd,
                                 bool bForward,
                                 bool bBlackChar,
                                 const std::vector<int>& oCutCoordsPrev,
                                 cv::Mat& oCost );

    virtual void fillScoreMatrix( const cv::Mat& oCost,
                                  cv::Mat& oScore,
                                  cv::Mat& oBackPtr );

    virtual void traceBack( const cv::Mat& oScore,
                            const cv::Mat& oBackPtr,
                            unsigned int nXStart,
                            std::vector<int>& oCutCoords );

    void segmentUsingNCut( cv::Mat& oGray,
						   bool bBlackChar,
						   std::vector< std::vector<int> >& oNCutCoords );

    void drawNCut( cv::Mat& oGrayOrBGR,
				   const std::vector< std::vector<int> >&  oNCutCoords,
				   CvScalar oColor,
				   bool b4Connected = true );

public:

    // Attributes
    float m_nNonLinearPentaly; // penalty for moving 1 step in x direction. i.e. larger value enforce straighter lines

    unsigned int m_nNCutStepMargin;
    unsigned int m_nNCutMinNumSegments;
    unsigned int m_nNCutMaxNumSegments;


protected:

    cv::Mat m_oCost;
    cv::Mat m_oScore;
    cv::Mat m_oBackPtr;


private:

    // Copy Constructor & Assignment Operator
    FTS_ANPR_NLCut( const FTS_ANPR_NLCut& ); //!< Always have these either private or implemented
    FTS_ANPR_NLCut& operator=( const FTS_ANPR_NLCut& ); //!< If not implemented, don't need arg names
};


#endif // _FTS_ANPR_NLCUT_H_
