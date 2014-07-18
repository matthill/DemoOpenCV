/*
 * fts_base_common.h
 *
 * A repository of useful functions
 *
 */

#include "fts_base_externals.h"
#include "fts_base_stackarray.h"


#ifndef _FTS_BASE_COMMON_H_
#define _FTS_BASE_COMMON_H_

// -------------------------------------------------------------
// -------------------------------------------------------------
// Prototypes:
// -------------------------------------------------------------
// -------------------------------------------------------------

//#define FTS_BASE_ASSERT_MULTIPLE_OF_8_BYTES( Type )
//    typedef char type_must_be_multiples_of_8_bytes[ sizeof(Type)? 1: -1 ];
//    (void) sizeof( type_must_be_multiples_of_8_bytes );

// LINUX
#define FTS_BASE_CV_MAT_ON_STACK( name, rows, cols, cv_type )            \
    CvMat name;                                                         \
    /* we need at least 2 rows for CvMat::step to be non-zero*/         \
    cvInitMatHeader( &name, 2, cols, cv_type );                         \
    unsigned char name##Data[ rows * name.step ];                       \
    cvInitMatHeader( &name, rows, cols, cv_type, name##Data, name.step )

// WIN
//	#define FTS_BASE_CV_MAT_ON_STACK( name, rows, cols, cv_type )            \
//    {																		\
//		CvMat name;                                                         \
//		/* we need at least 2 rows for CvMat::step to be non-zero*/         \
//		cvInitMatHeader( &name, 2, cols, cv_type );                         \
//		unsigned char ptr##name[ rows * name.step ];                       \
//		cvInitMatHeader( &name, rows, cols, cv_type, ptr##name, name.step ) \
//	}


const float FTS_BASE_PI = 3.1415926548f;

    inline float
FTS_BASE_ToDegrees( float radians ); //!< Takes radians, returns degrees

    inline float
FTS_BASE_ToRadians( float degrees ); //!< Takes degrees, returns radians

// Take square
#define FTS_BASE_SQ( a )  ( (a)*(a) )


    template< class T > inline void
FTS_BASE_Swap( T& a, T& b );

    template< class T > inline T
FTS_BASE_Euclidean( T a, T b );

    inline float
FTS_BASE_Euclidean( CvPoint2D32f a, CvPoint2D32f b );

    inline CvPoint2D32f
FTS_BASE_Rotate( const CvPoint2D32f& point, float rTheta );

    inline float
FTS_BASE_DotProduct( const CvPoint2D32f& v1, const CvPoint2D32f& v2 );

    inline CvPoint2D32f
FTS_BASE_VectorSum( const CvPoint2D32f& v1, const CvPoint2D32f& v2 );

    inline float
FTS_BASE_VectorMagnitude( const CvPoint2D32f& v );

    inline CvPoint2D32f
FTS_BASE_RectCentre( const CvRect& r );

    inline bool
FTS_BASE_FileExists( const std::string& sFilename );


// NOTE: must have AStartInclusive <= AEndInclusive, undefined behaviour otherwise
// NOTE: This function creates no intermediate objects except the return value.
    template< class T > inline T
FTS_BASE_OverlapLength( T AStartInclusive, T AEndInclusive,
                       T BStartInclusive, T BEndInclusive  );

    inline unsigned int
FTS_BASE_OverlapArea( CvRect A, CvRect B );


    template< class T > inline T&
FTS_BASE_Median( std::vector<T>& oV );

    template< class T > inline T&
FTS_BASE_Median( FTS_BASE_StackArray<T>& oV );

    template< class T > inline T&
FTS_BASE_MedianBiasHigh( FTS_BASE_StackArray<T>& oV );


inline int FTS_BASE_RoundToOdd ( float  r );
inline int FTS_BASE_RoundToOdd ( double r );

inline int FTS_BASE_RoundToEven( float  r );
inline int FTS_BASE_RoundToEven( double r );

    template< class T > inline void
FTS_BASE_DeleteMapValues( T& oMap );

    template< class T > inline void
FTS_BASE_DeleteValues( T& oContainer );

    template< class T > inline void
FTS_BASE_DeleteValues( std::stack<T>& oStack );


    inline bool
operator==( const CvRect& a, const CvRect& b );

    inline CvRect&
operator*=( const CvRect& oLHS, const float rScale );

    inline CvSize&
operator*=( CvSize& oLHS, const float rScale );

    inline CvSize
operator*( const CvSize& oSize, const float rScale );

    template< typename InputIterator >
inline int FTS_BASE_MaxElementIndex( InputIterator iFirst, InputIterator iLast );

    template< class T >
inline int FTS_BASE_MaxElementIndex( const T& oContainer );

    template< typename InputIterator >
inline int FTS_BASE_MinElementIndex( InputIterator iFirst, InputIterator iLast );

    template< class T >
inline int FTS_BASE_MinElementIndex( const T& oContainer );

    inline CvRect
FTS_BASE_Clip( const CvRect oRect, const CvRect oClipRect );

    inline CvRect
FTS_BASE_Clip( const CvRect oRect, unsigned int nClipWidth, unsigned int nClipHeight );



// -------------------------------------------------------------
// -------------------------------------------------------------
// Inline and template definitions
// -------------------------------------------------------------
// -------------------------------------------------------------

template< class T >
inline void FTS_BASE_Swap( T& a, T& b )
{
    T temp = a;
    a = b;
    b = temp;
}

template< class T >
inline T FTS_BASE_Euclidean( T a, T b )
{
	a *= a;
	b *= b;
	T c = a + b;
	c = sqrt( c );

	return c;
}

inline float FTS_BASE_Euclidean( CvPoint2D32f a, CvPoint2D32f b )
{
    return FTS_BASE_Euclidean( a.x-b.x,
                              a.y-b.y  );
}



inline CvPoint2D32f FTS_BASE_Rotate( const CvPoint2D32f& point, float rTheta )
{
	float rSinT = sin( rTheta );
	float rCosT = cos( rTheta );

	CvPoint2D32f rotated;

	rotated.x =  point.x *  rCosT
	          +  point.y * -rSinT;
	rotated.y =  point.x *  rSinT
	          +  point.y *  rCosT;

	return rotated;
}

inline float FTS_BASE_DotProduct( const CvPoint2D32f& v1, const CvPoint2D32f& v2 )
{
	float rDotProduct;

	rDotProduct = v1.x * v2.x
	            + v1.y * v2.y;

	return rDotProduct;
}

inline CvPoint2D32f FTS_BASE_VectorSum( const CvPoint2D32f& v1, const CvPoint2D32f& v2 )
{
	CvPoint2D32f sum;

	sum.x = v1.x + v2.x;
	sum.y = v1.y + v2.y;

	return sum;
}

inline float FTS_BASE_VectorMagnitude( const CvPoint2D32f& v )
{
	return sqrt( v.x * v.x + v.y * v.y );
}

inline CvPoint2D32f FTS_BASE_RectCentre( const CvRect& r )
{
    CvPoint2D32f ret;
    ret.x = (float) r.x
          + (float) r.width  / 2.0f;
    ret.y = (float) r.y
          + (float) r.height / 2.0f;

    return ret;
}


inline float FTS_BASE_ToDegrees( float radians )  //!< Takes radians, returns degrees
{
    return ( radians / FTS_BASE_PI * 180.0f );
}

inline float FTS_BASE_ToRadians( float degrees ) //!< Takes degrees, returns radians
{
    return ( degrees / 180.0f * FTS_BASE_PI );
}

template< class T > inline T
FTS_BASE_OverlapLength( T AStartInclusive, T AEndInclusive,
                       T BStartInclusive, T BEndInclusive  )
{
    if ( AEndInclusive < BStartInclusive )
    {
        return  ( AEndInclusive - AEndInclusive );  // don't do 'return 0' so that we can handle non-primitive types, like FTS_BASE_Time.
                                                    // the 'operator=()' is need later in the function anyway, so the user must supply it.
    }

    if ( BEndInclusive < AStartInclusive )
    {
        return  ( AEndInclusive - AEndInclusive );
    }

    // Gets here if there must be some overlap

    // Get max value for begin of overlapping interval
    if ( AStartInclusive >= BStartInclusive )  // then use AStartInclusive
    {
        // Get min value for end of overlapping interval
        if ( AEndInclusive <= BEndInclusive )    // then use BEndInclusive
        {
            return  ( AEndInclusive - AStartInclusive );
        }
        else // then use AEndInclusive
        {
            return  ( BEndInclusive - AStartInclusive );
        }
    }
    else // then use BStartInclusive
    {
        // Get min value for end of overlapping interval
        if ( AEndInclusive <= BEndInclusive )    // then use BEndInclusive
        {
            return  ( AEndInclusive - BStartInclusive );
        }
        else // then use AEndInclusive
        {
            return  ( BEndInclusive - BStartInclusive );
        }
    }
}

unsigned int FTS_BASE_OverlapArea( CvRect a, CvRect b )
{
    return FTS_BASE_OverlapLength( a.x,  a.x + a.width  - 1,
                                  b.x,  b.x + b.width  - 1  )
         * FTS_BASE_OverlapLength( a.y,  a.y + a.height - 1,
                                  b.y,  b.y + b.height - 1  );
}


// Modifies oV
template< class T >
inline T& FTS_BASE_Median( std::vector<T>& oV )
{
    // Round down to median
    // eg. 0 1 2 3 4, median idx is: ( 5 - 1 ) / 2  =  2
    // eg. 0 1 2 3,   median idx is: ( 4 - 1 ) / 2  =  1
    unsigned int nMedianIdx = (oV.size() - 1)  /  2;

    std::nth_element( oV.begin(),
                      oV.begin() + nMedianIdx,
                      oV.end  ()  );

    return oV.at( nMedianIdx );
}

template< class T >
inline T& FTS_BASE_Nth( std::vector<T>& oV, unsigned int nNth )
{
    assert( nNth < oV.size() );

    std::nth_element( oV.begin(),
                      oV.begin() + nNth,
                      oV.end  ()  );

    return oV.at( nNth );
}

template< class T, typename Compare >
inline T& FTS_BASE_Nth( std::vector<T>& oV, unsigned int nNth, Compare compare )
{
    assert( nNth < oV.size() );

    std::nth_element( oV.begin(),
                      oV.begin() + nNth,
                      oV.end  (),
                      compare );

    return oV.at( nNth );
}


template< class T >
inline T& FTS_BASE_Median( FTS_BASE_StackArray<T>& oV )
{
    // Round down to median
    // eg. 0 1 2 3 4, median idx is: ( 5 - 1 ) / 2  =  2
    // eg. 0 1 2 3,   median idx is: ( 4 - 1 ) / 2  =  1
    unsigned int nMedianIdx = ( oV.size() -1 ) / 2;	// DV: bias towards bigger values
    std::nth_element( oV.begin(),
                      oV.begin() + nMedianIdx,
                      oV.end  ()  );
    return oV.at( nMedianIdx );  // this throws when oV.size() == 0, so we don't have to explicitly check
}

template< class T >
inline T& FTS_BASE_MedianBiasHigh( FTS_BASE_StackArray<T>& oV )
{
    // Round down to median
    // eg. 0 1 2 3 4, median idx is: ( 5 - 1 ) / 2  =  2
    // eg. 0 1 2 3,   median idx is: ( 4 - 1 ) / 2  =  1
    unsigned int nMedianIdx = oV.size() / 2;	// DV: bias towards bigger values
    std::nth_element( oV.begin(),
                      oV.begin() + nMedianIdx,
                      oV.end  ()  );
    return oV.at( nMedianIdx );  // this throws when oV.size() == 0, so we don't have to explicitly check
}

template< class T >
inline T& FTS_BASE_Nth( FTS_BASE_StackArray<T>& oV, unsigned int nNth )
{
    assert( nNth < oV.size() );

    std::nth_element( oV.begin(),
                      oV.begin() + nNth,
                      oV.end  ()  );

    return oV.at( nNth );  // this throws when oV.size() == 0, so we don't have to explicitly check
}

template< class T, typename Compare >
inline T& FTS_BASE_Nth( FTS_BASE_StackArray<T>& oV, unsigned int nNth, Compare compare )
{
    assert( nNth < oV.size() );

    std::nth_element( oV.begin(),
                      oV.begin() + nNth,
                      oV.end  (),
                      compare );

    return oV.at( nNth );  // this throws when oV.size() == 0, so we don't have to explicitly check
}


inline bool FTS_BASE_FileExists( const std::string& sFilename )
{
	struct stat sFileInfo;

	int n = stat( sFilename.c_str(), &sFileInfo ); // Attempt to get the file attributes

	if( n == 0 )
	{
		return true;
	}

	return false;
}

inline unsigned int FTS_BASE_RoundUpToMultipleOfN( unsigned int n, unsigned int nBlockSize )
{
    assert( nBlockSize > 0 );

    return ( ( n + nBlockSize - 1 ) / nBlockSize ) * nBlockSize;
}


// nBytes      -- number of bytes
// nAlignBytes -- the byte alignment
// return -- the number of bytes rounded up to the nearest multiple of nAlignBytes
// eg, 9 bytes with and alignment of 4 returns 12
inline unsigned int FTS_BASE_AlignBytes( unsigned int nBytes, unsigned int nAlignBytes )
{
    assert( nAlignBytes > 0 );

    unsigned int nBlocks = ( nBytes - 1 )
                           / nAlignBytes
                           + 1;

    return nBlocks * nAlignBytes;
}

// Rounds to the nearest odd number
inline int FTS_BASE_RoundToOdd( float r )
{
    int n = (int)floor( r );

    return n + (n % 2 == 0);  // add 1 if it's even
}

// Rounds to the nearest even number
inline int FTS_BASE_RoundToEven( float r )
{
    int n = (int)floor( r );

    return n + (n % 2 == 1);  // add 1 if it's odd
}

// Rounds to the nearest odd number
inline int FTS_BASE_RoundToOdd( double r )
{
    int n = (int)floor( r );

    return n + (n % 2 == 0);  // add 1 if it's even
}

// Rounds to the nearest even number
inline int FTS_BASE_RoundToEven( double r )
{
    int n = (int)floor( r );

    return n + (n % 2 == 1);  // add 1 if it's odd
}


// Note: using FTS_BASE_DeleteValues( std::map<First, Second>& oMap ) would be
// bad choice because std::map<First, Second> uses default allocator, which
// means it won't take a map with a custom allocator.
template< class T >
inline void FTS_BASE_DeleteMapValues( T& oMap )
{
    typename T::iterator i  = oMap.begin();
    typename T::iterator iE = oMap.end();

    for( ; i != iE; ++i )
    {
        delete i->second;
    }
    oMap.clear();
}

// Default template to delete each element of a container and clear the container.
// Specialized versions exist for maps.
template< class T >
inline void FTS_BASE_DeleteValues( T& oContainer )
{
    typename T::iterator i  = oContainer.begin();
    typename T::iterator iE = oContainer.end();

    for( ; i != iE; ++i )
    {
        delete *i;
    }

    oContainer.clear();
}

// Default template to delete each element of a container and clear the container.
// Specialized versions exist for maps.
template< class T >
inline void FTS_BASE_DeleteValues( std::stack<T>& oStack )
{
    while( !oStack.empty() )
    {
        delete oStack.top();
               oStack.pop();
    }
}



inline bool operator==( const CvRect& a, const CvRect& b )
{
    return (     a.x      == b.x
             &&  a.y      == b.y
             &&  a.width  == b.width
             &&  a.height == b.height );
}

inline CvRect& operator*=( CvRect& oLHS, const float rScale )
{
    oLHS.x      *= rScale;
    oLHS.y      *= rScale;
    oLHS.width  *= rScale;
    oLHS.height *= rScale;

    return oLHS;
}

inline CvRect& operator/=( CvRect& oLHS, const float rScale )
{
    oLHS.x      /= rScale;
    oLHS.y      /= rScale;
    oLHS.width  /= rScale;
    oLHS.height /= rScale;

    return oLHS;
}

inline CvSize& operator*=( CvSize& oLHS, const float rScale )
{
    oLHS.width  *= rScale;
    oLHS.height *= rScale;

    return oLHS;
}

inline CvSize operator*( const CvSize& oSize, const float rScale )
{
    CvSize oRet;
    oRet.width  = oSize.width  * rScale;
    oRet.height = oSize.height * rScale;

    return oRet;
}

template< typename InputIterator >
inline int FTS_BASE_MaxElementIndex( InputIterator iFirst, InputIterator iLast )
{
    return std::distance(                    iFirst,
                           std::max_element( iFirst, iLast )  );
}


template< class T >
inline int FTS_BASE_MaxElementIndex( const T& oContainer )
{
    return std::distance(                    oContainer.begin(),
                           std::max_element( oContainer.begin(), oContainer.end() )  );
}

template< typename InputIterator >
inline int FTS_BASE_MinElementIndex( InputIterator iFirst, InputIterator iLast )
{
    return std::distance(                    iFirst,
                           std::min_element( iFirst, iLast )  );
}


template< class T >
inline int FTS_BASE_MinElementIndex( const T& oContainer )
{
    return std::distance(                    oContainer.begin(),
                           std::min_element( oContainer.begin(), oContainer.end() )  );
}


template< class T >
inline T FTS_BASE_Clip( const T& value, const T& lo, const T& hi )
{
    if( value < lo )
    {
        return lo;
    }

    if( value > hi )
    {
        return hi;
    }

    return value;
}

inline CvRect FTS_BASE_Clip( const CvRect oRect, const CvRect oClipRect )
{
    int x1 = FTS_BASE_Clip( oRect.x,                oClipRect.x, oClipRect.x + oClipRect.width );
    int x2 = FTS_BASE_Clip( oRect.x + oRect.width,  oClipRect.x, oClipRect.x + oClipRect.width );

    int y1 = FTS_BASE_Clip( oRect.y,                oClipRect.y, oClipRect.y + oClipRect.height );
    int y2 = FTS_BASE_Clip( oRect.y + oRect.height, oClipRect.y, oClipRect.y + oClipRect.height );

    return cvRect( x1, y1, x2-x1, y2-y1 );
}

inline CvRect FTS_BASE_Clip( const CvRect oRect, unsigned int nClipWidth, unsigned int nClipHeight )
{
    return FTS_BASE_Clip( oRect, cvRect( 0, 0, nClipWidth, nClipHeight ) );
}


#endif // _FTS_BASE_COMMON_H_














