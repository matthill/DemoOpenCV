
#include "fts_base_debug.h"

static unsigned int s_nPauseFromNthCall = 1;
static unsigned int s_nNthPause = 0;

#if defined(_MSC_VER) || defined(__MINGW32__)
//int gettimeofday(struct timeval* tp, void* tzp)
//{
//    unsigned long t;
//    t = timeGetTime();
//    tp->tv_sec = t / 1000;
//    tp->tv_usec = t % 1000;
//    * 0 indicates that the call succeeded. */
//    return 0;
//}
int gettimeofday(struct timeval *tv, struct timezone *tz)	
{
	FILETIME ft;
	unsigned __int64 tmpres = 0;
	static int tzflag;

	if (NULL != tv)
	{
		GetSystemTimeAsFileTime(&ft);

		tmpres |= ft.dwHighDateTime;
		tmpres <<= 32;
		tmpres |= ft.dwLowDateTime;

		/*converting file time to unix epoch*/
		tmpres -= DELTA_EPOCH_IN_MICROSECS; 
		tmpres /= 10;  /*convert into microseconds*/
		tv->tv_sec = (long)(tmpres / 1000000UL);
		tv->tv_usec = (long)(tmpres % 1000000UL);
	}

	if (NULL != tz)
	{
		if (!tzflag)
		{
			_tzset();
			tzflag++;
		}
		tz->tz_minuteswest = _timezone / 60;
		tz->tz_dsttime = _daylight;
	}

	return 0;
}
#endif

void FTS_PAUSE_FromNthCall( unsigned int nth )
{
    s_nPauseFromNthCall = nth;
}

bool FTS_PAUSE_PauseOnThisCall( unsigned int& nCurrAtNthPause )
{
    //TODO, maybe we need mutex for this
    ++s_nNthPause;

    nCurrAtNthPause = s_nNthPause;

    return  ( s_nNthPause >= s_nPauseFromNthCall );
}

void FTS_BASE_Debug::exit( int nStatus )  // allows for break point
{
    fprintf( FTS_ERR_TO_FILE,
             "------- To see stack trace, place break point at:  FTS_BASE_Debug::exit"
             "\n"
             "------- in file:  "__FILE__
             "\n"
             "------- at line:  %d"
             "\n"
             , __LINE__
             );

    std::exit( nStatus );
}

// string printing
// may have to specialise for std::string as well
void FTS_BASE_Debug::print( FILE* poFid, const char* var )
{
    fprintf( poFid, "%s", var );
}

// string printing
// may have to specialise for std::string as well
void FTS_BASE_Debug::print( FILE* poFid, char* var )
{
    fprintf( poFid, "%s", var );
}

// character printing
void FTS_BASE_Debug::print( FILE* poFid, char var )
{
    fprintf( poFid, "%c", var );
}

// string printing
// may have to specialise for std::string as well
void FTS_BASE_Debug::print( FILE* poFid, const std::string& var )
{
    fprintf( poFid, "%s", var.c_str() );
}

// speicalised for primitive types
void FTS_BASE_Debug::print( FILE* poFid, int var )
{
    fprintf( poFid, "%d", var );
}
void FTS_BASE_Debug::print( FILE* poFid, unsigned int var )
{
    fprintf( poFid, "%u", var );
}
void FTS_BASE_Debug::print( FILE* poFid, long unsigned int var )
{
    fprintf( poFid, "%lu", var );
}
void FTS_BASE_Debug::print( FILE* poFid, long int var )
{
    fprintf( poFid, "%ld", var );
}
void FTS_BASE_Debug::print( FILE* poFid,          long long  var )
{
    fprintf( poFid, "%lld", var );
}
void FTS_BASE_Debug::print( FILE* poFid, unsigned long long  var )
{
    fprintf( poFid, "%llu", var );
}
void FTS_BASE_Debug::print( FILE* poFid, float var )
{
    fprintf( poFid, "%f", var );
}
void FTS_BASE_Debug::print( FILE* poFid, double var )
{
    fprintf( poFid, "%f", var );
}
void FTS_BASE_Debug::print( FILE* poFid, const void* var )
{
    fprintf( poFid, "%p", var );
}
void FTS_BASE_Debug::print( FILE* poFid, void* var )
{
    fprintf( poFid, "%p", var );
}
void FTS_BASE_Debug::print( FILE* poFid, bool var )
{
    if ( var )
    {
        fprintf( poFid, "true" );
    }
    else
    {
        fprintf( poFid, "false" );
    }
}

void FTS_BASE_Debug::print( FILE* poFid, const CvSize& oSize )
{
    fprintf( poFid, "width = %d, height = %d", oSize.width, oSize.height );
}

void FTS_BASE_Debug::print( FILE* poFid, const CvRect& oRect )
{
	fprintf( poFid, "x = %d, y = %d, width = %d, height = %d", oRect.x, oRect.y, oRect.width, oRect.height );
}

void FTS_BASE_Debug::print( FILE* poFid, const CvPoint2D32f& oPoint )
{
    fprintf( poFid, "x = %f, y = %f", oPoint.x, oPoint.y );
}

void FTS_BASE_Debug::print( FILE* poFid, const CvPoint& oPoint )
{
    fprintf( poFid, "x = %d, y = %d", oPoint.x, oPoint.y );
}

std::string FTS_BASE_Debug::SimplifyFunctionName( const std::string& sIn )
{
    //
    // All printing are done in functions, so __PRETTY_FUNCTION__ should
    // always contain a '('
    // Lets find it an remove all the stuff trailing it.
    // -----------------------------------------------------------------------------

    // Special case for operator()
    std::size_t nParenthesisPos = sIn.find( "operator()" );

    if( nParenthesisPos != std::string::npos )
    {
        nParenthesisPos += 10;  // length of 'operator()' is 10
        nParenthesisPos = sIn.find( "(", nParenthesisPos );
    }
    else
    {
        nParenthesisPos = sIn.find( "(" );
    }

    // Just in case we're wrong
    if( nParenthesisPos == std::string::npos )
    {
        return sIn;
    }
	
    //
    // Construct a string that contains everything up to and including the function name
    // -----------------------------------------------------------------------------
    std::string sString = sIn.substr( 0, nParenthesisPos );

    std::string sString2;

    //
    // Remove all template parameters, i.e. '<....>', but retain the angle brackets
    // ----------------------------------------------------------------------------
    unsigned int i  = 0;
    unsigned int iE = sString.size();
    unsigned int iSta = i;

    while( i < iE )
    {
        if( sString.at(i) == '<' )
        {
            sString2.append(  sString.substr( iSta, i-iSta+1 )  );

            iSta = i;
            while( sString.at(i) != '>' )
            {
                ++i;
            }

            iSta = i;
        }
        ++i;
    }

    sString2.append(  sString.substr( iSta, iE )  );

    //
    // Remove the return type, which could also be pretty darn long.
    // ----------------------------------------------------------------------------
    std::size_t nSpacePos = sString2.find_last_of( " " );

    sString = sString2.substr( nSpacePos+1, sString2.size() - (nSpacePos+1) );

    return sString;
}






