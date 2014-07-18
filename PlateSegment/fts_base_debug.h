/*
 * Utilities for debugging
 *
 */
#ifndef _FTS_BASE_DEBUG_H_H
#define _FTS_BASE_DEBUG_H_H

#include "fts_base_externals.h"

//==================================================================================================
//==================================================================================================
//==================================================================================================
// Prototypes
//==================================================================================================
//==================================================================================================
//==================================================================================================

// function name prefix style
// __FUNCTION__ or __PRETTY_FUNCTION__ or NULL
#define FTS_P_FUNCTION_NAME_STYLE_DEFAULT   FTS_FUNCTION_NAME_SIMPLE
#define FTS_P_FUNCTION_NAME_STYLE           FTS_P_FUNCTION_NAME_STYLE_DEFAULT

#define FTS_ERR_FUNCTION_NAME_STYLE_DEFAULT   FTS_FUNCTION_NAME_SIMPLE
#define FTS_ERR_FUNCTION_NAME_STYLE           FTS_ERR_FUNCTION_NAME_STYLE_DEFAULT

    //! Use can redefine to any file pointer
    //! Redefine to NULL to not print anything
#define FTS_P_TO_FILE_DEFAULT  stderr
#define FTS_P_TO_FILE          FTS_P_TO_FILE_DEFAULT

#define FTS_ERR_TO_FILE_DEFAULT  stderr
#define FTS_ERR_TO_FILE          FTS_ERR_TO_FILE_DEFAULT

void FTS_PAUSE_FromNthCall( unsigned int nth );
bool FTS_PAUSE_PauseOnThisCall( unsigned int& nCurrAtNthPause );

// These all evaluate to a const char*
#define FTS_FUNCTION_NAME_VERBOSE  __PRETTY_FUNCTION__
#define FTS_FUNCTION_NAME_MINIMAL  __FUNCTION__
#ifndef WIN32
#define FTS_FUNCTION_NAME_SIMPLE   FTS_BASE_Debug::SimplifyFunctionName( __PRETTY_FUNCTION__ ).c_str()
#else
#define FTS_FUNCTION_NAME_SIMPLE   FTS_BASE_Debug::SimplifyFunctionName( __FUNCSIG__ ).c_str()
#endif


// We need this to stringify __LINE__ properly, and the reason to
// stringify __LINE__ is that we want a single fprintf (it's atomic
// in glibc) and use defined format specifiers and arguments passed
// in as 'args...', which means we can't break 'args...' into
// format specifier and argument components.
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)  // this extra wrapper causes x to be evaluated rather than a straight string sub


#define DASHES "------------------------------------------\n"  \
               "------------------------------------------\n"

// In order to prevent interleaved output from multiple thread, we
// need a single call to fprintf, since it's atmic in glibc.
#define FTS_PAUSE                                                    \
    {                                                               \
        unsigned int nCurrAtNthPause;                               \
        if (  FTS_PAUSE_PauseOnThisCall( nCurrAtNthPause )  )        \
        {                                                           \
            fprintf( FTS_ERR_TO_FILE,                                \
                     DASHES                                         \
                     "------- PAUSED:  no. %d"                      \
                     "\n"                                           \
                     "------- in:  %s"                              \
                     "\n"                                           \
                     "------- at line:  " TOSTRING(__LINE__)        \
                     "\n"                                           \
                     , nCurrAtNthPause                              \
                     , FTS_ERR_FUNCTION_NAME_STYLE                   \
                     );                                             \
            fflush(stdin);                                          \
            getc(stdin);                                            \
        }                                                           \
    }

// A terse version of pause
#define FTS_PAUSE_TERSE                                              \
    {                                                               \
        unsigned int nCurrAtNthPause;                               \
        if (  FTS_PAUSE_PauseOnThisCall( nCurrAtNthPause )  )        \
        {                                                           \
            fprintf( FTS_ERR_TO_FILE,                                \
                     "------- PAUSED:  no. %d"                      \
                     , nCurrAtNthPause                              \
                     );                                             \
            fflush(stdin);                                          \
            getc(stdin);                                            \
        }                                                           \
    }

//++ trungnt1 add to support snprintf in WINDOWS / WIN32
#ifdef _MSC_VER

#define snprintf c99_snprintf

inline int c99_vsnprintf(char* str, size_t size, const char* format, va_list ap)
{
    int count = -1;

    if (size != 0)
        count = _vsnprintf_s(str, size, _TRUNCATE, format, ap);
    if (count == -1)
        count = _vscprintf(format, ap);

    return count;
}

inline int c99_snprintf(char* str, size_t size, const char* format, ...)
{
    int count;
    va_list ap;

    va_start(ap, format);
    count = c99_vsnprintf(str, size, format, ap);
    va_end(ap);

    return count;
}

#endif // _MSC_VER
//-- trungnt1

// In order to prevent interleaved output from multiple thread, we
// need a single call to fprintf, since it's atmic in glibc.
#define FTS_EXIT( format_specifiers, /*args*/... )                 \
    {                                                               \
        fprintf( FTS_ERR_TO_FILE,                                    \
                 DASHES                                             \
                 "------- EXIT: " format_specifiers                 \
                 "\n"                                               \
                 "------- in:  %s"                                  \
                 "\n"                                               \
                 "------- at line:  " TOSTRING(__LINE__)            \
                 "\n"                                               \
                 , /*##args*/__VA_ARGS__                                    \
                 , FTS_ERR_FUNCTION_NAME_STYLE                       \
                 );                                                 \
        FTS_BASE_Debug::exit( EXIT_FAILURE );                        \
    }

#define FTS_ASSERT( test, /*args*/... )                                      \
    {                                                                   \
        if ( !(test) )  /* run the test */                              \
        {                                                               \
            /* Since this is a macro, we need to limit
               the amount of allocations on the stack.
               And since it really should not get in here
               we dynamically allocate the strin
            */ \
            const unsigned int nSize = 1000;                            \
            char* pcStr = new char[ nSize ];                            \
            /* safe snprintf to avoid buffer overrun */                 \
            if(  -1 != snprintf( pcStr, nSize, " " /*args*/__VA_ARGS__ )  )            \
            {                                                           \
                fprintf( FTS_ERR_TO_FILE,                                \
                         DASHES                                         \
                         "------- Assert Failed: " #test                \
                         "\n"                                           \
                         "------- in:  %s"                              \
                         "\n"                                           \
                         "------- at line:  " TOSTRING(__LINE__)        \
                         "\n"                                           \
                         "------- reason:  %s"                          \
                         "\n"                                           \
                         , FTS_ERR_FUNCTION_NAME_STYLE                   \
                         , pcStr                                        \
                         );                                             \
            }                                                           \
            else                                                        \
            {                                                           \
                fprintf( FTS_ERR_TO_FILE, "00000000000000000000000000"); \
                fprintf( FTS_ERR_TO_FILE,                                \
                         DASHES                                         \
                         "------- Assert Failed: " #test                \
                         "\n"                                           \
                         "------- in:  %s"                              \
                         "\n"                                           \
                         "------- at line:  " TOSTRING(__LINE__)        \
                         "\n"                                           \
                         , FTS_ERR_FUNCTION_NAME_STYLE                   \
                         );                                             \
                fprintf( FTS_ERR_TO_FILE,                                \
                         "------- reason:  " /*args*/__VA_ARGS__                       \
                         );                                             \
                fprintf( FTS_ERR_TO_FILE, "\n" );                        \
            }                                                           \
            delete [] pcStr;                                            \
            FTS_BASE_Debug::exit( EXIT_FAILURE );                        \
        }                                                               \
    }

#define FTS_ASSERT_ERRNO( test, /*args*/... )                                \
    {                                                                   \
        int __errorEnum = (test); /* evaluate the test */                   \
        if ( (__errorEnum) != 0 )                                         \
        {                                                               \
            /* Since this is a macro, we need to limit
               the amount of allocations on the stack.
               And since it really should not get in here
               we dynamically allocate the string
            */                                                          \
            const unsigned int nSize = 1000;                            \
            char* pcStr = new char[ nSize ];                            \
            /* safe snprintf to avoid buffer overrun */                 \
            if(  -1 != snprintf( pcStr, nSize, " " /*args*/__VA_ARGS__ )  )            \
            {                                                           \
                fprintf( FTS_ERR_TO_FILE,                                \
                         DASHES                                         \
                         "------- Assert Failed: " #test                \
                         "\n"                                           \
                         "------- Error code: %s"                       \
                         "\n"                                           \
                         "------- in:  %s"                              \
                         "\n"                                           \
                         "------- at line:  " TOSTRING(__LINE__)        \
                         "\n"                                           \
                         "------- reason:  %s"                          \
                         "\n"                                           \
                         , strerror( __errorEnum )                        \
                         , FTS_ERR_FUNCTION_NAME_STYLE                   \
                         , pcStr                                        \
                         );                                             \
            }                                                           \
            else                                                        \
            {                                                           \
                fprintf( FTS_ERR_TO_FILE, "00000000000000000000000000"); \
                fprintf( FTS_ERR_TO_FILE,                                \
                         DASHES                                         \
                         "------- Assert Failed: " #test                \
                         "\n"                                           \
                         "------- Error code: %s"                       \
                         "\n"                                           \
                         "------- in:  %s"                              \
                         "\n"                                           \
                         "------- at line:  " TOSTRING(__LINE__)        \
                         "\n"                                           \
                         , FTS_ERR_FUNCTION_NAME_STYLE                   \
                         , strerror( __errorEnum )                        \
                         );                                             \
                fprintf( FTS_ERR_TO_FILE,                                \
                         "------- reason:  " /*args*/__VA_ARGS__                      \
                         );                                             \
                fprintf( FTS_ERR_TO_FILE, "\n" );                        \
            }                                                           \
            delete [] pcStr;                                            \
            FTS_BASE_Debug::exit( EXIT_FAILURE );                        \
        }                                                               \
    }

// eg. FTS_P( temp ) --> produces: temp: (value of temp)
// If you defined a class, then it will call the class's .print() function
#define FTS_P_DEFAULT( var )  FTS_BASE_Debug::print( FTS_P_TO_FILE, FTS_P_FUNCTION_NAME_STYLE, #var, var )

#define FTS_P  FTS_P_DEFAULT

// Error message, similar to FTS_P(), but does not print variable name
//#define FTS_ERR( message )  FTS_BASE_Debug::print( FTS_ERR_TO_FILE, FTS_ERR_FUNCTION_NAME_STYLE, message)

    // eg. FTS_ERR( "error" );
    // eg. FTS_ERR( "error: %d %f", integer, decimal );
    // Scoped so you can do this: while (1) FTS_ERR( "blah" );

//#define FTS_ERR_DEFAULT( fmt, args... )                              \
//{                                                               \
//	timeval t;					\
//	gettimeofday( &t, NULL );	\
//	long int m_nSec  = t.tv_sec;			\
//	time_t sec = m_nSec;			\
//	struct tm oTimeinfo;			\
//	localtime_r( &sec, &oTimeinfo );	\
//	int nYYYY = oTimeinfo.tm_year +1900;	\
//	int nMM   = oTimeinfo.tm_mon +1;	\
//	std::ostringstream oss; \
//	oss << nYYYY << '-' << nMM << '-' << oTimeinfo.tm_mday;	\
//	oss << '_';	\
//	if( oTimeinfo.tm_hour < 10 ) oss << '0';	\
//	oss << oTimeinfo.tm_hour << ':';	\
//	if( oTimeinfo.tm_min < 10 ) oss << '0';	\
//	oss << oTimeinfo.tm_min << ':';	\
//	if( oTimeinfo.tm_sec < 10 ) oss << '0';	\
//	oss << oTimeinfo.tm_sec << '.';	\
//	if( (t.tv_usec/1000) < 10 ){ oss << '0';oss << '0';}	\
//	else if( (t.tv_usec/1000) < 100 ) oss << '0';	\
//	oss << (t.tv_usec/1000);	\
//	std::string sTime = oss.str();	\
//	std::stringstream ss;                                       \
//	ss << "%s %s:  " << fmt << "\n";                                  \
//	fprintf( FTS_ERR_TO_FILE,                                    \
//			 ss.str().c_str()                                   \
//			 , sTime.c_str()										\
//			 , FTS_ERR_FUNCTION_NAME_STYLE                       \
//			 , ## args );                                       \
//}

#if defined(_MSC_VER) || defined(__MINGW32__)
//#include <time.h>
//#include <WinSock2.h>
#include < time.h >
#include <windows.h> //I've ommited this line.
#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
  #define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
  #define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#endif
 
struct timezone 
{
  int  tz_minuteswest; /* minutes W of Greenwich */
  int  tz_dsttime;     /* type of dst correction */
};
int gettimeofday(struct timeval* tp, struct timezone* tzp);
#else
#include <sys/time.h>
#endif

#define FTS_ERR_DEFAULT( fmt, /*args*/... )                              \
{                                                               \
	timeval tv;					\
	gettimeofday( &tv, NULL );	\
	long int m_nSec  = tv.tv_sec;			\
	time_t t = time(0);						\
    struct tm* now = localtime( & t );		\
	int nYYYY = now->tm_year +1900;	\
	int nMM   = now->tm_mon +1;	\
	std::ostringstream oss; \
	oss << nYYYY << '-' << nMM << '-' << now->tm_mday;	\
	oss << '_';	\
	if( now->tm_hour < 10 ) oss << '0';	\
	oss << now->tm_hour << ':';	\
	if( now->tm_min < 10 ) oss << '0';	\
	oss << now->tm_min << ':';	\
	if( now->tm_sec < 10 ) oss << '0';	\
	oss << now->tm_sec << '.';	\
	if( (tv.tv_usec/1000) < 10 ){ oss << '0';oss << '0';}	\
	else if( (tv.tv_usec/1000) < 100 ) oss << '0';	\
	oss << (tv.tv_usec/1000);	\
	std::string sTime = oss.str();	\
	std::stringstream ss;                                       \
	ss << "%s %s:  " << fmt << "\n";                                  \
	fprintf( FTS_ERR_TO_FILE,                                    \
			 ss.str().c_str()                                   \
			 , sTime.c_str()										\
			 , FTS_ERR_FUNCTION_NAME_STYLE                       \
			 , /*## args*/__VA_ARGS__ );                                       \
}

#define FTS_ERR  FTS_ERR_DEFAULT

//#define FTS_ASSERT( truth ) FTS_BASE_Debug::assert( FTS_ERR_TO_FILE, FTS_ERR_FUNCTION_NAME_STYLE, #truth, truth )

//#define FTS_CV_SAFE_CALL( func )                                             \
//    do {                                                                        \
//        cvSetErrMode( CV_ErrModeParent );                                    \
//        cvSetErrStatus( CV_StsOk );                                          \
//        (func);                                                                \
//        if( cvGetErrStatus() < 0 )                                           \
//        {                                                                    \
//            FTS_ERR( "Opencv function call failed ( line %d )", __LINE__ );  \
//            FTS_BASE_Debug::exit( EXIT_FAILURE );                            \
//        }                                                                    \
//        cvSetErrMode( CV_ErrModeLeaf );                                      \
//    } while(0)
#define FTS_CV_SAFE_CALL( func, ... )                                             \
    do {                                                                        \
        cvSetErrMode( CV_ErrModeParent );                                    \
        cvSetErrStatus( CV_StsOk );                                          \
        func(__VA_ARGS__);                                                                \
        if( cvGetErrStatus() < 0 )                                           \
        {                                                                    \
            FTS_ERR( "Opencv function call failed ( line %d )", __LINE__ );  \
            FTS_BASE_Debug::exit( EXIT_FAILURE );                            \
        }                                                                    \
        cvSetErrMode( CV_ErrModeLeaf );                                      \
    } while(0)

#define  FTS_CV_ASSERT  FTS_CV_SAFE_CALL


class FTS_BASE_Debug
{
public:
        template <class T>
    static void print( FILE* poFid, const char* functionName, const char* variableName, const T& var );


public:
    static void exit( int nStatus );  // allows for break point

    static void print( FILE* poFid,   const char* var ); // printing for primitive types
    static void print( FILE* poFid,         char* var );
    static void print( FILE* poFid,         char  var );
    static void print( FILE* poFid, const std::string& var );
    static void print( FILE* poFid, const std::stringstream& ss );
    static void print( FILE* poFid,          int  var );
    static void print( FILE* poFid, unsigned int  var );
    static void print( FILE* poFid, long unsigned int var );
    static void print( FILE* poFid,     long int  var );
    static void print( FILE* poFid,          long long  var );
    static void print( FILE* poFid, unsigned long long  var );
    static void print( FILE* poFid,        float  var );
    static void print( FILE* poFid,       double  var );
    static void print( FILE* poFid,   const void* var );
    static void print( FILE* poFid,         void* var );
    static void print( FILE* poFid,         bool  var );
    static void print( FILE* poFid, const CvSize& oSize );
    static void print( FILE* poFid, const CvRect& oRect );
    static void print( FILE* poFid, const CvPoint2D32f& oPoint );
    static void print( FILE* poFid, const CvPoint& oPoint );

    template <class T> static void print( FILE* poFid, const T& var );

    static std::string SimplifyFunctionName( const std::string& sIn );

private:
        // Serves as a namespace, so instantiation is not allowed
    explicit FTS_BASE_Debug();
    ~FTS_BASE_Debug();
    FTS_BASE_Debug& operator=( const FTS_BASE_Debug& operand );
};


//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
// class FTS_BASE_Debug
//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------

template <class T>
void FTS_BASE_Debug::print( FILE* poFid, const char* functionName, const char* variableName, const T& var )
{
    if ( poFid == NULL ) return; // then we don't wish to print at all even if globalFid
    fprintf( poFid, "%s: %s: ", functionName, variableName );
    print( poFid, var );
    fprintf( poFid, "\n" );
}

//! a catch all printing function
template <class T>
void FTS_BASE_Debug::print( FILE* poFid,  const T& var )
{
    if ( poFid != NULL )
    {
        var.print( poFid );
    }
}


#endif
