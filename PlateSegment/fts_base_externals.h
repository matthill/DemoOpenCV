/*
 * fts_base_externals.h
 *
 * An array of bytes and geometric properties required to describe an (image) buffer.
 *
 */

//using namespace std;

// ANSI/GNU STL
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string>
#include <sstream>

#ifdef WIN32
//#include "stdafx.h"
#include "dirent.h"
#endif

#ifndef WIN32
// OS/Linux
#include <dirent.h> // file system operators, delete, create, list directory etc.
#include <setjmp.h>
#include "errno.h"
#include <unistd.h>
#include <iostream>
#include <sstream>
#include <fcntl.h>

// POSIX
#include <pthread.h> // Includes here, before #ifdef
#endif

// STL
#include <list>
#include <queue>
#include <vector>
#include <map>
#include <set>
#include <list>
#include <stack>
#include <numeric>

// C?
#include "limits.h"

// ANSI/GNU
#include <sys/types.h>
#include <sys/stat.h>

#ifndef WIN32
// OS/IO
#include <poll.h>
#endif

// STD
#include <math.h>

// SN
//#include "sn_base_debug.h"

// OPENCV
#ifndef WIN32
#include "cxcore.h"
#include "cv.h"
#include "highgui.h"
#else
#include "opencv/cxcore.h"
#include "opencv/cv.h"
#include "opencv/highgui.h"
#endif

// OPENCV 2
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

#ifndef WIN32
// OS kernel modules
#include <libgpsmm.h>
#endif

#ifndef WIN32
// JPEG
#include <jpeglib.h>
#include <jerror.h>
#endif

#ifndef WIN32
// Regex
extern "C"
{
    #include <regex.h>	
#include <cmath>
}
#endif

// boost
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>

//#include <baseapi.h>

using namespace std;
using namespace cv;


#ifndef _FTS_BASE_EXTERNALS_H_  // Encase everything in #ifdef to prevent multiple conflicting declarations
#define _FTS_BASE_EXTERNALS_H_


#endif // _FTS_BASE_EXTERNALS_H_
