#include "util.h"
#include <stdarg.h>





bool hasEnding(const std::string& fullString, const std::string& ending) {
	if (fullString.length() >= ending.length()) {
		return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
	} else {
		return false;
	}
}

void mkdir(const std::string& strDir) {
#ifdef _WIN32
	_mkdir(strDir.c_str());
#else 
	mkdir (strDir.c_str (), 0777); // notice that 777 is different than 0777	
#endif
}

bool isDirExist(const std::string& strDir) {
	struct stat info;

	//check item exits
	if ((stat(strDir.c_str(), &info) == 0) && (info.st_mode & S_IFDIR))
		return true;

	return false;
}

bool isFileExist(const std::string& strFile) {
	struct stat info;

	//check item exits
	if ((stat(strFile.c_str(), &info) == 0) && (info.st_mode & S_IFMT))
		return true;

	return false;
}

std::string createDirectory(std::string strParent, std::string strSub, std::string strDeviceID, std::string strDate, std::string strType) {
	std::string sFolder = strParent;
	if (sFolder[sFolder.size() - 1] != '\\')
		sFolder.append("\\");

	if (strSub.compare("") != 0) {
		sFolder.append(strSub);
		sFolder.append("\\");

		if (!isDirExist(sFolder)) {
			mkdir(sFolder);
		}
	}

	if (strDeviceID.compare("") != 0) {
		sFolder.append(strDeviceID);
		sFolder.append("\\");

		if (!isDirExist(sFolder)) {
			mkdir(sFolder);
		}
	}

	if (strDate.compare("") != 0) {
		sFolder.append(strDate);
		sFolder.append("\\");

		if (!isDirExist(sFolder)) {
			mkdir(sFolder);
		}
	}

	if (strType.compare("") != 0) {
		sFolder.append(strType);
		sFolder.append("\\");

		if (!isDirExist(sFolder)) {
			mkdir(sFolder);
		}
	}

	return sFolder;
}

void writeParamFile(cv::Ptr<cv::Algorithm> algorithm, const std::string &strParamFile) {
	cv::FileStorage fs(strParamFile, cv::FileStorage::WRITE);
	//cv::WriteStructContext ws(fs, strParam, CV_NODE_MAP);
	algorithm->write(fs);

}

void readParamFile(cv::Ptr<cv::Algorithm> algorithm, const std::string& strParamFile) {
	cv::FileStorage fs(strParamFile, cv::FileStorage::READ);

	if (fs.isOpened()) {
		algorithm->read(fs.root());
		fs.release();
	}
}


std::string getDateTimeString(time_t timer, const std::string& strFormat) {
	tm* timeinfo = localtime(&timer);
	char buffer[255];
	strftime(buffer, 255, strFormat.c_str(), timeinfo);

	return buffer;
}

bool stringCompare(const std::string &left, const std::string& right) {
	for (std::string::const_iterator lit = left.begin(), rit = right.begin(); lit != left.end() && rit != right.end(); ++lit, ++rit)
	if (tolower(*lit) < tolower(*rit))
		return true;
	else if (tolower(*lit) > tolower(*rit))
		return false;
	if (left.size() < right.size())
		return true;
	return false;
}

std::string formatString(const char* fmt, ...)
{
	char* buffer;
	va_list arglist;
    va_start(arglist, fmt);
	int len = _vscprintf( fmt, arglist ) + 1;	// _vscprintf doesn't count terminating '\0'
    buffer = (char*)malloc( len * sizeof(char) );
    int ret = vsprintf(buffer, fmt, arglist);
    va_end(arglist);	
    std::string retStr(buffer);
	free(buffer);
	return retStr;
}