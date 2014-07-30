#include "fts_anpr_object.h"
#include "opencv2/imgproc/imgproc.hpp"
#ifdef _WIN32
#include <direct.h>
#endif

#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <cmath>
#include <ctime>
#include <cstdio>

#define DISPLAY_RAW_BIN
#define DISPLAY_HIST


OcrResult::OcrResult()
{
	this->m_iPostProcessMinConfidence = 40;
	this->m_iPostProcessConfidenceSkipLevel = 75;
}

OcrResult::OcrResult(int minConf, int confSkipLevel)
{
	this->init(minConf, confSkipLevel);
}

OcrResult::~OcrResult()
{
}

void OcrResult::init(int minConf, int confSkipLevel)
{
	this->m_iPostProcessMinConfidence = minConf;
	this->m_iPostProcessConfidenceSkipLevel = confSkipLevel;
}

void OcrResult::clear()
{
	for (size_t i = 0; i < letters.size(); i++)
	{
		letters[i].clear();
	}
	letters.resize(0);
}


void OcrResult::addLetter(char letter, int charposition, float score)
{
	if (score < m_iPostProcessMinConfidence)
		return;

	insertLetter(letter, charposition, score);

	if (score < m_iPostProcessConfidenceSkipLevel)
	{
		float scoreDiff = ( m_iPostProcessConfidenceSkipLevel >= score ) ?
			m_iPostProcessConfidenceSkipLevel - score :
		score - m_iPostProcessConfidenceSkipLevel;
		float adjustedScore = scoreDiff + m_iPostProcessMinConfidence;
		insertLetter(SKIP_CHAR, charposition, adjustedScore );
	}

	//if (letter == '0')
	//{
	//  insertLetter('O', charposition, score - 0.5);
	//}
}

void OcrResult::insertLetter(char letter, int charposition, float score)
{
	score = score - m_iPostProcessMinConfidence;

	int existingIndex = -1;
	if ((int)letters.size() < charposition + 1)
	{
		for (int i = letters.size(); i < charposition + 1; i++)
		{
			vector<Letter> tmp;
			letters.push_back(tmp);
		}
	}

	for (size_t i = 0; i < letters[charposition].size(); i++)
	{
		if (letters[charposition][i].letter == letter &&
			letters[charposition][i].charposition == charposition)
		{
			existingIndex = i;
			break;
		}
	}

	if (existingIndex == -1)
	{
		Letter newLetter(letter, charposition, score);
		letters[charposition].push_back(newLetter);
	}
	else
	{
		letters[charposition][existingIndex].occurences = letters[charposition][existingIndex].occurences + 1;
		letters[charposition][existingIndex].totalscore = letters[charposition][existingIndex].totalscore + score;
	}
}

FTS_LOG_ITEMS::FTS_LOG_ITEMS()
{
	logLevel = ANPR_LOG_ERROR;
}

FTS_LOG_ITEMS::~FTS_LOG_ITEMS()
{
}

void FTS_LOG_ITEMS::printf(int level)
{
	for(size_t i = 0; i < vDebugLines.size(); i++)
	{
		if(vDebugLines[i].Level <= level)
			cout << vDebugLines[i].Content << endl;
	}
}

FTS_ANPR_OBJECT::FTS_ANPR_OBJECT()
	: oPlate()
	, oSrcRotated()
	, oLinesImg()
	, oMedOtsuThreshBinImg()
	, oLines()
	, oFirstBlobImg()
	, oFirstBlobOutliersRemovedBlobMergedAdjustHeightImg()
	, otsuHistByMean()
	, oTestEmptyBlob()
	, oCleanCharBin()
	, oFindMiddleCut()
	, oOverviewDebugImg()
	, oOverviewDebugImg_Resized()
	, middleLine()

	, nMedianBlobWidth( 0 )
	, nMedianBlobHeight( 0 )
	, rMediaBlobOtsu( 0.0 )
	, nMinX( 0 )
	, nMaxX( 0 )
	, nAdjustedMinX( INT_MAX )
	, nAdjustedMaxX( INT_MIN )
	, lPlateDetectTime ( 0 )
	, lPlateSegmentTime( 0 )
	, lPlateOcrTime    ( 0 )
	, rCropTime        ( 0.0 )
	, rRotateTime	   ( 0.0 )
	, rFindMiddleCutTime   ( 0.0 )
	, rDoSegmentTime       ( 0.0 )

	, plateRect()
	, oTopLineMiddleGapBB()
	, allHistograms()
	, oBestCharBoxes()
	, oCharPosLine()
	, oBestBinImages()
	, rawBins()
	, charRegions2D()
	, oReservedNoisyBlobs()
	, ocrResults()
	, ppResults()
{	
}

FTS_ANPR_OBJECT::~FTS_ANPR_OBJECT()
{
}

void mkdir(const std::string& strDir) 
{
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

std::string getDateTimeString(time_t timer, const std::string& strFormat) {
	tm* timeinfo = localtime(&timer);
	char buffer[255];
	strftime(buffer, 255, strFormat.c_str(), timeinfo);

	return buffer;
}

static std::string createDirectory(std::string strParent, std::string strSub, std::string strDeviceID, std::string strDate, std::string strType) {
	std::string sFolder = strParent;
	if (!isDirExist(sFolder)) 
	{
		mkdir(sFolder);
	}

	if (sFolder[sFolder.size() - 1] != '/')
		sFolder.append("/");

	if (strSub.compare("") != 0) {
		sFolder.append(strSub);
		sFolder.append("/");

		if (!isDirExist(sFolder)) {
			mkdir(sFolder);
		}
	}

	if (strDeviceID.compare("") != 0) {
		sFolder.append(strDeviceID);
		sFolder.append("/");

		if (!isDirExist(sFolder)) {
			mkdir(sFolder);
		}
	}

	if (strDate.compare("") != 0) {
		sFolder.append(strDate);
		sFolder.append("/");

		if (!isDirExist(sFolder)) {
			mkdir(sFolder);
		}
	}

	if (strType.compare("") != 0) {
		sFolder.append(strType);
		sFolder.append("/");

		if (!isDirExist(sFolder)) {
			mkdir(sFolder);
		}
	}

	return sFolder;
}

static void drawIntoArea(Mat &src, Mat &dst, const Rect& destinationRect)
{
	if(!src.data)
		return;
    Mat scaledSrc;
    // Destination image for the converted src image.
    Mat convertedSrc(src.rows,src.cols,CV_8UC3, Scalar(0,0,255));

    // Convert the src image into the correct destination image type
    // Could also use MixChannels here.
    // Expand to support range of image source types.
    if (src.type() != dst.type())
    {
        cvtColor(src, convertedSrc, CV_GRAY2RGB);
    }else{
        src.copyTo(convertedSrc);
    }

    // Resize the converted source image to the desired target width.
    resize(convertedSrc, scaledSrc,Size(destinationRect.width,destinationRect.height),1,1,INTER_AREA);

    // create a region of interest in the destination image to copy the newly sized and converted source image into.
    Mat ROI = dst(Rect(destinationRect.x, destinationRect.y, scaledSrc.cols, scaledSrc.rows));
    scaledSrc.copyTo(ROI);
}

const vector<FTS_ANPR_PPResult>& FTS_ANPR_OBJECT::getResults()
{
	return this->ppResults;
}

void FTS_ANPR_OBJECT::createOverviewDebugImg()
{
	if(!oSrcRotated.data) return;

	int cols = 0;

	int TILE_WIDTH  = oSrcRotated.cols+2;
	int TILE_HEIGHT = oSrcRotated.rows+2;

	int horizontal_resolution = TILE_WIDTH * 6;
	int tiles_per_col = 6;
	int vertical_resolution = TILE_HEIGHT * tiles_per_col;
	cout << horizontal_resolution <<   " : " << vertical_resolution << endl;
	oOverviewDebugImg = Mat::zeros(Size(horizontal_resolution, vertical_resolution), CV_8UC3);
	bitwise_not(oOverviewDebugImg, oOverviewDebugImg);

	//oSrc => COLUMN 1
	Rect destinationRect(1, 1, TILE_WIDTH-2, TILE_HEIGHT-2);
	drawIntoArea(oPlate, oOverviewDebugImg, destinationRect);

	//oSrcRotated
	destinationRect.y = TILE_HEIGHT+1;
	drawIntoArea(oSrcRotated, oOverviewDebugImg, destinationRect);

	//oLines
	destinationRect.y = TILE_HEIGHT*2+1;
	drawIntoArea(oLines, oOverviewDebugImg, destinationRect);

	//oMedOtsuThreshBinImg
	destinationRect.y = TILE_HEIGHT*3+1;
	drawIntoArea(oMedOtsuThreshBinImg, oOverviewDebugImg, destinationRect);

	//oFirstBlobOutliersRemovedBlobMergedAdjustHeightImg
	destinationRect.y = TILE_HEIGHT*4+1;
	drawIntoArea(oFirstBlobOutliersRemovedBlobMergedAdjustHeightImg, oOverviewDebugImg, destinationRect);

	//oFindMiddleCut
	destinationRect.y = TILE_HEIGHT*5+1;
	drawIntoArea(oFindMiddleCut, oOverviewDebugImg, destinationRect);

	cols++;

// COLUMN 2
#ifdef DISPLAY_RAW_BIN
	for( size_t i = 0; i < rawBins.size(); i++)
	{
		int col = i / tiles_per_col;
		int row = i % tiles_per_col;
		destinationRect.x = TILE_WIDTH*(cols+col)+1;
		destinationRect.y = TILE_HEIGHT*row+1;
		drawIntoArea(rawBins[i], oOverviewDebugImg, destinationRect);
	}
	cols += (rawBins.size() + tiles_per_col - 1)/ (tiles_per_col);
#endif

#ifdef DISPLAY_HIST
	for( size_t i = 0; i < allHistograms.size(); i++)
	{
		int col = i / tiles_per_col;
		int row = i % tiles_per_col;
		destinationRect.x = TILE_WIDTH*(cols+col)+1;
		destinationRect.y = TILE_HEIGHT*row+1;
		drawIntoArea(allHistograms[i], oOverviewDebugImg, destinationRect);
	}
	cols += (allHistograms.size() + tiles_per_col - 1)/ (tiles_per_col);
#endif

//	//Candidates BLOBS => COLUMN 3
//	destinationRect.x = TILE_WIDTH*cols+1;
//	//oFirstBlobImg
//	destinationRect.y = 1;
//	drawIntoArea(oFirstBlobImg, oOverviewDebugImg, destinationRect);
//
//	//oFirstBlobOutliersRemovedBlobMergedAdjustHeightImg
//	destinationRect.y = TILE_HEIGHT+1;
//	drawIntoArea(oFirstBlobOutliersRemovedBlobMergedAdjustHeightImg, oOverviewDebugImg, destinationRect);
//
////	//otsuHistByMean
////	destinationRect.y = TILE_HEIGHT*2+1;
////	drawIntoArea(otsuHistByMean, oOverviewDebugImg, destinationRect);
//
//	//oFindMiddleCut
//	destinationRect.y = TILE_HEIGHT*2+1;
//	drawIntoArea(oFindMiddleCut, oOverviewDebugImg, destinationRect);
//
//
//	//oTestEmptyBlob
//	destinationRect.y = TILE_HEIGHT*3+1;
//	drawIntoArea(oTestEmptyBlob, oOverviewDebugImg, destinationRect);
	
	//oCleanCharBin, padded so no 
//	Rect destinationRect(TILE_WIDTH*(1+cols), TILE_HEIGHT*4, TILE_WIDTH, TILE_HEIGHT);
//	drawIntoArea(oCleanCharBin, oOverviewDebugImg, destinationRect);

//	cols++;

	//BEST BIN IMAGES + BEST CHAR BOXES => COLUMN 4
//	destinationRect.x = TILE_WIDTH*(cols+1);

	// DV: 23/06/2014 - Use exact bounding boxes
//	vector<Mat> oBinsColor(oBestBinImages.size());
//	for( size_t i = 0; i < oBestBinImages.size(); i++ )
//	{
//		cvtColor( oBestBinImages[i], oBinsColor[i], CV_GRAY2BGR );
//		for( size_t j = 0; j < oBestCharBoxes.size(); j++ )
//		{
//			rectangle( oBinsColor[i], oBestCharBoxes[j], Scalar(0,255,0) );
//		}
//		destinationRect.y = TILE_HEIGHT*i;
//		drawIntoArea(oBinsColor[i], oOverviewDebugImg, destinationRect);
//	}
	destinationRect.x = TILE_WIDTH*(cols);
	vector<Mat> oBinsColor(charRegions2D.size());
	for( size_t i = 0; i < charRegions2D.size(); i++ )
	{
		cvtColor( oBestBinImages[i], oBinsColor[i], CV_GRAY2BGR );
		for( size_t j = 0; j < charRegions2D[i].size(); j++ )
		{
			rectangle( oBinsColor[i], charRegions2D[i][j], Scalar(0,255,0) );
		}
		destinationRect.y = TILE_HEIGHT*i;

		// DV: 21/07/2014 - draw top line middle gap bb
		if( oTopLineMiddleGapBB.width > 0 )
		{
//			rectangle( oBinsColor[i], oTopLineMiddleGapBB, Scalar(145,61,136), CV_FILLED );
			Point2f oCenter( oTopLineMiddleGapBB.x + (float)oTopLineMiddleGapBB.width/2, oTopLineMiddleGapBB.y + (float)oTopLineMiddleGapBB.height/2 );
			circle( oBinsColor[i], oCenter, 10, Scalar(145,61,136), CV_FILLED );
		}

		drawIntoArea(oBinsColor[i], oOverviewDebugImg, destinationRect);
	}

	//put debug text
	destinationRect.x = TILE_WIDTH*cols;
	destinationRect.y = TILE_HEIGHT*5+1;
	int fontFace = FONT_HERSHEY_SIMPLEX;
	double fontScale = 1;
	int thickness = 2;
	int bestPlateIndex = -1;
	for (unsigned int pp = 0; pp < ppResults.size(); pp++)
	{
		/*if( (int)pp >= topN )
		{

			break;
		}*/
		if (bestPlateIndex == -1 && ppResults[pp].matchesTemplate)
		{
			bestPlateIndex = pp;
		}			
	}

	stringstream ss1;
	stringstream ss2;
	if(ppResults.size() > 0 && bestPlateIndex >= 0)
	{
		ss1 << "OCR = "  << ppResults[bestPlateIndex].letters;
		ss2 << "Confidence = " << ppResults[bestPlateIndex].totalscore;
	}
	else if(ppResults.size() > 0)
	{		
		ss1 << "OCR = "  << ppResults[0].letters;
		ss2 << "Confidence = " << ppResults[0].totalscore;
	}
	putText( oOverviewDebugImg,
			 ss1.str(),
			 Point(destinationRect.x, destinationRect.y),
			 fontFace,
			 fontScale,
			 Scalar(0,0,255),
			 thickness,
			 8 );

	putText( oOverviewDebugImg,
			 ss2.str(),
			 Point(destinationRect.x, destinationRect.y + 40),
			 fontFace,
			 fontScale,
			 Scalar(0,0,255),
			 thickness,
			 8 );

	// DV 28/06/2014 - resize image
	resize( oOverviewDebugImg, oOverviewDebugImg_Resized, Size( 960, 700 ) );
}

//static void write(FileStorage& fs, const std::string& s, const vector<FTS_ANPR_PPResult>& ppResults)
//{
//	fs << "{";
//	for(int i = 0; i < ppResults.size(); i++)
//	{
//		fs << "{" << "Letters" << ppResults[i].letters << "Score" << ppResults[i].totalscore << "Match" << ppResults[i].matchesTemplate << "}";
//	}
//	fs << "}";
//}

void FTS_ANPR_OBJECT::write(std::string strOutputFolder, std::string strDeviceID, time_t timer, std::string frameID)
{
	char buf[255];
	cv::FileStorage fs;
	//time_t timer;
	//time(&timer);
	//get & create overview debug image folder
	std::string sCurDate = getDateTimeString(timer, "%Y%m%d");
	std::string sImgFolder = createDirectory(strOutputFolder, "RAW", strDeviceID, sCurDate, "IMG");
	//sprintf(buf, "%d", frameIndex);
	std::string sImgFilePath = getDateTimeString(timer, sImgFolder + "DBG" + "_%Y%m%d_%H%M%S_" + frameID + ".jpg");
	if(oOverviewDebugImg.empty())
		this->createOverviewDebugImg();
	cv::imwrite(sImgFilePath, this->oOverviewDebugImg);

	//get & create xml folder
	//std::string sXmlFolder = createDirectory(strOutputFolder, "RAW", strDeviceID, sCurDate, "");
	std::string sXmlFilePath = getDateTimeString(timer, sImgFolder + "DBG" + "_%Y%m%d_%H%M%S_" + frameID + ".xml");
	fs.open(sXmlFilePath, cv::FileStorage::WRITE);

	stringstream bbRect;
	bbRect << this->plateRect.x 
			<< "," << this->plateRect.y
			<< "," << this->plateRect.width
			<< "," << this->plateRect.height;
	stringstream ppRes;
	//ppRes << "";
	for(size_t i = 0; i < ppResults.size(); i++)
	{
		//if(i>0) ppRes << ", ";
		//ppRes << "Letters=" << ppResults[i].letters << "Score=" << ppResults[i].totalscore << "Match=" << (ppResults[i].matchesTemplate?1:0);
		sprintf(buf, "P=%s,S=%f,M=%d;", ppResults[i].letters.c_str(), ppResults[i].totalscore, (ppResults[i].matchesTemplate?1:0));
		ppRes << buf;
	}
	//ppRes << "";
	//printf(ppRes.str().c_str());

	fs  << "PlateRect"			<< bbRect.str()
		<< "MedianBlobWidth"	<< nMedianBlobWidth
		<< "MedianBlobHeight"	<< nMedianBlobHeight
		<< "MediaBlobOtsu"		<< rMediaBlobOtsu
		<< "AdjustedMinX"		<< nAdjustedMinX 
		<< "AdjustedMaxX"		<< nAdjustedMaxX
		<< "AnprResult"			<< ppRes.str()
		<< "PlateDetectTime"	<< (int)lPlateDetectTime
		<< "PlateSegmentTime"	<< (int)lPlateSegmentTime
		<< "PlateOcrTime"		<< (int)lPlateOcrTime;
	fs  << "DebugLog" << "[";
		for(size_t i = 0; i < oDebugLogs.vDebugLines.size(); i++)
		{
			fs << oDebugLogs.vDebugLines[i].Content;
		}
	fs << "]";
	fs.release();
}

//++24.06.2014 TrungNT1 add method to export 
void FTS_ANPR_OBJECT::extendImage(const cv::Mat &src, Rect &rect, Mat &dist)
{
	int w = rect.width;
	int h = rect.height;

	if (h > 2 * w)
	{
		dist = Mat::zeros(cv::Size(2 * w, h), CV_8UC1);
		rect.x = rect.x + w / 2 - h / 4;
		rect.width = h / 2;
	}

	Rect rectSrc(0, 0, src.cols, src.rows);
	if (rectSrc.contains(cv::Point(rect.x, rect.y)) &&
		rectSrc.contains(cv::Point(rect.x + rect.width - 1, rect.y + rect.height - 1)))
	{
		Mat it = src(rect);
		it.copyTo(dist);
	}
}

void FTS_ANPR_OBJECT::extendImage(const cv::Mat &src, cv::Mat& dist)
{
	const int w = src.rows;
	const int h = src.cols;
	if (h > 2 * w)
	{
		dist = Mat::zeros(cv::Size(2 * w, h), CV_8UC1);
		int x = w - src.rows / 2;
		Mat it = dist(cv::Rect(x, 0, w, h));
		src.copyTo(it);
	}
	else
	{
		src.copyTo(dist);
	}
}

static int s_sIdx = 0;
void FTS_ANPR_OBJECT::outputCharSegmentResult(std::string strOutputFolder)
{
	for( size_t i = 0; i < this->oBestCharBoxes.size(); i++ )
	{
		Mat buffer;
		ostringstream convert;
		string ind_str;
		convert << s_sIdx;
		ind_str = convert.str();

		extendImage(this->oSrcRotated, this->oBestCharBoxes[i], buffer);

		//buffer = oSrcRotated(oBestCharBoxes[i]);
		if (this->ocrResults.letters.size() > 0 && buffer.data 
			&& i < this->ocrResults.letters.size() && this->ocrResults.letters[i].size() > 0) 
		{
			string sClassName;
			stringstream ss;
			ss << this->ocrResults.letters[i][0].letter;
			ss >> sClassName;
#ifdef _DEBUG
			std::cout << "Export Char Image Index: " << i << " Character: " << sClassName << std::endl;
			//cv::imshow("Character", buffer);
#endif

			string sOutDir = createDirectory(strOutputFolder, "", "", "", sClassName);
			string sOutFile = sOutDir + "/" + sClassName + "_" + ind_str + ".jpg";
			s_sIdx++;
			imwrite(sOutFile, buffer);
		}
	}	
}
//--24.06.2014

void FTS_ANPR_OBJECT::setLogLevel(int level)
{
	if(level < ANPR_LOG_NONE) level = ANPR_LOG_NONE;
	if(level > ANPR_LOG_INFO) level = ANPR_LOG_INFO;
	this->oDebugLogs.setLogLevel(level);
}

void FTS_ANPR_OBJECT::printf(int level)
{
	/*for(int i = 0; i < oDevDebugLines.size(); i++)
	{
		if(vDebugLines[i].Level <= level)
			cout << vDebugLines[i].Content << endl;
	}*/
	oDebugLogs.printf(level);
}
