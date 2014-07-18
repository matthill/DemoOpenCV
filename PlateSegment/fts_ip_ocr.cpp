#include "fts_ip_ocr.h"
#include "fts_ip_util.h"

FTS_ANPR_TessOcr::FTS_ANPR_TessOcr(string tessDataDir, string ocrLanguage, int minCharToProcess, int ocrMinFontSize, bool bDebugOCR)
{
	//this->config = config;
	this->m_iPostProcessMinCharacters = minCharToProcess;
	this->m_iOcrMinFontSize = ocrMinFontSize;
	this->m_bDebugOcr = bDebugOCR;

	this->tesseract=new TessBaseAPI();

	// Tesseract requires the prefix directory to be set as an env variable
	vector<char> tessdataPrefix(tessDataDir.size() + 1);

	strcpy(tessdataPrefix.data(), tessDataDir.c_str());
	putenv(tessdataPrefix.data());
#ifdef WIN32 
	_putenv(tessdataPrefix.data());
#endif 

	int ret = this->tesseract->Init(tessDataDir.c_str(), ocrLanguage.c_str() );
	bool bRet = this->tesseract->SetVariable("save_blob_choices", "T");
	//this->tesseract->SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNPQRSTUVWXYZ1234567890");
	this->tesseract->SetPageSegMode(PSM_SINGLE_CHAR);

#ifdef POST_PROCESS
	this->postProcessor = NULL;
#endif
}

#ifdef POST_PROCESS
void FTS_ANPR_TessOcr::initPostProcess(string patternsFile, int minConfidence, int confidenceSkipLevel, int maxSubs, bool debugPP)
{	
	this->postProcessor = new FTS_ANPR_PostProcess(patternsFile, minConfidence, confidenceSkipLevel, maxSubs, debugPP);
}
#endif

FTS_ANPR_TessOcr::~FTS_ANPR_TessOcr()
{
	if(this->tesseract)
	{
		this->tesseract->Clear();
		this->tesseract->End();
		//delete tesseract; 
	}
#ifdef POST_PROCESS
	if(this->postProcessor)
	{
		delete postProcessor;
	}
#endif
}

void FTS_ANPR_TessOcr::clean()
{
	this->tesseract->Clear();
#ifdef POST_PROCESS
	if(this->postProcessor)
	{
		this->postProcessor->clear();
	}
#endif
}

void FTS_ANPR_TessOcr::performOCR(	const vector<Mat>& thresholds,
									const vector< vector<Rect> >& charRegions,
									//OcrResult& ocrResults )		//21.06
									FTS_ANPR_OBJECT& oAnprObject )	//27.06
{
	//timespec startTime;
	//getTime(&startTime);

#ifdef POST_PROCESS
	if(this->postProcessor) postProcessor->clear();
#endif

	oAnprObject.ocrResults.clear();
#ifdef POST_PROCESS
	oAnprObject.ocrResults.init(this->postProcessor->m_iPostProcessMinConfidence,			//min confidence level
								this->postProcessor->m_iPostProcessConfidenceSkipLevel	//confidence skip level
								);
#endif

	// Don't waste time on OCR processing if it is impossible to get sufficient characters
	if ( (int)charRegions[0].size() < m_iPostProcessMinCharacters)
	{
		oAnprObject.oDebugLogs.warn( "NUMBER OF CHARACTERS = %d, IGNORE\n", (int)charRegions[0].size() );
		return;
	}

	//#ifdef POST_PROCESS
	//  int curLetterSize = postProcessor->getLetterSize();
	//#endif

	for (unsigned int i = 0; i < thresholds.size(); i++)
	{
		// Make it black text on white background
		Mat oBinInversed;
		bitwise_not(thresholds[i], oBinInversed);
		tesseract->SetImage( (uchar*) oBinInversed.data,
							oBinInversed.size().width,
							oBinInversed.size().height,
							oBinInversed.channels(),
							oBinInversed.step1() );

		// DV: 16/06/2014 - Use bounding boxes after gaps are removed
		for (unsigned int j = 0; j < charRegions[i].size(); j++)
		{
			Rect expandedRegion = FTS_IP_Util::expandRectXY( charRegions[i][j], 2, 2, oBinInversed.cols, oBinInversed.rows) ;

			tesseract->SetRectangle(expandedRegion.x, expandedRegion.y, expandedRegion.width, expandedRegion.height);
			tesseract->Recognize(NULL);

			tesseract::ResultIterator* ri = tesseract->GetIterator();
			tesseract::PageIteratorLevel level = tesseract::RIL_SYMBOL;
			do
			{
				const char* symbol = ri->GetUTF8Text(level);
				float conf = ri->Confidence(level);

				bool dontcare;
				int fontindex = 0;
				int pointsize = 0;
				const char* fontName = ri->WordFontAttributes(&dontcare, &dontcare, &dontcare, &dontcare, &dontcare, &dontcare, &pointsize, &fontindex);

				if(symbol != 0  && pointsize >= m_iOcrMinFontSize)
				{
//#ifdef POST_PROCESS
					//if(this->postProcessor) postProcessor->addLetter(*symbol, j, conf);
//#endif
					oAnprObject.ocrResults.addLetter(*symbol, j, conf);

					if (this->m_bDebugOcr)
					{
						oAnprObject.oDebugLogs.info("charpos%d: threshold %d:  symbol %c, conf: %f font: %s (index %d) size %dpx", 
							j, i, *symbol, conf, fontName, fontindex, pointsize);
					}

					/*bool indent = false;
					do 
					{
						const char* choice = ri->GetUTF8Text(level);
//#ifdef POST_PROCESS
						//if(this->postProcessor) postProcessor->addLetter(*choice, j, ri->Confidence(level));
//#endif
						oAnprObject.ocrResults.addLetter(*choice, j, ri->Confidence(level));

						if (this->m_bDebugOcr)
						{				
							oAnprObject.oDebugLogs.info("%s\t-%c conf: %f\n", indent?"\t\t ": "", *choice, ri->Confidence(level));
						}

						indent = true;
					} 
					while(ri->Next(level));*/	//28.06 trung rem out => non used
				}

				/*if (this->m_bDebugOcr)
					oAnprObject.oDebugLogs.info("---------------------------------------------\n"); */ //28.06 trung remove

				//delete[] symbol;
			} 
			while((ri->Next(level)));

			delete ri;
		}
	}

	/*if (config->debugTiming)
	{
	timespec endTime;
	getTime(&endTime);
	cout << "OCR Time: " << diffclock(startTime, endTime) << "ms." << endl;
	}*/  
}
