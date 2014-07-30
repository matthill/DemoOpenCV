
#ifndef _FTS_IP_VERTICALHISTOGRAM_H
#define _FTS_IP_VERTICALHISTOGRAM_H

#include "fts_base_externals.h"

using namespace cv;
using namespace std;

class FTS_IP_VerticalHistogram
{
	struct Valley
	{
	  int startIndex;
	  int endIndex;
	  int width;
	  int pixelsWithin;
	};

	enum FTS_IP_HistogramDirection { RISING, FALLING, FLAT };

  public:

	FTS_IP_VerticalHistogram();
    FTS_IP_VerticalHistogram(Mat inputImage, Mat mask);
    virtual ~FTS_IP_VerticalHistogram();

    Mat histoImg;

    // Returns the lowest X position between two points.
    int getLocalMinimum(int leftX, int rightX) const;
    // Returns the highest X position between two points.
    int getLocalMaximum(int leftX, int rightX) const;

    int getHeightAt(int x) const;

    void analyzeImage(Mat inputImage, Mat mask);

  private:
    vector<int> colHeights;
    int highestPeak;
    int lowestValley;
    vector<Valley> valleys;


    void findValleys();

    int getHistogramDirection(int index);
};

#endif // _FTS_IP_VERTICALHISTOGRAM_H
