
#include "fts_ip_verticalhistogram.h"

FTS_IP_VerticalHistogram::FTS_IP_VerticalHistogram()
{
	// Nothing
}

FTS_IP_VerticalHistogram::FTS_IP_VerticalHistogram(Mat inputImage, Mat mask)
{
	analyzeImage(inputImage, mask);
}

FTS_IP_VerticalHistogram::~FTS_IP_VerticalHistogram()
{
	histoImg.release();
	colHeights.clear();
}

void FTS_IP_VerticalHistogram::analyzeImage(Mat inputImage, Mat mask)
{
	highestPeak = 0;
	lowestValley = inputImage.rows;

	histoImg = Mat::zeros(inputImage.size(), CV_8U);

	int columnCount;

	for (int col = 0; col < inputImage.cols; col++)
	{
		columnCount = 0;

		for (int row = 0; row < inputImage.rows; row++)
		{
			if (inputImage.at<uchar>(row, col) > 0 && mask.at<uchar>(row, col) > 0)
			{
				columnCount++;
			}
		}

		this->colHeights.push_back(columnCount);

		if (columnCount < lowestValley)
		{
			lowestValley = columnCount;
		}
		if( columnCount > highestPeak )
		{
			highestPeak = columnCount;
		}

		for (; columnCount > 0; columnCount--)
		{
			histoImg.at<uchar>(inputImage.rows - columnCount, col) = 255;
		}
	}
}

int FTS_IP_VerticalHistogram::getLocalMinimum(int leftX, int rightX) const
{
	int minimum = histoImg.rows + 1;
	int lowestX = leftX;

	for (int i = leftX; i <= rightX; i++)
	{
		if (colHeights[i] < minimum)
		{
			lowestX = i;
			minimum = colHeights[i];
		}
	}

	return lowestX;
}

int FTS_IP_VerticalHistogram::getLocalMaximum(int leftX, int rightX) const
{
	int maximum = -1;
	int highestX = leftX;

	for (int i = leftX; i <= rightX; i++)
	{
		if (colHeights[i] > maximum)
		{
			highestX = i;
			maximum = colHeights[i];
		}
	}

	return highestX;
}

int FTS_IP_VerticalHistogram::getHeightAt(int x) const
{
	return colHeights[x];
}

void FTS_IP_VerticalHistogram::findValleys()
{
	int totalWidth = colHeights.size();

	int midpoint = ((highestPeak - lowestValley) / 2) + lowestValley;

	FTS_IP_HistogramDirection prevDirection = FALLING;

	int relativePeakHeight = 0;

	for (int i = 0; i < totalWidth; i++)
	{
		bool aboveMidpoint = (colHeights[i] >= midpoint);

		if (aboveMidpoint)
		{
			if (colHeights[i] > relativePeakHeight)
				relativePeakHeight = colHeights[i];

			prevDirection = FLAT;
		}
		else
		{
			relativePeakHeight = 0;

			int direction = getHistogramDirection(i);

			if ((prevDirection == FALLING || prevDirection == FLAT) && direction == RISING)
			{
			}
			else if ((prevDirection == FALLING || prevDirection == FLAT) && direction == RISING)
			{
			}
		}
	}
}

int FTS_IP_VerticalHistogram::getHistogramDirection(int index)
{
	int EXTRA_WIDTH_TO_AVERAGE = 2;

	float trailingAverage = 0;
	float forwardAverage = 0;

	int trailStartIndex = index - EXTRA_WIDTH_TO_AVERAGE;
	if (trailStartIndex < 0)
		trailStartIndex = 0;
	unsigned int forwardEndIndex = index + EXTRA_WIDTH_TO_AVERAGE;
	if (forwardEndIndex >= colHeights.size())
		forwardEndIndex = colHeights.size() - 1;

	for (int i = index; i >= trailStartIndex; i--)
	{
		trailingAverage += colHeights[i];
	}
	trailingAverage = trailingAverage / ((float) (1 + index - trailStartIndex));

	for( unsigned int i = index; i <= forwardEndIndex; i++)
	{
		forwardAverage += colHeights[i];
	}
	forwardAverage = forwardAverage / ((float) (1 + forwardEndIndex - index));

	float diff = forwardAverage - trailingAverage;
	float minDiff = ((float) (highestPeak - lowestValley)) * 0.10;

	if (diff > minDiff)
		return RISING;
	else if (diff < minDiff)
		return FALLING;
	else
		return FLAT;
}
