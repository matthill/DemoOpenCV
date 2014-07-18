
#ifndef FTSALPR_POSTPROCESS_H
#define FTSALPR_POSTPROCESS_H

//#include "TRexpp.h"
//#include "constants.h"
//#include "utility.h"
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <map>
#include <string>
//#include "config.h"

using namespace std;

#define USE_BOOST_REGEX

#ifndef USE_BOOST_REGEX
#include "TRexpp.h"
#else
#include <boost/regex.hpp>
#endif

#include "fts_anpr_object.h"

bool wordCompare( const FTS_ANPR_PPResult &left, const FTS_ANPR_PPResult &right );
bool letterCompare( const Letter &left, const Letter &right );

class FTS_ANPR_RegexRule
{
  public:
    FTS_ANPR_RegexRule(string region, string pattern);

    bool match(string text);
    string filterSkips(string text);

  private:
    int numchars;
#ifndef USE_BOOST_REGEX
    TRexpp trexp;
#else
	boost::regex re;
#endif
    string original;
    string regex;
    string region;
    vector<int> skipPositions;
};

class FTS_ANPR_PostProcess
{
  public:
    FTS_ANPR_PostProcess(/*Config* config*/string patternsFile, int minConfidence, int confidenceSkipLevel, int maxSubs, bool debugPP);
    ~FTS_ANPR_PostProcess();

	//void addBatchLetter(const OcrResult& ocrResults);
    //void addLetter(char letter, int charposition, float score);

    void clear();
    //void analyze(string templateregion, int topn, vector<vector<Letter>>& letters, vector<FTS_ANPR_PPResult>& allPossibilities);
	void analyze(string templateregion, int topn, FTS_ANPR_OBJECT& oAnprObject);

    string bestChars;
    bool matchesTemplate;

    //const vector<FTS_ANPR_PPResult> getResults();

	//int getLetterSize() { return letters.size(); }
	//vector<vector<Letter>>& getLetters() { return letters; }
	int m_iPostProcessMinConfidence;
	int m_iPostProcessConfidenceSkipLevel;
	bool m_bDebugPostProcess;
	int m_iPostProcessMaxSubstitutions;

  private:
    //Config* config;
    //void getTopN();
    void findAllPermutations(	vector< vector<Letter> >& letters,
								vector<Letter> prevletters, 
								int charPos, 
								int substitutionsLeft, 
								vector<FTS_ANPR_PPResult>& allPossibilities);

    //void insertLetter(char letter, int charPosition, float score);

    map<string, vector<FTS_ANPR_RegexRule*> > rules;

    float calculateMaxConfidenceScore(vector< vector<Letter> >& letters);

    //vector<vector<Letter> > letters;
    //vector<int> unknownCharPositions;

    //vector<FTS_ANPR_PPResult> allPossibilities;

	// Functions used to prune the list of letters (based on topn) to improve performance
    vector<int> getMaxDepth(int topn, vector< vector<Letter> >& letters);
    int getPermutationCount(vector<int> depth);
    int getNextLeastDrop(vector<int> depth, vector< vector<Letter> >& letters);
};

/*
class LetterScores
{
  public:
    LetterScores(int numCharPositions);

    void addScore(char letter, int charposition, float score);

    vector<char> getBestScore();
    float getConfidence();

  private:
    int numCharPositions;

    vector<char> letters;
    vector<int> charpositions;
    vector<float> scores;
};
*/
#endif // FTSALPR_POSTPROCESS_H
