
#include "fts_ip_postprocess.h"
#include <algorithm>

FTS_ANPR_PostProcess::FTS_ANPR_PostProcess(/*Config* config*/string patternsFile, int minConfidence, int confidenceSkipLevel, int maxSubs, bool debugPP)
{
	//this->config = config;

	//stringstream filename;
	//filename << config->getPostProcessRuntimeDir() << "/" << config->country << ".patterns";  

	this->m_iPostProcessConfidenceSkipLevel = confidenceSkipLevel;
	this->m_iPostProcessMaxSubstitutions = maxSubs;
	this->m_iPostProcessMinConfidence = minConfidence;
	this->m_bDebugPostProcess = debugPP;

	std::ifstream infile(patternsFile.c_str());

	string region, pattern;
	while (infile >> region >> pattern)
	{
		FTS_ANPR_RegexRule* rule = new FTS_ANPR_RegexRule(region, pattern);
		//cout << "REGION: " << region << " PATTERN: " << pattern << endl;

		if (rules.find(region) == rules.end())
		{
			vector<FTS_ANPR_RegexRule*> newRule;
			newRule.push_back(rule);
			rules[region] = newRule;
		}
		else
		{
			vector<FTS_ANPR_RegexRule*> oldRule = rules[region];
			oldRule.push_back(rule);
			rules[region] = oldRule;
		}
	}

	infile.close();

	//vector<RegexRule> test = rules["base"];
	//for (int i = 0; i < test.size(); i++)
	//  cout << "Rule: " << test[i].regex << endl;
}

FTS_ANPR_PostProcess::~FTS_ANPR_PostProcess()
{
	// TODO: Delete all entries in rules vector
	map<string, vector<FTS_ANPR_RegexRule*> >::iterator iter;

	for (iter = rules.begin(); iter != rules.end(); ++iter)
	{
		for (size_t i = 0; i < iter->second.size(); i++)
		{
			delete iter->second[i];
		}
	}
}

//void FTS_ANPR_PostProcess::addBatchLetter(const vector<OcrResult>& ocrResults)
//{
//	for(int i = 0; i < ocrResults.size(); i++)
//	{
//		this->addLetter(ocrResults[i].letter, ocrResults[i].charposition, ocrResults[i].totalscore);
//	}
//}

void FTS_ANPR_PostProcess::clear()
{
	/*for (int i = 0; i < letters.size(); i++)
	{
		letters[i].clear();
	}
	letters.resize(0);*/

	//unknownCharPositions.clear();
	//unknownCharPositions.resize(0);
	/*allPossibilities.clear();*/ //25.06 trungnt1 remove object members
	//allPossibilities.resize(0);

	bestChars = "";
	matchesTemplate = false;
}

//void FTS_ANPR_PostProcess::analyze(string templateregion, int topn, vector< vector<Letter> >& letters, vector<FTS_ANPR_PPResult>& allPossibilities)
void FTS_ANPR_PostProcess::analyze(string templateregion, int topn, FTS_ANPR_OBJECT& oAnprObject)
{
	//timespec startTime;
	//getTime(&startTime);

	// Get a list of missing positions
	vector<int> unknownCharPositions;
	unknownCharPositions.clear();
	unknownCharPositions.resize(0);
	for (int i = oAnprObject.ocrResults.letters.size() -1; i >= 0; i--)
	{
		if (oAnprObject.ocrResults.letters[i].size() == 0)
		{
			unknownCharPositions.push_back(i);
		}
	}

	if (oAnprObject.ocrResults.letters.size() == 0)
		return;

	// Sort the letters as they are
	for (size_t i = 0; i < oAnprObject.ocrResults.letters.size(); i++)
	{
		if (oAnprObject.ocrResults.letters[i].size() > 0)
			sort(oAnprObject.ocrResults.letters[i].begin(), oAnprObject.ocrResults.letters[i].end(), letterCompare);
	}

	if (this->m_bDebugPostProcess)
	{
		// Print all letters
		for (size_t i = 0; i < oAnprObject.ocrResults.letters.size(); i++)
		{
			for (size_t j = 0; j < oAnprObject.ocrResults.letters[i].size(); j++)
			{
				//cout << "PostProcess Letter: " << oAnprObject.ocrResults.letters[i][j].charposition << " " << oAnprObject.ocrResults.letters[i][j].letter 
				//	 << " -- score: " << oAnprObject.ocrResults.letters[i][j].totalscore 
				//	 << " -- occurences: " << oAnprObject.ocrResults.letters[i][j].occurences << endl;
				stringstream ss;
				ss << "PostProcess Letter: " << oAnprObject.ocrResults.letters[i][j].charposition 
				   << " " << oAnprObject.ocrResults.letters[i][j].letter 
				   << " -- score: " << oAnprObject.ocrResults.letters[i][j].totalscore 
				   << " -- occurences: " << oAnprObject.ocrResults.letters[i][j].occurences;
				oAnprObject.oDebugLogs.info(ss.str().c_str());
			}
		}
	}

	// Prune the letters based on the topN value.
	// If our topN value is 3, for example, we can get rid of a lot of low scoring letters
	// because it would be impossible for them to be a part of our topN results.
	vector<int> maxDepth = getMaxDepth(topn, oAnprObject.ocrResults.letters);

	for (size_t i = 0; i < oAnprObject.ocrResults.letters.size(); i++)
	{
		for (int k = oAnprObject.ocrResults.letters[i].size() - 1; k > maxDepth[i]; k--)
		{
			oAnprObject.ocrResults.letters[i].erase(oAnprObject.ocrResults.letters[i].begin() + k);
		}
	}

	//getTopN();
	vector<Letter> tmp;
	findAllPermutations(oAnprObject.ocrResults.letters, tmp, 0, m_iPostProcessMaxSubstitutions, oAnprObject.ppResults);

	//timespec sortStartTime;
	//getTime(&sortStartTime);

	int numelements = topn;
	//if (allPossibilities.size() < topn)
	numelements = oAnprObject.ppResults.size() - 1;

	//trungnt1 fix if allPossibilities.size() == 1 => not sort
	if(numelements > 0)
		//partial_sort( allPossibilities.begin(), allPossibilities.begin() + numelements, allPossibilities.end(), wordCompare );
		sort( oAnprObject.ppResults.begin(), oAnprObject.ppResults.end(), wordCompare );

	//if (config->debugTiming)
	//{
	//	timespec sortEndTime;
	//	getTime(&sortEndTime);
	//	cout << " -- PostProcess Sort Time: " << diffclock(sortStartTime, sortEndTime) << "ms." << endl;
	//}

	matchesTemplate = false;

	if (templateregion != "")
	{
		vector<FTS_ANPR_RegexRule*> regionRules = rules[templateregion];

		for (size_t i = 0; i < oAnprObject.ppResults.size(); i++)
		{
			for (size_t j = 0; j < regionRules.size(); j++)
			{
				oAnprObject.ppResults[i].matchesTemplate = regionRules[j]->match(oAnprObject.ppResults[i].letters);
				if (oAnprObject.ppResults[i].matchesTemplate)
				{
					oAnprObject.ppResults[i].letters = regionRules[j]->filterSkips(oAnprObject.ppResults[i].letters);
					//bestChars = regionRules[j]->filterSkips(ppResults[i].letters);
					matchesTemplate = true;
					break;
				}
			}

			if (i >= topn - 1)
				break;
			//if (matchesTemplate || i >= TOP_N - 1)
			//break;
		}
	}

	if (matchesTemplate)
	{
		for (size_t z = 0; z < oAnprObject.ppResults.size(); z++)
		{
			if (oAnprObject.ppResults[z].matchesTemplate)
			{
				bestChars = oAnprObject.ppResults[z].letters;
				break;
			}
		}
	}
	else
	{
		bestChars = oAnprObject.ppResults[0].letters;
	}

	// Now adjust the confidence scores to a percentage value
	if (oAnprObject.ppResults.size() > 0)
	{
		float maxPercentScore = calculateMaxConfidenceScore(oAnprObject.ocrResults.letters);
		float highestRelativeScore = (float) oAnprObject.ppResults[0].totalscore;

		for (size_t i = 0; i < oAnprObject.ppResults.size(); i++)
		{
			oAnprObject.ppResults[i].totalscore = maxPercentScore * (oAnprObject.ppResults[i].totalscore / highestRelativeScore);
		}
	}

	if (this->m_bDebugPostProcess)
	{
		// Print top words
		for (size_t i = 0; i < oAnprObject.ppResults.size(); i++)
		{
			//cout << "Top " << topn << " Possibilities: " << oAnprObject.ppResults[i].letters << " :\t" << oAnprObject.ppResults[i].totalscore;
			stringstream ss;
			ss << "Top " << topn << " Possibilities: " << oAnprObject.ppResults[i].letters << " : " << oAnprObject.ppResults[i].totalscore;
			if (oAnprObject.ppResults[i].letters == bestChars)				
				ss << " <--- ";
			oAnprObject.oDebugLogs.info(ss.str().c_str());

			if (i >= topn - 1)
				break;
		}
		//cout << oAnprObject.ppResults.size() << " total permutations" << endl;
		oAnprObject.oDebugLogs.info("%d total permutations", oAnprObject.ppResults.size());;
	}

	/*if (config->debugTiming)
	{
		timespec endTime;
		getTime(&endTime);
		cout << "PostProcess Time: " << diffclock(startTime, endTime) << "ms." << endl;
	}*/

	if (this->m_bDebugPostProcess)
		//cout << "PostProcess Analysis Complete: " << bestChars << " -- MATCH: " << matchesTemplate << endl;
		oAnprObject.oDebugLogs.info("PostProcess Analysis Complete: %s -- MATCH: %d", bestChars.c_str(), matchesTemplate);
}

float FTS_ANPR_PostProcess::calculateMaxConfidenceScore(vector< vector<Letter> >& letters)
{
	// Take the best score for each char position and average it.
	float totalScore = 0;
	int numScores = 0;
	// Get a list of missing positions
	for (size_t i = 0; i < letters.size(); i++)
	{
		if (letters[i].size() > 0)
		{
			totalScore += (letters[i][0].totalscore / letters[i][0].occurences) + m_iPostProcessMinConfidence;
			numScores++;
		}
	}

	if (numScores == 0)
		return 0;

	return totalScore / ((float) numScores);
}

// Finds the minimum number of letters to include in the recursive sorting algorithm.
// For example, if I have letters
//	A-200 B-100 C-100
//	X-99 Y-95   Z-90
//	Q-55        R-80
// And my topN value was 3, this would return:
// 0, 1, 1
// Which represents:
// 	A-200 B-100 C-100
//	      Y-95  Z-90
vector<int> FTS_ANPR_PostProcess::getMaxDepth(int topn, vector< vector<Letter> >& letters)
{
	vector<int> depth;
	for (size_t i = 0; i < letters.size(); i++)
		depth.push_back(0);

	int nextLeastDropCharPos = getNextLeastDrop(depth, letters);
	while (nextLeastDropCharPos != -1)
	{
		if (getPermutationCount(depth) >= topn)
			break;

		depth[nextLeastDropCharPos] = depth[nextLeastDropCharPos] + 1;

		nextLeastDropCharPos = getNextLeastDrop(depth, letters);
	}

	return depth;
}

int FTS_ANPR_PostProcess::getPermutationCount(vector<int> depth)
{
	int permutationCount = 1;
	for (size_t i = 0; i < depth.size(); i++)
	{
		permutationCount *= (depth[i] + 1);
	}

	return permutationCount;
}

int FTS_ANPR_PostProcess::getNextLeastDrop(vector<int> depth, vector< vector<Letter> >& letters)
{
	int nextLeastDropCharPos = -1;
	float leastNextDrop = 99999999999;

	for (size_t i = 0; i < letters.size(); i++)
	{
		if (depth[i] + 1 >= letters[i].size())
			continue;

		float drop = letters[i][depth[i]].totalscore - letters[i][depth[i]+1].totalscore;

		if (drop < leastNextDrop)
		{
			nextLeastDropCharPos = i;
			leastNextDrop = drop;
		}
	}

	return nextLeastDropCharPos;
}

//const vector<FTS_ANPR_PPResult> FTS_ANPR_PostProcess::getResults()
//{
//	return this->allPossibilities;
//}

void FTS_ANPR_PostProcess::findAllPermutations(vector< vector<Letter> >& letters,
												vector<Letter> prevletters, 
												int charPos, 
												int substitutionsLeft, 
												vector<FTS_ANPR_PPResult>& allPossibilities)
{
	if (substitutionsLeft < 0)
		return;

	// Add my letter to the chain and recurse
	for (size_t i = 0; i < letters[charPos].size(); i++)
	{
		if (charPos == (int)letters.size() - 1)
		{
			// Last letter, add the word
			FTS_ANPR_PPResult possibility;
			possibility.letters = "";
			possibility.totalscore = 0;
			possibility.matchesTemplate = false;
			for (size_t z = 0; z < prevletters.size(); z++)
			{
				if (prevletters[z].letter != SKIP_CHAR)
					possibility.letters = possibility.letters + prevletters[z].letter;
				else
					possibility.letters = possibility.letters + "*";
				possibility.totalscore = possibility.totalscore + prevletters[z].totalscore;
			}

			if (letters[charPos][i].letter != SKIP_CHAR)
				possibility.letters = possibility.letters + letters[charPos][i].letter;
			else
				possibility.letters = possibility.letters + "*";
			possibility.totalscore = possibility.totalscore +letters[charPos][i].totalscore;

			allPossibilities.push_back(possibility);
		}
		else
		{
			prevletters.push_back(letters[charPos][i]);

			float scorePercentDiff = abs( letters[charPos][0].totalscore - letters[charPos][i].totalscore ) / letters[charPos][0].totalscore;
			if (i != 0 && letters[charPos][i].letter != SKIP_CHAR && scorePercentDiff > 0.10f )
				findAllPermutations(letters, prevletters, charPos + 1, substitutionsLeft - 1, allPossibilities);
			else
				findAllPermutations(letters, prevletters, charPos + 1, substitutionsLeft, allPossibilities);

			prevletters.pop_back();
		}
	}

	if (letters[charPos].size() == 0)
	{
		// No letters for this char position...
		// Just pass it along
		findAllPermutations(letters, prevletters, charPos + 1, substitutionsLeft, allPossibilities);
	}
}

bool wordCompare( const FTS_ANPR_PPResult &left, const FTS_ANPR_PPResult &right )
{
	if (left.totalscore < right.totalscore)
		return false;
	return true;
}

bool letterCompare( const Letter &left, const Letter &right )
{
	if (left.totalscore < right.totalscore)
		return false;
	return true;
}

//CLASS FTS_ANPR_RegexRule
FTS_ANPR_RegexRule::FTS_ANPR_RegexRule(string region, string pattern)
{
	this->original = pattern;
	this->region = region;

	numchars = 0;
	for (size_t i = 0; i < pattern.size(); i++)
	{
		if (pattern.at(i) == '[')
		{
			while (pattern.at(i) != ']' )
			{
				this->regex = this->regex + pattern.at(i);
				i++;
			}
			this->regex = this->regex + ']';
		}
		else if (pattern.at(i) == '?')
		{
			this->regex = this->regex + '.';
			this->skipPositions.push_back(numchars);
		}
		else if (pattern.at(i) == '@')
		{
			this->regex = this->regex + "[A-Z]";
		}
		else if (pattern.at(i) == '#')
		{
			this->regex = this->regex + "\\d";
		}

		numchars++;
	}

#ifndef USE_BOOST_REGEX
	trexp.Compile(this->regex.c_str());
#else
	try
	{
		// Set up the regular expression for case-insensitivity
		re.assign(this->regex.c_str(), boost::regex_constants::icase);
	}
	catch (boost::regex_error& e)
	{
		cout << this->regex << " is not a valid regular expression: \""
			 << e.what() << "\"" << endl;
		return;
	}
#endif

	//cout << "AA " << this->region << ": " << original << " regex: " << regex << endl;
	//for (int z = 0; z < this->skipPositions.size(); z++)
	//	cout << "AA Skip position: " << skipPositions[z] << endl;
}

bool FTS_ANPR_RegexRule::match(string text)
{
	if ((int)text.length() != numchars)
		return false;

#ifndef USE_BOOST_REGEX
	return trexp.Match(text.c_str());
#else
	return boost::regex_match(text.c_str(), re);
    //{
        //cout << re << " matches " << text << endl;
    //}
#endif

	//#include <iostream>
	//#include <string>
	//#include <boost/regex.hpp>  // Boost.Regex lib
	//
	//using namespace std;
	//
	//int main( )
	//      try
	//      {
	//         // Set up the regular expression for case-insensitivity
	//         re.assign(sre, boost::regex_constants::icase);
	//      }
	//      catch (boost::regex_error& e)
	//      {
	//         cout << sre << " is not a valid regular expression: \""
	//              << e.what() << "\"" << endl;
	//         continue;
	//      }
	//      if (boost::regex_match(s, re))
	//      {
	//         cout << re << " matches " << s << endl;
	//      }
	//   }
	//}
}

string FTS_ANPR_RegexRule::filterSkips(string text)
{
	string response = "";
	for (size_t i = 0; i < text.size(); i++)
	{
		bool skip = false;
		for (size_t j = 0; j < skipPositions.size(); j++)
		{
			if (skipPositions[j] == i)
			{
				skip = true;
				break;
			}
		}

		if (skip == false)
			response = response + text[i];
	}

	return response;
}
