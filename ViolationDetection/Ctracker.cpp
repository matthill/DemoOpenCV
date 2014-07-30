#include "Ctracker.h"

//#include <vld.h>

using namespace cv;
using namespace std;
const Scalar Colors[] = { Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(0, 0, 255), Scalar(255, 255, 0), Scalar(0, 255, 255), Scalar(255, 0, 255), Scalar(255, 127, 255), Scalar(127, 0, 255), Scalar(127, 0, 127) };
size_t CTrack::NextTrackID = 0;
// ---------------------------------------------------------------------------
// Конструктор трека.
// При создании, трек начинается с какой то точки,
// эта точка и передается конструктору в качестве аргумента.
// ---------------------------------------------------------------------------
CTrack::CTrack(Point2f pt, float dt, float Accel_noise_mag) : isCount(false), isIn(false), isOut(false), isCaught(false)
{
	track_id = NextTrackID;

	NextTrackID++;
	// Каждый трек имеет свой фильтр Кальмана,
	// при помощи которого делается прогноз, где должна быть следующая точка.
	KF = new TKalmanFilter(pt, dt, Accel_noise_mag);
	// Здесь хранятся координаты точки, в которой трек прогнозирует следующее наблюдение (детект).
	prediction = pt;
	skipped_frames = 0;
}
// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
CTrack::~CTrack()
{
	// Освобождаем фильтр Кальмана.
	delete KF;
}

// ---------------------------------------------------------------------------
// Трекер. Производит управление треками. Создает, удаляет, уточняет.
// ---------------------------------------------------------------------------
CTracker::CTracker(float _dt, float _Accel_noise_mag, double _dist_thres, double _cos_thres, int _maximum_allowed_skipped_frames, int _max_trace_length, double _very_large_cost)
{
	dt = _dt;
	Accel_noise_mag = _Accel_noise_mag;
	dist_thres = _dist_thres;
	cos_thres = _cos_thres;
	maximum_allowed_skipped_frames = _maximum_allowed_skipped_frames;
	max_trace_length = _max_trace_length;
	very_large_cost = _very_large_cost;
}
// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
void CTracker::Update(vector<Point2d>& detections)
{
	if (detections.empty())
		return;
	// -----------------------------------
	// Если треков еще нет, то начнем для каждой точки по треку
	// -----------------------------------
	if (tracks.size() == 0)
	{
		// Если еще нет ни одного трека
		for (int i = 0; i<detections.size(); i++)
		{
			CTrack* tr = new CTrack(detections[i], dt, Accel_noise_mag);
			tracks.push_back(tr);
		}
	}

	// -----------------------------------
	// Здесь треки уже есть в любом случае
	// -----------------------------------
	int N = tracks.size();		// треки
	int M = detections.size();	// детекты

	// Матрица расстояний от N-ного трека до M-ного детекта.
	vector< vector<double> > Cost(N, vector<double>(M));
	//vector<int> assignment; // назначения
	// HOANG: Add assignment into attribute list of CTracker
	this->assignment.clear();

	// -----------------------------------
	// Треки уже есть, составим матрицу расстояний
	// -----------------------------------
	double dist;
	// Hoang: Utilize the formula of the magnitude of decay function in Tensor Voting
	double sigma = 2;
	double c = ( -16 * std::log(0.1)*(sigma - 1) ) / (3.14*3.14);
	for (int i = 0; i<tracks.size(); i++)
	{
		// Point2d prediction=tracks[i]->prediction;
		// cout << prediction << endl;
		for (int j = 0; j<detections.size(); j++)
		{
			Point2d diff = (tracks[i]->prediction - detections[j]);
			dist = sqrtf(diff.x*diff.x + diff.y*diff.y);
			int iTraceLength = tracks[i]->trace.size();
			if (iTraceLength > 1) {
				Point2d v1 = tracks[i]->trace[iTraceLength - 1] - tracks[i]->trace[iTraceLength - 2];
				Point2d v2 = detections[j] - tracks[i]->trace[iTraceLength - 1];

				double cos_v1_v2 = (v1.x*v2.x + v1.y*v2.y) / (std::sqrt((v1.x*v1.x + v1.y*v1.y)*(v2.x*v2.x + v2.y*v2.y)));

				if (dist > dist_thres || std::abs(cos_v1_v2) < cos_thres || (dist > dist_thres/2 &&  cos_v1_v2 < -0.3))
				{
					Cost[i][j] = this->very_large_cost;
				}
				else
				{
					/*double angle_v1_v2 = std::acos(cos_v1_v2);
					double sin_v1_v2 = std::sin(angle_v1_v2);
					double s = angle_v1_v2*dist / sin_v1_v2;
					double curvature = 2 * sin_v1_v2 / dist;
					Cost[i][j] = std::exp((s*s + c*curvature*curvature) / (sigma*sigma));*/
					Cost[i][j] = dist*std::exp((1 - std::abs(cos_v1_v2)) / 2);
				}
			}
			else 
			{
				if (dist > dist_thres)
				{
					Cost[i][j] = this->very_large_cost;
				}
				else
				{
					Cost[i][j] = dist;
				}
			}
		}
	}
	// -----------------------------------
	// Решаем задачу о назначениях (треки и прогнозы фильтра)
	// -----------------------------------
	AssignmentProblemSolver APS;
	//APS.Solve(Cost, assignment, AssignmentProblemSolver::optimal);
	APS.Solve(Cost, assignment, AssignmentProblemSolver::many_forbidden_assignments);
	

	// -----------------------------------
	// почистим assignment от пар с большим расстоянием
	// -----------------------------------
	// Не назначенные треки
	vector<int> not_assigned_tracks;

	for (int i = 0; i<assignment.size(); i++)
	{
		if (assignment[i] != -1)
		{
			if (Cost[i][assignment[i]]>dist_thres)
			{
				assignment[i] = -1;
				// Отмечаем неназначенные треки, и увеличиваем счетчик пропущеных кадров,
				// когда количество пропущенных кадров превысит пороговое значение, трек стирается.
				not_assigned_tracks.push_back(i);
			}
		}
		else
		{
			// Если треку не назначен детект, то увеличиваем счетчик пропущеных кадров.
			tracks[i]->skipped_frames++;
		}

	}

	// -----------------------------------
	// Если трек долго не получает детектов, удаляем
	// -----------------------------------
	for (int i = 0; i<tracks.size(); i++)
	{
		if (tracks[i]->skipped_frames>maximum_allowed_skipped_frames)
		{
			tracks[i]->trace.clear();
			delete tracks[i];
			tracks.erase(tracks.begin() + i);
			assignment.erase(assignment.begin() + i);
			i--;
		}
	}
	// -----------------------------------
	// Выявляем неназначенные детекты
	// -----------------------------------
	vector<int> not_assigned_detections;
	vector<int>::iterator it;
	for (int i = 0; i<detections.size(); i++)
	{
		it = find(assignment.begin(), assignment.end(), i);
		if (it == assignment.end())
		{
			not_assigned_detections.push_back(i);
		}
	}

	// -----------------------------------
	// и начинаем для них новые треки
	// -----------------------------------
	if (not_assigned_detections.size() != 0)
	{
		for (int i = 0; i<not_assigned_detections.size(); i++)
		{
			CTrack* tr = new CTrack(detections[not_assigned_detections[i]], dt, Accel_noise_mag);
			tracks.push_back(tr);
		}
	}

	// Апдейтим состояние фильтров

	for (int i = 0; i<assignment.size(); i++)
	{
		// Если трек апдейтился меньше одного раза, то состояние фильтра некорректно.

		tracks[i]->KF->GetPrediction();

		if (assignment[i] != -1) // Если назначение есть то апдейтим по нему
		{
			tracks[i]->skipped_frames = 0;
			tracks[i]->prediction = tracks[i]->KF->Update(detections[assignment[i]], 1);
		}
		else				  // Если нет, то продолжаем прогнозировать
		{
			tracks[i]->prediction = tracks[i]->KF->Update(Point2f(0, 0), 0);
		}

		if (tracks[i]->trace.size()>max_trace_length)
		{
			tracks[i]->trace.erase(tracks[i]->trace.begin(), tracks[i]->trace.end() - max_trace_length);
		}

		tracks[i]->trace.push_back(tracks[i]->prediction);
		tracks[i]->KF->LastResult = tracks[i]->prediction;
	}

}
// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
void CTracker::updateSkipedSkippedFrames(){
	for (int i = 0; i<tracks.size(); i++)
	{
		tracks[i]->skipped_frames++;
		if (tracks[i]->skipped_frames>maximum_allowed_skipped_frames)
		{
			tracks[i]->trace.clear();
			delete tracks[i];
			tracks.erase(tracks.begin() + i);
			i--;
		}
	}
}
// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
void CTracker::drawTrackToImage(cv::Mat &img)
{
	for (int i = 0; i < tracks.size(); i++)
	{
		// Draw tracks
		if (tracks[i]->trace.size()>1)
		{
			for (int j = 0; j < tracks[i]->trace.size() - 1; j++)
			{
				line(img, tracks[i]->trace[j], tracks[i]->trace[j + 1], Colors[tracks[i]->track_id % 9], 2, CV_AA);
			}
		}
	}
}
// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
void CTracker::setPlateForTrackers(const cv::Mat& originalImage, std::vector<cv::Rect> &listPlateObjs){
	cv::Rect boundingBox;
	for (size_t i = 0; i < assignment.size(); i++)
	{
		int ind = assignment[i];
		if (ind > -1 && ind < listPlateObjs.size() && !tracks[i]->isHavePlate){
			cv::Mat plate = originalImage(listPlateObjs[ind]).clone();
			
			tracks[i]->imgPlate = plate;
			tracks[i]->isHavePlate = true;
		}
	}
}
CTracker::~CTracker(void)
{
	for (size_t i = 0; i<tracks.size(); i++)
	{
		delete tracks[i];
	}
	tracks.clear();
}
