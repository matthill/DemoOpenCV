#include "FTSVideoAlgorithm.h"

//std::vector<std::thread> workers;
//std::mutex mutexTaskQueue;
std::queue<ViolationEvent> taskQueue;
int stop_count = 0;