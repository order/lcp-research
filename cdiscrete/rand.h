#ifndef __RAND_INCLUDED__
#define __RAND_INCLUDED__
#include <random>
#include <chrono>


unsigned SEED = std::chrono::system_clock::now().time_since_epoch().count();
std::mt19937 MT_GEN(SEED);

#endif
