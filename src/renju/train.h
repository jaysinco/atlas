#pragma once
#include "network.h"
#include <atomic>
#include <mutex>

struct SelfPlayMeta
{
    std::atomic<bool> stop;
    std::mutex lck;
    int rounds;
    int turns;
};

void selfplay(Player& player, DataSet& dataset, int nthreads, SelfPlayMeta& meta);
void train(std::shared_ptr<FIRNet> net);
