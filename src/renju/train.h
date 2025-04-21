#pragma once
#include "network.h"

struct SelfPlayMeta
{
    int rounds;
    int turns;
};

void selfPlay(Player& player, DataSet& dataset, SelfPlayMeta& meta);
void train(std::shared_ptr<FIRNet> net);
