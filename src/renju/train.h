#pragma once
#include "network.h"

int selfplay(Player& player, DataSet& dataset, int itermax);
void train(std::shared_ptr<FIRNet> net);
