#pragma once
#include "network.h"

constexpr float kCpuct = 1.0;
constexpr int kEpochPerGame = 1;
constexpr int kExploreStep = 30;
constexpr int kTestPureItermax = 1000;
constexpr int kTrainDeepItermax = 400;
constexpr int kMinutePerLog = 5;
constexpr int kMinutePerSave = 30;
constexpr int kMinutePerBenchmark = 45;
constexpr bool kDebugTrainData = false;

int selfplay(std::shared_ptr<FIRNet> net, DataSet& dataset, int itermax);
void train(std::shared_ptr<FIRNet> net);
