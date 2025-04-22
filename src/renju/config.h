#pragma once

/*
    3 * 3 board looks like:

      0 1 2
    ------- Col
    0|0 1 2
    1|3 4 5
    2|6 7 8

    Row  => move z(5) = (r(1), c(2))
*/

constexpr int kFiveInRow = 5;
constexpr int kBoardMaxCol = 9;
constexpr int kBoardMaxRow = kBoardMaxCol;
constexpr int kBoardSize = kBoardMaxRow * kBoardMaxCol;
constexpr bool kBoardRichSymbol = true;
constexpr int kInputFeatureNum = 2;  // self, opponent[[, lastmove], color]
constexpr unsigned char kNoMoveYet = 0xff;
constexpr int kMCTSThreadNum = 8;
constexpr float kNoiseRate = 0.2;
constexpr float kDirichletAlpha = 0.03;
constexpr bool kDebugMCTSProb = false;

constexpr int kResidualLayers = 7;
constexpr int kResidualFilters = 64;
constexpr int kBatchSize = 512;
constexpr int kBufferSize = 10000;
constexpr float kInitLearningRate = 1e-3;
constexpr float kWeightDecay = 1e-4;
constexpr int kDropStepLR1 = 2000;
constexpr int kDropStepLR2 = 8000;
constexpr int kDropStepLR3 = 10000;
constexpr int kEpochPerGame = 1;
constexpr int kExploreStep = 30;
constexpr int kTestPureItermax = 1000;
constexpr int kTrainDeepItermax = 400;
constexpr float kTrainCpuct = 2.4;
constexpr int kMinutePerLog = 5;
constexpr int kMinutePerSave = 30;
constexpr int kMinutePerBenchmark = 45;
constexpr bool kDebugTrainData = false;
