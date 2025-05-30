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
constexpr int kBoardMaxCol = 15;
constexpr int kBoardMaxRow = kBoardMaxCol;
constexpr int kBoardSize = kBoardMaxRow * kBoardMaxCol;
constexpr bool kBoardRichSymbol = true;
constexpr int kInputFeatureNum = 4;  // self, opponent[[, lastmove], color]
constexpr unsigned char kNoMoveYet = 0xff;
constexpr float kNoiseRate = 0.2;
constexpr float kDirichletAlpha = 0.03;
constexpr int kResidualLayers = kBoardMaxCol + 1;
constexpr int kResidualFilters = 128;
constexpr int kTrainDataBufferSize = 51200;
constexpr int kTrainBatchSize = 64;
constexpr float kWeightDecay = 1e-3;
constexpr bool kDebugMCTSProb = false;
constexpr bool kDebugTrainData = false;
constexpr bool kCheckBatchEval = false;

struct ActionMeta
{
    float temperature = 1e-3;
    bool add_noise_prior = false;
    float value = -10;
    float* move_priors = nullptr;
};

struct TrainMeta
{
    bool use_gpu = false;
    int thread_num = 16;
    int itermax = 1280;
    float c_puct = 2.4;
    float learning_rate = 1e-3;
    int explore_step = 30;
    int selfplay_turns_per_train = 5;
    float selfplay_avg_rounds = 0;
};
