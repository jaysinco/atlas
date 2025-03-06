#include "utils/args.h"
#include "mcts.h"
#include "train.h"
#include <iostream>

std::random_device g_random_device;
std::mt19937 g_random_engine(g_random_device());

void runCmdTrain(int64_t verno)
{
    ILOG("verno={}", verno);
    auto net = std::make_shared<FIRNet>(verno);
    train(net);
}

void runCmdPlay(int color, int64_t verno, int itermax)
{
    ILOG("color={}, verno={}, itermax={}", color, verno, itermax);
    auto net = std::make_shared<FIRNet>(verno);
    auto p1 = MCTSDeepPlayer(net, itermax, kCpuct);
    if (color == 0) {
        auto p0 = HumanPlayer("human");
        play(p0, p1, false);
    } else if (color == 1) {
        auto p0 = HumanPlayer("human");
        play(p1, p0, false);
    } else if (color == -1) {
        auto p0 = MCTSDeepPlayer(net, itermax, kCpuct);
        play(p0, p1, false);
    }
}

void runCmdBenchmark(int64_t verno1, int64_t verno2, int itermax)
{
    ILOG("verno1={}, verno2={}, itermax={}", verno1, verno2, itermax);
    auto net1 = std::make_shared<FIRNet>(verno1);
    auto net2 = std::make_shared<FIRNet>(verno2);
    auto p1 = MCTSDeepPlayer(net1, itermax, kCpuct);
    auto p2 = MCTSDeepPlayer(net2, itermax, kCpuct);
    benchmark(p1, p2, 10, false);
}

int main(int argc, char** argv)
{
    MY_TRY;
    utils::Args args(argc, argv);

    auto& train_args = args.addSub("train", "train model from scatch or checkpoint");
    train_args.positional("verno", utils::value<int64_t>(),
                          "verno of checkpoint; 0 to train from scratch", 1);

    auto& play_args = args.addSub("play", "play with trained model");
    play_args.positional("color", utils::value<int>(),
                         "first hand color; human(0), computer(1), selfplay(-1)", 1);
    play_args.positional("verno", utils::value<int64_t>(), "verno of checkpoint", 1);
    play_args.positional("itermax", utils::value<int>()->default_value(kTrainDeepItermax),
                         "itermax for mcts deep player", 1);

    auto& benchmark_args = args.addSub("benchmark", "benchmark between two mcts deep players");
    benchmark_args.positional("verno1", utils::value<int64_t>(), "verno of checkpoint to compare",
                              1);
    benchmark_args.positional("verno2", utils::value<int64_t>(), "see above", 1);
    benchmark_args.positional("itermax", utils::value<int>()->default_value(kTrainDeepItermax),
                              "itermax for mcts deep player", 1);

    args.parse();

    // run cmd
    if (args.hasSub("train")) {
        auto verno = train_args.get<int64_t>("verno");
        runCmdTrain(verno);
    } else if (args.hasSub("play")) {
        auto color = play_args.get<int>("color");
        auto verno = play_args.get<int64_t>("verno");
        auto itermax = play_args.get<int>("itermax");
        runCmdPlay(color, verno, itermax);
    } else if (args.hasSub("benchmark")) {
        auto verno1 = benchmark_args.get<int64_t>("verno1");
        auto verno2 = benchmark_args.get<int64_t>("verno2");
        auto itermax = benchmark_args.get<int>("itermax");
        runCmdBenchmark(verno1, verno2, itermax);
    }
    MY_CATCH;
}