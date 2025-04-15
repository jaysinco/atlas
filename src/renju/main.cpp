#include "toolkit/args.h"
#include "mcts.h"
#include "train.h"
#include <iostream>

std::random_device g_random_device;
std::mt19937 g_random_engine(g_random_device());

void runTrain(int64_t verno)
{
    ILOG("verno={}", verno);
    auto net = std::make_shared<FIRNet>(verno);
    train(net);
}

void runPlay(int color, int64_t verno, int itermax)
{
    ILOG("color={}, verno={}, itermax={}", color, verno, itermax);
    std::shared_ptr<Player> p0, p1;
    std::shared_ptr<FIRNet> net;
    if (verno > 0) {
        net = std::make_shared<FIRNet>(verno);
        if (color == 0) {
            p0 = std::make_shared<HumanPlayer>("human");
            p1 = std::make_shared<MCTSDeepPlayer>(net, itermax, kCpuct);
        } else if (color == 1) {
            p0 = std::make_shared<MCTSDeepPlayer>(net, itermax, kCpuct);
            p1 = std::make_shared<HumanPlayer>("human");
        } else {
            p0 = std::make_shared<MCTSDeepPlayer>(net, itermax, kCpuct);
            p1 = std::make_shared<MCTSDeepPlayer>(net, itermax, kCpuct);
        }
    } else {
        if (color == 0) {
            p0 = std::make_shared<HumanPlayer>("human");
            p1 = std::make_shared<MCTSPurePlayer>(itermax, kCpuct);
        } else if (color == 1) {
            p0 = std::make_shared<MCTSPurePlayer>(itermax, kCpuct);
            p1 = std::make_shared<HumanPlayer>("human");
        } else {
            p0 = std::make_shared<MCTSPurePlayer>(itermax, kCpuct);
            p1 = std::make_shared<MCTSPurePlayer>(itermax, kCpuct);
        }
    }
    play(*p0, *p1, false);
}

void runBenchmark(int64_t verno1, int64_t verno2, int itermax, int round)
{
    ILOG("verno1={}, verno2={}, itermax={}, round={}", verno1, verno2, itermax, round);
    std::shared_ptr<Player> p1, p2;
    if (verno1 > 0) {
        auto net1 = std::make_shared<FIRNet>(verno1);
        p1 = std::make_shared<MCTSDeepPlayer>(net1, itermax, kCpuct);
    } else {
        p1 = std::make_shared<MCTSPurePlayer>(itermax, kCpuct);
    }
    if (verno2 > 0) {
        auto net2 = std::make_shared<FIRNet>(verno2);
        p2 = std::make_shared<MCTSDeepPlayer>(net2, itermax, kCpuct);
    } else {
        p2 = std::make_shared<MCTSPurePlayer>(itermax, kCpuct);
    }
    benchmark(*p1, *p2, round, false);
}

int main(int argc, char** argv)
{
    MY_TRY;
    toolkit::Args args(argc, argv);

    auto& train_args = args.addSub("train", "train model from scatch or checkpoint");
    train_args.positional("verno", po::value<int64_t>(),
                          "verno of checkpoint; 0 for training from scratch", 1);

    auto& play_args = args.addSub("play", "play with trained model");
    play_args.positional("color", po::value<int>(),
                         "who play first hand; 0 for human, 1 for computer, 2 for selfplay", 1);
    play_args.positional("verno", po::value<int64_t>(), "verno of checkpoint; 0 for pure mtsc", 1);
    play_args.positional("itermax", po::value<int>()->default_value(kTrainDeepItermax),
                         "itermax for mcts player", 1);

    auto& benchmark_args = args.addSub("benchmark", "benchmark between two mcts players");
    benchmark_args.positional("verno1", po::value<int64_t>(),
                              "verno of checkpoint; 0 for pure mtsc", 1);
    benchmark_args.positional("verno2", po::value<int64_t>(), "see above", 1);
    benchmark_args.positional("itermax", po::value<int>()->default_value(kTrainDeepItermax),
                              "itermax for mcts player", 1);
    benchmark_args.positional("round", po::value<int>()->default_value(10),
                              "num of benchmark rounds", 1);

    args.parse();

    // run cmd
    if (args.hasSub("train")) {
        auto verno = train_args.get<int64_t>("verno");
        runTrain(verno);
    } else if (args.hasSub("play")) {
        auto color = play_args.get<int>("color");
        auto verno = play_args.get<int64_t>("verno");
        auto itermax = play_args.get<int>("itermax");
        runPlay(color, verno, itermax);
    } else if (args.hasSub("benchmark")) {
        auto verno1 = benchmark_args.get<int64_t>("verno1");
        auto verno2 = benchmark_args.get<int64_t>("verno2");
        auto itermax = benchmark_args.get<int>("itermax");
        auto round = benchmark_args.get<int>("round");
        runBenchmark(verno1, verno2, itermax, round);
    }
    MY_CATCH_RTI;
}