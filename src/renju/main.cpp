#include "toolkit/args.h"
#include "mcts.h"
#include "train.h"
#include <iostream>
#include <regex>
#include <optional>

std::random_device g_random_device;
std::mt19937 g_random_engine(g_random_device());

std::shared_ptr<Player> createPlayer(std::string const& setup, bool use_gpu)
{
    if (setup == "human") {
        return std::make_shared<HumanPlayer>("human");
    }
    static std::regex pattern(R"(i(\d+)u([\d.]+)t(\d+)(?:@(\d+))?)");
    int itermax;
    float c_puct;
    int nthreads;
    std::optional<int64_t> verno;
    std::smatch matches;
    if (std::regex_match(setup, matches, pattern)) {
        itermax = std::stoi(matches[1].str());
        c_puct = std::stof(matches[2].str());
        nthreads = std::stoi(matches[3].str());
        if (matches.size() > 4 && matches[4].matched) {
            verno = std::stoi(matches[4].str());
        }

    } else {
        MY_THROW("failed to parse player setup: {}", setup);
    }
    ILOG("itermax={}, c_puct={}, nthreads={}, use_gpu={}{}", itermax, c_puct, nthreads, use_gpu,
         verno ? FSTR(", verno={}", *verno) : "");
    std::shared_ptr<Player> player;
    if (verno) {
        auto net = std::make_shared<FIRNet>(*verno, use_gpu, (nthreads + 1) / 2);
        player = std::make_shared<MCTSDeepPlayer>(net, itermax, c_puct, nthreads);
    } else {
        player = std::make_shared<MCTSPurePlayer>(itermax, c_puct, nthreads);
    }
    return player;
}

void runTrain(int64_t verno, bool use_gpu, TrainMeta& meta)
{
    ILOG("verno={}", verno);
    auto net = std::make_shared<FIRNet>(verno, use_gpu, (meta.thread_num + 1) / 2);
    train(net, meta);
}

void runPlay(std::string const& setup1, std::string const& setup2, bool use_gpu)
{
    ILOG("player1='{}', player2='{}'", setup1, setup2);
    std::shared_ptr<Player> player1 = createPlayer(setup1, use_gpu);
    std::shared_ptr<Player> player2 = setup1 == setup2 ? player1 : createPlayer(setup2, use_gpu);
    play(*player1, *player2, false);
}

void runBenchmark(std::string const& setup1, std::string const& setup2, int round, bool use_gpu)
{
    ILOG("player1='{}', player2='{}', round={}", setup1, setup2, round);
    std::shared_ptr<Player> player1 = createPlayer(setup1, use_gpu);
    std::shared_ptr<Player> player2 = createPlayer(setup2, use_gpu);
    benchmark(*player1, *player2, round, false);
}

int main(int argc, char** argv)
{
    MY_TRY;
    toolkit::Args args(argc, argv);
    args.optional("gpu,g", po::bool_switch()->default_value(false), "use gpu");

    auto& train_args = args.addSub("train", "train model from scatch or checkpoint");
    train_args.optional("thread,t", po::value<int>()->default_value(16), "mcts thread num");
    train_args.optional("itermax,i", po::value<int>()->default_value(1200), "mcts itermax");
    train_args.positional("verno", po::value<int64_t>(),
                          "verno of checkpoint, 0 to train from scratch", 1);

    auto& play_args = args.addSub("play", "play game with ai or selfplay");
    play_args.positional("player1", po::value<std::string>(),
                         "player setup like i1000u1.25t8@1, human", 1);
    play_args.positional("player2", po::value<std::string>()->default_value(""),
                         "see above, omit if selfplay", 1);

    auto& mark_args = args.addSub("mark", "benchmark between two players");
    mark_args.positional("player1", po::value<std::string>(),
                         "player setup like i1000u1.25t8@1, human", 1);
    mark_args.positional("player2", po::value<std::string>(), "see above", 1);
    mark_args.positional("round", po::value<int>()->default_value(10), "num of benchmark rounds",
                         1);

    args.parse();

    // run cmd
    auto use_gpu = args.get<bool>("gpu");
    if (args.hasSub("train")) {
        auto verno = train_args.get<int64_t>("verno");
        TrainMeta meta;
        meta.itermax = train_args.get<int>("itermax");
        meta.thread_num = train_args.get<int>("thread");
        runTrain(verno, use_gpu, meta);
    } else if (args.hasSub("play")) {
        auto player1 = play_args.get<std::string>("player1");
        auto player2 = play_args.get<std::string>("player2");
        if (player2 == "") {
            player2 = player1;
        }
        runPlay(player1, player2, use_gpu);
    } else if (args.hasSub("mark")) {
        auto player1 = mark_args.get<std::string>("player1");
        auto player2 = mark_args.get<std::string>("player2");
        auto round = mark_args.get<int>("round");
        runBenchmark(player1, player2, round, use_gpu);
    }
    MY_CATCH_RTI;
}