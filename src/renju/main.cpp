#include "toolkit/args.h"
#include "mcts.h"
#include "train.h"
#include <iostream>
#include <regex>
#include <optional>

std::random_device g_random_device;
std::mt19937 g_random_engine(g_random_device());

std::shared_ptr<Player> createPlayer(std::string const& setup)
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
    ILOG("itermax={}, c_puct={}, nthreads={}{}", itermax, c_puct, nthreads,
         verno ? FSTR(", verno={}", *verno) : "");
    std::shared_ptr<Player> player;
    if (verno) {
        auto net = std::make_shared<FIRNet>(*verno);
        player = std::make_shared<MCTSDeepPlayer>(net, itermax, c_puct, nthreads);
    } else {
        player = std::make_shared<MCTSPurePlayer>(itermax, c_puct, nthreads);
    }
    return player;
}

void runTrain(int64_t verno)
{
    ILOG("verno={}", verno);
    auto net = std::make_shared<FIRNet>(verno);
    train(net);
}

void runPlay(std::string const& setup1, std::string const& setup2)
{
    ILOG("player1='{}', player2='{}'", setup1, setup2);
    std::shared_ptr<Player> player1 = createPlayer(setup1);
    std::shared_ptr<Player> player2 = setup1 == setup2 ? player1 : createPlayer(setup2);
    play(*player1, *player2, false);
}

void runBenchmark(std::string const& setup1, std::string const& setup2, int round)
{
    ILOG("player1='{}', player2='{}', round={}", setup1, setup2, round);
    std::shared_ptr<Player> player1 = createPlayer(setup1);
    std::shared_ptr<Player> player2 = createPlayer(setup2);
    benchmark(*player1, *player2, round, false);
}

int main(int argc, char** argv)
{
    MY_TRY;
    toolkit::Args args(argc, argv);

    auto& train_args = args.addSub("train", "train model from scatch or checkpoint");
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
    if (args.hasSub("train")) {
        auto verno = train_args.get<int64_t>("verno");
        runTrain(verno);
    } else if (args.hasSub("play")) {
        auto player1 = play_args.get<std::string>("player1");
        auto player2 = play_args.get<std::string>("player2");
        if (player2 == "") {
            player2 = player1;
        }
        runPlay(player1, player2);
    } else if (args.hasSub("mark")) {
        auto player1 = mark_args.get<std::string>("player1");
        auto player2 = mark_args.get<std::string>("player2");
        auto round = mark_args.get<int>("round");
        runBenchmark(player1, player2, round);
    }
    MY_CATCH_RTI;
}