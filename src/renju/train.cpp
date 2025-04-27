#include "train.h"
#include "mcts.h"
#include <chrono>
#include <numeric>
#include <thread>

static void selfPlayOne(Player& player, DataSet& dataset, TrainMeta& train_meta)
{
    State game;
    std::vector<SampleData> record;
    float ind = -1.0f;
    int step = 0;
    player.reset();
    while (!game.over()) {
        ++step;
        ind *= -1.0f;
        SampleData one_step;
        *one_step.v_label = ind;
        game.fillFeatureArray(one_step.data);
        ActionMeta act_meta;
        act_meta.temperature = step <= train_meta.explore_step ? 1.0f : 1e-3;
        act_meta.add_noise_prior = true;
        act_meta.move_priors = one_step.p_label;
        Move act = player.play(game, act_meta);
        record.push_back(one_step);
        game.next(act);
        if (kDebugTrainData) {
            std::cout << game << std::endl;
        }
    }
    if (game.getWinner() != Color::kEmpty) {
        if (ind < 0) {
            for (auto& step: record) {
                (*step.v_label) *= -1;
            }
        }
    } else {
        for (auto& step: record) {
            (*step.v_label) = 0.0f;
        }
    }
    for (auto& step: record) {
        if (kDebugTrainData) {
            std::cout << step << std::endl;
        }
        dataset.addWithTransform(&step);
    }
}

static void selfPlay(Player& player, DataSet& dataset, TrainMeta& train_meta)
{
    for (int i = 0; i < train_meta.selfplay_turns_per_train; ++i) {
        selfPlayOne(player, dataset, train_meta);
    }
}

void train(std::shared_ptr<FIRNet> net, TrainMeta& meta)
{
    ILOG("start training...");

    auto last_log = std::chrono::system_clock::now();
    auto last_save = std::chrono::system_clock::now();
    auto last_benchmark = std::chrono::system_clock::now();

    DataSet dataset;

    int test_itermax = meta.itermax;
    auto test_player = MCTSPurePlayer(test_itermax, meta.c_puct, meta.thread_num);
    auto net_player = MCTSDeepPlayer(net, meta.itermax, meta.c_puct, meta.thread_num);

    for (;;) {
        selfPlay(net_player, dataset, meta);
        if (dataset.total() > kTrainBatchSize) {
            for (int epoch = 0; epoch < 1; ++epoch) {
                auto batch = new MiniBatch();
                dataset.makeMiniBatch(batch);
                float loss = net->step(batch, meta);
                ILOG("loss={}, dataset_total={}, update_cnt={}, avg_turn={}", loss, dataset.total(),
                     net->verno(), meta.selfplay_avg_rounds);
                delete batch;
            }
        }
        float lose_prob = 1 - benchmark(net_player, test_player, 10);
        ILOG("benchmark 10 games against {}, lose_prob={}", test_player.name(), lose_prob);
        net->saveModel();
    }
}