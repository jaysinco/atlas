#include "train.h"
#include "mcts.h"
#include <chrono>
#include <numeric>
#include <thread>

static void selfPlayOne(Player& player, DataSet& dataset, TrainingMeta& train_meta)
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
        act_meta.temperature = step <= kExploreStep ? 1.0f : 1e-3;
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

static void selfPlay(Player& player, DataSet& dataset, TrainingMeta& train_meta)
{
    for (int i = 0; i < 1; ++i) {
        selfPlayOne(player, dataset, train_meta);
    }
}

static bool triggerTimer(std::chrono::time_point<std::chrono::system_clock>& last, int per_minute)
{
    auto now = std::chrono::system_clock::now();
    auto sec = std::chrono::duration_cast<std::chrono::seconds>(now - last).count();
    if (sec >= per_minute * 60) {
        last = now;
        return true;
    }
    return false;
}

void train(std::shared_ptr<FIRNet> net, TrainingMeta& meta)
{
    ILOG("start training...");

    auto last_log = std::chrono::system_clock::now();
    auto last_save = std::chrono::system_clock::now();
    auto last_benchmark = std::chrono::system_clock::now();

    DataSet dataset;

    int test_itermax = kTestPureItermax;
    auto test_player = MCTSPurePlayer(test_itermax, kTrainCpuct, kTrainThreadNum);
    auto net_player = MCTSDeepPlayer(net, kTrainDeepItermax, kTrainCpuct, kTrainThreadNum);

    for (;;) {
        selfPlay(net_player, dataset, meta);
        if (dataset.total() > kBatchSize) {
            for (int epoch = 0; epoch < kEpochPerGame; ++epoch) {
                auto batch = new MiniBatch();
                dataset.makeMiniBatch(batch);
                float loss = net->train(batch, meta);
                if (triggerTimer(last_log, kMinutePerLog)) {
                    int avg_turn = static_cast<float>(meta.selfplay_turns) / meta.selfplay_rounds;
                    ILOG("loss={}, dataset_total={}, update_cnt={}, avg_turn={}, game_cnt={}", loss,
                         dataset.total(), net->verno(), avg_turn, meta.selfplay_rounds);
                }
                delete batch;
            }
        }
        if (triggerTimer(last_benchmark, kMinutePerBenchmark)) {
            float lose_prob = 1 - benchmark(net_player, test_player, 10);
            ILOG("benchmark 10 games against {}, lose_prob={}", test_player.name(), lose_prob);
            if (lose_prob < 1e-3 && test_itermax < 15 * kTestPureItermax) {
                test_itermax += kTestPureItermax;
                test_player.setItermax(test_itermax);
            }
        }
        if (triggerTimer(last_save, kMinutePerSave)) {
            net->save();
        }
    }
}