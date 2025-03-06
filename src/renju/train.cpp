#include "train.h"
#include "mcts.h"
#include <chrono>
#include <numeric>

int selfplay(std::shared_ptr<FIRNet> net, DataSet& dataset, int itermax)
{
    State game;
    std::vector<SampleData> record;
    MCTSNode* root = new MCTSNode(nullptr, 1.0f);
    float ind = -1.0f;
    int step = 0;
    while (!game.over()) {
        ++step;
        ind *= -1.0f;
        SampleData one_step;
        *one_step.v_label = ind;
        game.fillFeatureArray(one_step.data);
        MCTSDeepPlayer::think(itermax, kCpuct, game, net, root, true);
        Move act = root->actByProb(one_step.p_label, step <= kExploreStep ? 1.0f : 1e-3);
        record.push_back(one_step);
        game.next(act);
        auto temp = root->cut(act);
        delete root;
        root = temp;
        if (kDebugTrainData) {
            std::cout << game << std::endl;
        }
    }
    delete root;
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
        dataset.pushWithTransform(&step);
    }
    return step;
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

void train(std::shared_ptr<FIRNet> net)
{
    ILOG("start training...");

    auto last_log = std::chrono::system_clock::now();
    auto last_save = std::chrono::system_clock::now();
    auto last_benchmark = std::chrono::system_clock::now();

    int64_t game_cnt = 0;
    std::vector<int> steps_buf;
    DataSet dataset;

    int test_itermax = kTestPureItermax;
    auto test_player = MCTSPurePlayer(test_itermax, kCpuct);
    auto net_player = MCTSDeepPlayer(net, kTrainDeepItermax, kCpuct);

    for (;;) {
        ++game_cnt;
        int step = selfplay(net, dataset, kTrainDeepItermax);
        steps_buf.push_back(step);
        if (dataset.total() > kBatchSize) {
            for (int epoch = 0; epoch < kEpochPerGame; ++epoch) {
                auto batch = new MiniBatch();
                dataset.makeMiniBatch(batch);
                float loss = net->trainStep(batch);
                if (triggerTimer(last_log, kMinutePerLog)) {
                    int avg_turn =
                        std::round(std::accumulate(steps_buf.begin(), steps_buf.end(), 0) /
                                   static_cast<float>(steps_buf.size()));
                    steps_buf.clear();
                    ILOG("loss={}, dataset_total={}, update_cnt={}, avg_turn={}, game_cnt={}", loss,
                         dataset.total(), net->verno(), avg_turn, game_cnt);
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
            net->saveParam();
        }
    }
}