#include <torch/torch.h>
#include "network.h"
#include <iomanip>
#include <filesystem>
#include <moodycamel/blockingconcurrentqueue.h>
#include <future>

void SampleData::flipVerticing()
{
    for (int row = 0; row < kBoardMaxRow; ++row) {
        for (int col = 0; col < kBoardMaxCol / 2; ++col) {
            int a = row * kBoardMaxCol + col;
            int b = row * kBoardMaxCol + kBoardMaxCol - col - 1;
            std::iter_swap(data + a, data + b);
            std::iter_swap(data + kBoardSize + a, data + kBoardSize + b);
            if (kInputFeatureNum > 2) {
                std::iter_swap(data + 2 * kBoardSize + a, data + 2 * kBoardSize + b);
            }
            std::iter_swap(p_label + a, p_label + b);
        }
    }
}

void SampleData::transpose()
{
    for (int row = 0; row < kBoardMaxRow; ++row) {
        for (int col = row + 1; col < kBoardMaxCol; ++col) {
            int a = row * kBoardMaxCol + col;
            int b = col * kBoardMaxCol + row;
            std::iter_swap(data + a, data + b);
            std::iter_swap(data + kBoardSize + a, data + kBoardSize + b);
            if (kInputFeatureNum > 2) {
                std::iter_swap(data + 2 * kBoardSize + a, data + 2 * kBoardSize + b);
            }
            std::iter_swap(p_label + a, p_label + b);
        }
    }
}

std::ostream& operator<<(std::ostream& out, SampleData const& sample)
{
    Move last(kNoMoveYet);
    float first = -1.0f;
    for (int row = 0; row < kBoardMaxRow; ++row) {
        for (int col = 0; col < kBoardMaxCol; ++col) {
            if (sample.data[row * kBoardMaxCol + col] > 0) {
                out << Color::kBlack;
            } else if (sample.data[kBoardSize + row * kBoardMaxCol + col] > 0) {
                out << Color::kWhite;
            } else {
                out << Color::kEmpty;
            }
            if (kInputFeatureNum > 2) {
                if (sample.data[2 * kBoardSize + row * kBoardMaxCol + col] > 0) {
                    last = Move(row, col);
                }
            }
            if (kInputFeatureNum > 3) {
                if (first < 0) {
                    first = sample.data[3 * kBoardSize + row * kBoardMaxCol + col];
                }
            }
        }
        out << "｜";
        for (int col = 0; col < kBoardMaxCol; ++col) {
            out << " " << std::setw(5) << std::fixed << std::setprecision(1)
                << sample.p_label[row * kBoardMaxCol + col] * 100 << "%,";
        }
        out << std::endl;
    }
    out << "↑value=" << sample.v_label[0];
    if (kInputFeatureNum > 2) {
        out << ", last_move=";
        if (last.z() == kNoMoveYet) {
            out << "None";
        } else {
            out << last;
        }
    }
    if (kInputFeatureNum > 3) {
        out << ", fist_hand=" << first;
    }
    out << std::endl;
    return out;
}

std::ostream& operator<<(std::ostream& out, MiniBatch const& batch)
{
    for (int i = 0; i < kTrainBatchSize; ++i) {
        SampleData item;
        std::copy(batch.data + i * kInputFeatureNum * kBoardSize,
                  batch.data + (i + 1) * kInputFeatureNum * kBoardSize, item.data);
        std::copy(batch.p_label + i * kBoardSize, batch.p_label + (i + 1) * kBoardSize,
                  item.p_label);
        std::copy(batch.v_label + i, batch.v_label + (i + 1), item.v_label);
        out << item << std::endl;
    }
    return out;
}

void DataSet::addWithTransform(SampleData* data)
{
    for (int i = 0; i < 4; ++i) {
        data->transpose();
        add(data);
        data->flipVerticing();
        add(data);
    }
}

void DataSet::makeMiniBatch(MiniBatch* batch) const
{
    if (index_ < kTrainBatchSize) {
        MY_THROW("not enough data to make mini batch");
    }
    std::uniform_int_distribution<int> uniform(0, size() - 1);
    for (int i = 0; i < kTrainBatchSize; i++) {
        int c = uniform(g_random_engine);
        SampleData* r = buf_ + c;
        std::copy(std::begin(r->data), std::end(r->data),
                  batch->data + kInputFeatureNum * kBoardSize * i);
        std::copy(std::begin(r->p_label), std::end(r->p_label), batch->p_label + kBoardSize * i);
        std::copy(std::begin(r->v_label), std::end(r->v_label), batch->v_label + i);
    }
}

std::ostream& operator<<(std::ostream& out, DataSet const& ds)
{
    for (int i = 0; i < ds.size(); ++i) {
        out << ds.get(i) << std::endl;
    }
    return out;
}

class ResidualBlockImpl: public torch::nn::Module
{
public:
    explicit ResidualBlockImpl(int channel_n)
    {
        using namespace torch::nn;
        conv_ = register_module(
            "conv", Sequential(Conv2d(Conv2dOptions(channel_n, channel_n, 3).padding(1)),
                               BatchNorm2d(BatchNorm2dOptions(channel_n)), ReLU(),
                               Conv2d(Conv2dOptions(channel_n, channel_n, 3).padding(1)),
                               BatchNorm2d(BatchNorm2dOptions(channel_n))));
    }

    torch::Tensor forward(torch::Tensor input)
    {
        auto x = conv_->forward(input);
        return torch::relu(x + input);
    }

private:
    torch::nn::Sequential conv_ = nullptr;
};

TORCH_MODULE(ResidualBlock);

class FIRModelImpl: public torch::nn::Module
{
public:
    FIRModelImpl(int res_layers, int res_filters)
    {
        using namespace torch::nn;

        mid_conv_ = register_module(
            "mid_conv",
            Sequential(Conv2d(Conv2dOptions(kInputFeatureNum, res_filters, 3).padding(1)),
                       BatchNorm2d(BatchNorm2dOptions(res_filters)), ReLU()));

        mid_res_blks_ = register_module("mid_res_blks", Sequential());
        for (int i = 0; i < res_layers; ++i) {
            mid_res_blks_->push_back(ResidualBlock(res_filters));
        }

        act_ = register_module("act", Sequential(Conv2d(Conv2dOptions(res_filters, 2, 1)),
                                                 BatchNorm2d(BatchNorm2dOptions(2)), ReLU(),
                                                 Flatten(), Linear(2 * kBoardSize, kBoardSize)));

        val_ = register_module("val", Sequential(Conv2d(Conv2dOptions(res_filters, 1, 1)),
                                                 BatchNorm2d(BatchNorm2dOptions(1)), ReLU(),
                                                 Flatten(), Linear(1 * kBoardSize, res_filters),
                                                 ReLU(), Linear(res_filters, 1), Tanh()));
    }

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor input)
    {
        auto x = mid_conv_->forward(input);
        x = mid_res_blks_->forward(x);
        return {act_->forward(x), val_->forward(x)};
    }

private:
    torch::nn::Sequential mid_conv_ = nullptr;
    torch::nn::Sequential mid_res_blks_ = nullptr;
    torch::nn::Sequential act_ = nullptr;
    torch::nn::Sequential val_ = nullptr;
};

TORCH_MODULE(FIRModel);

struct EvalTask
{
    float* state_feature;
    float* value;
    std::promise<MyErrCode> promise;
};

struct FIRNet::Impl
{
    Impl(int64_t verno, bool use_gpu, int max_eval_batch_size)
        : update_cnt(verno),
          device(use_gpu ? torch::kCUDA : torch::kCPU,
                 use_gpu ? torch::cuda::device_count() - 1 : -1),
          max_eval_batch_size(max_eval_batch_size),
          model(kResidualLayers, kResidualFilters),
          optimizer(model->parameters(), torch::optim::AdamOptions().weight_decay(kWeightDecay))
    {
        if (use_gpu) {
            model->to(device);
        }
    }

    int64_t update_cnt;
    torch::Device const device;
    int const max_eval_batch_size;
    FIRModel model;
    torch::optim::Adam optimizer;
    moodycamel::BlockingConcurrentQueue<EvalTask*> eval_queue;
    std::thread eval_thread;
};

FIRNet::FIRNet(int64_t verno, bool use_gpu, int max_eval_batch_size)
    : impl_(new Impl(verno, use_gpu, max_eval_batch_size))
{
    if (impl_->update_cnt > 0) {
        loadModel();
    }
    impl_->eval_thread = std::thread([this]() { evalThreadEntry(); });
}

FIRNet::~FIRNet()
{
    impl_->eval_queue.enqueue(nullptr);
    impl_->eval_thread.join();
    delete impl_;
}

int64_t FIRNet::verno() const { return impl_->update_cnt; }

void FIRNet::setLearningRate(float lr)
{
    for (auto& param_group: impl_->optimizer.param_groups()) {
        if (param_group.has_options()) {
            auto& options = dynamic_cast<torch::optim::AdamOptions&>(param_group.options());
            options.lr(lr);
        }
    }
}

std::string FIRNet::saveModelPath() const
{
    std::string fp = FSTR("FIR-{}x{}-r{}c{}@{}.pt", kBoardMaxRow, kBoardMaxCol, kResidualLayers,
                          kResidualFilters, impl_->update_cnt);
    return (toolkit::getTempDir() / fp).string();
}

void FIRNet::loadModel()
{
    auto fp = saveModelPath();
    ILOG("loading checkpoint from {}", fp);
    if (!std::filesystem::exists(fp)) {
        MY_THROW("file not exist: {}", fp);
    }
    torch::serialize::InputArchive input_archive;
    input_archive.load_from(fp, impl_->device);
    impl_->model->load(input_archive);
}

void FIRNet::saveModel()
{
    auto fp = saveModelPath();
    ILOG("saving checkpoint into {}", fp);
    torch::serialize::OutputArchive output_archive;
    impl_->model->save(output_archive);
    output_archive.save_to(fp);
}

void FIRNet::train(bool on)
{
    if (on) {
        impl_->model->train();
    } else {
        impl_->model->eval();
    }
}

void FIRNet::evalThreadEntry()
{
    std::vector<EvalTask*> batch_task;
    batch_task.reserve(impl_->max_eval_batch_size);
    torch::Tensor batch_buf;
    if (impl_->device.is_cuda()) {
        batch_buf =
            torch::empty({impl_->max_eval_batch_size, kInputFeatureNum, kBoardMaxRow, kBoardMaxCol},
                         torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));
    }
    auto batch_data =
        torch::empty({impl_->max_eval_batch_size, kInputFeatureNum, kBoardMaxRow, kBoardMaxCol},
                     torch::TensorOptions().dtype(torch::kFloat32).device(impl_->device));

    while (true) {
        EvalTask* curr_task = nullptr;
        bool has_task = impl_->eval_queue.try_dequeue(curr_task);
        if (has_task) {
            if (curr_task == nullptr) {
                break;
            }

            int idx = batch_task.size();
            batch_task.push_back(curr_task);
            float* data_ptr =
                (impl_->device.is_cuda() ? batch_buf : batch_data).mutable_data_ptr<float>();
            memcpy(data_ptr + idx * kInputFeatureNum * kBoardSize, curr_task->state_feature,
                   kInputFeatureNum * kBoardSize * sizeof(float));

            if (idx + 1 < impl_->max_eval_batch_size) {
                continue;
            }
        } else {
            if (batch_task.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
        }

        int idx = 0;
        int task_size = batch_task.size();
        try {
            torch::NoGradGuard no_grad;
            if (impl_->device.is_cuda()) {
                batch_data.slice(0, 0, task_size).copy_(batch_buf.slice(0, 0, task_size));
            }
            auto&& [x_act, x_val] = impl_->model(batch_data.slice(0, 0, task_size));
            x_act = torch::softmax(x_act, 1);
            for (; idx < task_size; ++idx) {
                auto act_prob = torch::from_blob(
                    batch_task[idx]->state_feature, {kBoardSize}, [](void* buf) {},
                    torch::kFloat32);
                act_prob.copy_(x_act[idx]);
                batch_task[idx]->value[0] = x_val[idx][0].item<float>();
                batch_task[idx]->promise.set_value(MyErrCode::kOk);
            }
        } catch (std::exception const& e) {
            ILOG("eval failed: {}", e.what());
            for (; idx < task_size; ++idx) {
                batch_task[idx]->promise.set_value(MyErrCode::kFailed);
            }
        }

        batch_task.clear();
    }
    DLOG("eval thread exit");
}

MyErrCode FIRNet::eval(float state_feature[kInputFeatureNum * kBoardSize], float value[1],
                       bool batched)
{
    if (batched) {
        EvalTask task;
        task.state_feature = state_feature;
        task.value = value;
        auto future = task.promise.get_future();
        impl_->eval_queue.enqueue(&task);
        return future.get();
    } else {
        torch::NoGradGuard no_grad;
        auto data = torch::from_blob(
            state_feature, {1, kInputFeatureNum, kBoardMaxRow, kBoardMaxCol}, [](void* buf) {},
            torch::kFloat32);
        auto&& [x_act, x_val] = impl_->model(data.to(impl_->device));
        x_act = torch::softmax(x_act, 1).cpu();
        auto act_prob = x_act[0];
        for (int i = 0; i < kBoardSize; ++i) {
            state_feature[i] = act_prob[i].item<float>();
        }
        value[0] = x_val[0][0].item<float>();
        return MyErrCode::kOk;
    }
}

float FIRNet::step(MiniBatch* batch, TrainMeta& meta)
{
    auto data = torch::from_blob(
        batch->data, {kTrainBatchSize, kInputFeatureNum, kBoardMaxRow, kBoardMaxCol},
        [](void* buf) {}, torch::kFloat32);
    auto plc_label = torch::from_blob(
        batch->p_label, {kTrainBatchSize, kBoardSize}, [](void* buf) {}, torch::kFloat32);
    auto val_label = torch::from_blob(
        batch->p_label, {kTrainBatchSize, 1}, [](void* buf) {}, torch::kFloat32);

    auto&& [x_act_logits, x_val] = impl_->model(data);
    auto value_loss = torch::mse_loss(x_val, val_label);
    auto policy_loss = torch::cross_entropy_loss(x_act_logits, plc_label);
    auto loss = value_loss + policy_loss;

    setLearningRate(meta.learning_rate);
    impl_->optimizer.zero_grad();
    loss.backward();
    impl_->optimizer.step();

    ++impl_->update_cnt;
    return loss.item<float>();
}