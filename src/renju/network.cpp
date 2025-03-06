#include "network.h"
#include <iomanip>
#include <filesystem>
#include <torch/torch.h>

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
                    assert(last.z() == kNoMoveYet);
                    last = Move(row, col);
                }
            }
            if (kInputFeatureNum > 3) {
                if (first < 0) {
                    first = sample.data[3 * kBoardSize + row * kBoardMaxCol + col];
                } else {
                    assert(first == sample.data[3 * kBoardSize + row * kBoardMaxCol + col]);
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
    for (int i = 0; i < kBatchSize; ++i) {
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

void DataSet::pushWithTransform(SampleData* data)
{
    for (int i = 0; i < 4; ++i) {
        data->transpose();
        pushBack(data);
        data->flipVerticing();
        pushBack(data);
    }
}

void DataSet::makeMiniBatch(MiniBatch* batch) const
{
    assert(index_ > kBatchSize);
    std::uniform_int_distribution<int> uniform(0, size() - 1);
    for (int i = 0; i < kBatchSize; i++) {
        int c = uniform(g_random_engine);
        SampleData* r = buf_ + c;
        std::copy(std::begin(r->data), std::end(r->data),
                  batch->data + kInputFeatureNum * kBoardSize * i);
        std::copy(std::begin(r->p_label), std::end(r->p_label), batch->p_label + kBoardSize * i);
        std::copy(std::begin(r->v_label), std::end(r->v_label), batch->v_label + i);
    }
}

std::ostream& operator<<(std::ostream& out, DataSet const& set)
{
    for (int i = 0; i < set.size(); ++i) {
        out << set.get(i) << std::endl;
    }
    return out;
}

class ResidualBlock: public torch::nn::Module
{
public:
    explicit ResidualBlock(int channel_n);
    torch::Tensor forward(torch::Tensor input);

private:
    torch::nn::Sequential conv1_;
    torch::nn::Sequential conv2_;
};

ResidualBlock::ResidualBlock(int channel_n)
    : conv1_(torch::nn::Conv2d(torch::nn::Conv2dOptions(channel_n, channel_n, 3).padding(1)),
             torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(channel_n)), torch::nn::ReLU()),
      conv2_(torch::nn::Conv2d(torch::nn::Conv2dOptions(channel_n, channel_n, 3).padding(1)),
             torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(channel_n)))
{
    register_module("conv1", conv1_);
    register_module("conv2", conv2_);
}

torch::Tensor ResidualBlock::forward(torch::Tensor input)
{
    auto x = conv1_->forward(input);
    x = conv2_->forward(x);
    x = x + input;
    return torch::relu(x);
}

class FIRNetModule: public torch::nn::Module
{
public:
    explicit FIRNetModule(int res_layers, int res_filters);
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor input);

private:
    torch::nn::Conv2d mid_conv_;
    torch::nn::ModuleList mid_res_blks_;
    torch::nn::Sequential act_;
    torch::nn::Sequential val_;
};

FIRNetModule::FIRNetModule(int res_layers, int res_filters)
    : mid_conv_(torch::nn::Conv2dOptions(kInputFeatureNum, res_filters, 3).padding(1)),
      act_(torch::nn::Conv2d(torch::nn::Conv2dOptions(res_filters, 2, 1)),
           torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(2)), torch::nn::ReLU(),
           torch::nn::Flatten(), torch::nn::Linear(2 * kBoardSize, kBoardSize)),
      val_(torch::nn::Conv2d(torch::nn::Conv2dOptions(res_filters, 1, 1)),
           torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(1)), torch::nn::ReLU(),
           torch::nn::Flatten(), torch::nn::Linear(1 * kBoardSize, res_filters), torch::nn::ReLU(),
           torch::nn::Linear(res_filters, 1), torch::nn::Tanh())
{
    for (int i = 0; i < res_layers; ++i) {
        mid_res_blks_->push_back(ResidualBlock(res_filters));
    }
    register_module("mid_conv", mid_conv_);
    register_module("mid_res_blks", mid_res_blks_);
    register_module("act", act_);
    register_module("val", val_);
}

std::pair<torch::Tensor, torch::Tensor> FIRNetModule::forward(torch::Tensor input)
{
    auto x = torch::relu(mid_conv_->forward(input));
    for (auto& r: *mid_res_blks_) {
        x = r->as<ResidualBlock>()->forward(x);
    }
    auto x_act = act_->forward(x);
    x_act = torch::softmax(x_act, 1);
    auto x_val = val_->forward(x);
    return {x_act, x_val};
}

struct FIRNet::Impl
{
    explicit Impl(int64_t verno)
        : module(kResidualLayers, kResidualFilters),
          update_cnt(verno),
          optimizer(module.parameters(), kWeightDecay)
    {
    }

    FIRNetModule module;
    int64_t update_cnt;
    torch::optim::Adam optimizer;
};

FIRNet::FIRNet(int64_t verno): impl_(new Impl(verno))
{
    if (impl_->update_cnt > 0) {
        loadParam();
    }
    this->setLR(calcInitLR());
}

int64_t FIRNet::verno() const { return impl_->update_cnt; }

float FIRNet::calcInitLR() const
{
    float multiplier;
    if (impl_->update_cnt < kDropStepLR1) {
        multiplier = 1.0f;
    } else if (impl_->update_cnt >= kDropStepLR1 && impl_->update_cnt < kDropStepLR2) {
        multiplier = 1e-1;
    } else if (impl_->update_cnt >= kDropStepLR2 && impl_->update_cnt < kDropStepLR3) {
        multiplier = 1e-2;
    } else {
        multiplier = 1e-3;
    }
    float lr = kInitLearningRate * multiplier;
    ILOG("init learning_rate={}", lr);
    return lr;
}

void FIRNet::adjustLR()
{
    float multiplier = 1.0f;
    switch (impl_->update_cnt) {
        case kDropStepLR1:
            multiplier = 1e-1;
            break;
        case kDropStepLR2:
            multiplier = 1e-2;
            break;
        case kDropStepLR3:
            multiplier = 1e-3;
            break;
    }
    if (multiplier < 1.0f) {
        float lr = kInitLearningRate * multiplier;
        this->setLR(lr);
        ILOG("adjusted learning_rate={}", lr);
    }
}

void FIRNet::setLR(float lr)
{
    for (auto& group: impl_->optimizer.param_groups()) {
        if (group.has_options()) {
            auto& options = static_cast<torch::optim::AdamOptions&>(group.options());
            options.lr(lr);
        }
    }
}

FIRNet::~FIRNet() { delete impl_; };

std::string FIRNet::makeParamFileName() const
{
    std::ostringstream filename;
    filename << "FIR-" << kBoardMaxRow << "x" << kBoardMaxCol << "-r" << kResidualLayers << "c"
             << kResidualFilters << "@" << impl_->update_cnt << ".pt";
    return utils::ws2s(utils::getExeDir()) + "/" + filename.str();
}

void FIRNet::loadParam()
{
    auto file_name = makeParamFileName();
    ILOG("loading parameters from {}", file_name);
    if (!std::filesystem::exists(file_name)) {
        MY_THROW("file not exist: {}", file_name);
    }
    torch::serialize::InputArchive input_archive;
    input_archive.load_from(file_name);
    impl_->module.load(input_archive);
}

void FIRNet::saveParam()
{
    auto file_name = makeParamFileName();
    ILOG("saving parameters into {}", file_name);
    torch::serialize::OutputArchive output_archive;
    impl_->module.save(output_archive);
    output_archive.save_to(file_name);
}

static void mappingData(int id, float data[kInputFeatureNum * kBoardSize])
{
    int n = 0;
    while (true) {
        if (n == id) {
            break;
        }
        // transpose
        for (int row = 0; row < kBoardMaxRow; ++row) {
            for (int col = row + 1; col < kBoardMaxCol; ++col) {
                int a = row * kBoardMaxCol + col;
                int b = col * kBoardMaxCol + row;
                std::iter_swap(data + a, data + b);
                std::iter_swap(data + kBoardSize + a, data + kBoardSize + b);
                if (kInputFeatureNum > 2) {
                    std::iter_swap(data + 2 * kBoardSize + a, data + 2 * kBoardSize + b);
                }
            }
        }
        ++n;
        if (n == id) {
            break;
        }
        // flip_verticing
        for (int row = 0; row < kBoardMaxRow; ++row) {
            for (int col = 0; col < kBoardMaxCol / 2; ++col) {
                int a = row * kBoardMaxCol + col;
                int b = row * kBoardMaxCol + kBoardMaxCol - col - 1;
                std::iter_swap(data + a, data + b);
                std::iter_swap(data + kBoardSize + a, data + kBoardSize + b);
                if (kInputFeatureNum > 2) {
                    std::iter_swap(data + 2 * kBoardSize + a, data + 2 * kBoardSize + b);
                }
            }
        }
        ++n;
    }
}

static Move mappingMove(int id, Move mv)
{
    int n = 0, r, c;
    while (true) {
        if (n == id) {
            break;
        }
        // transpose
        r = mv.c(), c = mv.r();
        mv = Move(r, c);
        ++n;
        if (n == id) {
            break;
        }
        // flip_verticing
        r = mv.r(), c = kBoardMaxCol - mv.c() - 1;
        mv = Move(r, c);
        ++n;
    }
    return mv;
}

void FIRNet::evalState(State const& state, float value[1],
                       std::vector<std::pair<Move, float>>& net_move_priors)
{
    torch::NoGradGuard no_grad;
    impl_->module.eval();
    float buf[kInputFeatureNum * kBoardSize] = {0.0f};
    state.fillFeatureArray(buf);
    std::uniform_int_distribution<int> uniform(0, 7);
    int transform_id = uniform(g_random_engine);
    mappingData(transform_id, buf);
    auto data = torch::from_blob(
        buf, {1, kInputFeatureNum, kBoardMaxRow, kBoardMaxCol}, [](void* buf) {}, torch::kFloat32);
    auto&& [x_act, x_val] = impl_->module.forward(data);
    float priors_sum = 0.0f;
    for (auto const mv: state.getOptions()) {
        Move mapped = mappingMove(transform_id, mv);
        float prior = x_act[0][mapped.z()].item<float>();
        net_move_priors.emplace_back(mv, prior);
        priors_sum += prior;
    }
    if (priors_sum < 1e-8) {
        ILOG("wield policy prob, lr might be too large: sum={}, available_move_n={}", priors_sum,
             net_move_priors.size());
        for (auto& item: net_move_priors) {
            item.second = 1.0f / static_cast<float>(net_move_priors.size());
        }
    } else {
        for (auto& item: net_move_priors) {
            item.second /= priors_sum;
        }
    }
    value[0] = x_val[0][0].item<float>();
}

float FIRNet::trainStep(MiniBatch* batch)
{
    impl_->module.train();
    auto data = torch::from_blob(
        batch->data, {kBatchSize, kInputFeatureNum, kBoardMaxRow, kBoardMaxCol}, [](void* buf) {},
        torch::kFloat32);
    auto plc_label = torch::from_blob(
        batch->p_label, {kBatchSize, kBoardSize}, [](void* buf) {}, torch::kFloat32);
    auto val_label = torch::from_blob(
        batch->p_label, {kBatchSize, 1}, [](void* buf) {}, torch::kFloat32);
    auto&& [x_act, x_val] = impl_->module.forward(data);
    auto value_loss = torch::mse_loss(x_val, val_label);
    auto policy_loss = -torch::mean(torch::sum(plc_label * torch::log(x_act), 1));
    auto loss = value_loss + policy_loss;
    impl_->optimizer.zero_grad();
    loss.backward();
    impl_->optimizer.step();
    adjustLR();
    ++impl_->update_cnt;
    return loss.item<float>();
}