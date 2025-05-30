#include "common.h"

namespace
{

struct TensorDataset: public torch::data::Dataset<TensorDataset>
{
    TensorDataset(torch::Tensor data, torch::Tensor target): data(data), target(target) {}

    ExampleType get(size_t index) override { return {data[index], target[index]}; }

    c10::optional<size_t> size() const override { return data.size(0); }

    torch::Tensor data;
    torch::Tensor target;
};

}  // namespace

MyErrCode linearRegression(int argc, char** argv)
{
    toolkit::Args args(argc, argv);
    args.parse();

    // torch::manual_seed(1);

    std::vector<float> w_v{2.0, -3.4, 5.8, 4.2, 9.9};
    int const n_feature = w_v.size();
    float const b = 4.2;
    float const lr = 0.03;
    int const total_size = 1000;
    int const batch_size = 16;
    int const n_epoch = 3;

    auto w = torch::from_blob(w_v.data(), {n_feature, 1}, torch::kFloat32);
    auto x = torch::normal(0, 1, {total_size, n_feature});
    auto y = torch::matmul(x, w) + b;
    y += torch::normal(0, 0.01, y.sizes());

    auto dataset = TensorDataset{x, y}.map(torch::data::transforms::Stack<>());
    auto loader = torch::data::make_data_loader<torch::data::samplers::DistributedRandomSampler>(
        std::move(dataset), batch_size);

    auto net = torch::nn::Sequential({
        {"layer0", torch::nn::Linear(n_feature, 1)},
    });
    auto layer0 = net[0]->as<torch::nn::Linear>();
    layer0->weight.detach().normal_(0, 0.01);
    layer0->bias.detach().fill_(0);
    auto loss = torch::nn::MSELoss();
    auto optimizer = torch::optim::SGD(net->parameters(), lr);

    MY_TIMER_BEGIN(INFO, "process")
    for (int i = 0; i < n_epoch; ++i) {
        for (auto& batch: *loader) {
            auto l = loss(net->forward(batch.data), batch.target);
            optimizer.zero_grad();
            l.backward();
            optimizer.step();
        }
        ILOG("epoch {}, loss={}", i, loss(net->forward(x), y).item<double>());
    }
    MY_TIMER_END()

    ILOG("w = \n{}", layer0->weight);
    ILOG("b = \n{}", layer0->bias);

    return MyErrCode::kOk;
}
