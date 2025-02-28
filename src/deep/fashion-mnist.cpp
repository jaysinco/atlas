#include "common.h"
#include <filesystem>
#include "toolkit/toolkit.h"

class FashionMnistDataset: public torch::data::Dataset<FashionMnistDataset>
{
public:
    FashionMnistDataset(std::filesystem::path const& data_root, bool is_train)
    {
        auto image_path =
            data_root / (is_train ? L"train-images-idx3-ubyte" : L"t10k-images-idx3-ubyte");
        auto label_path =
            data_root / (is_train ? L"train-labels-idx1-ubyte" : L"t10k-labels-idx1-ubyte");

        if (!std::filesystem::exists(image_path)) {
            MY_THROW("file not found: {}", image_path.string());
        }
        if (!std::filesystem::exists(label_path)) {
            MY_THROW("file not found: {}", label_path.string());
        }

        data_ = readImages(image_path);
        target_ = readLabels(label_path);
    }

    ExampleType get(size_t index) override { return {data_[index], target_[index]}; }

    c10::optional<size_t> size() const override { return data_.size(0); }

private:
    static int32_t reverseInt(int32_t i)
    {
        uint8_t c1, c2, c3, c4;
        c1 = i & 255;
        c2 = (i >> 8) & 255;
        c3 = (i >> 16) & 255;
        c4 = (i >> 24) & 255;
        return (static_cast<int32_t>(c1) << 24) + (static_cast<int32_t>(c2) << 16) +
               (static_cast<int32_t>(c3) << 8) + c4;
    }

    static torch::Tensor readImages(std::filesystem::path const& fp)
    {
        std::ifstream file(fp, std::ios::in | std::ios::binary);
        int32_t header[4];
        for (int i = 0; i < 4; ++i) {
            file.read(reinterpret_cast<char*>(&header[i]), sizeof(int32_t));
            header[i] = reverseInt(header[i]);
        }
        int32_t n = header[1];
        int32_t nr = header[2];
        int32_t nc = header[3];
        ILOG("read {}x{}x{} images from {}", n, nr, nc, fp.string());
        size_t numel = n * nr * nc;
        auto buf = new uint8_t[numel];
        for (int i = 0; i < n; ++i) {
            file.read(reinterpret_cast<char*>(buf + nr * nc * i), nr * nc * sizeof(uint8_t));
        }
        return torch::from_blob(
            buf, {n, 1, nr, nc}, [](void* buf) { delete[] static_cast<uint8_t*>(buf); },
            torch::kUInt8);
    }

    static torch::Tensor readLabels(std::filesystem::path const& fp)
    {
        std::ifstream file(fp, std::ios::in | std::ios::binary);
        int32_t header[2];
        for (int i = 0; i < 2; ++i) {
            file.read(reinterpret_cast<char*>(&header[i]), sizeof(int32_t));
            header[i] = reverseInt(header[i]);
        }
        int32_t n = header[1];
        ILOG("read {} labels from {}", n, fp.string());
        auto buf = new uint8_t[n];
        for (int i = 0; i < n; ++i) {
            file.read(reinterpret_cast<char*>(&buf[i]), sizeof(uint8_t));
        }
        torch::Tensor labels = torch::from_blob(
            buf, {n}, [](void* buf) { delete[] static_cast<uint8_t*>(buf); }, torch::kUInt8);
        return labels.to(torch::kLong);
    }

    torch::Tensor data_;
    torch::Tensor target_;
};

struct Net: torch::nn::Module
{
    Net()
        : conv1(torch::nn::Conv2dOptions(1, 10, 5)),
          conv2(torch::nn::Conv2dOptions(10, 20, 5)),
          fc1(320, 50),
          fc2(50, 10),
          bn1(10),
          bn2(20),
          bn3(50)
    {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("bn1", bn1);
        register_module("bn2", bn2);
        register_module("bn3", bn3);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(torch::max_pool2d(bn1->forward(conv1->forward(x)), 2));
        x = torch::relu(torch::max_pool2d(bn2->forward(conv2->forward(x)), 2));
        x = x.reshape({-1, 320});
        x = torch::relu(bn3->forward(fc1->forward(x)));
        x = torch::dropout(x, 0.5, is_training());
        x = fc2->forward(x);
        return torch::log_softmax(x, 1);
    }

    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
    torch::nn::BatchNorm2d bn1;
    torch::nn::BatchNorm2d bn2;
    torch::nn::BatchNorm1d bn3;
};

template <typename DataLoader>
void train(int curr_epoch, Net& model, torch::Device device, DataLoader& data_loader,
           torch::optim::Optimizer& optimizer, size_t dataset_size)
{
    model.train();
    size_t batch_idx = 0;
    for (auto& batch: data_loader) {
        auto data = batch.data.to(device);
        auto targets = batch.target.to(device);
        optimizer.zero_grad();
        auto output = model.forward(data);
        auto loss = torch::nll_loss(output, targets);
        AT_ASSERT(!std::isnan(loss.template item<float>()));
        loss.backward();
        optimizer.step();

        if (batch_idx++ % 200 == 0) {
            ILOG("epoch {}: {:5}/{:5} loss={:.4f}", curr_epoch, batch_idx * batch.data.size(0),
                 dataset_size, loss.template item<float>());
        }
    }
}

template <typename DataLoader>
void test(int curr_epoch, Net& model, torch::Device device, DataLoader& data_loader,
          size_t dataset_size)
{
    torch::NoGradGuard no_grad;
    model.eval();
    double test_loss = 0;
    int correct = 0;
    for (auto const& batch: data_loader) {
        auto data = batch.data.to(device);
        auto targets = batch.target.to(device);
        auto output = model.forward(data);
        test_loss +=
            torch::nll_loss(output, targets, {}, torch::Reduction::Sum).template item<float>();
        auto pred = output.argmax(1);
        correct += pred.eq(targets).sum().template item<int>();
    }

    ILOG("epoch {}: {:5}/{:5} loss={:.4f} accuracy={:.1f}%", curr_epoch, dataset_size, dataset_size,
         test_loss / dataset_size, static_cast<double>(correct) / dataset_size * 100);
}

MyErrCode fashionMnist(int argc, char** argv)
{
    torch::manual_seed(1);

    auto data_root = CURR_FILE_DIR() / ".temp" / "FashionMNIST" / "raw";
    FashionMnistDataset train_raw_dataset(data_root, true);
    FashionMnistDataset test_raw_dataset(data_root, false);

    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        ILOG("cuda available, training on gpu");
        device_type = torch::kCUDA;
    } else {
        ILOG("training on cpu");
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    auto saved_model_path = toolkit::getTempDir() / "fashion-mnist.pt";
    Net model;
    model.to(device);
    if (std::filesystem::exists(saved_model_path)) {
        ILOG("load model from {}", saved_model_path.string());
        torch::serialize::InputArchive ia;
        ia.load_from(saved_model_path);
        model.load(ia);
    }

    auto train_dataset = train_raw_dataset.map(torch::data::transforms::Normalize<>(0, 255))
                             .map(torch::data::transforms::Stack<>());
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(train_dataset), 64);

    auto test_dataset = test_raw_dataset.map(torch::data::transforms::Normalize<>(0, 255))
                            .map(torch::data::transforms::Stack<>());
    auto test_loader = torch::data::make_data_loader(std::move(test_dataset), 1000);

    size_t const train_dataset_size = train_raw_dataset.size().value();
    size_t const test_dataset_size = test_raw_dataset.size().value();

    torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(0.05).momentum(0.5));

    MY_TIMER_BEGIN(INFO, "process")
    test(0, model, device, *test_loader, test_dataset_size);
    for (size_t epoch = 1; epoch <= 10; ++epoch) {
        train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
        test(epoch, model, device, *test_loader, test_dataset_size);
    }
    MY_TIMER_END()

    ILOG("save model to {}", saved_model_path.string());
    torch::serialize::OutputArchive oa;
    model.save(oa);
    oa.save_to(saved_model_path);

    return MyErrCode::kOk;
}
