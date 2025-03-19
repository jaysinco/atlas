#include "common.h"
#include <filesystem>
#include <random>

namespace
{

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
            MY_THROW("file not found: {}", image_path);
        }
        if (!std::filesystem::exists(label_path)) {
            MY_THROW("file not found: {}", label_path);
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
        ILOG("read {}x{}x{} images from {}", n, nr, nc, fp);
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
        ILOG("read {} labels from {}", n, fp);
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

struct FashionMnistNetImpl: torch::nn::Module
{
    FashionMnistNetImpl()
    {
        using namespace torch::nn;
        net_ = Sequential({
            {"9b645b6d", Conv2d(Conv2dOptions(1, 32, 3).padding(1))},
            {"ace20aff", ReLU()},
            {"eca26f36", BatchNorm2d(BatchNorm2dOptions(32))},
            {"ccaa4b6b", Conv2d(Conv2dOptions(32, 32, 3).padding(1))},
            {"12366812", ReLU()},
            {"e409fcfc", BatchNorm2d(BatchNorm2dOptions(32))},
            {"ba196546", MaxPool2d(MaxPool2dOptions({2, 2}))},
            {"35d3dff9", Dropout(DropoutOptions(0.25))},
            {"867fb601", Conv2d(Conv2dOptions(32, 64, 3).padding(1))},
            {"6db104af", ReLU()},
            {"0638916e", BatchNorm2d(BatchNorm2dOptions(64))},
            {"04d1b721", Conv2d(Conv2dOptions(64, 64, 3).padding(1))},
            {"c1871057", ReLU()},
            {"d774e4b8", BatchNorm2d(BatchNorm2dOptions(64))},
            {"66fec474", MaxPool2d(MaxPool2dOptions({2, 2}))},
            {"30b7caf7", Dropout(DropoutOptions(0.25))},
            {"430bcfba", Flatten()},
            {"9ea993be", Linear(LinearOptions(64 * (28 / 4) * (28 / 4), 512))},
            {"c1e8ca8f", ReLU()},
            {"4f0d34d4", BatchNorm1d(BatchNorm1dOptions(512))},
            {"9ef1ce6d", Dropout(DropoutOptions(0.5))},
            {"e167c0ad", Linear(LinearOptions(512, 10))},
            {"50a37880", LogSoftmax(LogSoftmaxOptions(1))},
        });
        register_module("074a5943", net_);
    }

    torch::Tensor forward(torch::Tensor x) { return net_->forward(x); }

private:
    torch::nn::Sequential net_;
};

TORCH_MODULE(FashionMnistNet);

template <typename DataLoader>
void train(int curr_epoch, FashionMnistNet& model, torch::Device device, DataLoader& data_loader,
           torch::optim::Optimizer& optimizer, size_t dataset_size)
{
    model->train();
    size_t batch_idx = 0;
    for (auto& batch: data_loader) {
        auto data = batch.data.to(device);
        auto targets = batch.target.to(device);
        optimizer.zero_grad();
        auto output = model(data);
        auto loss = torch::nll_loss(output, targets);
        AT_ASSERT(!std::isnan(loss.template item<float>()));
        loss.backward();
        optimizer.step();

        if (batch_idx++ % 100 == 0) {
            ILOG("epoch {}: {:5}/{:5} loss={:.4f}", curr_epoch, batch_idx * batch.data.size(0),
                 dataset_size, loss.template item<float>());
        }
    }
}

template <typename DataLoader>
void test(int curr_epoch, FashionMnistNet& model, torch::Device device, DataLoader& data_loader,
          size_t dataset_size)
{
    torch::NoGradGuard no_grad;
    model->eval();
    double test_loss = 0;
    int correct = 0;
    for (auto const& batch: data_loader) {
        auto data = batch.data.to(device);
        auto targets = batch.target.to(device);
        auto output = model(data);
        test_loss +=
            torch::nll_loss(output, targets, {}, torch::Reduction::Sum).template item<float>();
        auto pred = output.argmax(1);
        correct += pred.eq(targets).sum().template item<int>();
    }

    ILOG("epoch {}: {:5}/{:5} loss={:.4f} accuracy={:.1f}%", curr_epoch, dataset_size, dataset_size,
         test_loss / dataset_size, static_cast<double>(correct) / dataset_size * 100);
}

void testOne(FashionMnistNet& model, torch::Device device, FashionMnistDataset& dataset)
{
    static std::map<unsigned, std::string> const kLabelDesc = {
        {0, "T-shirt"}, {1, "Trouser"}, {2, "Pullover"}, {3, "Dress"}, {4, "Coat"},
        {5, "Sandal"},  {6, "Shirt"},   {7, "Sneaker"},  {8, "Bag"},   {9, "Ankle boot"},
    };
    static std::random_device rd;
    static std::mt19937 e2(rd());

    std::uniform_int_distribution<int> dist(0, dataset.size().value() - 1);
    int idx = dist(e2);
    auto example = dataset.get(idx);
    int nc = example.data.size(1);
    int nw = example.data.size(2);
    int target = example.target.item<int>();
    int predict;
    {
        torch::NoGradGuard no_grad;
        model->eval();
        auto data = example.data.unsqueeze(0).to(torch::kFloat).to(device) / 255.0f;
        auto output = model(data);
        predict = output.argmax(1).item<int>();
    }

    for (int i = 0; i < nc * nw; i++) {
        std::cout << (" .:-=+*#%@"[example.data.data_ptr<uint8_t>()[i] / 26])
                  << (((i + 1) % nw) ? "" : "\n");
    }
    ILOG("predict [{}] as '{}', really is '{}'", idx, kLabelDesc.at(predict),
         kLabelDesc.at(target));
}

}  // namespace

MyErrCode fashionMnist(int argc, char** argv)
{
    toolkit::Args args(argc, argv);
    args.parse();

    auto data_root = CURR_FILE_DIR() / ".temp" / "FashionMNIST" / "raw";
    FashionMnistDataset train_raw_dataset(data_root, true);
    FashionMnistDataset test_raw_dataset(data_root, false);

    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        ILOG("cuda available, training on {} gpu", torch::cuda::device_count());
        device_type = torch::kCUDA;
    } else {
        ILOG("training on cpu");
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    auto saved_model_path = toolkit::getTempDir() / "fashion-mnist.pt";
    FashionMnistNet model;
    model->to(device);
    if (std::filesystem::exists(saved_model_path)) {
        ILOG("load model from {}", saved_model_path);
        torch::serialize::InputArchive ia;
        ia.load_from(saved_model_path.string());
        model->load(ia);
    }

    auto train_dataset = train_raw_dataset.map(torch::data::transforms::Normalize<>(0, 255))
                             .map(torch::data::transforms::Stack<>());
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), 128);

    auto test_dataset = test_raw_dataset.map(torch::data::transforms::Normalize<>(0, 255))
                            .map(torch::data::transforms::Stack<>());
    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), 1000);

    size_t const train_dataset_size = train_raw_dataset.size().value();
    size_t const test_dataset_size = test_raw_dataset.size().value();

    torch::optim::RMSprop optimizer(model->parameters(),
                                    torch::optim::RMSpropOptions(0.0005).weight_decay(1e-6));

    MY_TIMER_BEGIN(INFO, "process")
    test(0, model, device, *test_loader, test_dataset_size);
    for (size_t epoch = 1; epoch <= 10; ++epoch) {
        train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
        test(epoch, model, device, *test_loader, test_dataset_size);
        test(epoch, model, device, *train_loader, train_dataset_size);
    }
    MY_TIMER_END()

    for (int i = 0; i < 1; ++i) {
        testOne(model, device, test_raw_dataset);
    }

    ILOG("save model to {}", saved_model_path);
    torch::serialize::OutputArchive oa;
    model->save(oa);
    oa.save_to(saved_model_path.string());

    return MyErrCode::kOk;
}
