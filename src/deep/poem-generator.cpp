#include "common.h"
#include <sentencepiece_trainer.h>
#include "toolkit/sqlite-helper.h"

#define BATCH_SIZE 1
#define VOCAB_SIZE 6500
#define TOKEN_UNK_ID 0
#define TOKEN_BOS_ID 1
#define TOKEN_EOS_ID 2

namespace sentencepiece::util
{
std::string toString(sentencepiece::util::Status const& s) { return s.ToString(); }

}  // namespace sentencepiece::util

namespace
{

class StrSentenceIterator: public sentencepiece::SentenceIterator
{
public:
    StrSentenceIterator(std::vector<std::string> const& data): data_(data), total_(data.size()) {}

    ~StrSentenceIterator() override = default;

    bool done() const override { return idx_ >= total_; }

    void Next() override { ++idx_; }

    std::string const& value() const override { return data_[idx_]; }

    sentencepiece::util::Status status() const override { return {}; }

private:
    std::vector<std::string> const& data_;
    int idx_ = 0;
    int total_;
};

class PoemDataset: public torch::data::Dataset<PoemDataset>
{
public:
    PoemDataset(std::filesystem::path const& poem_db, std::filesystem::path const& tk_model)
    {
        if (loadPoemToken(poem_db, tk_model, tokenizer_, poem_tk_) != MyErrCode::kOk) {
            MY_THROW("failed to load poem poken");
        }
    }

    ExampleType get(size_t index) override
    {
        auto& p = poem_tk_->at(index);
        torch::Tensor data = torch::from_blob(
            p.data(), {static_cast<int>(p.size())}, [](void* buf) {}, torch::kInt32);

        std::vector<int> n;
        for (int i = 1; i < p.size(); ++i) {
            n.push_back(p[i]);
        }
        n.push_back(TOKEN_EOS_ID);
        torch::Tensor target = torch::from_blob(
            n.data(), {static_cast<int64_t>(n.size())}, [](void* buf) {}, torch::kInt32);

        return {data.to(torch::kLong), target.to(torch::kLong)};
    }

    c10::optional<size_t> size() const override { return poem_tk_->size(); }

    std::string decode(std::vector<int> const& ids)
    {
        std::string text;
        if (auto err = tokenizer_->Decode(ids, &text); !err.ok()) {
            MY_THROW("failed to decode: {}", err);
        }
        return text;
    }

    std::string decode(torch::Tensor const& tids)
    {
        torch::Tensor data = tids.to(torch::kInt32);
        std::vector<int> ids;
        for (int i = 0; i < data.numel(); ++i) {
            ids.push_back(data.const_data_ptr<int>()[i]);
        }
        return decode(ids);
    }

private:
    static MyErrCode loadPoemToken(
        std::filesystem::path const& poem_db, std::filesystem::path const& tk_model,
        std::shared_ptr<sentencepiece::SentencePieceProcessor>& tokenizer,
        std::shared_ptr<std::vector<std::vector<int>>>& poem_tk)
    {
        std::vector<std::string> poem_str;
        CHECK_ERR_RET(loadPoemStr(poem_db, poem_str));
        CHECK_ERR_RET(prepareTokenizer(tk_model.string(), poem_str));

        tokenizer = std::make_shared<sentencepiece::SentencePieceProcessor>();
        if (auto err = tokenizer->Load(tk_model.string()); !err.ok()) {
            ELOG("failed to load tokenizer: {}", err);
            return MyErrCode::kFailed;
        }

        poem_tk = std::make_shared<std::vector<std::vector<int>>>();
        for (auto const& poem: poem_str) {
            std::vector<int> pieces;
            if (auto err = tokenizer->Encode(poem, &pieces); !err.ok()) {
                ELOG("failed to encode '{}': {}", poem, err);
                return MyErrCode::kFailed;
            }
            pieces.insert(pieces.begin(), TOKEN_BOS_ID);
            poem_tk->push_back(std::move(pieces));
        }

        return MyErrCode::kOk;
    }

    static MyErrCode loadPoemStr(std::filesystem::path const& poem_db,
                                 std::vector<std::string>& poem_str)
    {
        toolkit::RowsValue rows;
        CHECK_ERR_RET(toolkit::SQLiteHelper::querySQL(
            poem_db, {"select rhythmic, content from ci order by value;", {{}}}, rows))
        for (auto& row: rows) {
            auto s = FSTR("《{}》 {}", row.at(0).asStr(), row.at(1).asStr());
            poem_str.push_back(s);
        }
        ILOG("{} poems loaded", poem_str.size());
        return MyErrCode::kOk;
    }

    static MyErrCode prepareTokenizer(std::string const& tk_model,
                                      std::vector<std::string> const& poem_str, bool force = false)
    {
        if (std::filesystem::exists(tk_model) && !force) {
            return MyErrCode::kOk;
        }
        ILOG("training tokenizer model...");
        StrSentenceIterator iter(poem_str);
        auto err = sentencepiece::SentencePieceTrainer::Train(
            {
                {"model_prefix", tk_model.substr(0, tk_model.size() - 6)},
                {"vocab_size", std::to_string(VOCAB_SIZE)},
                {"character_coverage", "0.9995"},
                {"model_type", "unigram"},
                {"minloglevel", "1"},
                {"add_dummy_prefix", "false"},
                {"normalization_rule_name", "identity"},
                {"unk_id", std::to_string(TOKEN_UNK_ID)},
                {"bos_id", std::to_string(TOKEN_BOS_ID)},
                {"eos_id", std::to_string(TOKEN_EOS_ID)},
                {"pad_id", "-1"},
            },
            &iter);
        if (!err.ok()) {
            ELOG("failed to train tokenizer: {}", err);
            return MyErrCode::kFailed;
        }
        ILOG("tokenizer model is written to {}", tk_model);
        return MyErrCode::kOk;
    }

    std::shared_ptr<sentencepiece::SentencePieceProcessor> tokenizer_;
    std::shared_ptr<std::vector<std::vector<int>>> poem_tk_;
};

struct PoemNetImpl: torch::nn::Module
{
    PoemNetImpl(int embed_sz = 200, int lstm_hidden = 500, int lstm_layers = 2)
        : lstm_hidden_(lstm_hidden), lstm_layers_(lstm_layers)
    {
        lstm_h_ = torch::zeros({lstm_layers_, BATCH_SIZE, lstm_hidden_});
        lstm_c_ = torch::zeros({lstm_layers_, BATCH_SIZE, lstm_hidden_});

        using namespace torch::nn;
        embed_ = register_module("embed", Embedding(EmbeddingOptions(VOCAB_SIZE, embed_sz)));
        lstm_ = register_module(
            "lstm",
            LSTM(LSTMOptions(embed_sz, lstm_hidden_).num_layers(lstm_layers_).batch_first(true)));
        fc0_ = register_module("fc0", Linear(LinearOptions(lstm_hidden_, VOCAB_SIZE)));
    }

    void reset()
    {
        lstm_h_ = lstm_h_.detach();
        lstm_c_ = lstm_c_.detach();
        lstm_h_.zero_();
        lstm_c_.zero_();
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = embed_(x);
        auto xhc = lstm_->forward(x, std::make_tuple(lstm_h_, lstm_c_));
        auto hc = std::get<1>(xhc);
        lstm_h_ = std::get<0>(hc);
        lstm_c_ = std::get<1>(hc);
        x = std::get<0>(xhc);
        x = fc0_(x);
        x = torch::log_softmax(x, 2);
        return x;
    }

    void to(torch::Device device, torch::Dtype dtype, bool non_blocking = false) override
    {
        torch::nn::Module::to(device, dtype, non_blocking);
        lstm_h_ = lstm_h_.to(device, dtype, non_blocking);
        lstm_c_ = lstm_c_.to(device, dtype, non_blocking);
    }

    void to(torch::Dtype dtype, bool non_blocking = false) override
    {
        torch::nn::Module::to(dtype, non_blocking);
        lstm_h_ = lstm_h_.to(dtype, non_blocking);
        lstm_c_ = lstm_c_.to(dtype, non_blocking);
    }

    void to(torch::Device device, bool non_blocking = false) override
    {
        torch::nn::Module::to(device, non_blocking);
        lstm_h_ = lstm_h_.to(device, non_blocking);
        lstm_c_ = lstm_c_.to(device, non_blocking);
    }

private:
    int const lstm_hidden_;
    int const lstm_layers_;
    torch::nn::Embedding embed_ = nullptr;
    torch::nn::LSTM lstm_ = nullptr;
    torch::nn::Linear fc0_ = nullptr;
    torch::Tensor lstm_h_;
    torch::Tensor lstm_c_;
};

TORCH_MODULE(PoemNet);

template <typename DataLoader>
void train(int curr_epoch, PoemNet& model, PoemDataset dataset, torch::Device device,
           DataLoader& data_loader, torch::optim::Optimizer& optimizer)
{
    model->train();
    size_t dataset_size = *dataset.size();
    size_t batch_idx = 0;
    double sum_loss = 0;

    for (auto& batch: data_loader) {
        model->reset();
        auto data = batch.data.to(device);
        auto targets = batch.target.to(device);
        optimizer.zero_grad();
        auto output = model(data);
        auto loss = torch::nll_loss(output.view({-1, VOCAB_SIZE}), targets.view({-1}));
        loss.backward();
        optimizer.step();
        sum_loss += loss.template item<float>();

        ++batch_idx;
        size_t trained_size = batch_idx * batch.data.size(0);
        if (batch_idx % 1000 == 0 || trained_size == dataset_size) {
            ILOG("epoch {}: {:5}/{:5} loss={:.4f}", curr_epoch, trained_size, dataset_size,
                 sum_loss / batch_idx);
        }
    }
}

std::pair<std::string, bool> sample(PoemNet& model, PoemDataset& dataset, torch::Device device,
                                    std::string const& prefix = "", int max_output = 600)
{
    torch::NoGradGuard no_grad;
    model->eval();
    model->reset();
    bool reach_eos = false;
    std::vector<int> ids = {TOKEN_BOS_ID};
    for (int i = 0; i < max_output; ++i) {
        torch::Tensor data =
            torch::tensor(ids.back(), torch::TensorOptions(torch::kLong).device(device))
                .reshape({1, 1});
        auto log_probs = model(data);
        torch::Tensor probs = torch::exp(log_probs.squeeze());
        int64_t next_token = torch::multinomial(probs, 1, true).item<int64_t>();
        if (next_token == TOKEN_EOS_ID) {
            reach_eos = true;
            break;
        }
        ids.push_back(next_token);
    }
    return std::make_pair(dataset.decode(ids), reach_eos);
}

void test(int curr_epoch, PoemNet& model, PoemDataset& dataset, torch::Device device)
{
    auto s = sample(model, dataset, device);
    ILOG("epoch {}: {}", curr_epoch, s.first);
}

}  // namespace

MyErrCode poemGenerator(int argc, char** argv)
{
    toolkit::Args args(argc, argv);
    args.parse();

    auto poem_db = toolkit::getDataDir() / "ci.db";
    auto tk_model = toolkit::getDataDir() / "poem.model";
    PoemDataset dataset{poem_db, tk_model};
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        dataset.map(torch::data::transforms::Stack<>()), BATCH_SIZE);

    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        ILOG("cuda available, training on {} gpu", torch::cuda::device_count());
        device_type = torch::kCUDA;
    } else {
        ILOG("training on cpu");
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    auto saved_model_path = toolkit::getTempDir() / "poem-generator.pt";
    PoemNet model;
    model->to(device);
    if (std::filesystem::exists(saved_model_path)) {
        ILOG("load model from {}", saved_model_path);
        torch::serialize::InputArchive ia;
        ia.load_from(saved_model_path.string());
        model->load(ia);
    }

    torch::optim::RMSprop optimizer(model->parameters(), torch::optim::RMSpropOptions(1e-3));

    MY_TIMER_BEGIN(INFO, "process")
    test(0, model, dataset, device);
    for (size_t epoch = 1; epoch <= 10; ++epoch) {
        train(epoch, model, dataset, device, *data_loader, optimizer);
        test(epoch, model, dataset, device);
    }
    MY_TIMER_END()

    ILOG("save model to {}", saved_model_path);
    torch::serialize::OutputArchive oa;
    model->save(oa);
    oa.save_to(saved_model_path.string());

    return MyErrCode::kOk;
}
