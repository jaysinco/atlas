#include "common.h"
#include <sentencepiece_trainer.h>
#include "toolkit/sqlite-helper.h"

#define BATCH_SIZE 1
#define MAX_SEQ_LEN 300
#define VOCAB_SIZE 5120
#define TOKEN_UNK_ID 0
#define TOKEN_BOS_ID 1
#define TOKEN_EOS_ID 2
#define TOKEN_PAD_ID 3

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
        n.push_back(TOKEN_UNK_ID);
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
            pieces.push_back(TOKEN_EOS_ID);
            pieces.resize(MAX_SEQ_LEN, TOKEN_PAD_ID);
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
                {"pad_id", std::to_string(TOKEN_PAD_ID)},
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

struct DyTOptions
{
    explicit DyTOptions(int64_t embed_dim, float init_alpha)
        : embed_dim_(embed_dim), init_alpha_(init_alpha)
    {
    }

    TORCH_ARG(int64_t, embed_dim);
    TORCH_ARG(float, init_alpha);
};

struct DyTImpl: torch::nn::Module
{
    DyTImpl(DyTOptions const& o)
    {
        alpha_ = register_parameter("alpha", torch::ones(1) * o.init_alpha());
        gamma_ = register_parameter("gamma", torch::ones(o.embed_dim()));
        beta_ = register_parameter("beta", torch::zeros(o.embed_dim()));
    }

    // `x` has the shape of [batch_size, seq_len, embed_dim]
    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::tanh(alpha_ * x);
        return gamma_ * x + beta_;
    }

private:
    torch::Tensor alpha_;
    torch::Tensor gamma_;
    torch::Tensor beta_;
};

TORCH_MODULE(DyT);

struct MultiHeadSelfAttentionOptions
{
    MultiHeadSelfAttentionOptions(int64_t embed_dim, int64_t num_heads)
        : embed_dim_(embed_dim), num_heads_(num_heads)
    {
    }

    TORCH_ARG(int64_t, embed_dim);
    TORCH_ARG(int64_t, num_heads);
};

struct MultiHeadSelfAttentionImpl: torch::nn::Module
{
    MultiHeadSelfAttentionImpl(MultiHeadSelfAttentionOptions const& o)
        : num_heads_(o.num_heads()), embed_dim_(o.embed_dim())
    {
        if (embed_dim_ % num_heads_ != 0) {
            MY_THROW(R"(embed_dim % num_heads != 0)");
        }
        head_dim_ = embed_dim_ / num_heads_;
        q_proj_ = register_module("q_proj", torch::nn::Linear(embed_dim_, embed_dim_));
        k_proj_ = register_module("k_proj", torch::nn::Linear(embed_dim_, embed_dim_));
        v_proj_ = register_module("v_proj", torch::nn::Linear(embed_dim_, embed_dim_));
        out_proj_ = register_module("out_proj", torch::nn::Linear(embed_dim_, embed_dim_));
        drop_ = register_module("drop", torch::nn::Dropout(0.1));
    }

    // `x` has the shape of [batch_size, seq_len, embed_dim]
    // `mask` has the shape of [batch_size, num_heads, seq_len, seq_len]
    //        True value indicates that the corresponding position is not allowed to attend
    torch::Tensor forward(torch::Tensor x, torch::Tensor mask = {})
    {
        auto batch_size = x.size(0);
        auto seq_len = x.size(1);

        auto q = q_proj_(x);
        auto k = k_proj_(x);
        auto v = v_proj_(x);

        q = q.view({batch_size, seq_len, num_heads_, head_dim_}).transpose(1, 2);
        k = k.view({batch_size, seq_len, num_heads_, head_dim_}).transpose(1, 2);
        v = v.view({batch_size, seq_len, num_heads_, head_dim_}).transpose(1, 2);

        auto scores = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(head_dim_);
        if (mask.defined()) {
            scores = scores.masked_fill(mask, -INFINITY);
        }

        auto attn_weights = torch::softmax(scores, -1);
        attn_weights = drop_(attn_weights);
        auto output = torch::matmul(attn_weights, v);
        output = output.transpose(1, 2).contiguous().view({batch_size, seq_len, embed_dim_});

        return out_proj_(output);
    }

private:
    int64_t num_heads_;
    int64_t embed_dim_;
    int64_t head_dim_;

    torch::nn::Linear q_proj_ = nullptr;
    torch::nn::Linear k_proj_ = nullptr;
    torch::nn::Linear v_proj_ = nullptr;
    torch::nn::Linear out_proj_ = nullptr;
    torch::nn::Dropout drop_ = nullptr;
};

TORCH_MODULE(MultiHeadSelfAttention);

struct TransformerBlockOptions: MultiHeadSelfAttentionOptions
{
    TransformerBlockOptions(int64_t embed_dim, int64_t num_heads, int64_t hidden_dim)
        : MultiHeadSelfAttentionOptions(embed_dim, num_heads), hidden_dim_(hidden_dim)
    {
    }

    TORCH_ARG(int64_t, hidden_dim);
};

struct TransformerBlockImpl: torch::nn::Module
{
    TransformerBlockImpl(TransformerBlockOptions const& o)
    {
        fc1_ = register_module("fc1", torch::nn::Linear(o.embed_dim(), o.hidden_dim()));
        fc2_ = register_module("fc2", torch::nn::Linear(o.hidden_dim(), o.embed_dim()));
        dy1_ = register_module("dy1", DyT(DyTOptions(o.embed_dim(), 0.5)));
        dy2_ = register_module("dy2", DyT(DyTOptions(o.embed_dim(), 0.5)));
        attn_ = register_module("attn", MultiHeadSelfAttention(o));
        drop_ = register_module("drop", torch::nn::Dropout(0.1));
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor mask = {})
    {
        x = attn_(dy1_(x), mask);
        x = dy2_(x + drop_(x));
        x = torch::gelu(fc1_(x));
        x = fc2_(drop_(x));
        x = x + drop_(x);
        return x;
    }

private:
    MultiHeadSelfAttention attn_ = nullptr;
    torch::nn::Linear fc1_ = nullptr, fc2_ = nullptr;
    torch::nn::Dropout drop_ = nullptr;
    DyT dy1_ = nullptr, dy2_ = nullptr;
};

TORCH_MODULE(TransformerBlock);

struct TransformerStackOptions: TransformerBlockOptions
{
    TransformerStackOptions(int64_t num_layers, int64_t embed_dim, int64_t num_heads,
                            int64_t hidden_dim)
        : TransformerBlockOptions(embed_dim, num_heads, hidden_dim), num_layers_(num_layers)
    {
    }

    TORCH_ARG(int64_t, num_layers);
};

struct TransformerStackImpl: torch::nn::Module
{
    TransformerStackImpl(TransformerStackOptions const& o)
    {
        for (int i = 0; i < o.num_layers(); ++i) {
            layers_->push_back(register_module("layer_" + std::to_string(i), TransformerBlock(o)));
        }
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor mask = {})
    {
        for (auto& layer: *layers_) {
            x = layer->as<TransformerBlock>()->forward(x, mask);
        }
        return x;
    }

private:
    torch::nn::ModuleList layers_;
};

TORCH_MODULE(TransformerStack);

struct PoemNetImpl: torch::nn::Module
{
    PoemNetImpl(): o_(3, 512, 4, 1024)
    {
        token_embed_ = register_module(
            "token_embed",
            torch::nn::Embedding(torch::nn::EmbeddingOptions(VOCAB_SIZE, o_.embed_dim())));
        pos_embed_ = register_module("pos_embed", torch::nn::Embedding(torch::nn::EmbeddingOptions(
                                                      MAX_SEQ_LEN, o_.embed_dim())));
        tf_ = register_module("tf", TransformerStack(o_));
        drop_ = register_module("drop", torch::nn::Dropout(0.1));
        fc1_ = register_module("fc1", torch::nn::Linear(o_.embed_dim(), VOCAB_SIZE));
        dy1_ = register_module("dy1", DyT(DyTOptions(o_.embed_dim(), 0.5)));
    }

    // `x` has the shape of [batch_size, seq_len]
    torch::Tensor forward(torch::Tensor x)
    {
        int64_t batch_size = x.size(0);
        int64_t seq_len = x.size(1);
        auto pos_ids = torch::arange(seq_len, torch::TensorOptions(torch::kLong).device(x.device()))
                           .expand({batch_size, seq_len});
        auto mask =
            torch::triu(torch::ones({seq_len, seq_len}, torch::TensorOptions(x.device())), 1)
                .to(torch::kBool);

        x = token_embed_(x) + pos_embed_(pos_ids);
        x = tf_(drop_(x), mask);
        x = fc1_(dy1_(x));
        return x;
    }

private:
    const TransformerStackOptions o_;
    torch::nn::Embedding token_embed_ = nullptr;
    torch::nn::Embedding pos_embed_ = nullptr;
    TransformerStack tf_ = nullptr;
    torch::nn::Dropout drop_ = nullptr;
    torch::nn::Linear fc1_ = nullptr;
    DyT dy1_ = nullptr;
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
        auto data = batch.data.to(device);
        auto targets = batch.target.to(device);
        optimizer.zero_grad();
        auto logits = model(data);
        auto log_probs = torch::log_softmax(logits, -1);
        auto loss = torch::nll_loss(log_probs.view({-1, VOCAB_SIZE}), targets.view({-1}), {},
                                    torch::Reduction::Mean, TOKEN_PAD_ID);
        loss.backward();
        optimizer.step();
        sum_loss += loss.template item<float>();

        ++batch_idx;
        size_t trained_size = batch_idx * batch.data.size(0);
        if (batch_idx % (1024 / BATCH_SIZE) == 0 || trained_size == dataset_size) {
            ILOG("epoch {}: {:5}/{:5} loss={:.4f}", curr_epoch, trained_size, dataset_size,
                 sum_loss / batch_idx);
        }
    }
}

int64_t sampleNext(torch::Tensor const& logits, double temperature = 0.7, double top_p = 0.7,
                   int top_k = 50)
{
    torch::Tensor scaled_logits = logits / temperature;
    torch::Tensor probs = torch::softmax(scaled_logits, -1);
    if (top_k > 0) {
        auto [topk_values, topk_indices] = torch::topk(probs, top_k);
        torch::Tensor mask = torch::zeros_like(probs).scatter(-1, topk_indices, topk_values);
        probs = mask / mask.sum(-1, true);
    }
    if (top_p < 1.0) {
        auto [sorted_probs, sorted_indices] = torch::sort(probs, -1, true);
        torch::Tensor cum_probs = torch::cumsum(sorted_probs, -1);
        torch::Tensor mask = cum_probs <= top_p;
        if (mask.sum().item<int>() == 0) {
            mask[0] = 1;
        }
        torch::Tensor expanded_mask =
            torch::zeros_like(probs, torch::kBool).scatter(-1, sorted_indices, mask);
        probs *= expanded_mask;
        probs /= probs.sum(-1, true);
    }
    torch::Tensor next_token = torch::multinomial(probs, 1, true);
    return next_token.item<int64_t>();
}

std::pair<std::string, bool> sample(PoemNet& model, PoemDataset& dataset, torch::Device device,
                                    std::string const& prefix = "")
{
    torch::NoGradGuard no_grad;
    model->eval();
    bool reach_eos = false;
    std::vector<int> ids = {};
    torch::Tensor data = torch::full({1, MAX_SEQ_LEN}, TOKEN_PAD_ID,
                                     torch::TensorOptions(torch::kLong).device(device));
    data.index({0, 0}) = TOKEN_BOS_ID;
    for (int i = 1; i < MAX_SEQ_LEN; ++i) {
        auto logits = model(data).index({0, i - 1});
        int64_t next_token = sampleNext(logits);
        if (next_token == TOKEN_EOS_ID) {
            reach_eos = true;
            break;
        }
        data.index({0, i}) = next_token;
        ids.push_back(next_token);
    }
    return std::make_pair(dataset.decode(ids), reach_eos);
}

void test(int curr_epoch, PoemNet& model, PoemDataset& dataset, torch::Device device)
{
    for (int i = 0; i < 5; ++i) {
        auto s = sample(model, dataset, device);
        ILOG("{}{}", s.first, s.second ? "<eos>" : "");
    }
}

void save(int curr_epoch, PoemNet& model, std::filesystem::path const& saved_path)
{
    torch::serialize::OutputArchive oa;
    model->save(oa);
    oa.save_to(saved_path.string());
    ILOG("epoch {}: model saved to {}", curr_epoch, saved_path);
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

    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        auto gpu_cnt = torch::cuda::device_count();
        ILOG("cuda available, training on {} gpu", gpu_cnt);
        device = torch::Device(torch::kCUDA, gpu_cnt - 1);
    } else {
        ILOG("training on cpu");
    }

    auto saved_model_path = toolkit::getTempDir() / "poem-generator.pt";
    PoemNet model;
    model->to(device);
    if (std::filesystem::exists(saved_model_path)) {
        ILOG("load model from {}", saved_model_path);
        torch::serialize::InputArchive ia;
        ia.load_from(saved_model_path.string());
        model->load(ia);
    }

    torch::optim::RMSprop optimizer(model->parameters(), torch::optim::RMSpropOptions(1e-5));

    test(0, model, dataset, device);
    for (size_t epoch = 1; true; ++epoch) {
        MY_TIMER_BEGIN(INFO, "epoch {}", epoch)
        train(epoch, model, dataset, device, *data_loader, optimizer);
        MY_TIMER_END()
        test(epoch, model, dataset, device);
        save(epoch, model, saved_model_path);
    }

    return MyErrCode::kOk;
}
