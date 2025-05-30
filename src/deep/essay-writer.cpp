#include "common.h"
#include <sentencepiece_trainer.h>
#include "toolkit/sqlite-helper.h"

#define BATCH_SIZE 8
#define MAX_SEQ_LEN 256
#define SLIDE_WINDOW MAX_SEQ_LEN

#define TF_NUM_LAYERS 12
#define TF_NUM_HEADS 12
#define TF_EMBED_DIM 768
#define TF_HIDDEN_DIM 1024

#define VOCAB_SIZE 16384
#define TOKEN_UNK_ID 0
#define TOKEN_BOS_ID 1
#define TOKEN_EOS_ID 2
#define TOKEN_PAD_ID 3

namespace sentencepiece::util
{
std::string toString(sentencepiece::util::Status const& s);

}  // namespace sentencepiece::util

namespace
{

class EssayIterator: public sentencepiece::SentenceIterator
{
public:
    EssayIterator(std::filesystem::path const& essay_db): essay_db_(essay_db)
    {
        toolkit::RowsValue rows;
        if (toolkit::SQLiteHelper::querySQL(essay_db_, {"select count(*) from essays;", {{}}},
                                            rows) != MyErrCode::kOk) {
            MY_THROW("failed to query essay count");
        }
        if (rows.empty() || rows[0].empty()) {
            total_ = 0;
        } else {
            total_ = rows[0][0].asInt();
        }
    }

    ~EssayIterator() override = default;

    bool done() const override { return idx_ > total_; }

    void Next() override { ++idx_; }

    std::string const& value() const override
    {
        toolkit::RowsValue rows;
        if (toolkit::SQLiteHelper::querySQL(
                essay_db_, {"select title, content from essays where id = ?;", {{idx_}}}, rows) !=
            MyErrCode::kOk) {
            MY_THROW("failed to query essay when id={}", idx_);
        }
        if (rows.empty() || rows[0].empty()) {
            MY_THROW("failed to query essay when id={}", idx_);
        }
        curr_ = FSTR("《{}》 {}", rows.at(0).at(0).asStr(), rows.at(0).at(1).asStr());
        return curr_;
    }

    sentencepiece::util::Status status() const override { return {}; }

private:
    std::filesystem::path const& essay_db_;
    int total_;
    int idx_ = 1;
    mutable std::string curr_;
};

class EssayDataset: public torch::data::Dataset<EssayDataset>
{
public:
    EssayDataset(std::filesystem::path const& essay_db, std::filesystem::path const& tk_model)
        : essay_db_(essay_db), tk_model_(tk_model)

    {
        if (prepareTokenizer() != MyErrCode::kOk) {
            MY_THROW("failed to prepare essay tokenizer");
        }
    }

    ExampleType get(size_t index) override
    {
        auto& p = essay_tk_->at(index);
        torch::Tensor data = torch::from_blob(
            p.data(), {static_cast<int>(MAX_SEQ_LEN)}, [](void* buf) {}, torch::kInt32);

        std::vector<int> n;
        for (int i = 1; i < MAX_SEQ_LEN + 1; ++i) {
            n.push_back(p[i]);
        }
        torch::Tensor target = torch::from_blob(
            n.data(), {static_cast<int64_t>(n.size())}, [](void* buf) {}, torch::kInt32);

        return {data.to(torch::kLong), target.to(torch::kLong)};
    }

    c10::optional<size_t> size() const override { return essay_tk_->size(); }

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

    MyErrCode load()
    {
        essay_tk_ = std::make_shared<std::vector<std::vector<int>>>();
        int64_t sum_tokens = 0;
        int64_t count = 0;
        EssayIterator iter(essay_db_);
        for (; !iter.done(); iter.Next()) {
            std::string const& essay = iter.value();
            std::vector<int> pieces;
            if (auto err = tokenizer_->Encode(essay, &pieces); !err.ok()) {
                ELOG("failed to encode '{}': {}", essay, err);
                return MyErrCode::kFailed;
            }
            pieces.insert(pieces.begin(), TOKEN_BOS_ID);
            pieces.push_back(TOKEN_EOS_ID);
            sum_tokens += pieces.size();
            ++count;
            CHECK_ERR_RET(splitEssay(std::move(pieces)));
        }

        ILOG("{} essays loaded, {:.1f} average tokens", count,
             static_cast<float>(sum_tokens) / count);
        return MyErrCode::kOk;
    }

private:
    MyErrCode splitEssay(std::vector<int>&& pieces)
    {
        if (pieces.size() > MAX_SEQ_LEN + 1) {
            for (int i = 0; i < pieces.size(); i += SLIDE_WINDOW) {
                std::vector<int> p(MAX_SEQ_LEN + 1, TOKEN_PAD_ID);
                int len = std::min(MAX_SEQ_LEN + 1, static_cast<int>(pieces.size()) - i);
                std::copy(pieces.begin() + i, pieces.begin() + i + len, p.data());
                essay_tk_->push_back(std::move(p));
            }
        } else {
            pieces.resize(MAX_SEQ_LEN + 1, TOKEN_PAD_ID);
            essay_tk_->push_back(std::move(pieces));
        }
        return MyErrCode::kOk;
    }

    MyErrCode prepareTokenizer(bool force = false)
    {
        std::string tk_model = tk_model_.string();
        if (!std::filesystem::exists(tk_model_) || force) {
            ILOG("training tokenizer model...");
            EssayIterator iter(essay_db_);
            auto err = sentencepiece::SentencePieceTrainer::Train(
                {
                    {"model_prefix", tk_model.substr(0, tk_model.size() - 6)},
                    {"vocab_size", std::to_string(VOCAB_SIZE)},
                    {"character_coverage", "0.9995"},
                    {"model_type", "unigram"},
                    {"minloglevel", "0"},
                    {"add_dummy_prefix", "false"},
                    {"normalization_rule_name", "nmt_nfkc"},
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
        }
        tokenizer_ = std::make_shared<sentencepiece::SentencePieceProcessor>();
        if (auto err = tokenizer_->Load(tk_model); !err.ok()) {
            ELOG("failed to load tokenizer: {}", err);
            return MyErrCode::kFailed;
        }
        return MyErrCode::kOk;
    }

    std::filesystem::path const essay_db_;
    std::filesystem::path const tk_model_;
    std::shared_ptr<sentencepiece::SentencePieceProcessor> tokenizer_;
    std::shared_ptr<std::vector<std::vector<int>>> essay_tk_;
};

struct LayerNormOptions
{
    explicit LayerNormOptions(int64_t embed_dim): embed_dim_(embed_dim) {}

    TORCH_ARG(int64_t, embed_dim);
};

struct LayerNormImpl: torch::nn::Module
{
    LayerNormImpl(LayerNormOptions const& o)
    {
        weight_ = register_parameter("weight", torch::ones(o.embed_dim()));
        bias_ = register_parameter("bias", torch::zeros(o.embed_dim()));
    }

    // `x` has the shape of [batch_size, seq_len, embed_dim]
    torch::Tensor forward(torch::Tensor x)
    {
        return torch::layer_norm(x, weight_.sizes(), weight_, bias_);
    }

private:
    torch::Tensor weight_;
    torch::Tensor bias_;
};

TORCH_MODULE(LayerNorm);

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
        ln1_ = register_module("ln1", LayerNorm(LayerNormOptions(o.embed_dim())));
        ln2_ = register_module("ln2", LayerNorm(LayerNormOptions(o.embed_dim())));
        attn_ = register_module("attn", MultiHeadSelfAttention(o));
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor mask = {})
    {
        auto a = attn_(ln1_(x), mask);
        x = x + a;
        auto m = torch::gelu(fc1_(ln2_(x)));
        x = x + fc2_(m);
        return x;
    }

private:
    MultiHeadSelfAttention attn_ = nullptr;
    torch::nn::Linear fc1_ = nullptr, fc2_ = nullptr;
    LayerNorm ln1_ = nullptr, ln2_ = nullptr;
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

struct EssayNetImpl: torch::nn::Module
{
    EssayNetImpl(): o_(TF_NUM_LAYERS, TF_EMBED_DIM, TF_NUM_HEADS, TF_HIDDEN_DIM)
    {
        token_embed_ = register_module(
            "token_embed",
            torch::nn::Embedding(torch::nn::EmbeddingOptions(VOCAB_SIZE, o_.embed_dim())));
        pos_embed_ = register_module("pos_embed", torch::nn::Embedding(torch::nn::EmbeddingOptions(
                                                      MAX_SEQ_LEN, o_.embed_dim())));
        tf_ = register_module("tf", TransformerStack(o_));
        fc1_ = register_module("fc1", torch::nn::Linear(o_.embed_dim(), VOCAB_SIZE));
        ln1_ = register_module("ln1", LayerNorm(LayerNormOptions(o_.embed_dim())));
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
        x = tf_(x, mask);
        x = fc1_(ln1_(x));
        return x;
    }

private:
    TransformerStackOptions const o_;
    torch::nn::Embedding token_embed_ = nullptr;
    torch::nn::Embedding pos_embed_ = nullptr;
    TransformerStack tf_ = nullptr;
    torch::nn::Linear fc1_ = nullptr;
    LayerNorm ln1_ = nullptr;
};

TORCH_MODULE(EssayNet);

template <typename DataLoader>
void train(int curr_epoch, EssayNet& model, EssayDataset dataset, torch::Device device,
           DataLoader& data_loader, torch::optim::Optimizer& optimizer)
{
    model->train();
    size_t dataset_size = *dataset.size();
    size_t batch_idx = 0;
    double smooth_loss = -std::log(1.0 / VOCAB_SIZE);

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
        smooth_loss = smooth_loss * 0.9 + loss.template item<float>() * 0.1;

        ++batch_idx;
        size_t trained_size = batch_idx * batch.data.size(0);
        if (batch_idx % (10240 / BATCH_SIZE) == 0 || trained_size == dataset_size) {
            ILOG("epoch {}: {:5}/{:5} loss={:.4f}", curr_epoch, trained_size, dataset_size,
                 smooth_loss);
        }
    }
}

int64_t sampleNext(torch::Tensor const& logits, double temperature = 0.7, int top_k = 50,
                   double top_p = -1)
{
    torch::Tensor scaled_logits = logits / temperature;
    torch::Tensor probs = torch::softmax(scaled_logits, -1);
    if (top_k > 0) {
        auto [topk_values, topk_indices] = torch::topk(probs, top_k);
        torch::Tensor mask = torch::zeros_like(probs).scatter(-1, topk_indices, topk_values);
        probs = mask / mask.sum(-1, true);
    } else if (top_p > 0 && top_p < 1) {
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

std::pair<std::string, bool> sample(EssayNet& model, EssayDataset& dataset, torch::Device device,
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

void test(int curr_epoch, EssayNet& model, EssayDataset& dataset, torch::Device device)
{
    for (int i = 0; i < 5; ++i) {
        auto s = sample(model, dataset, device);
        ILOG("{}{}", s.first, s.second ? "<eos>" : "");
    }
}

void save(int curr_epoch, EssayNet& model, std::filesystem::path const& saved_path)
{
    torch::serialize::OutputArchive oa;
    model->save(oa);
    oa.save_to(saved_path.string());
    ILOG("epoch {}: model saved to {}", curr_epoch, saved_path);
}

}  // namespace

MyErrCode essayWriter(int argc, char** argv)
{
    toolkit::Args args(argc, argv);
    args.parse();

    auto essay_db = CURR_FILE_DIR() / ".temp" / "essays.db";
    auto tk_model = CURR_FILE_DIR() / ".temp" / "essay.model";

    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        auto gpu_cnt = torch::cuda::device_count();
        ILOG("cuda available, training on {} gpu", gpu_cnt);
        device = torch::Device(torch::kCUDA, gpu_cnt - 1);
    } else {
        ILOG("training on cpu");
    }

    auto saved_model_path = CURR_FILE_DIR() / ".temp" / "essay-writer.pt";
    EssayNet model;
    model->to(device);
    if (std::filesystem::exists(saved_model_path)) {
        ILOG("load model from {}", saved_model_path);
        torch::serialize::InputArchive ia;
        ia.load_from(saved_model_path.string(), device);
        model->load(ia);
    }
    CHECK_ERR_RET(describeModel(*model));

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-5));

    EssayDataset dataset{essay_db, tk_model};
    test(0, model, dataset, device);
    CHECK_ERR_RET(dataset.load());
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        dataset.map(torch::data::transforms::Stack<>()), BATCH_SIZE);
    for (size_t epoch = 1; true; ++epoch) {
        MY_TIMER_BEGIN(INFO, "epoch {}", epoch)
        train(epoch, model, dataset, device, *data_loader, optimizer);
        MY_TIMER_END()
        test(epoch, model, dataset, device);
        save(epoch, model, saved_model_path);
    }

    return MyErrCode::kOk;
}
