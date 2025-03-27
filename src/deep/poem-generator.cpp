#include "common.h"
#include <sentencepiece_trainer.h>
#include "toolkit/sqlite-helper.h"

#define VOCAB_SIZE 6500
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
        auto& p = poem_tk_.at(index);
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

    c10::optional<size_t> size() const override { return poem_tk_.size(); }

    std::string decode(std::vector<int> const& ids)
    {
        std::string text;
        if (auto err = tokenizer_.Decode(ids, &text); !err.ok()) {
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
    static MyErrCode loadPoemToken(std::filesystem::path const& poem_db,
                                   std::filesystem::path const& tk_model,
                                   sentencepiece::SentencePieceProcessor& tokenizer,
                                   std::vector<std::vector<int>>& poem_tk)
    {
        std::vector<std::string> poem_str;
        CHECK_ERR_RET(loadPoemStr(poem_db, poem_str));
        CHECK_ERR_RET(prepareTokenizer(tk_model.string(), poem_str));

        if (auto err = tokenizer.Load(tk_model.string()); !err.ok()) {
            ELOG("failed to load tokenizer: {}", err);
            return MyErrCode::kFailed;
        }

        for (auto const& poem: poem_str) {
            std::vector<int> pieces;
            if (auto err = tokenizer.Encode(poem, &pieces); !err.ok()) {
                ELOG("failed to encode '{}': {}", poem, err);
                return MyErrCode::kFailed;
            }
            pieces.insert(pieces.begin(), TOKEN_BOS_ID);
            poem_tk.push_back(std::move(pieces));
        }

        return MyErrCode::kOk;
    }

    static MyErrCode loadPoemStr(std::filesystem::path const& poem_db,
                                 std::vector<std::string>& poem_str)
    {
        toolkit::RowsValue rows;
        CHECK_ERR_RET(toolkit::SQLiteHelper::querySQL(
            poem_db, {"select content from ci order by value;", {{}}}, rows))
        for (auto& row: rows) {
            poem_str.push_back(row.at(0).asStr());
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

    sentencepiece::SentencePieceProcessor tokenizer_;
    std::vector<std::vector<int>> poem_tk_;
};

}  // namespace

MyErrCode poemGenerator(int argc, char** argv)
{
    toolkit::Args args(argc, argv);
    args.parse();

    auto poem_db = toolkit::getDataDir() / "ci.db";
    auto tk_model = toolkit::getDataDir() / "poem.model";
    PoemDataset ds(poem_db, tk_model);

    auto exam = ds.get(0);
    ILOG("data:   {}", ds.decode(exam.data));
    ILOG("target: {}", ds.decode(exam.target));

    return MyErrCode::kOk;
}
