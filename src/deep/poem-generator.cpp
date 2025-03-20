#include "common.h"
#include <sentencepiece_trainer.h>
#include "toolkit/sqlite-helper.h"

namespace sentencepiece::util
{
std::string toString(sentencepiece::util::Status const& s) { return s.ToString(); }

}  // namespace sentencepiece::util

namespace
{

class PoemSentenceIterator: public sentencepiece::SentenceIterator
{
public:
    PoemSentenceIterator(std::vector<std::string> const& data): data_(data), total_(data.size()) {}

    ~PoemSentenceIterator() override = default;

    bool done() const override { return idx_ >= total_; }

    void Next() override { ++idx_; }

    std::string const& value() const override { return data_[idx_]; }

    sentencepiece::util::Status status() const override { return {}; }

private:
    std::vector<std::string> const& data_;
    int idx_ = 0;
    int total_;
};

MyErrCode getPoemData(std::vector<std::string>& poem_data)
{
    auto db_path = toolkit::getDataDir() / "ci.db";
    toolkit::RowsValue rows;
    CHECK_ERR_RET(toolkit::SQLiteHelper::querySQL(
        db_path, {"select content from ci order by value;", {{}}}, rows))
    for (auto& row: rows) {
        poem_data.push_back(row.at(0).asStr());
    }
    ILOG("{} poems loaded", poem_data.size());
    return MyErrCode::kOk;
}

MyErrCode prepareTokenizer(std::string const& model_fp, std::vector<std::string> const& poem_data,
                           bool force = false)
{
    if (std::filesystem::exists(model_fp) && !force) {
        return MyErrCode::kOk;
    }
    ILOG("training tokenizer model...");
    PoemSentenceIterator iter(poem_data);
    auto err = sentencepiece::SentencePieceTrainer::Train(
        {
            {"model_prefix", model_fp.substr(0, model_fp.size() - 6)},
            {"vocab_size", "6500"},
            {"character_coverage", "0.9995"},
            {"model_type", "unigram"},
            {"minloglevel", "1"},
        },
        &iter);
    if (!err.ok()) {
        ELOG("failed to train tokenizer: {}", err);
        return MyErrCode::kFailed;
    }
    ILOG("tokenizer model is written to {}", model_fp);
    return MyErrCode::kOk;
}

}  // namespace

MyErrCode poemGenerator(int argc, char** argv)
{
    toolkit::Args args(argc, argv);
    args.parse();

    std::vector<std::string> poem_data;
    CHECK_ERR_RET(getPoemData(poem_data));

    auto tokenizer_model = toolkit::getDataDir() / "poem.model";
    CHECK_ERR_RET(prepareTokenizer(tokenizer_model, poem_data));

    sentencepiece::SentencePieceProcessor processor;
    if (auto err = processor.Load(tokenizer_model.string()); !err.ok()) {
        ELOG("failed to load tokenizer: {}", err);
        return MyErrCode::kFailed;
    }

    std::vector<std::string> pieces;
    if (auto err = processor.Encode("天涯共此时", &pieces); !err.ok()) {
        ELOG("failed to encode: {}", err);
        return MyErrCode::kFailed;
    }
    for (auto& token: pieces) {
        ILOG("\"{}\"", token);
    }

    return MyErrCode::kOk;
}
