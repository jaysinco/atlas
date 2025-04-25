#pragma once
#include "game.h"

struct SampleData
{
    float data[kInputFeatureNum * kBoardSize] = {0.0f};
    float p_label[kBoardSize] = {0.0f};
    float v_label[1] = {0.0f};

    void flipVerticing();
    void transpose();
};

std::ostream& operator<<(std::ostream& out, SampleData const& sample);

struct MiniBatch
{
    float data[kBatchSize * kInputFeatureNum * kBoardSize] = {0.0f};
    float p_label[kBatchSize * kBoardSize] = {0.0f};
    float v_label[kBatchSize * 1] = {0.0f};
};

std::ostream& operator<<(std::ostream& out, MiniBatch const& batch);

class DataSet
{
public:
    DataSet() { buf_ = new SampleData[kBufferSize]; }

    ~DataSet() { delete[] buf_; }

    int size() const { return (index_ > kBufferSize) ? kBufferSize : index_; }

    int64_t total() const { return index_; }

    void add(SampleData const* data)
    {
        buf_[index_ % kBufferSize] = *data;
        ++index_;
    }

    void addWithTransform(SampleData* data);

    SampleData const& get(int i) const
    {
        if (i >= size() || i < 0) {
            MY_THROW("invalid index: {}", i);
        }
        return buf_[i];
    }

    void makeMiniBatch(MiniBatch* batch) const;

private:
    int64_t index_ = 0;
    SampleData* buf_;
};

std::ostream& operator<<(std::ostream& out, DataSet const& ds);

struct TrainingMeta
{
    float learning_rate = 1e-3;
    int selfplay_rounds = 0;
    int selfplay_turns = 0;
};

class FIRNet
{
public:
    explicit FIRNet(int64_t verno);
    ~FIRNet();
    int64_t verno() const;
    float train(MiniBatch* batch, TrainingMeta& meta);
    void eval(State const& state, float value[1], std::vector<std::pair<Move, float>>& act_priors);
    void save();

private:
    void load();
    void setLearningRate(float lr);
    std::string savePath() const;

    struct Impl;
    Impl* impl_;
};
