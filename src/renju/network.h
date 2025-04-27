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
    float data[kTrainBatchSize * kInputFeatureNum * kBoardSize] = {0.0f};
    float p_label[kTrainBatchSize * kBoardSize] = {0.0f};
    float v_label[kTrainBatchSize * 1] = {0.0f};
};

std::ostream& operator<<(std::ostream& out, MiniBatch const& batch);

class DataSet
{
public:
    DataSet() { buf_ = new SampleData[kTrainDataBufferSize]; }

    ~DataSet() { delete[] buf_; }

    int size() const { return (index_ > kTrainDataBufferSize) ? kTrainDataBufferSize : index_; }

    int64_t total() const { return index_; }

    void add(SampleData const* data)
    {
        buf_[index_ % kTrainDataBufferSize] = *data;
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

class FIRNet
{
public:
    explicit FIRNet(int64_t verno, bool use_gpu, int eval_batch_size);
    ~FIRNet();
    float step(MiniBatch* batch, TrainMeta& meta);
    MyErrCode eval(State const& state, float const state_feature[kInputFeatureNum * kBoardSize],
                   float value[1], std::vector<std::pair<Move, float>>& act_priors);
    void saveModel();
    int64_t verno() const;

private:
    void loadModel();
    void setLearningRate(float lr);
    std::string saveModelPath() const;
    void evalThreadEntry();

    struct Impl;
    Impl* impl_;
};
