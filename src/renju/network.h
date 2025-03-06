#pragma once
#include "game.h"

constexpr int kResidualLayers = 7;
constexpr int kResidualFilters = 64;
constexpr int kBatchSize = 512;
constexpr int kBufferSize = 10000;
constexpr float kInitLearningRate = 1e-3;
constexpr float kWeightDecay = 1e-4;
constexpr int kDropStepLR1 = 2000;
constexpr int kDropStepLR2 = 8000;
constexpr int kDropStepLR3 = 10000;

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
private:
    int64_t index_ = 0;
    SampleData* buf_;

public:
    DataSet() { buf_ = new SampleData[kBufferSize]; }

    ~DataSet() { delete[] buf_; }

    int size() const { return (index_ > kBufferSize) ? kBufferSize : index_; }

    int64_t total() const { return index_; }

    void pushBack(SampleData const* data)
    {
        buf_[index_ % kBufferSize] = *data;
        ++index_;
    }

    void pushWithTransform(SampleData* data);

    SampleData const& get(int i) const
    {
        assert(i < size());
        return buf_[i];
    }

    void makeMiniBatch(MiniBatch* batch) const;
};

std::ostream& operator<<(std::ostream& out, DataSet const& set);

class FIRNet
{
public:
    explicit FIRNet(int64_t verno);
    ~FIRNet();

    int64_t verno() const;
    void saveParam();
    void loadParam();
    void setLR(float lr);
    float calcInitLR() const;
    void adjustLR();
    std::string makeParamFileName() const;
    float trainStep(MiniBatch* batch);
    void evalState(State const& state, float value[1],
                   std::vector<std::pair<Move, float>>& move_priors);

private:
    struct Impl;
    Impl* impl_;
};
