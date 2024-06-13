#include "./common.cuh"
#include <fmt/ostream.h>
#include <cuda_runtime.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <iostream>

std::ostream& operator<<(std::ostream& out, nvinfer1::Dims const& shape)
{
    for (int i = 0; i < shape.nbDims; ++i) {
        if (i != 0) {
            out << ",";
        }
        out << shape.d[i];
    }
    return out;
}

template <>
struct fmt::formatter<nvinfer1::Dims>: ostream_formatter
{
};

std::ostream& operator<<(std::ostream& out, nvinfer1::TensorIOMode mode)
{
    switch (mode) {
        case nvinfer1::TensorIOMode::kNONE:
            out << "NONE";
            break;
        case nvinfer1::TensorIOMode::kINPUT:
            out << "IN";
            break;
        case nvinfer1::TensorIOMode::kOUTPUT:
            out << "OUT";
            break;
    }
    return out;
}

template <>
struct fmt::formatter<nvinfer1::TensorIOMode>: ostream_formatter
{
};

std::ostream& operator<<(std::ostream& out, nvinfer1::DataType type)
{
    switch (type) {
        case nvinfer1::DataType::kFLOAT:
            out << "FLOAT";
            break;
        case nvinfer1::DataType::kHALF:
            out << "HALF";
            break;
        case nvinfer1::DataType::kINT8:
            out << "INT8";
            break;
        case nvinfer1::DataType::kINT32:
            out << "INT32";
            break;
        case nvinfer1::DataType::kBOOL:
            out << "BOOL";
            break;
        case nvinfer1::DataType::kUINT8:
            out << "UINT8";
            break;
    }
    return out;
}

void readPGMFile(std::string const& fileName, uint8_t* buffer, int32_t inH, int32_t inW)
{
    std::ifstream infile(fileName, std::ifstream::binary);
    std::string magic, w, h, max;
    infile >> magic >> w >> h >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(buffer), inH * inW);
}

void printPGMData(uint8_t* buffer, int32_t inH, int32_t inW)
{
    for (int i = 0; i < inH * inW; i++) {
        std::cout << (" .:-=+*#%@"[buffer[i] / 26]) << (((i + 1) % inW) ? "" : "\n");
    }
}

template <>
struct fmt::formatter<nvinfer1::DataType>: ostream_formatter
{
};

class MyLogger: public nvinfer1::ILogger
{
    void log(Severity severity, char const* msg) noexcept override
    {
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
            case Severity::kERROR:
                ELOG(msg);
                break;
            case Severity::kWARNING:
                WLOG(msg);
                break;
            case Severity::kINFO:
                ILOG(msg);
                break;
            case Severity::kVERBOSE:
                DLOG(msg);
                break;
        }
    }
};

int build(MyLogger& logger, nvinfer1::IHostMemory*& plan)
{
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    if (!builder) {
        return -1;
    }
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(
        1U << static_cast<int>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    if (!network) {
        return -1;
    }
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    if (!parser) {
        return -1;
    }
    auto fpath = toolkit::getDataDir() / "mnist.onnx";
    bool parsed = parser->parseFromFile(
        fpath.c_str(), static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING));
    for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
        ELOG(parser->getError(i)->desc());
    }
    if (!parsed) {
        return -1;
    }
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    if (!config) {
        return -1;
    }
    plan = builder->buildSerializedNetwork(*network, *config);
    if (!plan) {
        return -1;
    }

    delete config;
    delete parser;
    delete network;
    delete builder;

    return 0;
}

int verifyOutput(std::vector<float>& outData)
{
    float sum = 0.0;
    for (float& i: outData) {
        i = exp(i);
        sum += i;
    }

    std::cout << "Output:" << std::endl;
    for (int i = 0; i < outData.size(); i++) {
        outData[i] /= sum;

        std::cout << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4)
                  << outData[i] << " "
                  << "Class " << i << ": "
                  << std::string(static_cast<int>(std::floor(outData[i] * 10 + 0.5F)), '*')
                  << std::endl;
    }
    return 0;
}

int process(nvinfer1::IExecutionContext* context)
{
    int inputH = 28;
    int inputW = 28;
    srand(static_cast<unsigned>(time(nullptr)));
    std::vector<uint8_t> fileData(inputH * inputW);
    auto fpath = toolkit::getDataDir() / "mnist-input" / (std::to_string(rand() % 10) + ".pgm");
    readPGMFile(fpath.string(), fileData.data(), inputH, inputW);
    printPGMData(fileData.data(), inputH, inputW);

    std::vector<float> inData(inputH * inputW);
    std::vector<float> outData(10);

    float* d_in;
    float* d_out;
    CHECK(cudaMalloc(&d_in, sizeof(float) * inputH * inputW));
    CHECK(cudaMalloc(&d_out, sizeof(float) * 10));

    for (int i = 0; i < inputH * inputW; i++) {
        inData[i] = 1.0 - fileData[i] / 255.0;
    }
    CHECK(cudaMemcpy(d_in, inData.data(), sizeof(float) * inputH * inputW, cudaMemcpyHostToDevice));

    // context->setTensorAddress("Input3", d_in);
    // context->setTensorAddress("Plus214_Output_0", d_out);
    void* bindings[] = {d_in, d_out};
    bool status = context->executeV2(bindings);
    if (!status) {
        return false;
    }

    CHECK(cudaMemcpy(outData.data(), d_out, sizeof(float) * 10, cudaMemcpyDeviceToHost));
    if (int ret = verifyOutput(outData) != 0) {
        return ret;
    }

    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));

    return 0;
}

int infer(MyLogger& logger, nvinfer1::IHostMemory* plan)
{
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    if (!runtime) {
        return -1;
    }
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(plan->data(), plan->size());
    if (!engine) {
        return -1;
    }
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        return -1;
    }
    for (int32_t i = 0, e = engine->getNbIOTensors(); i < e; i++) {
        auto const name = engine->getIOTensorName(i);
        nvinfer1::Dims shape = engine->getTensorShape(name);
        nvinfer1::TensorIOMode mode = engine->getTensorIOMode(name);
        nvinfer1::DataType type = engine->getTensorDataType(name);
        ILOG("[{:>3s}] {} = {}({})", mode, name, type, shape);
    }

    if (int ret = process(context) != 0) {
        return ret;
    }

    delete context;
    delete engine;
    delete runtime;

    return 0;
}

int trt_mnist(int argc, char** argv)
{
    MyLogger logger;
    nvinfer1::IHostMemory* plan;

    if (int ret = build(logger, plan) != 0) {
        return ret;
    }
    if (int ret = infer(logger, plan) != 0) {
        return ret;
    }

    delete plan;

    return 0;
}
