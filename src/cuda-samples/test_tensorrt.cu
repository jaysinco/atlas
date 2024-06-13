#include "./common.cuh"
#include <cuda_runtime.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"

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
        // context->setTensorAddress(name, buffers.getDeviceBuffer(name));
        ILOG("IONAME={}", name);
    }

    delete context;
    delete engine;
    delete runtime;

    return 0;
}

int test_tensorrt(int argc, char** argv)
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
