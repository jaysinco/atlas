#pragma once
#include <cstdint>
#include "toolkit/error.h"

MyErrCode helloWorld(int argc, char** argv);
MyErrCode checkDevice(int argc, char** argv);
MyErrCode sumMatrix(int argc, char** argv);
MyErrCode reduceInteger(int argc, char** argv);
MyErrCode nestedHelloWorld(int argc, char** argv);
MyErrCode globalVariable(int argc, char** argv);
MyErrCode cufftTest(int argc, char** argv);
MyErrCode juliaSet(int argc, char** argv);
MyErrCode dotProduct(int argc, char** argv);
MyErrCode rayTracing(int argc, char** argv);
MyErrCode txiGaussian(int argc, char** argv);
MyErrCode txiGuided(int argc, char** argv);
MyErrCode trtMnist(int argc, char** argv);
MyErrCode contrastLG(int argc, char** argv);