#include "./common.h"

MyErrCode describeModel(torch::nn::Module const& model, std::string const& prefix)
{
    size_t param_count = 0;
    for (auto const& param: model.parameters(true)) {
        param_count += param.numel();
    }
    ILOG("{}[{}] ({} params)", prefix, model.name(), param_count);
    for (auto const& param: model.named_parameters(false)) {
        ILOG("{}..{}: {} ({} params)", prefix, param.key(), param.value().sizes(),
             param.value().numel());
    }
    for (auto const& child: model.named_children()) {
        describeModel(*child.value(), prefix + "..");
    }
    return MyErrCode::kOk;
}

InceptionImpl::InceptionImpl(InceptionOptions const& o)
{
    using namespace torch::nn;
    b1_1_ = register_module("b1_1", Conv2d(Conv2dOptions(o.c0(), o.c1(), 1)));
    b2_1_ = register_module("b2_1", Conv2d(Conv2dOptions(o.c0(), o.c2()[0], 1)));
    b2_2_ = register_module("b2_2", Conv2d(Conv2dOptions(o.c2()[0], o.c2()[1], 3).padding(1)));
    b3_1_ = register_module("b3_1", Conv2d(Conv2dOptions(o.c0(), o.c3()[0], 1)));
    b3_2_ = register_module("b3_2", Conv2d(Conv2dOptions(o.c3()[0], o.c3()[1], 5).padding(2)));
    b4_1_ = register_module("b4_1", MaxPool2d(MaxPool2dOptions(3).stride(1).padding(1)));
    b4_2_ = register_module("b4_2", Conv2d(Conv2dOptions(o.c0(), o.c4(), 1)));
}

torch::Tensor InceptionImpl::forward(torch::Tensor x)
{
    auto b1 = torch::relu(b1_1_(x));
    auto b2 = torch::relu(b2_2_(torch::relu(b2_1_(x))));
    auto b3 = torch::relu(b3_2_(torch::relu(b3_1_(x))));
    auto b4 = torch::relu(b4_2_(b4_1_(x)));
    return torch::cat({b1, b2, b3, b4}, 1);
}

ResidualImpl::ResidualImpl(ResidualOptions const& o)
{
    using namespace torch::nn;
    conv1_ = register_module(
        "conv1", Conv2d(Conv2dOptions(o.in_c(), o.out_c(), 3).padding(1).stride(o.stride())));

    conv2_ = register_module(
        "conv2", Conv2d(Conv2dOptions(o.out_c(), o.out_c(), 3).padding(1).stride(o.stride())));

    if (o.in_c() != o.out_c()) {
        conv3_ = register_module("conv3",
                                 Conv2d(Conv2dOptions(o.in_c(), o.out_c(), 1).stride(o.stride())));
    }

    bn1_ = register_module("bn1", BatchNorm2d(BatchNorm2dOptions(o.out_c())));
    bn2_ = register_module("bn2", BatchNorm2d(BatchNorm2dOptions(o.out_c())));
}

torch::Tensor ResidualImpl::forward(torch::Tensor x)
{
    auto y = torch::relu(bn1_(conv1_(x)));
    y = bn2_(conv2_(y));
    if (conv3_) {
        x = conv3_(x);
    }
    return torch::relu(y + x);
}
