/*
 * TFLite-Micro op resolver for the ConvFSENet (sliced/no-compress) streaming model.
 * int8 op set: ADD, CONCATENATION, CONV_2D, DEPTHWISE_CONV_2D, GATHER_ND, LOGISTIC,
 * QUANTIZE, RESHAPE, SLICE, TRANSPOSE. (POW dropped via extractor='mag'; REDUCE_ALL
 * + SELECT_V2 dropped by the strided-slice FIFO tap — see build_convfsenet_deploy.)
 */
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

extern "C" tflite::MicroOpResolver &MODEL_GetOpsResolver(void)
{
    static tflite::MicroMutableOpResolver<10> s_resolver;
    s_resolver.AddAdd();
    s_resolver.AddConcatenation();
    s_resolver.AddConv2D();
    s_resolver.AddDepthwiseConv2D();
    s_resolver.AddGatherNd();
    s_resolver.AddLogistic();
    s_resolver.AddQuantize();
    s_resolver.AddReshape();
    s_resolver.AddSlice();
    s_resolver.AddTranspose();
    return s_resolver;
}
