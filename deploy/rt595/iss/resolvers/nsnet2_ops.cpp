/*
 * TFLite-Micro op resolver for the NSNet2 (plain GRU) streaming model.
 * int8 op set (host flatbuffer inspection): ADD, CONCATENATION, FULLY_CONNECTED,
 * LOGISTIC, MUL, QUANTIZE, RESHAPE, SLICE, SUB, TANH.
 */
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

extern "C" tflite::MicroOpResolver &MODEL_GetOpsResolver(void)
{
    static tflite::MicroMutableOpResolver<10> s_resolver;
    s_resolver.AddAdd();
    s_resolver.AddConcatenation();
    s_resolver.AddFullyConnected();
    s_resolver.AddLogistic();
    s_resolver.AddMul();
    s_resolver.AddQuantize();
    s_resolver.AddReshape();
    s_resolver.AddSlice();
    s_resolver.AddSub();
    s_resolver.AddTanh();
    return s_resolver;
}
