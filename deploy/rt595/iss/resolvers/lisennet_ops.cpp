/*
 * TFLite-Micro op resolver for the LiSenNet streaming SE model.
 * The exact op set of the int8 graph (from `pyocd`-free host inspection of the
 * .tflite): ADD, CONCATENATION, CONV_2D, DEPTHWISE_CONV_2D, LOGISTIC, MUL, PAD,
 * QUANTIZE, RESHAPE, SLICE, SUB, TANH, TRANSPOSE, TRANSPOSE_CONV. Keep this list in
 * sync if the exported graph's ops change (gen_io_layout.py prints them).
 */
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

extern "C" tflite::MicroOpResolver &MODEL_GetOpsResolver(void)
{
    static tflite::MicroMutableOpResolver<14> s_resolver;
    s_resolver.AddAdd();
    s_resolver.AddConcatenation();
    s_resolver.AddConv2D();
    s_resolver.AddDepthwiseConv2D();
    s_resolver.AddLogistic();
    s_resolver.AddMul();
    s_resolver.AddPad();
    s_resolver.AddQuantize();
    s_resolver.AddReshape();
    s_resolver.AddSlice();
    s_resolver.AddSub();
    s_resolver.AddTanh();
    s_resolver.AddTranspose();
    s_resolver.AddTransposeConv();
    return s_resolver;
}
