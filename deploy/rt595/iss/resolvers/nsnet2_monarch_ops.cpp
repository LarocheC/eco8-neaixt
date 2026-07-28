/*
 * TFLite-Micro op resolver for the SPARSE (monarch) NSNet2 streaming model.
 *
 * The structured-linear variants do NOT lower to FULLY_CONNECTED like the dense
 * model. torch_structured's MonarchLinear becomes BATCH_MATMUL surrounded by
 * RESHAPE/TRANSPOSE (the block-permute), so the op set is materially different:
 * dense = 50 nodes / 16 FULLY_CONNECTED, monarch_8 = 128 nodes / 24 BATCH_MATMUL
 * + 24 TRANSPOSE + 35 RESHAPE.
 *
 * Op set from the flatbuffer histogram of nsnet2_monarch8_streaming.tflite.
 */
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

extern "C" tflite::MicroOpResolver &MODEL_GetOpsResolver(void)
{
    static tflite::MicroMutableOpResolver<12> s_resolver;
    s_resolver.AddAdd();
    s_resolver.AddBatchMatMul();
    s_resolver.AddConcatenation();
    s_resolver.AddLogistic();
    s_resolver.AddMul();
    s_resolver.AddPad();
    s_resolver.AddQuantize();
    s_resolver.AddReshape();
    s_resolver.AddSlice();
    s_resolver.AddSub();
    s_resolver.AddTanh();
    s_resolver.AddTranspose();
    return s_resolver;
}
