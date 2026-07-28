/* never-called float kernel stubs: satisfy TFLM float-path refs so the broken
   xa_nn *_f32 objects (with a /DISCARD/ section) are not pulled into the link. */
int xa_nn_elm_mul_f32xf32_f32(void){return -1;}
int xa_nn_vec_activation_min_max_f32_f32(void){return -1;}
