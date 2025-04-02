#include "../include/tensor.h"

void sgd_step(graph *network, float learning_rate) {
  for (uint64_t i = 0; i < network->size; i++) {
    tensor *t = network->nodes[i];
    if (t->op == NOOP) {
      tensor *lrs = tensor_fill(t->vw->shape, -learning_rate, false);
      tensor *updated_grads = tensor_mul(lrs, t->grads, false);
      tensor *updated_params = tensor_add(updated_grads, t, false);
      tensor_copy_buffer(t, updated_params);
      tensor_free(lrs);
      tensor_free(updated_grads);
      tensor_free(updated_params);
    }
  }
}
