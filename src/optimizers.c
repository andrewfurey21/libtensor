#include "../include/tensor.h"
#include "assert.h"

void tsgd(toptimizer *optim) { // maybe just opt_params and net.
  // for (uint64_t i = 0; i < optim->net->size; i++) {
    // tt *t = optim->net->nodes[i];
    //
    // tt *lrs = tt_fill(t->view->shape, -optim->opt_params->learning_rate, false);
    // tt *updated_grads = tt_mul(lrs, t->grads);
    //
    // tt *updated_params = tt_add(updated_grads, t);
    //
    // tt_copy_buffer(t, updated_params);
    //
    // tt_free(lrs);
    // tt_free(updated_grads);
    // tt_free(updated_params);
  // }
}

toptimizer *toptimizer_create(tgraph *net, uint64_t size,
                              toptimizer_params *opt_params,
                              void (*step)(toptimizer *)) {
  // assert(lr > 0 && lr <= 1 && "Learning rate should be between 0 and 1.");
  //  TODO: free/copy opt_params, check lrs etc.
  assert(size > 0 && "Must have 1 or more param.");
  toptimizer *optim = (toptimizer *)malloc(sizeof(toptimizer));
  optim->net = net;
  optim->opt_params = opt_params;
  optim->step = step;
  return optim;
}

void toptimizer_free(toptimizer *topt) {
  free(topt); // dont free net
}

