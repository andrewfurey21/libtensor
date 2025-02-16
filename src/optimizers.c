#include "../include/tensor.h"
#include "assert.h"

// void sgd(optimizer *optim) { // maybe just opt_params and net.
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
// }

// optimizer *optimizer_create(tgraph *net, uint64_t size,
//                               toptimizer_params *opt_params,
//                               void (*step)(optimizer *)) {
//   // assert(lr > 0 && lr <= 1 && "Learning rate should be between 0 and 1.");
//   //  TODO: free/copy opt_params, check lrs etc.
//   assert(size > 0 && "Must have 1 or more param.");
//   optimizer *optim = (optimizer *)malloc(sizeof(optimizer));
//   optim->net = net;
//   optim->opt_params = opt_params;
//   optim->step = step;
//   return optim;
// }
//
// void toptimizer_free(optimizer *opt) {
//   free(opt); // dont free net
// }

