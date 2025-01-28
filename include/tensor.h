#include "stdarg.h"
#include "stdbool.h"
#include "stdint.h"
#include "stdlib.h"

#ifndef _TENSOR_H
#define _TENSOR_H

#define MAX_ITEMS 4

enum top { // need enough for: conv, batchnorm, maxpool, linear
  NOOP = 0,
  RELU,
  SUM_REDUCE,
  RESHAPE,
  EXPAND,
  ADD,
  SUB,
  MUL,
  RECIPROCAL,
  MAX_POOL_2D,
  NEG,
  CONV_2D,
  MATMUL,
  SQUARE,
  SQRT,
  EXP,
  LOG,
};

size_t top_radix(enum top);
void print_op_string(enum top op);

typedef struct {
  int32_t *items;
  uint32_t size;
} ttuple;

ttuple *ttuple_build(uint32_t size, ...);
ttuple *ttuple_zeros(uint32_t size);
ttuple *ttuple_ones(uint32_t size);
ttuple *ttuple_add(ttuple *a, ttuple *b);
uint64_t ttuple_prod(ttuple *s);
ttuple *ttuple_copy(ttuple *other);
bool ttuple_equal(ttuple *a, ttuple *b);
ttuple *ttuple_div(ttuple *a, ttuple *b);
void ttuple_free(ttuple *s);
void ttuple_print(ttuple *s);
ttuple* ttuple_add_one(ttuple* s);

// ttuple* ttuple_permute(ttuple* shape, ttuple* axes);
// bool tshape_duplicates(struct tshape* axes);

typedef struct {
  float *buffer;
  uint64_t refcount;
  uint64_t size;
} tstorage;

typedef struct {
  ttuple *shape;
  ttuple *strides;
  uint64_t offset;
} tview;

typedef struct tt tt;
struct tt {
  tstorage *data;
  tview *view;

  tt **parents;
  void (*_backwards)(tt *);
  enum top op;

  bool requires_grad;
  struct tt *grads;
};

// TODO: empty, logical index to physical index, setitem/item, arange, tostring
// (cache in repr, use inside print), view/reshape

tt *tt_zeros(ttuple *s, bool requires_grad);
tt *tt_ones(ttuple *s, bool requires_grad);
tt *tt_from_buffer(ttuple *s, float *buffer, bool requires_grads);
float tt_getindex(tt *self, ttuple *s);
void tt_setindex(tt *self, ttuple *s, float num);
tt *tt_fill(ttuple *s, float fill_value, bool requires_grad);
tt *tt_linspace(ttuple *s, float min, float max, bool requires_grad);
tt *tt_uniform(ttuple *s, float min, float max, bool requires_grad);
tt *tt_uniformint(ttuple *s, float min, float max, bool requires_grad);
void tt_copy_buffer(tt *dest, tt *src);
tt *tt_copy(tt *original, bool requires_grad);
void tt_to_zeros(tt *t);
void tt_to_n(tt *t, float n);
void tt_print(tt *t, bool show_buffer, bool show_grads);
tt *tt_view(tt *tensor, tview *view);
void tt_free(tt *t);
bool tt_equal(tt* a, tt*b);
tt* tt_linear_init(ttuple* shape, int in_features, bool requires_grad);
tt* tt_conv_init(ttuple* shape, int in_channels, int kernel_size, bool requires_grad);

// ops
tt *tt_add(tt *a, tt *b);
tt *tt_sub(tt *a, tt *b);
tt *tt_mul(tt *a, tt *b);
tt *tt_sum(tt *a, int axis);
tt *tt_relu(tt *a);
tt *tt_reshape(tt *a, ttuple *new_shape);
tt *tt_neg(tt *a);
tt *tt_expand(tt *a, uint64_t axis, uint64_t amount);
tt *tt_maxpool2d(tt *input, int kernel_size);
tt *tt_conv2d(tt *input, tt *kernels);
tt *tt_matmul(tt *input, tt *other);
tt* tt_square(tt* input);
tt* tt_sqrt(tt* input);
tt* tt_exp(tt* input);
tt* tt_log(tt* input);
tt *tt_reciprocal(tt *a);

// computational graph
typedef struct {
  struct tt **nodes;
  size_t size;
  bool training;
} tgraph;

tgraph *tgraph_build(tt *x);
void tgraph_free(tgraph *net);
void tgraph_zeroed(tgraph *net);
void tgraph_backprop(tgraph *net);
void tgraph_print(tgraph *net, bool no_buffer, bool show_grads);

// nn
typedef struct {
  float learning_rate;
} toptimizer_params;

typedef struct toptimizer toptimizer;
struct toptimizer {
  tgraph *net;
  toptimizer_params *opt_params;
  void (*step)(toptimizer *optim);
};

toptimizer *toptimizer_build(tt **params, uint64_t size,
                             toptimizer_params *opt_params,
                             void (*step)(toptimizer *));
void toptimizer_free(toptimizer *topt);

#endif
