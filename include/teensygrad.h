#include "stdint.h"
#include "stdarg.h"
#include "stdbool.h"
#include "stdlib.h"
#include <stdint.h>

#ifndef _TEENSYGRAD_H
#define _TEENSYGRAD_H


enum top {//need enough for: conv, batchnorm, maxpool, linear, relu
    NOOP=0,
    RELU,// TODO: remove, use min/max instead.
    NEG,
    SUM_REDUCE,
    RESHAPE,
    EXPAND,
    ADD,
    MUL,
};

size_t top_radix(enum top);
void print_op_string(enum top op);

typedef struct {
    int32_t* items;
    uint32_t size;
} ttuple;

ttuple* ttuple_build(uint32_t size, ...);
uint64_t ttuple_mul(ttuple* s);
ttuple* ttuple_copy(ttuple* other);
bool ttuple_equal(ttuple* a, ttuple* b);
void ttuple_free(ttuple* s);
void ttuple_print(ttuple* s);

//ttuple* ttuple_permute(ttuple* shape, ttuple* axes);
//bool tshape_duplicates(struct tshape* axes);

typedef struct {
    float* buffer;
    uint64_t refcount;
    uint64_t size;
} tstorage;

struct tt {
    ttuple* shape;
    ttuple* strides;
    uint64_t offset;
    tstorage* data;

    struct tt** parents;
    void (*_backwards)(struct tt*);
    enum top op;

    bool requires_grad;
    struct tt* grads;
};

typedef struct tt tt;

// TODO: empty, logical index to physical index, setitem/item, arange, tostring (cache in repr, use inside print), view/reshape

tt* tt_zeros(ttuple* s, bool requires_grad);
tt* tt_ones(ttuple* s, bool requires_grad);
tt* tt_from_buffer(ttuple* s, float* buffer, bool requires_grads);
tt* tt_fill(ttuple* s, float fill_value, bool requires_grad);
tt* tt_linspace(ttuple* s, float min, float max, bool requires_grad);
tt* tt_uniform(ttuple* s, float min, float max, bool requires_grad);
tt* tt_uniformint(ttuple* s, float min, float max, bool requires_grad);
void tt_copy_buffer(tt* dest, tt* src);
tt* tt_copy(tt* original, bool requires_grad);
void tt_to_zeros(tt* t);
void tt_to_n(struct tt* t, float n);
void tt_print(tt* t);
void tt_free(tt* t);

//elementwise ops
struct tt* tt_add(struct tt* a, struct tt* b);
struct tt* tt_neg(struct tt* a);
struct tt* tt_mul(struct tt* a, struct tt* b);
struct tt* tt_max(struct tt* a, struct tt* b);
//reduce ops
struct tt* tt_sum(struct tt* a, struct tshape* axes);
//movement ops
// TODO: use new storage abstraction
struct tt* tt_permute(struct tt* t, struct tshape* axes);
struct tt* tt_expand(struct tt* a, struct tshape* shape);
struct tt* tt_reshape(struct tt* a, struct tshape* new_shape);


// //computational graph
// struct tgraph {
//     struct tt** nodes;
//     size_t size;
//     bool training;
// };
// struct tgraph* tgraph_build(struct tt* x);
// void tgraph_free(struct tgraph* net);
// void tgraph_zeroed(struct tgraph* net);
// //backprop
// void tgraph_backprop(struct tgraph* net);
//
// struct toptimizer {
//     struct tgraph* net;
//     struct toptimizer_params* opt_params;
//     void (*step)(struct toptimizer* optim);
// };
//
// struct toptimizer_params {
//     float learning_rate;
// };
//
// struct toptimizer* toptimizer_build(struct tt** params, uint64_t size, struct toptimizer_params* opt_params , void (*step)(struct toptimizer*));
// void toptimizer_free(struct toptimizer* topt);
//
// //optimization steps
// void tsgd(struct toptimizer* optim);
#endif
