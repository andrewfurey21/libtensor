#include "stdlib.h"
#include "stdint.h"
#include "stdbool.h"
#include "stdio.h"
#include "assert.h"
#include "string.h"

#include "teensygrad.h"

uint64_t buflen(struct teensy_shape* s) {
    uint64_t size = 1;
    for (uint32_t i = 0; i < s->size; i++) {
        size *= s->dims[i];
    }
    return size;
}

struct teensy_tensor* teensy_tensor_zeros(struct teensy_shape* s, bool requires_grad, struct teensy_tensor** parents, enum teensy_op op) {
    uint64_t size = buflen(s);
    float* buffer = (float*)calloc(size, size*(uint64_t)sizeof(float));

    struct teensy_shape* teensy_shape_copy = teensy_shape_create(s->dims, s->size);

    struct teensy_tensor* grads;
    if (requires_grad) {
        grads = teensy_tensor_zeros(s, false, NULL, NOOP);
    }

    struct teensy_tensor* t = (struct teensy_tensor*)malloc(sizeof(struct teensy_tensor));
    //TODO: copy everything!
    t->shape = teensy_shape_copy;
    t->buffer = buffer;
    t->size = size;
    t->requires_grad = requires_grad;
    t->parents = parents;
    t->op = op;
    t->grads = grads;
    t->_backwards = NULL;
    return t;
}

void teensy_tensor_destroy(struct teensy_tensor* t) {
    teensy_shape_destroy(t->shape);
    free(t->buffer);
    free(t->parents);
    if (t->requires_grad) {
        teensy_tensor_destroy(t->grads);
    }
    free(t);
}

struct teensy_tensor* teensy_tensor_ones(struct teensy_shape* s, bool requires_grad, struct teensy_tensor** parents, enum teensy_op op) {
    struct teensy_tensor* ones = teensy_tensor_zeros(s, requires_grad, parents, op);
    for (size_t i = 0; i < ones->size; i++) {
        ones->buffer[i] = 1.0f;
    }
    return ones;
}

struct teensy_tensor* teensy_tensor_from_buffer(struct teensy_shape* s, float* buffer, bool requires_grad) {
    struct teensy_tensor* ret = (struct teensy_tensor*)malloc(sizeof(struct teensy_tensor));
    struct teensy_shape* teensy_shape_copy = teensy_shape_create(s->dims, s->size);
    ret->shape = teensy_shape_copy;
    uint64_t size = buflen(s);

    ret->buffer = buffer;
    ret->size = size;

    struct teensy_tensor* grads = NULL;
    if (requires_grad) {
        grads = teensy_tensor_zeros(s, false, NULL, NOOP);
    }
    ret->op = NOOP;
    ret->parents = NULL;
    ret->requires_grad = requires_grad;
    ret->_backwards = NULL;
    ret->grads = grads;
    return ret;
}

//TODO: clean up, print teensy_shape as well, better formatting
void teensy_tensor_print(struct teensy_tensor* t) {
    printf("Teensy_Tensor:[");
    for (int i = 0; i < t->size; i++) {
        printf("%f, ", t->buffer[i]);
    }
    printf("]\n");
}

//TODO:zeroing out function
//TODO:take requires grad into account!
////TODO:could easily be vectorized.
void _add_backwards(struct teensy_tensor* self) {
    struct teensy_tensor* grads_0 = teensy_tensor_add(self->grads, self->parents[0]->grads, false);
    struct teensy_tensor* grads_1 = teensy_tensor_add(self->grads, self->parents[1]->grads, false);

    teensy_tensor_destroy(self->parents[0]->grads);
    teensy_tensor_destroy(self->parents[1]->grads);

    self->parents[0]->grads = grads_0;
    self->parents[1]->grads = grads_1;
}

struct teensy_tensor* teensy_tensor_add(struct teensy_tensor* a, struct teensy_tensor* b, bool requires_grad) {
    assert(teensy_tensor_same_shape(a, b) && "Teensy_Tensors are not the same teensy_shape.");
    struct teensy_shape* teensy_shape_copy = teensy_shape_create(a->shape->dims, a->shape->size);

    struct teensy_tensor** parents = (struct teensy_tensor**)malloc(op_radix(ADD)*sizeof(struct teensy_tensor*));
    parents[0] = a;
    parents[1] = b;

    struct teensy_tensor* t = teensy_tensor_zeros(teensy_shape_copy, requires_grad, parents, ADD);
    t->_backwards = &_add_backwards;
    for (uint64_t i = 0; i < a->size; i++) {
        t->buffer[i] = a->buffer[i] + b->buffer[i];
    }

    return t;
}

void _mul_backwards(struct teensy_tensor* self) {
    struct teensy_tensor* grads_1 = teensy_tensor_mul(self->grads, self->parents[0], false);
    struct teensy_tensor* grads_0 = teensy_tensor_mul(self->grads, self->parents[1], false);

    struct teensy_tensor* acc_grads_0 = teensy_tensor_add(grads_0, self->parents[0]->grads, false);
    struct teensy_tensor* acc_grads_1 = teensy_tensor_add(grads_1, self->parents[1]->grads, false);

    teensy_tensor_destroy(self->parents[0]->grads);
    teensy_tensor_destroy(self->parents[1]->grads);

    teensy_tensor_destroy(grads_1);
    teensy_tensor_destroy(grads_0);

    self->parents[0]->grads = acc_grads_0;
    self->parents[1]->grads = acc_grads_1;
}


struct teensy_tensor* teensy_tensor_mul(struct teensy_tensor* a, struct teensy_tensor* b, bool requires_grad) {
    assert(teensy_tensor_same_shape(a, b) && "Teensy_Tensors are not the same teensy_shape.");
    struct teensy_shape* teensy_shape_copy = teensy_shape_create(a->shape->dims, a->shape->size);

    struct teensy_tensor** parents = (struct teensy_tensor**)malloc(op_radix(MUL)*sizeof(struct teensy_tensor*));
    parents[0] = a;
    parents[1] = b;

    struct teensy_tensor* t = teensy_tensor_zeros(teensy_shape_copy, requires_grad, parents, MUL);
    t->_backwards = &_mul_backwards;

    for (uint64_t i = 0; i < a->size; i++) {
        t->buffer[i] = a->buffer[i] * b->buffer[i];
    }

    return t;
}

void _relu_backwards(struct teensy_tensor* self) {
    struct teensy_tensor* grads = teensy_tensor_zeros(self->shape, false, NULL, NOOP);
    for (size_t i = 0; i < self->parents[0]->size; i++) {
        if (grads->buffer[i] < self->parents[0]->buffer[i]) {
            grads->buffer[i] = 1;
        }
    }
    teensy_tensor_destroy(self->parents[0]->grads);
    self->parents[0]->grads = grads;
}

struct teensy_tensor* teensy_tensor_relu(struct teensy_tensor* a, bool requires_grad) {
    struct teensy_shape* teensy_shape_copy = teensy_shape_create(a->shape->dims, a->shape->size);

    struct teensy_tensor** parents = (struct teensy_tensor**)malloc(op_radix(RELU)*sizeof(struct teensy_tensor*));
    parents[0] = a;

    struct teensy_tensor* t = teensy_tensor_zeros(teensy_shape_copy, requires_grad, parents, RELU);
    t->_backwards = &_relu_backwards;

    for (uint64_t i = 0; i < a->size; i++) {
        t->buffer[i] = a->buffer[i] * (a->buffer[i] > 0);
    }

    return t;
}

void _sum_backwards(struct teensy_tensor* self) {
    //struct teensy_tensor* grads = teensy_tensor_add(self->grads, self->parents[0]->grads, false);
    struct teensy_tensor* grads = teensy_tensor_ones(self->parents[0]->shape, false, NULL, NOOP);

    teensy_tensor_destroy(self->parents[0]->grads);

    self->parents[0]->grads = grads;
}

struct teensy_tensor* teensy_tensor_sum(struct teensy_tensor* a, bool requires_grad) {
    struct teensy_shape* teensy_shape_copy = teensy_shape_create_1d(1);

    struct teensy_tensor** parents = (struct teensy_tensor**)malloc(op_radix(SUM_REDUCE)*sizeof(struct teensy_tensor*));
    parents[0] = a;

    struct teensy_tensor* t = teensy_tensor_zeros(teensy_shape_copy, requires_grad, parents, SUM_REDUCE);
    t->_backwards = &_sum_backwards;

    double sum = 0.0f;
    for (uint64_t i = 0; i < a->size; i++) {
        sum += a->buffer[i];
    }

    t->buffer[0] = sum;

    return t;
}

void teensy_tensor_to_zeros(struct teensy_tensor* t) {
    memset(t->buffer, 0, t->size);
}

bool teensy_tensor_same_shape(struct teensy_tensor* a, struct teensy_tensor* b) {
    if (a->shape->size != b->shape->size) {
        return false;
    }
    for (uint32_t i = 0; i < a->shape->size; i++) {
        if (a->shape->dims[i] != b->shape->dims[i]) {
            return false;
        }
    }
    return true;
}
