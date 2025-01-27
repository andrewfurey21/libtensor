#include "stdint.h"
#include "assert.h"
#include "stdlib.h"
#include "stdio.h"
#include "stdarg.h"
#include "../include/tensor.h"
#include <stdint.h>


ttuple* ttuple_build(uint32_t size, ...) {
    assert(MAX_ITEMS >= size);
    assert(size > 0 && "Size must be positive");
    va_list ap;

    ttuple* ret = (ttuple*)malloc(sizeof(ttuple));
    ret->size = size;
    ret->items = (int32_t*)malloc(size * sizeof(int32_t));
    va_start(ap, size);

    for (int i = 0; i < size; i++) {
        ret->items[i] = va_arg(ap, uint32_t);
    }
    va_end(ap);
    return ret;
}

ttuple* ttuple_zeros(uint32_t size) {
    assert(MAX_ITEMS >= size);
    assert(size > 0 && "Size must be positive");
    ttuple* ret = (ttuple*)malloc(sizeof(ttuple));
    ret->size = size;
    ret->items = (int32_t*)calloc(size, sizeof(int32_t));
    return ret;
}

ttuple* ttuple_ones(uint32_t size) {
    ttuple* ret = (ttuple*)malloc(sizeof(ttuple));
    ret->size = size;
    ret->items = (int32_t*)malloc(size * sizeof(int32_t));
    for (int i = 0; i < size; i++) {
        ret->items[i] = 1;
    }
    return ret;
}

ttuple* ttuple_add(ttuple* a, ttuple* b) {
    assert(a->size == b->size && "Cannot add tuples of different sizes");
    ttuple* ret = ttuple_zeros(a->size);
    for (uint32_t i = 0; i < a->size; i++) {
        ret->items[i] = a->items[i] + b->items[i];
    }
    return ret;
}

uint64_t ttuple_prod(ttuple* s) {
    uint64_t size = 1;
    for (uint32_t i = 0; i < s->size; i++) {
        size *= s->items[i];
    }
    return size;
}

ttuple* ttuple_copy(ttuple* other) {
    ttuple* copy = (ttuple*)malloc(sizeof(ttuple));
    copy->size = other->size;
    copy->items = (int32_t*)malloc(sizeof(int32_t) * copy->size);
    for (uint32_t i = 0; i < MAX_ITEMS || i < copy->size; i++) {
        copy->items[i] = other->items[i];
    }
    return copy;
}

bool ttuple_equal(ttuple* a, ttuple* b) {
    if (a->size != b->size) return false;
    for (int i = 0; i < a->size && i < MAX_ITEMS; i++) {
        if (a->items[i] != b->items[i]) return false;
    }
    return true;
}

ttuple* ttuple_div(ttuple* a, ttuple* b) {
    assert(a->size == b->size);
    ttuple* copy = ttuple_zeros(a->size);
    for (int i = 0; i < a->size; i++) {
        copy->items[i] = a->items[i] / b->items[i];
    }
    return copy;
}

void ttuple_free(ttuple* s) {
    free(s->items);
    free(s);
}

void ttuple_print(ttuple* s) {
    assert(s->size <= MAX_ITEMS && "Too many dimensions in tshape.");
    printf("(");
    for (size_t i = 0; i < s->size; i++) {
        printf("%d", s->items[i]);
        if (i < s->size - 1) printf(",");
    }
    printf(")\n");
}

ttuple* ttuple_add_one(ttuple* s) {
    assert(s->size < MAX_ITEMS - 1);
    ttuple* new_shape = ttuple_ones(s->size + 1);
    for (int i = 1; i < new_shape->size; i++) {
        new_shape->items[i] = s->items[i-1];
    }
    return new_shape;
}
