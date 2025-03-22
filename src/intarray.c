#include "stdint.h"
#include "assert.h"
#include "stdlib.h"
#include "stdio.h"
#include "stdarg.h"
#include "assert.h"
#include "../include/tensor.h"
#include <stdint.h>


intarray* intarray_build(uint32_t size, ...) {
    assert(MAX_ITEMS >= size);
    assert(size > 0 && "Size must be positive");
    va_list ap;

    intarray* ret = (intarray*)malloc(sizeof(intarray));
    ret->size = size;
    ret->items = (int32_t*)malloc(size * sizeof(int32_t));
    va_start(ap, size);

    for (int i = 0; i < size; i++) {
        ret->items[i] = va_arg(ap, uint32_t);
    }
    va_end(ap);
    return ret;
}

intarray* intarray_zeros(uint32_t size) {
    assert(MAX_ITEMS >= size);
    assert(size > 0 && "Size must be positive");
    intarray* ret = (intarray*)malloc(sizeof(intarray));
    ret->size = size;
    ret->items = (int32_t*)calloc(size, sizeof(int32_t));
    return ret;
}

intarray* intarray_ones(uint32_t size) {
    intarray* ret = (intarray*)malloc(sizeof(intarray));
    ret->size = size;
    ret->items = (int32_t*)malloc(size * sizeof(int32_t));
    for (int i = 0; i < size; i++) {
        ret->items[i] = 1;
    }
    return ret;
}

intarray* intarray_add(intarray* a, intarray* b) {
    assert(a->size == b->size && "Cannot add tuples of different sizes");
    intarray* ret = intarray_zeros(a->size);
    for (uint32_t i = 0; i < a->size; i++) {
        ret->items[i] = a->items[i] + b->items[i];
    }
    return ret;
}

uint64_t intarray_prod(intarray* s) {
    uint64_t size = 1;
    for (uint32_t i = 0; i < s->size; i++) {
        size *= s->items[i];
    }
    return size;
}

intarray* intarray_copy(intarray* other) {
    intarray* copy = (intarray*)malloc(sizeof(intarray));
    copy->size = other->size;
    copy->items = (int32_t*)malloc(sizeof(int32_t) * copy->size);
    for (uint32_t i = 0; i < MAX_ITEMS || i < copy->size; i++) {
        copy->items[i] = other->items[i];
    }
    return copy;
}

bool intarray_equal(intarray* a, intarray* b) {
    if (a->size != b->size) return false;
    for (int i = 0; i < a->size && i < MAX_ITEMS; i++) {
        if (a->items[i] != b->items[i]) return false;
    }
    return true;
}

intarray* intarray_div(intarray* a, intarray* b) {
    assert(a->size == b->size);
    intarray* copy = intarray_zeros(a->size);
    for (int i = 0; i < a->size; i++) {
        copy->items[i] = a->items[i] / b->items[i];
    }
    return copy;
}

void intarray_free(intarray* s) {
    free(s->items);
    free(s);
}

void intarray_print(intarray* s) {
    assert(s->size <= MAX_ITEMS && "Too many dimensions in tshape.");
    printf("(");
    for (size_t i = 0; i < s->size; i++) {
        printf("%d", s->items[i]);
        if (i < s->size - 1) printf(",");
    }
    printf(")\n");
}

intarray* intarray_pad_left(intarray* s, int new_size) {
    assert(new_size >= s->size && new_size <= MAX_ITEMS);
    intarray* new_shape = intarray_ones(new_size);
    for (int i = new_size - s->size; i < new_shape->size; i++) {
        new_shape->items[i] = s->items[i-(new_size-s->size)];
    }
    return new_shape;
}

intarray* intarray_squeeze(intarray* s) {
    assert(s->size > 0 && s->size <= MAX_ITEMS);
    int count = 0;
    for (int i = 0; i < s->size; i++) {
        count += s->items[i] == 1;
    }

    intarray* new = intarray_zeros(s->size - count);

    count = 0;
    for (int i = 0; i < s->size; i++) {
        if (s->items[i] != 1) {
            new->items[count++] = s->items[i];
        }
    }
    return new;
}
