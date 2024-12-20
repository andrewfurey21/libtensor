#include "../include/tensor.h"
#include "assert.h"
#include "stdio.h"

size_t top_radix(enum top op) {
  switch (op) {
  case NOOP:
    return 0;
  case RELU:
  case SUM_REDUCE:
  case RESHAPE:
  case EXPAND:
  case MAX_POOL_2D:
  case SQUARE:
  case SQRT:
  case EXP:
  case LOG:
  case RECIPROCAL:
    return 1;
  case ADD:
  case MUL:
  case CONV_2D:
  case SUB:
    return 2;
  default:
    assert(false && "This op is not implemented.");
  }
}

// TODO: theres gota be a macro for this right?
void print_op_string(enum top op) {
  switch (op) {
  case NOOP:
    printf("LEAF NODE\n");
    return;
  case RELU:
    printf("RELU\n");
    return;
  case NEG:
    printf("NEGATE\n");
    return;
  case SUM_REDUCE:
    printf("SUM REDUCE\n");
    return;
  case RESHAPE:
    printf("RESHAPE\n");
    return;
  case EXPAND:
    printf("EXPAND\n");
    return;
  case ADD:
    printf("ADD\n");
    return;
  case SUB:
    printf("SUB\n");
    return;
  case MUL:
    printf("MUL\n");
    return;
  case RECIPROCAL:
    printf("RECIPROCAL\n");
    return;
  case MAX_POOL_2D:
    printf("MAX POOL 2D\n");
    return;
  case CONV_2D:
    printf("CONV 2D\n");
    return;
  case SQUARE:
    printf("SQUARE\n");
    return;
  case SQRT:
    printf("SQRT\n");
    return;
  case EXP:
    printf("EXP\n");
    return;
  case LOG:
    printf("LOG\n");
    return;
  default:
    assert(false && "This op is not implemented.");
  }
}
