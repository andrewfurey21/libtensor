#include "stdbool.h"
#include "assert.h"
#include "stdio.h"
#include "../include/tensor.h"
#define MAX_NODES 100

bool already_visited(graph* net, tensor* t) {
    for (size_t i = 0; i < net->size; i++) {
        if (net->nodes[i] == t) {
            return true;
        }
    }
    return false;
}

void topo_sort(graph* net, tensor* current) {
    for (size_t i = 0; i < tensor_op_operands(current->op); i++) {
        tensor* parent = current->parents[i];
        if (!already_visited(net, parent) && parent->requires_grad) {//all tensors in graph require grads
            topo_sort(net, parent);
        }
    }
    net->nodes[net->size] = current;
    net->size += 1;
    assert(net->size < MAX_NODES && "Too many nodes in the graph.");
}

// TODO: 
// 1. should memory of tensors be created lazily? 
// maybe only allocate memory once. when graph is built. never reallocate.
// 2. keep storage abstraction. this is where you build in broadcasting. make storage.c
// tensor.c should only have operations and their backwards functions.
graph* graph_build(tensor* x) {
    assert(x->requires_grad && "Will not build graph on something that doesn't require gradients");
    graph* net = (graph*)malloc(sizeof(graph));
    net->nodes = (tensor**)malloc(sizeof(tensor*)*MAX_NODES);
    net->size = 0;
    net->training = true;
    topo_sort(net, x);
    return net;
}

void graph_free(graph* net) {
    for (size_t i = 0; i < net->size; i++) {
        tensor_free(net->nodes[i]);
    }
    free(net);
}

void graph_zeroed(graph* net) {
    if (!net->training) return;
    for (uint32_t i = 0; i < net->size; i++) {
        struct tensor* t = net->nodes[i];
        tensor_to_zeros(t->grads);
    }
}

void graph_backprop(graph* net) {
    if (!net->training) return;
    intarray* unit_shape = intarray_build(1, 1);
    tensor* current = net->nodes[net->size-1];
    assert(intarray_equal(current->dview->shape, unit_shape) && "Last tensor must be scalar");
    assert(current->requires_grad && "Can't do backprop on tensor without grads");
    free(unit_shape);

    tensor* grads = tensor_ones(current->dview->shape, false);
    tensor_free(current->grads);
    current->grads = grads;

    for (int32_t i = net->size-2; i >= 0; i--) {
        if (current->op) {
            current->_backwards(current);
        }
        current = net->nodes[i];
    }
}


void graph_print(graph* net, bool no_buffer, bool show_grads) {
    for (int i = 0; i < net->size; i++) {
        tensor_print(net->nodes[i], no_buffer, show_grads);
        if (i < net->size-1) printf(" | \n");
    }
}
