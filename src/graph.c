#include "stdbool.h"
#include "assert.h"
#include "stdio.h"
#include "../include/tensor.h"
#define MAX_NODES 100

bool already_visited(tgraph* net, tt* t) {
    for (size_t i = 0; i < net->size; i++) {
        if (net->nodes[i] == t) {
            return true;
        }
    }
    return false;
}

void topo_sort(tgraph* net, tt* current) {
    for (size_t i = 0; i < top_radix(current->op); i++) {
        tt* parent = current->parents[i];
        if (!already_visited(net, parent) && parent->requires_grad) {//all tensors in graph require grads
            topo_sort(net, parent);
        }
    }
    net->nodes[net->size] = current;
    net->size += 1;
    assert(net->size < MAX_NODES && "Too many nodes in the tgraph.");
}

tgraph* tgraph_build(tt* x) {
    assert(x->requires_grad && "Will not build graph on something that doesn't require gradients");
    tgraph* net = (tgraph*)malloc(sizeof(tgraph));
    net->nodes = (tt**)malloc(sizeof(tt*)*MAX_NODES);
    net->size = 0;
    net->training = true;
    topo_sort(net, x);
    return net;
}

void tgraph_free(tgraph* net) {
    for (size_t i = 0; i < net->size; i++) {
        tt_free(net->nodes[i]);
    }
    free(net);
}

void tgraph_zeroed(tgraph* net) {
    if (!net->training) return;
    for (uint32_t i = 0; i < net->size; i++) {
        struct tt* t = net->nodes[i];
        tt_to_zeros(t->grads);
    }
}

void tgraph_backprop(tgraph* net) {
    if (!net->training) return;
    ttuple* unit_shape = ttuple_build(1, 1);
    tt* current = net->nodes[net->size-1];
    assert(ttuple_equal(current->view->shape, unit_shape) && "Last tensor must be scalar");
    assert(current->requires_grad && "Can't do backprop on tensor without grads");
    free(unit_shape);

    tt* grads = tt_ones(current->view->shape, false);
    tt_free(current->grads);
    current->grads = grads;

    for (int32_t i = net->size-2; i >= 0; i--) {
        if (current->op) {
            current->_backwards(current);
        }
        current = net->nodes[i];
    }
}


// TODO: sort out the graph nicer.
// use bfs instead of dfs
void tgraph_print(tgraph* net, bool no_buffer, bool show_grads) {
    for (int i = 0; i < net->size; i++) {
        tt_print(net->nodes[i], no_buffer, show_grads);
        if (i < net->size-1) printf(" | \n");
    }
}

// gather_params
// save_tensors
