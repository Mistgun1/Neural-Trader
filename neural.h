#ifndef NEURAL_H
#define NEURAL_H
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <math.h>
#include <pthread.h>
#include <string.h>

typedef struct neuron{
    double value;
    int weight_count;
    double* weights;
    double bias;
}neuron;

typedef struct layer{
    int neuron_count;
    neuron* neurons;
}layer;

#define foreach_layer(LAYER_COUNT, _iter) for(int _iter = 0; _iter < LAYER_COUNT; _iter++)
#define foreach_neuron(layer, _iter) for(int _iter = 0; _iter < (layer).neuron_count; _iter++)
#define foreach_weight(neuron, _iter) for(int _iter = 0; _iter < (neuron).weight_count; _iter++)
#define get_neuron(layer, index) (layer).neurons[index]
#define get_neuron_count(layer) (layer).neuron_count
#define get_neuron_value(layer, index) get_neuron(layer, index).value
#define get_neuron_bias(neuron) get_neuron(neuron, 0).bias
#define get_neuron_weights(neuron, index) get_neuron(neuron, index).weights
#define random_function(min, max) rand() / (double)RAND_MAX * (max - min) + min

// Adjust for better results
#define LAYER_COUNT 3
#define EPOCH 100
#define LEARNING_RATE 0.0001
#define DELTA 0.0001 // Derivative small step -- the smaller the more accurate
#define STEP_SIZE 10

double sigmoid_func(double x);
double random_func(double min, double max);
layer* create_layer(int neuron_count, layer* previous_layer,bool random_fill);
void free_layer(layer* layer);
void calculate_neuron_value(neuron* neuron, layer* previous_layer);
void calculate_layer(struct layer* layer, struct layer* previous_layer);
double cost_func(layer* layers[LAYER_COUNT], int correct_answer);
void gradient_descent(layer* gradient[LAYER_COUNT - 1], int step_size, layer* layers[LAYER_COUNT], int correct_answer, int epoch);
void calculate_average_gradient(layer* gradient[LAYER_COUNT - 1], layer* average_gradient[LAYER_COUNT - 1], int step_size);
void clear_gradient(layer* gradient[LAYER_COUNT - 1]);
#endif // NEURAL_H
