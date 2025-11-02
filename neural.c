#include "neural.h"

double sigmoid_func(double x){
    return 1 / (1 + exp(-x));
}

double random_func(double min, double max){
    return min + (max - min) * ((double)rand() / (double)RAND_MAX);
}

layer* create_layer(int neuron_count, layer* previous_layer,bool random_fill){
    layer* new_layer = malloc(sizeof(layer));
    new_layer->neuron_count = neuron_count;
    new_layer->neurons = malloc(sizeof(neuron) * neuron_count);
    if (previous_layer != NULL){
        foreach_neuron(*new_layer, i){
            new_layer->neurons[i].weights = malloc(sizeof(double) * get_neuron_count(*previous_layer));
            new_layer->neurons[i].bias = random_func(-1, 1);
            new_layer->neurons[i].weight_count = get_neuron_count(*previous_layer);
            if (random_fill){
                foreach_weight(new_layer->neurons[i], j){
                    new_layer->neurons[i].weights[j] = random_func(-1, 1);
                }
            }
        }
    }
    return new_layer;
}

void free_layer(layer* layer){
    foreach_neuron(*layer, i){
        free(layer->neurons[i].weights);
    }
    free(layer->neurons);
    free(layer);
}

void calculate_neuron_value(neuron* neuron, layer* previous_layer){
    double sum = 0;
    foreach_neuron(*previous_layer, i){
        sum += neuron->weights[i] * previous_layer->neurons[i].value;
    }
    neuron->value = sigmoid_func(sum + neuron->bias);
}

void calculate_layer(struct layer* layer, struct layer* previous_layer){
    for (int i = 0; i < layer->neuron_count; i++){
        calculate_neuron_value(layer->neurons + i, previous_layer);
    }
}

double cost_func(layer* layers[LAYER_COUNT], int correct_answer){    
    for (int i = 1; i < LAYER_COUNT; i++){
        calculate_layer(layers[i], layers[i - 1]);
    }
    double sum = 0;
    for(int i = 0; i < layers[LAYER_COUNT - 1]->neuron_count; i++){
        if (correct_answer == i){
            sum += pow(layers[LAYER_COUNT - 1]->neurons[i].value - 1, 2);
        }
        else{
            sum += pow(layers[LAYER_COUNT - 1]->neurons[i].value, 2);
        }
    }
    return sum;
}

void gradient_descent(layer* gradient[LAYER_COUNT - 1], int step_size, layer* layers[LAYER_COUNT], int correct_answer, int epoch){
    
    // gradient is a layer array with 1 less layer than layers (the first layer is the input layer)
    // loop using layers and change gradient layer
    double temp_cost; 
    layer* temp_layers[LAYER_COUNT];
    foreach_layer(LAYER_COUNT, i){
        temp_layers[i] = malloc(sizeof(layer));
        temp_layers[i]->neuron_count = layers[i]->neuron_count;
        temp_layers[i]->neurons = malloc(sizeof(neuron) * layers[i]->neuron_count);
        foreach_neuron(*layers[i], j){
            temp_layers[i]->neurons[j].bias = layers[i]->neurons[j].bias;
            temp_layers[i]->neurons[j].weight_count = layers[i]->neurons[j].weight_count;
            temp_layers[i]->neurons[j].weights = malloc(sizeof(double) * layers[i]->neurons[j].weight_count);
            memcpy(temp_layers[i]->neurons[j].weights, layers[i]->neurons[j].weights, sizeof(double) * layers[i]->neurons[j].weight_count);
        }
    }

    foreach_layer(LAYER_COUNT - 1, i){
        foreach_neuron(*layers[i + 1], j){
            // partial derivation of cost function with respect to weight
            foreach_neuron(*layers[i], k){
                temp_cost = cost_func(temp_layers, correct_answer);
                temp_layers[i + 1]->neurons[j].weights[k] += DELTA;
                gradient[i]->neurons[j].weights[k] = LEARNING_RATE * ((cost_func(temp_layers, correct_answer) - temp_cost) / DELTA );
                temp_layers[i + 1]->neurons[j].weights[k] -= DELTA;
            }
            // partial derivation of cost function with respect to bias
            temp_cost = cost_func(temp_layers, correct_answer);
            temp_layers[i + 1]->neurons[j].bias += DELTA;
            gradient[i]->neurons[j].bias = LEARNING_RATE * ((cost_func(temp_layers, correct_answer) - temp_cost) / DELTA );
            temp_layers[i + 1]->neurons[j].bias -= DELTA;
        }
    }
    foreach_layer(LAYER_COUNT, i){
        free_layer(temp_layers[i]);
    }
}

void calculate_average_gradient(layer* gradient[LAYER_COUNT - 1], layer* average_gradient[LAYER_COUNT - 1], int step_size){
    foreach_layer(LAYER_COUNT - 1, i){
        foreach_neuron(*gradient[i], j){
            foreach_weight(gradient[i]->neurons[j], k){
                average_gradient[i]->neurons[j].weights[k] += gradient[i]->neurons[j].weights[k] / step_size;
            }
            average_gradient[i]->neurons[j].bias += gradient[i]->neurons[j].bias / step_size;
        }
    }
}

void clear_gradient(layer* gradient[LAYER_COUNT - 1]){
    foreach_layer(LAYER_COUNT - 1, i){
        foreach_neuron(*gradient[i], j){
            memset(gradient[i]->neurons[j].weights, 0, sizeof(double) * gradient[i]->neurons[j].weight_count);
            gradient[i]->neurons[j].bias = 0.0;
        }
    }
}

