#include "neural.h"
#include "data.h"
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define TRAINING_CANDLE_DATA 10000
#define NUMBER_OF_CANDLES_INPUT 100
#define TRADE_PERIOD 20
#define TRADE_THRESHOLD 0.5

// Global hyperparameters
double LEARNING_RATE_OPT = 0.001;

// Normalize input data - simple min-max normalization
void normalize(double* data, int length) {
    double min = data[0], max = data[0];
    for (int i = 1; i < length; i++) {
        if (data[i] < min) min = data[i];
        if (data[i] > max) max = data[i];
    }
    double range = max - min;
    if (range < 1e-8) range = 1; // avoid div by zero
    for (int i = 0; i < length; i++) {
        data[i] = (data[i] - min) / range;
    }
}

// Simplified backpropagation for one sample and updating weights directly
void backpropagate(layer* layers[LAYER_COUNT], int correct_trade) {
    // Forward pass already handled in cost_func or elsewhere
    // Here implement backward pass:
    // Compute errors at output layer and propagate backwards updating weights and biases

    // Allocate delta arrays per layer
    double* deltas[LAYER_COUNT];
    for (int i = 0; i < LAYER_COUNT; i++) {
        deltas[i] = malloc(sizeof(double) * layers[i]->neuron_count);
        memset(deltas[i], 0, sizeof(double) * layers[i]->neuron_count);
    }

    // Output layer error calculation (assume 3 output neurons classifying trade)
    for (int i = 0; i < layers[LAYER_COUNT-1]->neuron_count; i++) {
        double output = layers[LAYER_COUNT-1]->neurons[i].value;
        double target = (i == correct_trade) ? 1.0 : 0.0;
        deltas[LAYER_COUNT-1][i] = (output - target) * output * (1 - output); // Sigmoid derivative
    }

    // Backward propagate errors and update weights
    for (int layer_i = LAYER_COUNT-2; layer_i >= 0; layer_i--) {
        layer* current = layers[layer_i];
        layer* next = layers[layer_i + 1];
        for (int i = 0; i < current->neuron_count; i++) {
            double error_sum = 0.0;
            for (int j = 0; j < next->neuron_count; j++) {
                error_sum += next->neurons[j].weights[i] * deltas[layer_i + 1][j];
            }
            double output = current->neurons[i].value;
            deltas[layer_i][i] = error_sum * output * (1 - output);
        }
    }

    // Update weights and biases using gradients & learning rate
    for (int layer_i = 1; layer_i < LAYER_COUNT; layer_i++) {
        layer* current = layers[layer_i];
        layer* previous = layers[layer_i - 1];
        for (int i = 0; i < current->neuron_count; i++) {
            for (int j = 0; j < current->neurons[i].weight_count; j++) {
                double grad = deltas[layer_i][i] * previous->neurons[j].value;
                current->neurons[i].weights[j] -= LEARNING_RATE_OPT * grad;
            }
            current->neurons[i].bias -= LEARNING_RATE_OPT * deltas[layer_i][i];
        }
    }

    // Free delta buffers
    for (int i = 0; i < LAYER_COUNT; i++) {
        free(deltas[i]);
    }
}

// Create labels function with fixed index calculations corrected
int* create_correct_trades(double* data) {
    int length = TRAINING_CANDLE_DATA;
    int batch_count = length / NUMBER_OF_CANDLES_INPUT;
    int* correct_trades = malloc(sizeof(int) * batch_count);
    for (int i = 0; i < batch_count; i++) {
        double sum = 0;
        for (int j = 0; j < TRADE_PERIOD; j++) {
            int idx = i * NUMBER_OF_CANDLES_INPUT + j;
            if (idx < length) sum += data[idx];
        }
        if (sum > TRADE_THRESHOLD) {
            correct_trades[i] = 0; // buy
        } else if (sum < -TRADE_THRESHOLD) {
            correct_trades[i] = 1; // sell
        } else {
            correct_trades[i] = 2; // no entry
        }
    }
    return correct_trades;
}

void train(layer* layers[LAYER_COUNT], double* training_candles, int* correct_trades, int epochs) {
    int batches = TRAINING_CANDLE_DATA / NUMBER_OF_CANDLES_INPUT;

    // Normalize entire training data once
    normalize(training_candles, TRAINING_CANDLE_DATA);

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int b = 0; b < batches; b++) {
            // Load input for batch
            for (int i = 0; i < NUMBER_OF_CANDLES_INPUT; i++) {
                layers[0]->neurons[i].value = training_candles[b * NUMBER_OF_CANDLES_INPUT + i];
            }
            // Forward pass
            for (int i = 1; i < LAYER_COUNT; i++) {
                calculate_layer(layers[i], layers[i - 1]);
            }

            // Backpropagation and weights updates
            backpropagate(layers, correct_trades[b]);
        }
    }
}

int main() {
    layer* layers[LAYER_COUNT];
    layers[0] = create_layer(NUMBER_OF_CANDLES_INPUT, NULL, true);
    layers[1] = create_layer(10, layers[0], true);
    layers[2] = create_layer(3, layers[1], true);

    double* data = convert_csv_data("data.csv");
    if (!data) {
        fprintf(stderr, "Failed to load data.csv\n");
        return 1;
    }
    int* correct_trades = create_correct_trades(data);

    int epochs = 50;
    train(layers, data, correct_trades, epochs);

    free_correct_trades(correct_trades);
    free_data(data);

    // Save model to file (simple binary write)
    FILE* fptr = fopen("traindata.bin", "wb");
    for (int i = 1; i < LAYER_COUNT; i++) {
        for (int n = 0; n < layers[i]->neuron_count; n++) {
            fwrite(&layers[i]->neurons[n].bias, sizeof(double), 1, fptr);
            fwrite(layers[i]->neurons[n].weights, sizeof(double), layers[i]->neurons[n].weight_count, fptr);
        }
    }
    fclose(fptr);

    // Free all layers
    for (int i = 0; i < LAYER_COUNT; i++) {
        free_layer(layers[i]);
    }

    return 0;
}
