#include "neural.h"

#define TRAINING_CANDLE_DATA 10000
#define NUMBER_OF_CANDLES 100

struct train_thread_args{
    layer** layers;
    layer** gradient;
    double* training_candles;
    int correct_trade;
};

void* train_thread(void* arg){
    struct train_thread_args* args = (struct train_thread_args*)arg;
    for(int j = 0; j < NUMBER_OF_CANDLES; j++){
        args->layers[0]->neurons[j].value = args->training_candles[j];
    }
    gradient_descent(args->gradient ,STEP_SIZE ,args->layers ,args->correct_trade, EPOCH);
    return NULL;
}

struct subtract_thread_args{
    layer** layers;
    int neuron_index;
    layer** average_gradient;
    int current_layer;
};

void* subtract_thread(void* arg){
    struct subtract_thread_args* args = (struct subtract_thread_args*)arg;
    foreach_weight(args->layers[args->current_layer + 1]->neurons[args->neuron_index],l){
        args->layers[args->current_layer + 1]->neurons[args->neuron_index].weights[l] -= args->average_gradient[args->current_layer]->neurons[args->neuron_index].weights[l];
    }
    args->layers[args->current_layer + 1]->neurons[args->neuron_index].bias -= args->average_gradient[args->current_layer]->neurons[args->neuron_index].bias;
    return NULL;
}

void train(layer* layers[LAYER_COUNT] ,int step_size, double* training_candles, int* correct_trades){
    int train_candle_count = TRAINING_CANDLE_DATA;
    int correct_answer = 0;
    int step_counter = 0;

    layer* gradient[LAYER_COUNT - 1];
    layer* average_gradient[LAYER_COUNT - 1];
    foreach_layer(LAYER_COUNT - 1, i){
        gradient[i] = create_layer(layers[i + 1]->neuron_count, layers[i], false);
        average_gradient[i] = create_layer(layers[i + 1]->neuron_count, layers[i], false);
    }

    layer* thread_gradients[STEP_SIZE][LAYER_COUNT - 1];
    for (int i = 0; i < STEP_SIZE; i++){
        foreach_layer(LAYER_COUNT - 1, j){
            thread_gradients[i][j] = create_layer(layers[j + 1]->neuron_count, layers[j], false);
        }
    }
    

    double candles_training_array[TRAINING_CANDLE_DATA / NUMBER_OF_CANDLES][NUMBER_OF_CANDLES];
    for (int i = 0; i < TRAINING_CANDLE_DATA; i++){
        for (int j = 0; j < NUMBER_OF_CANDLES; j++){
            candles_training_array[i][j] = training_candles[i];
        }
    }

    for(int i = 0; i < (TRAINING_CANDLE_DATA / NUMBER_OF_CANDLES) / STEP_SIZE; i++){
        pthread_t thread[STEP_SIZE];
        for (int j = 0; j < STEP_SIZE; j++){
            struct train_thread_args args;
            args.layers = layers;
            args.gradient = thread_gradients[j];
            args.training_candles = candles_training_array[i * STEP_SIZE + j];
            args.correct_trade = correct_trades[i * STEP_SIZE + j];
            pthread_create(&thread[j], NULL, train_thread, (void*)&args);
        }
        for (int j = 0; j < STEP_SIZE; j++){
            pthread_join(thread[j], NULL);
        }
        for (int j = 0; j < STEP_SIZE; j++){
            calculate_average_gradient(thread_gradients[j], average_gradient, STEP_SIZE);
        }

        foreach_layer(LAYER_COUNT - 1, j){
            pthread_t thread[layers[j + 1]->neuron_count];
            for (int k = 0; k < layers[j + 1]->neuron_count; k++){
                struct subtract_thread_args args;
                args.layers = layers;
                args.neuron_index = k;
                args.average_gradient = average_gradient;
                args.current_layer = j;
                pthread_create(&thread[k], NULL, subtract_thread, (void*)&args);
            }
            foreach_neuron(*layers[j + 1], k){
                pthread_join(thread[k], NULL);
            }
        }
        clear_gradient(average_gradient);
    }
    foreach_layer(LAYER_COUNT - 1, i){
        for(int j = 0; j < STEP_SIZE; j++){
            free_layer(thread_gradients[j][i]);
        }
    }
}
