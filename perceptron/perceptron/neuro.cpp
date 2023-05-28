#include "neuro.h"
#include <cmath>
#include <vector>
#include <fstream>

#define SIGMOID_PARAM 1

MLP::MLP(const vector<size_t> &layer_sizes, const string& log_file_name) : layer_sizes(layer_sizes),
                                                                                             log_file_name(
                                                                                                     log_file_name) {
    for (size_t i = 0; i < layer_sizes.size() - 1; i++) {
        weights.emplace_back(layer_sizes[i] * layer_sizes[i + 1]);
        biases.emplace_back(layer_sizes[i + 1]);
        weighted_inputs.emplace_back(layer_sizes[i + 1]);

        default_random_engine generator;
        normal_distribution<double> distribution(0.0, 1.0);
        for (double &j: weights[i]) {
            j = distribution(generator);
        }
        for (double &j: biases[i]) {
            j = distribution(generator);
        }
    }
    results = ofstream(log_file_name, ofstream::trunc);
//    mse_file << "Epoch, Value" << endl;
    results << "Epoch";
    for (int i = 0; i < layer_sizes[layer_sizes.size() - 1]; i++) {
        results << ",Predicted " << i << ",Real " << i;
    }
    results << endl;
}

double MLP::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-SIGMOID_PARAM * x));
}

double MLP::sigmoid_derivative(double x) {
    return SIGMOID_PARAM * sigmoid(x) * (1.0 - sigmoid(x));
}

vector<double> MLP::feedforward(const vector<double> &input) {
    if (input.size() != layer_sizes[0]) {
        throw runtime_error("Input size does not match first layer size");
    }

    activations = {input};
//    cout << input[0] << endl;
    // Calculate the activations for each layer
    for (size_t i = 0; i < layer_sizes.size() - 1; i++) {
        for (size_t j = 0; j < layer_sizes[i + 1]; j++) {
            double weighted_input = 0.0;
            for (size_t k = 0; k < layer_sizes[i]; k++) {
                weighted_input += weights[i][j * layer_sizes[i] + k] * activations[i][k];
            }
            weighted_inputs[i][j] = weighted_input + biases[i][j];
        }

        vector<double> layer_activations(layer_sizes[i + 1]);
        for (size_t j = 0; j < layer_sizes[i + 1]; j++) {
            layer_activations[j] = sigmoid(weighted_inputs[i][j]);
        }
        activations.push_back(layer_activations);
//        f << layer_activations[0] << endl;
    }
    return activations.back();
}

void MLP::calculate_output_error(vector<double> output, const vector<Neuron*> &targets, vector<double> &output_error) {
    for (size_t j = 0; j < output.size(); j++) {
        if (targets[j] != nullptr) {
            output_error[j] = output[j] - targets[j]->value;
            if (log) {
                results << "," << output[j] << "," << targets[j]->value;
            }
        } else {
            output_error[j] = 0;
            if (log) {
                results << "," << "None" << "," << "None";
            }
        }
    }
    if (log) {
        results << endl;
    }
//    double mean_squared_error_part = 0.0;
//    for (double j: output_error) {
//        mean_squared_error_part += pow(j, 2);
//    }
//    mean_squared_error_part /= output_error.size();
//    mean_squared_error += mean_squared_error_part;
//
//    double mean_absolute_error_part = 0.0;
//    for (double j: output_error) {
//        mean_absolute_error_part += abs(j);
//    }
//    mean_absolute_error_part /= output_error.size();
//    mean_absolute_error += mean_absolute_error_part;
}

void MLP::calculate_layer_error(size_t layer_number, vector<double> &layer_error, vector<double> &prev_layer_error,
                                vector<double> output_error) {
    if (layer_number == layer_sizes.size() - 1) {
        for (size_t j = 0; j < layer_sizes[layer_number]; j++) {
            layer_error[j] = output_error[j] * sigmoid_derivative(weighted_inputs[layer_number - 1][j]);
        }
    } else {
        for (size_t k = 0; k < layer_sizes[layer_number]; k++) {
            double weighted_error = 0.0;
            for (size_t l = 0; l < layer_sizes[layer_number - 1]; l++) {
                weighted_error +=
                        weights[layer_number - 1][k * layer_sizes[layer_number - 1] + l] * prev_layer_error[l];
            }
            layer_error[k] = sigmoid_derivative(weighted_inputs[layer_number - 1][k]) * weighted_error;
        }
    }
}

void MLP::apply_backpropagation(const vector<double> &inputs, const vector<double> &output,
                                const vector<double> &output_error,
                                double learning_rate) {
    vector<double> prev_layer_error;
    for (size_t j = layer_sizes.size() - 1; j > 0; j--) {
        vector<double> layer_error(layer_sizes[j]);
        calculate_layer_error(j, layer_error, prev_layer_error, output_error);
        prev_layer_error = layer_error;
        for (size_t k = 0; k < layer_sizes[j]; k++) {
            for (size_t l = 0; l < layer_sizes[j - 1]; l++) {
                weights[j - 1][k * layer_sizes[j - 1] + l] -=
                        learning_rate * layer_error[k] * activations[j - 1][l];
            }
            biases[j - 1][k] -= learning_rate * layer_error[k];
        }
    }
}

void MLP::train(const vector<vector<double>> &inputs, const vector<vector<Neuron*>> &targets, size_t epochs,
                double learning_rate) {
    if (inputs.size() != targets.size()) {
        throw runtime_error("Number of input rows does not match number of target rows");
    }
    for (size_t epoch = 0; epoch < epochs; epoch++) {
        // Shuffle the input and target rows
        vector<size_t> indices(inputs.size());
        for (size_t i = 0; i < inputs.size(); i++) {
            indices[i] = i;
        }
        shuffle(indices.begin(), indices.end(), default_random_engine());
        double epoch_loss = 0.0;
        for (size_t i = 0; i < inputs.size(); i++) {
            if (log) {
                results << epoch + 1;
            }
            vector<double> output = feedforward(inputs[indices[i]]);
            vector<double> output_error(output.size());
            calculate_output_error(output, targets[indices[i]], output_error);
            apply_backpropagation(inputs[indices[i]], output, output_error, learning_rate);
        }

//        mean_squared_error /= inputs.size();
//        mean_absolute_error /= inputs.size();
//        mse_file << epoch + 1 << ", " << mean_squared_error << endl;
//        mae_file << epoch + 1 << ", " << mean_absolute_error << endl;
    }
}

Neuron::Neuron(double value) : value(value) {
}
