#ifndef PERCEPTRON_NEURO_H
#define PERCEPTRON_NEURO_H


#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>

using namespace std;

class Neuron {
public:
    double value = 0;

    explicit Neuron(double value);
};

class MLP {
private:
    vector<vector<double>> biases;
    vector<vector<double>> activations;
    ofstream results;
    bool log = false;
    string log_file_name;

    double sigmoid(double x);

    void apply_backpropagation(const vector<double> &inputs, const vector<double> &output,
                               const vector<double> &output_error,
                               double learning_rate);

    void calculate_output_error(vector<double> output, const vector<Neuron*> &targets, vector<double> &output_error);

    virtual void
    calculate_layer_error(size_t layer_number, vector<double> &layer_error, vector<double> &prev_layer_error,
                          vector<double> output_error);

protected:
    vector<size_t> layer_sizes;
    vector<vector<double>> weights;
    vector<vector<double>> weighted_inputs;

    double sigmoid_derivative(double x);

public:
    explicit MLP(const vector<size_t> &layer_sizes, const string &log_file_name = "training_results.csv");

    vector<double> feedforward(const vector<double> &input);

    void train(const vector<vector<double>> &inputs, const vector<vector<Neuron*>> &targets, size_t epochs,
               double learning_rate);

    void enable_logging() {
        log = true;
    }

    void disable_logging() {
        log = false;
    }
};

#endif //PERCEPTRON_NEURO_H
