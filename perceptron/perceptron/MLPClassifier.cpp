#include "MLPClassifier.h"

void
MLPClassifier::calculate_layer_error(size_t layer_number, vector<double> &layer_error, vector<double> &prev_layer_error,
                                     vector<double> output_error) {
    if (layer_number == layer_sizes.size() - 1) {
        for (size_t j = 0; j < layer_sizes[layer_number]; j++) {
            layer_error[j] = output_error[j];
        }
    } else {
        for (size_t k = 0; k < layer_sizes[layer_number]; k++) {
            double weighted_error = 0.0;
            for (size_t l = 0; l < layer_sizes[layer_number - 1]; l++) {
                weighted_error +=
                        weights[layer_number - 1][k * layer_sizes[layer_number - 1] + l] * prev_layer_error[l];
            }
            layer_error[k] = weighted_error;
        }
    }
}

MLPClassifier::MLPClassifier(const vector<size_t> &layerSizes, const string &logFileName) : MLP(layerSizes,
                                                                                                logFileName) {
}