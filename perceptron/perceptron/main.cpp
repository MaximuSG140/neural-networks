#include <iostream>
#include <vector>
#include <string>
#include "csv.h"
#include "neuro.h"

using namespace std;

void parse(const string &file, vector<vector<double>> &dataset_without_target, vector<vector<Neuron*>> &targets) {
    vector<vector<std::string>> dataset = parse_csv(file);
    vector<vector<std::string>> dataset_without_target_string = remove_column(dataset, 0);
    dataset_without_target_string = remove_row(dataset_without_target_string, 0);
    vector<vector<string>> targets_string = get_last_two_columns(dataset_without_target_string);
    dataset_without_target_string = remove_column(dataset_without_target_string, 23);
    dataset_without_target_string = remove_column(dataset_without_target_string, 23);

    dataset_without_target = convert_to_doubles(dataset_without_target_string);
    targets = convert_to_neurons(targets_string);
}

int main() {
    vector<vector<double>> training_dataset_without_target;
    vector<vector<Neuron*>> training_targets;
    parse("training_dataset.csv", training_dataset_without_target, training_targets);

    vector<vector<double>> testing_dataset_without_target;
    vector<vector<Neuron*>> testing_targets;
    parse("testing_dataset.csv", testing_dataset_without_target, testing_targets);

    size_t features_num = get_num_columns(training_dataset_without_target);
    MLP mlp({features_num, 40, 40, 2});
    mlp.enable_logging();
    mlp.train(training_dataset_without_target, training_targets, 3200, 0.001);
    ofstream testing_results("testing_results.csv");
    testing_results << "Number";
    for (int i = 0; i < 2; i++) {
        testing_results << ",Predicted " << i << ",Real " << i;
    }
    testing_results << endl;
    for (size_t i = 0; i < testing_dataset_without_target.size(); i++) {
        vector<double> result = mlp.feedforward(testing_dataset_without_target[i]);
        cout << endl;
        cout << "Predicted: " << result[0] << " " << result[1] << endl;
        cout << "Real: " << testing_targets[i][0]->value << " " << testing_targets[i][1]->value << endl;
        testing_results << i << "," << result[0] << "," << testing_targets[i][0]->value << "," << result[1] << ","
                        << testing_targets[i][1]->value << endl;
    }
    return 0;
}