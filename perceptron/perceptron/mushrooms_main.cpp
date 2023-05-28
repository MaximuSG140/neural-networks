#include <iostream>
#include <vector>
#include <string>
#include "csv.h"
#include "MLPClassifier.h"

using namespace std;

void parse(const string &file, vector<vector<double>> &dataset_without_target, vector<vector<double>> &targets) {
    vector<vector<std::string>> dataset = parse_csv(file);
    vector<vector<std::string>> dataset_without_target_string = remove_row(dataset, 0);
    vector<vector<string>> targets_string = get_subtable(dataset_without_target_string, 0,
                                                         dataset_without_target_string.size(), 0, 1);
    dataset_without_target_string = remove_column(dataset_without_target_string, 0);

    dataset_without_target = convert_to_doubles(dataset_without_target_string);
    targets = convert_to_doubles(targets_string);
}

int main() {
    vector<vector<double>> training_dataset_without_target;
    vector<vector<double>> training_targets;
    parse("training_mushrooms_dataset.csv", training_dataset_without_target, training_targets);
//    cerr << training_dataset_without_target.size() << endl;
//    cerr << training_targets.size() << endl << endl;
    vector<vector<double>> testing_dataset_without_target;
    vector<vector<double>> testing_targets;
    parse("testing_mushrooms_dataset.csv", testing_dataset_without_target, testing_targets);

    size_t features_num = get_num_columns(training_dataset_without_target);
    MLPClassifier mlp({features_num, 40, 40, 1}, "mushroom_training_results.csv");
    mlp.enable_logging();
    mlp.train(training_dataset_without_target, training_targets, 10, 0.1);

    ofstream testing_results("mushroom_testing_results.csv");
    testing_results << "Number";
    for (int i = 0; i < 1; i++) {
        testing_results << ",Predicted " << i << ",Real " << i;
    }
    testing_results << endl;
    for (size_t i = 0; i < testing_dataset_without_target.size(); i++) {
        vector<double> result = mlp.feedforward(testing_dataset_without_target[i]);
        cout << endl;
        cout << "Predicted: " << result[0] << endl;
        cout << "Real: " << testing_targets[i][0] << endl;
        testing_results << i << "," << result[0] << "," << testing_targets[i][0] << endl;
    }
    return 0;
}