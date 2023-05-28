#ifndef PERCEPTRON_CSV_H
#define PERCEPTRON_CSV_H


#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include "neuro.h"

using namespace std;

vector<vector<string>> parse_csv(const string &filename, char delimiter = ',');

vector<vector<string>> remove_column(const vector<vector<string>> &data, int col_index);

vector<vector<string>> remove_row(const vector<vector<string>> &data, int row_index);

vector<string> get_column(const vector<vector<string>> &data, int col_index);

vector<vector<double>> convert_to_doubles(const vector<vector<string>> &data);

vector<double> convert_to_doubles(const vector<string> &data);

size_t get_num_columns(const vector<vector<double>> &data);

void print_table(const vector<vector<string>> &data);

vector<vector<string>> get_last_two_columns(const vector<vector<string>> &table);

double* vectorOfVectorsToArray(vector<vector<double>> v);

vector<double> get_row(vector<vector<double>> matrix, int row);

void normalize_column(vector<vector<double>> &matrix, vector<vector<double>> &normalized_matrix, size_t col_index);

pair<vector<vector<double>>, vector<vector<double>>> split(const vector<vector<double>> &matrix, size_t split_row);

std::vector<std::vector<std::string>> get_subtable(const std::vector<std::vector<std::string>>& table,
                                                   size_t startRow, size_t endRow,
                                                   size_t startColumn, size_t endColumn);
vector<vector<Neuron*>> convert_to_neurons(const vector<vector<string>> &data);

#endif //PERCEPTRON_CSV_H
