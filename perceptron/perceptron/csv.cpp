#include "csv.h"
#include "neuro.h"

vector<vector<string>> parse_csv(const string &filename, char delimiter) {
    // Open the CSV file for reading
    ifstream file(filename);

    // Initialize the result vector
    vector<vector<string>> result;
    bool skip = false;

    // Read each line of the file and split it into fields
    string line;
    while (getline(file, line)) {
        vector<string> fields;

        // Use a string stream to split the line into fields
        stringstream ss(line);
        string field;
        while (getline(ss, field, delimiter)) {
            fields.push_back(field);
        }
        result.push_back(fields);
    }

    // Close the file and return the result vector
    file.close();
    return result;
}


vector<vector<string>> remove_column(const vector<vector<string>> &data, int col_index) {
    // Create a new vector to hold the result
    vector<vector<string>> result;

    // Iterate over the rows of the input data
    for (const auto &i: data) {
        // Create a new row vector to hold the filtered columns
        vector<string> row;

        // Iterate over the columns of the input data
        for (size_t j = 0; j < i.size(); ++j) {
            // Skip the column to be removed
            if (j == col_index) {
                continue;
            }
            row.push_back(i[j]);
        }

        // Add the filtered row to the result vector
        result.push_back(row);
    }

    return result;
}


vector<vector<string>> remove_row(const vector<vector<string>> &data, int row_index) {
    // Create a new vector to hold the result
    vector<vector<string>> result;

    // Iterate over the rows of the input data
    for (size_t i = 0; i < data.size(); ++i) {
        // Skip the row to be removed
        if (i == row_index) {
            continue;
        }

        // Add the row to the result vector
        result.push_back(data[i]);
    }

    return result;
}

vector<string> get_column(const vector<vector<string>> &data, int col_index) {
    // Create a new vector to hold the result
    vector<string> result;

    // Iterate over the rows of the input data
    for (const auto &i: data) {
        // Add the value in the desired column to the result vector
        result.push_back(i[col_index]);
    }

    return result;
}

vector<vector<double>> convert_to_doubles(const vector<vector<string>> &data) {
    // Create a new vector to hold the result
    vector<vector<double>> result;

    // Iterate over the rows of the input data
    for (const auto &i: data) {
        // Create a new row vector to hold the converted values
        vector<double> row;
        // Iterate over the columns of the input data
        for (const auto &j: i) {
            // Convert the string value to a double
            double value;
            istringstream(j) >> value;
            // Add the converted value to the row vector
            row.push_back(value);
        }

        // Add the converted row to the result vector
        result.push_back(row);
    }

    return result;
}

vector<vector<Neuron*>> convert_to_neurons(const vector<vector<string>> &data) {
    // Create a new vector to hold the result
    vector<vector<Neuron*>> result;

    // Iterate over the rows of the input data
    for (const auto &i: data) {
        // Create a new row vector to hold the converted values
        vector<Neuron*> row;
        // Iterate over the columns of the input data
        for (const auto &j: i) {
            // Convert the string value to a double
            if (j.empty()) {
                row.push_back(nullptr);
            } else {
                double value;
                istringstream(j) >> value;
                // Add the converted value to the row vector
                row.push_back(new Neuron(value));
            }
        }

        // Add the converted row to the result vector
        result.push_back(row);
    }

    return result;
}

vector<double> convert_to_doubles(const vector<string> &data) {
    // Create a new vector to hold the result
    vector<double> result;

    // Iterate over the input data
    for (const auto &value: data) {
        // Convert the string value to a double
        double double_value;
        istringstream(value) >> double_value;

        // Add the converted value to the result vector
        result.push_back(double_value);
    }

    return result;
}

size_t get_num_columns(const vector<vector<double>> &data) {
    // If there are no rows, return 0
    if (data.empty()) {
        return 0;
    }

    // Return the number of columns in the first row
    return data[0].size();
}

void print_table(const vector<vector<string>> &data) {
    for (const auto &row: data) {
        for (const auto &field: row) {
            cout << field << " ";
        }
        cout << endl;
    }
}

vector<vector<string>> get_last_two_columns(const vector<vector<string>> &table) {
    vector<vector<string>> result(table.size());
    for (size_t i = 0; i < table.size(); i++) {
        result[i].resize(2);
        result[i][0] = table[i][table[i].size() - 2];
        result[i][1] = table[i][table[i].size() - 1];
    }
    return result;
}

double* vectorOfVectorsToArray(vector<vector<double>> v) {
    int rows = v.size();
    int cols = v[0].size();

    auto* arr = new double[rows * cols];

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            arr[i * cols + j] = v[i][j];
        }
    }

    return arr;
}

vector<double> get_row(vector<vector<double>> matrix, int row) {
    return matrix[row];
}

void normalize_column(vector<vector<double>> &matrix, vector<vector<double>> &normalized_matrix, size_t col_index) {
    // Find the minimum and maximum values in the column
    double min_val = matrix[0][col_index];
    double max_val = matrix[0][col_index];
    for (size_t i = 1; i < matrix.size(); i++) {
        min_val = min(min_val, matrix[i][col_index]);
        max_val = max(max_val, matrix[i][col_index]);
    }

    // Subtract the minimum value from every element in the column and divide by the difference of max and min values
    double diff = max_val - min_val;
    for (size_t i = 0; i < matrix.size(); i++) {
        normalized_matrix[i][col_index] = (matrix[i][col_index] - min_val) / diff;
    }
}

pair<vector<vector<double>>, vector<vector<double>>> split(const vector<vector<double>> &matrix, size_t split_row) {
    if (split_row >= matrix.size()) {
        throw runtime_error("Split row index is out of range");
    }
    vector<vector<double>> first_half(matrix.begin(), matrix.begin() + split_row + 1);
    vector<vector<double>> second_half(matrix.begin() + split_row + 1, matrix.end());
    return make_pair(first_half, second_half);
}

std::vector<std::vector<std::string>> get_subtable(const std::vector<std::vector<std::string>>& table,
                                                   size_t startRow, size_t endRow,
                                                   size_t startColumn, size_t endColumn) {
    std::vector<std::vector<std::string>> subtable;

    for (size_t i = startRow; i < endRow && i < table.size(); i++) {
        std::vector<std::string> subtableRow;

        for (size_t j = startColumn; j < endColumn && j < table[i].size(); j++) {
            subtableRow.push_back(table[i][j]);
        }

        subtable.push_back(subtableRow);
    }

    return subtable;
}