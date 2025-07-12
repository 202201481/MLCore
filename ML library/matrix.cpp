#include "matrix.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <random>
#include <thread>
#include <numeric>

// Private helper method - calculate offset for multi-dimensional indexing
template<typename T>
size_t Matrix<T>::calcOffset(const std::vector<size_t>& indices) const {
    if (indices.size() != shape.size()) {
        throw std::invalid_argument("Number of indices must match number of dimensions");
    }
    
    size_t offset = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= shape[i]) {
            throw std::out_of_range("Index out of bounds");
        }
        offset += indices[i] * strides[i];
    }
    return offset;
}

// Private helper method - calculate strides for multi-dimensional array
template<typename T>
void Matrix<T>::calcStrides() {
    strides.resize(shape.size());
    if (!shape.empty()) {
        strides[shape.size() - 1] = 1;
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
}

// Constructors
template<typename T>
Matrix<T>::Matrix() : data(), shape(), strides() {}

template<typename T>
Matrix<T>::Matrix(const std::vector<size_t>& shape) : shape(shape) {
    if (shape.empty()) {
        throw std::invalid_argument("Shape cannot be empty");
    }
    
    size_t total_size = 1;
    for (size_t dim : shape) {
        if (dim == 0) {
            throw std::invalid_argument("Shape dimensions must be positive");
        }
        total_size *= dim;
    }
    
    data.resize(total_size);
    calcStrides();
}

template<typename T>
Matrix<T>::Matrix(const std::vector<size_t>& shape, T value) : shape(shape) {
    if (shape.empty()) {
        throw std::invalid_argument("Shape cannot be empty");
    }
    
    size_t total_size = 1;
    for (size_t dim : shape) {
        if (dim == 0) {
            throw std::invalid_argument("Shape dimensions must be positive");
        }
        total_size *= dim;
    }
    
    data.resize(total_size, value);
    calcStrides();
}

template<typename T>
Matrix<T>::Matrix(std::initializer_list<T> list) {
    data = std::vector<T>(list);
    shape = {data.size()};
    calcStrides();
}

template<typename T>
Matrix<T>::Matrix(const Matrix<T>& other) : data(other.data), shape(other.shape), strides(other.strides) {}

template<typename T>
Matrix<T>::Matrix(Matrix<T>&& other) noexcept : data(std::move(other.data)), shape(std::move(other.shape)), strides(std::move(other.strides)) {}

// Assignment operators
template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& other) {
    if (this != &other) {
        data = other.data;
        shape = other.shape;
        strides = other.strides;
    }
    return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::operator=(Matrix<T>&& other) noexcept {
    if (this != &other) {
        data = std::move(other.data);
        shape = std::move(other.shape);
        strides = std::move(other.strides);
    }
    return *this;
}

template<typename T>
Matrix<T> Matrix<T>::fromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::vector<std::vector<T>> rows;
    std::string line;
    
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<T> row;
        T value;
        
        while (iss >> value) {
            row.push_back(value);
        }
        
        if (!row.empty()) {
            rows.push_back(row);
        }
    }
    
    if (rows.empty()) {
        throw std::runtime_error("File is empty or contains no valid data");
    }
    
    size_t cols = rows[0].size();
    for (const auto& row : rows) {
        if (row.size() != cols) {
            throw std::runtime_error("Inconsistent number of columns in file");
        }
    }
    
    std::vector<size_t> result_shape = {rows.size(), cols};
    Matrix<T> result(result_shape);
    for (size_t i = 0; i < rows.size(); ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.at({i, j}) = rows[i][j];
        }
    }
    
    return result;
}

// Element access methods
template<typename T>
T& Matrix<T>::at(const std::vector<size_t>& indices) {
    return data[calcOffset(indices)];
}

template<typename T>
const T& Matrix<T>::at(const std::vector<size_t>& indices) const {
    return data[calcOffset(indices)];
}

template<typename T>
T& Matrix<T>::operator[](const std::vector<size_t>& indices) {
    return data[calcOffset(indices)];
}

template<typename T>
const T& Matrix<T>::operator[](const std::vector<size_t>& indices) const {
    return data[calcOffset(indices)];
}

// Basic query methods
template<typename T>
bool Matrix<T>::any(size_t axis) const {
    if (data.empty()) return false;
    
    if (axis == static_cast<size_t>(-1)) {
        // Check all elements
        for (const auto& element : data) {
            if constexpr (std::is_arithmetic_v<T>) {
                if (element != T(0)) return true;
            } else {
                if (element) return true;
            }
        }
        return false;
    } else {
        // Axis-specific implementation
        if (shape.size() != 2) {
            throw std::runtime_error("Axis-specific any() currently only supports 2D matrices");
        }
        
        if (axis == 0) {
            // Check along rows (any element in each column)
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t i = 0; i < shape[0]; ++i) {
                    if constexpr (std::is_arithmetic_v<T>) {
                        if (at({i, j}) != T(0)) return true;
                    } else {
                        if (at({i, j})) return true;
                    }
                }
            }
            return false;
        } else if (axis == 1) {
            // Check along columns (any element in each row)
            for (size_t i = 0; i < shape[0]; ++i) {
                for (size_t j = 0; j < shape[1]; ++j) {
                    if constexpr (std::is_arithmetic_v<T>) {
                        if (at({i, j}) != T(0)) return true;
                    } else {
                        if (at({i, j})) return true;
                    }
                }
            }
            return false;
        } else {
            throw std::invalid_argument("Invalid axis for 2D matrix");
        }
    }
}

template<typename T>
bool Matrix<T>::all(size_t axis) const {
    if (data.empty()) return true;
    
    if (axis == static_cast<size_t>(-1)) {
        // Check all elements
        for (const auto& element : data) {
            if constexpr (std::is_arithmetic_v<T>) {
                if (element == T(0)) return false;
            } else {
                if (!element) return false;
            }
        }
        return true;
    } else {
        // Axis-specific implementation
        if (shape.size() != 2) {
            throw std::runtime_error("Axis-specific all() currently only supports 2D matrices");
        }
        
        if (axis == 0) {
            // Check along rows (all elements in each column)
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t i = 0; i < shape[0]; ++i) {
                    if constexpr (std::is_arithmetic_v<T>) {
                        if (at({i, j}) == T(0)) return false;
                    } else {
                        if (!at({i, j})) return false;
                    }
                }
            }
            return true;
        } else if (axis == 1) {
            // Check along columns (all elements in each row)
            for (size_t i = 0; i < shape[0]; ++i) {
                for (size_t j = 0; j < shape[1]; ++j) {
                    if constexpr (std::is_arithmetic_v<T>) {
                        if (at({i, j}) == T(0)) return false;
                    } else {
                        if (!at({i, j})) return false;
                    }
                }
            }
            return true;
        } else {
            throw std::invalid_argument("Invalid axis for 2D matrix");
        }
    }
}

// Matrix operations
template<typename T>
Matrix<T> Matrix<T>::reshape(const std::vector<size_t>& newShape) const {
    size_t new_size = 1;
    for (size_t dim : newShape) {
        new_size *= dim;
    }
    
    if (new_size != data.size()) {
        throw std::invalid_argument("New shape must have same total size");
    }
    
    Matrix<T> result;
    result.data = data;
    result.shape = newShape;
    result.calcStrides();
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::transpose(const std::vector<size_t>& axes) const {
    if (shape.size() != 2) {
        throw std::runtime_error("Transpose currently only supports 2D matrices");
    }
    
    std::vector<size_t> result_shape = {shape[1], shape[0]};
    Matrix<T> result(result_shape);
    
    const size_t parallel_th = 1000;
    
    if (shape[0] >= parallel_th) {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t rows_per_thread = shape[0] / num_threads;
        
        auto worker = [&](size_t start_row, size_t end_row) {
            for (size_t i = start_row; i < end_row; ++i) {
                for (size_t j = 0; j < shape[1]; ++j) {
                    result.at({j, i}) = at({i, j});
                }
            }
        };
        
        for (size_t t = 0; t < num_threads - 1; ++t) {
            threads.emplace_back(worker, t * rows_per_thread, (t + 1) * rows_per_thread);
        }
        worker((num_threads - 1) * rows_per_thread, shape[0]);
        
        for (auto& thread : threads) thread.join();
    } else {
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                result.at({j, i}) = at({i, j});
            }
        }
    }
    
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::slice(const std::vector<std::pair<size_t,size_t>>& ranges) const {
    if (ranges.size() != shape.size()) {
        throw std::invalid_argument("Number of ranges must match number of dimensions");
    }
    
    std::vector<size_t> new_shape;
    for (size_t i = 0; i < ranges.size(); ++i) {
        if (ranges[i].second <= ranges[i].first || ranges[i].second > shape[i]) {
            throw std::invalid_argument("Invalid range");
        }
        new_shape.push_back(ranges[i].second - ranges[i].first);
    }
    
    Matrix<T> result(new_shape);
    
    // Simple 2D case for now
    if (shape.size() == 2) {
        for (size_t i = 0; i < new_shape[0]; ++i) {
            for (size_t j = 0; j < new_shape[1]; ++j) {
                result.at({i, j}) = at({ranges[0].first + i, ranges[1].first + j});
            }
        }
    }
    
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::rotate90(int k) const {
    if (shape.size() != 2) {
        throw std::runtime_error("rotate90 only supports 2D matrices");
    }
    
    k = k % 4;
    if (k < 0) k += 4;
    
    Matrix<T> result = *this;
    
    for (int rotation = 0; rotation < k; ++rotation) {
        std::vector<size_t> temp_shape = {result.shape[1], result.shape[0]};
        Matrix<T> temp(temp_shape);
        for (size_t i = 0; i < result.shape[0]; ++i) {
            for (size_t j = 0; j < result.shape[1]; ++j) {
                temp.at({j, result.shape[0] - 1 - i}) = result.at({i, j});
            }
        }
        result = temp;
    }
    
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::flip(bool horizontal) const {
    if (shape.size() != 2) {
        throw std::runtime_error("flip only supports 2D matrices");
    }
    
    Matrix<T> result(shape);
    
    if (horizontal) {
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                result.at({i, j}) = at({i, shape[1] - 1 - j});
            }
        }
    } else {
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                result.at({i, j}) = at({shape[0] - 1 - i, j});
            }
        }
    }
    
    return result;
}

// Element-wise operations
template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix& other) const {
    if (shape != other.shape) {
        throw std::invalid_argument("Matrix shapes must match for addition");
    }
    
    Matrix<T> result(shape);
    const size_t parallel_th = 8000;
    
    if (data.size() >= parallel_th) {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t elements_per_thread = data.size() / num_threads;
        
        auto worker = [&](size_t start, size_t end) {
            for (size_t i = start; i < end; ++i) {
                result.data[i] = data[i] + other.data[i];
            }
        };
        
        for (size_t t = 0; t < num_threads - 1; ++t) {
            threads.emplace_back(worker, t * elements_per_thread, (t + 1) * elements_per_thread);
        }
        worker((num_threads - 1) * elements_per_thread, data.size());
        
        for (auto& thread : threads) thread.join();
    } else {
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] + other.data[i];
        }
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator-(const Matrix& other) const {
    if (shape != other.shape) {
        throw std::invalid_argument("Matrix shapes must match for subtraction");
    }
    
    Matrix<T> result(shape);
    const size_t parallel_th = 8000;
    
    if (data.size() >= parallel_th) {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t elements_per_thread = data.size() / num_threads;
        
        auto worker = [&](size_t start, size_t end) {
            for (size_t i = start; i < end; ++i) {
                result.data[i] = data[i] - other.data[i];
            }
        };
        
        for (size_t t = 0; t < num_threads - 1; ++t) {
            threads.emplace_back(worker, t * elements_per_thread, (t + 1) * elements_per_thread);
        }
        worker((num_threads - 1) * elements_per_thread, data.size());
        
        for (auto& thread : threads) thread.join();
    } else {
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] - other.data[i];
        }
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix& other) const {
    if (shape != other.shape) {
        throw std::invalid_argument("Matrix shapes must match for element-wise multiplication");
    }
    
    Matrix<T> result(shape);
    const size_t parallel_th = 8000;
    
    if (data.size() >= parallel_th) {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t elements_per_thread = data.size() / num_threads;
        
        auto worker = [&](size_t start, size_t end) {
            for (size_t i = start; i < end; ++i) {
                result.data[i] = data[i] * other.data[i];
            }
        };
        
        for (size_t t = 0; t < num_threads - 1; ++t) {
            threads.emplace_back(worker, t * elements_per_thread, (t + 1) * elements_per_thread);
        }
        worker((num_threads - 1) * elements_per_thread, data.size());
        
        for (auto& thread : threads) thread.join();
    } else {
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] * other.data[i];
        }
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator/(const Matrix& other) const {
    if (shape != other.shape) {
        throw std::invalid_argument("Matrix shapes must match for division");
    }
    
    Matrix<T> result(shape);
    for (size_t i = 0; i < data.size(); ++i) {
        if (other.data[i] == T(0)) {
            throw std::runtime_error("Division by zero");
        }
        result.data[i] = data[i] / other.data[i];
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator+(T scalar) const {
    Matrix<T> result(shape);
    const size_t parallel_th = 10000;
    
    if (data.size() >= parallel_th) {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t elements_per_thread = data.size() / num_threads;
        
        auto worker = [&](size_t start, size_t end) {
            for (size_t i = start; i < end; ++i) {
                result.data[i] = data[i] + scalar;
            }
        };
        
        for (size_t t = 0; t < num_threads - 1; ++t) {
            threads.emplace_back(worker, t * elements_per_thread, (t + 1) * elements_per_thread);
        }
        worker((num_threads - 1) * elements_per_thread, data.size());
        
        for (auto& thread : threads) thread.join();
    } else {
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] + scalar;
        }
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator-(T scalar) const {
    Matrix<T> result(shape);
    const size_t parallel_th = 10000;
    
    if (data.size() >= parallel_th) {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t elements_per_thread = data.size() / num_threads;
        
        auto worker = [&](size_t start, size_t end) {
            for (size_t i = start; i < end; ++i) {
                result.data[i] = data[i] - scalar;
            }
        };
        
        for (size_t t = 0; t < num_threads - 1; ++t) {
            threads.emplace_back(worker, t * elements_per_thread, (t + 1) * elements_per_thread);
        }
        worker((num_threads - 1) * elements_per_thread, data.size());
        
        for (auto& thread : threads) thread.join();
    } else {
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] - scalar;
        }
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(T scalar) const {
    Matrix<T> result(shape);
    const size_t parallel_th = 10000;
    
    if (data.size() >= parallel_th) {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t elements_per_thread = data.size() / num_threads;
        
        auto worker = [&](size_t start, size_t end) {
            for (size_t i = start; i < end; ++i) {
                result.data[i] = data[i] * scalar;
            }
        };
        
        for (size_t t = 0; t < num_threads - 1; ++t) {
            threads.emplace_back(worker, t * elements_per_thread, (t + 1) * elements_per_thread);
        }
        worker((num_threads - 1) * elements_per_thread, data.size());
        
        for (auto& thread : threads) thread.join();
    } else {
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] * scalar;
        }
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator/(T scalar) const {
    if (scalar == T(0)) {
        throw std::runtime_error("Division by zero");
    }
    
    Matrix<T> result(shape);
    const size_t parallel_th = 10000;
    
    if (data.size() >= parallel_th) {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t elements_per_thread = data.size() / num_threads;
        
        auto worker = [&](size_t start, size_t end) {
            for (size_t i = start; i < end; ++i) {
                result.data[i] = data[i] / scalar;
            }
        };
        
        for (size_t t = 0; t < num_threads - 1; ++t) {
            threads.emplace_back(worker, t * elements_per_thread, (t + 1) * elements_per_thread);
        }
        worker((num_threads - 1) * elements_per_thread, data.size());
        
        for (auto& thread : threads) thread.join();
    } else {
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] / scalar;
        }
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator+() const {
    return *this;
}

template<typename T>
Matrix<T> Matrix<T>::operator-() const {
    Matrix<T> result(shape);
    const size_t parallel_th = 10000;
    
    if (data.size() >= parallel_th) {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t elements_per_thread = data.size() / num_threads;
        
        auto worker = [&](size_t start, size_t end) {
            for (size_t i = start; i < end; ++i) {
                result.data[i] = -data[i];
            }
        };
        
        for (size_t t = 0; t < num_threads - 1; ++t) {
            threads.emplace_back(worker, t * elements_per_thread, (t + 1) * elements_per_thread);
        }
        worker((num_threads - 1) * elements_per_thread, data.size());
        
        for (auto& thread : threads) thread.join();
    } else {
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = -data[i];
        }
    }
    return result;
}

// Math element-wise operations
template<typename T>
Matrix<T> Matrix<T>::exp() const {
    Matrix<T> result(shape);
    const size_t parallel_th = 8000;
    
    if (data.size() >= parallel_th) {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t elements_per_thread = data.size() / num_threads;
        
        auto worker = [&](size_t start, size_t end) {
            for (size_t i = start; i < end; ++i) {
                result.data[i] = std::exp(data[i]);
            }
        };
        
        for (size_t t = 0; t < num_threads - 1; ++t) {
            threads.emplace_back(worker, t * elements_per_thread, (t + 1) * elements_per_thread);
        }
        worker((num_threads - 1) * elements_per_thread, data.size());
        
        for (auto& thread : threads) thread.join();
    } else {
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = std::exp(data[i]);
        }
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::log() const {
    Matrix<T> result(shape);
    const size_t parallel_th = 8000;
    
    if (data.size() >= parallel_th) {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t elements_per_thread = data.size() / num_threads;
        
        auto worker = [&](size_t start, size_t end) {
            for (size_t i = start; i < end; ++i) {
                if (data[i] <= T(0)) {
                    throw std::runtime_error("Logarithm of non-positive number");
                }
                result.data[i] = std::log(data[i]);
            }
        };
        
        for (size_t t = 0; t < num_threads - 1; ++t) {
            threads.emplace_back(worker, t * elements_per_thread, (t + 1) * elements_per_thread);
        }
        worker((num_threads - 1) * elements_per_thread, data.size());
        
        for (auto& thread : threads) thread.join();
    } else {
        for (size_t i = 0; i < data.size(); ++i) {
            if (data[i] <= T(0)) {
                throw std::runtime_error("Logarithm of non-positive number");
            }
            result.data[i] = std::log(data[i]);
        }
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::sqrt() const {
    Matrix<T> result(shape);
    const size_t parallel_th = 8000;
    
    if (data.size() >= parallel_th) {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t elements_per_thread = data.size() / num_threads;
        
        auto worker = [&](size_t start, size_t end) {
            for (size_t i = start; i < end; ++i) {
                if (data[i] < T(0)) {
                    throw std::runtime_error("Square root of negative number");
                }
                result.data[i] = std::sqrt(data[i]);
            }
        };
        
        for (size_t t = 0; t < num_threads - 1; ++t) {
            threads.emplace_back(worker, t * elements_per_thread, (t + 1) * elements_per_thread);
        }
        worker((num_threads - 1) * elements_per_thread, data.size());
        
        for (auto& thread : threads) thread.join();
    } else {
        for (size_t i = 0; i < data.size(); ++i) {
            if (data[i] < T(0)) {
                throw std::runtime_error("Square root of negative number");
            }
            result.data[i] = std::sqrt(data[i]);
        }
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::pow(T expo) const {
    Matrix<T> result(shape);
    const size_t parallel_th = 6000;
    
    if (data.size() >= parallel_th) {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t elements_per_thread = data.size() / num_threads;
        
        auto worker = [&](size_t start, size_t end) {
            for (size_t i = start; i < end; ++i) {
                result.data[i] = std::pow(data[i], expo);
            }
        };
        
        for (size_t t = 0; t < num_threads - 1; ++t) {
            threads.emplace_back(worker, t * elements_per_thread, (t + 1) * elements_per_thread);
        }
        worker((num_threads - 1) * elements_per_thread, data.size());
        
        for (auto& thread : threads) thread.join();
    } else {
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = std::pow(data[i], expo);
        }
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::round() const {
    Matrix<T> result(shape);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = std::round(data[i]);
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::ceil() const {
    Matrix<T> result(shape);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = std::ceil(data[i]);
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::floor() const {
    Matrix<T> result(shape);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = std::floor(data[i]);
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::abs() const {
    Matrix<T> result(shape);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = std::abs(data[i]);
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::sin() const {
    Matrix<T> result(shape);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = std::sin(data[i]);
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::cos() const {
    Matrix<T> result(shape);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = std::cos(data[i]);
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::tan() const {
    Matrix<T> result(shape);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = std::tan(data[i]);
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::arcsin() const {
    Matrix<T> result(shape);
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i] < T(-1) || data[i] > T(1)) {
            throw std::runtime_error("arcsin input out of range [-1, 1]");
        }
        result.data[i] = std::asin(data[i]);
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::arccos() const {
    Matrix<T> result(shape);
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i] < T(-1) || data[i] > T(1)) {
            throw std::runtime_error("arccos input out of range [-1, 1]");
        }
        result.data[i] = std::acos(data[i]);
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::arctan() const {
    Matrix<T> result(shape);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = std::atan(data[i]);
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::diff(size_t n, size_t axis) const {
    // Simplified 1D difference for now
    if (shape.size() != 1) {
        throw std::runtime_error("diff currently only supports 1D arrays");
    }
    
    if (data.size() <= n) {
        return Matrix<T>({0});
    }
    
    std::vector<size_t> result_shape = {data.size() - n};
    Matrix<T> result(result_shape);
    for (size_t i = 0; i < result.data.size(); ++i) {
        result.data[i] = data[i + n] - data[i];
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::gradient(const std::vector<T>& spacing) const {
    // Simplified 1D gradient
    if (shape.size() != 1) {
        throw std::runtime_error("gradient currently only supports 1D arrays");
    }
    
    T h = spacing.empty() ? T(1) : spacing[0];
    Matrix<T> result(shape);
    
    if (data.size() == 1) {
        result.data[0] = T(0);
        return result;
    }
    
    // Forward difference for first point
    result.data[0] = (data[1] - data[0]) / h;
    
    // Central difference for middle points
    for (size_t i = 1; i < data.size() - 1; ++i) {
        result.data[i] = (data[i + 1] - data[i - 1]) / (T(2) * h);
    }
    
    // Backward difference for last point
    result.data[data.size() - 1] = (data[data.size() - 1] - data[data.size() - 2]) / h;
    
    return result;
}

// Matrix multiplication
template<typename T>
Matrix<T> Matrix<T>::matmul(const Matrix& other) const {
    if (shape.size() != 2 || other.shape.size() != 2) {
        throw std::runtime_error("matmul requires 2D matrices");
    }
    
    if (shape[1] != other.shape[0]) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    
    std::vector<size_t> result_shape = {shape[0], other.shape[1]};
    Matrix<T> result(result_shape);
    
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < other.shape[1]; ++j) {
            T sum = T(0);
            for (size_t k = 0; k < shape[1]; ++k) {
                sum += at({i, k}) * other.at({k, j});
            }
            result.at({i, j}) = sum;
        }
    }
    
    return result;
}

// Comparison operations
template<typename T>
bool Matrix<T>::operator==(const Matrix& other) const {
    return shape == other.shape && data == other.data;
}

template<typename T>
bool Matrix<T>::operator!=(const Matrix& other) const {
    return !(*this == other);
}

// Note: isnan, isinf, isfinite functions commented out due to Matrix<bool> compatibility issues
// TODO: These need to be reimplemented with a different approach or header needs to be updated

/*
template<typename T>
Matrix<int> Matrix<T>::isnan() const {
    std::vector<size_t> result_shape = shape;
    Matrix<int> result(result_shape);
    
    // Create indices vector for accessing elements
    std::vector<size_t> indices(shape.size(), 0);
    
    for (size_t i = 0; i < data.size(); ++i) {
        // Convert linear index back to multi-dimensional indices
        size_t temp = i;
        for (int dim = static_cast<int>(shape.size()) - 1; dim >= 0; --dim) {
            indices[dim] = temp % shape[dim];
            temp /= shape[dim];
        }
        
        if constexpr (std::is_floating_point_v<T>) {
            result.at(indices) = std::isnan(data[i]) ? 1 : 0;
        } else {
            result.at(indices) = 0;
        }
    }
    return result;
}
*/

/*
template<typename T>
Matrix<bool> Matrix<T>::isinf() const {
    std::vector<size_t> result_shape = shape;
    Matrix<bool> result(result_shape);
    
    // Create indices vector for accessing elements
    std::vector<size_t> indices(shape.size(), 0);
    
    for (size_t i = 0; i < data.size(); ++i) {
        // Convert linear index back to multi-dimensional indices
        size_t temp = i;
        for (int dim = static_cast<int>(shape.size()) - 1; dim >= 0; --dim) {
            indices[dim] = temp % shape[dim];
            temp /= shape[dim];
        }
        
        if constexpr (std::is_floating_point_v<T>) {
            result.at(indices) = std::isinf(data[i]);
        } else {
            result.at(indices) = false;
        }
    }
    return result;
}
*/

/*
template<typename T>
Matrix<bool> Matrix<T>::isfinite() const {
    std::vector<size_t> result_shape = shape;
    Matrix<bool> result(result_shape);
    
    // Create indices vector for accessing elements
    std::vector<size_t> indices(shape.size(), 0);
    
    for (size_t i = 0; i < data.size(); ++i) {
        // Convert linear index back to multi-dimensional indices
        size_t temp = i;
        for (int dim = static_cast<int>(shape.size()) - 1; dim >= 0; --dim) {
            indices[dim] = temp % shape[dim];
            temp /= shape[dim];
        }
        
        if constexpr (std::is_floating_point_v<T>) {
            result.at(indices) = std::isfinite(data[i]);
        } else {
            result.at(indices) = true;
        }
    }
    return result;
}
*/

// Basic statistics
template<typename T>
T Matrix<T>::sum() const {
    return std::accumulate(data.begin(), data.end(), T(0));
}

template<typename T>
T Matrix<T>::max() const {
    if (data.empty()) {
        throw std::runtime_error("Cannot find max of empty matrix");
    }
    return *std::max_element(data.begin(), data.end());
}

template<typename T>
T Matrix<T>::min() const {
    if (data.empty()) {
        throw std::runtime_error("Cannot find min of empty matrix");
    }
    return *std::min_element(data.begin(), data.end());
}

template<typename T>
Matrix<T> Matrix<T>::sum(size_t axis) const {
    // Simplified implementation for 2D case
    if (shape.size() != 2) {
        throw std::runtime_error("Axis-based sum currently only supports 2D matrices");
    }
    
    const size_t parallel_th = 500;
    
    if (axis == 0) {
        // Sum along rows (result is 1D with shape[1] elements)
        std::vector<size_t> result_shape = {shape[1]};
        Matrix<T> result(result_shape);
        
        if (shape[1] >= parallel_th) {
            size_t num_threads = std::thread::hardware_concurrency();
            std::vector<std::thread> threads;
            size_t cols_per_thread = shape[1] / num_threads;
            
            auto worker = [&](size_t start_col, size_t end_col) {
                for (size_t j = start_col; j < end_col; ++j) {
                    T sum_val = T(0);
                    for (size_t i = 0; i < shape[0]; ++i) {
                        sum_val += at({i, j});
                    }
                    result.data[j] = sum_val;
                }
            };
            
            for (size_t t = 0; t < num_threads - 1; ++t) {
                threads.emplace_back(worker, t * cols_per_thread, (t + 1) * cols_per_thread);
            }
            worker((num_threads - 1) * cols_per_thread, shape[1]);
            
            for (auto& thread : threads) thread.join();
        } else {
            for (size_t j = 0; j < shape[1]; ++j) {
                T sum_val = T(0);
                for (size_t i = 0; i < shape[0]; ++i) {
                    sum_val += at({i, j});
                }
                result.data[j] = sum_val;
            }
        }
        return result;
    } else if (axis == 1) {
        // Sum along columns (result is 1D with shape[0] elements)
        std::vector<size_t> result_shape = {shape[0]};
        Matrix<T> result(result_shape);
        
        if (shape[0] >= parallel_th) {
            size_t num_threads = std::thread::hardware_concurrency();
            std::vector<std::thread> threads;
            size_t rows_per_thread = shape[0] / num_threads;
            
            auto worker = [&](size_t start_row, size_t end_row) {
                for (size_t i = start_row; i < end_row; ++i) {
                    T sum_val = T(0);
                    for (size_t j = 0; j < shape[1]; ++j) {
                        sum_val += at({i, j});
                    }
                    result.data[i] = sum_val;
                }
            };
            
            for (size_t t = 0; t < num_threads - 1; ++t) {
                threads.emplace_back(worker, t * rows_per_thread, (t + 1) * rows_per_thread);
            }
            worker((num_threads - 1) * rows_per_thread, shape[0]);
            
            for (auto& thread : threads) thread.join();
        } else {
            for (size_t i = 0; i < shape[0]; ++i) {
                T sum_val = T(0);
                for (size_t j = 0; j < shape[1]; ++j) {
                    sum_val += at({i, j});
                }
                result.data[i] = sum_val;
            }
        }
        return result;
    } else {
        throw std::invalid_argument("Invalid axis for 2D matrix");
    }
}

template<typename T>
Matrix<T> Matrix<T>::max(size_t axis) const {
    // Similar to sum but with max operation
    if (shape.size() != 2) {
        throw std::runtime_error("Axis-based max currently only supports 2D matrices");
    }
    
    if (axis == 0) {
        std::vector<size_t> result_shape = {shape[1]};
        Matrix<T> result(result_shape);
        for (size_t j = 0; j < shape[1]; ++j) {
            T max_val = at({0, j});
            for (size_t i = 1; i < shape[0]; ++i) {
                max_val = std::max(max_val, at({i, j}));
            }
            result.data[j] = max_val;
        }
        return result;
    } else if (axis == 1) {
        std::vector<size_t> result_shape = {shape[0]};
        Matrix<T> result(result_shape);
        for (size_t i = 0; i < shape[0]; ++i) {
            T max_val = at({i, 0});
            for (size_t j = 1; j < shape[1]; ++j) {
                max_val = std::max(max_val, at({i, j}));
            }
            result.data[i] = max_val;
        }
        return result;
    } else {
        throw std::invalid_argument("Invalid axis for 2D matrix");
    }
}

template<typename T>
Matrix<T> Matrix<T>::min(size_t axis) const {
    // Similar to max but with min operation
    if (shape.size() != 2) {
        throw std::runtime_error("Axis-based min currently only supports 2D matrices");
    }
    
    if (axis == 0) {
        std::vector<size_t> result_shape = {shape[1]};
        Matrix<T> result(result_shape);
        for (size_t j = 0; j < shape[1]; ++j) {
            T min_val = at({0, j});
            for (size_t i = 1; i < shape[0]; ++i) {
                min_val = std::min(min_val, at({i, j}));
            }
            result.data[j] = min_val;
        }
        return result;
    } else if (axis == 1) {
        std::vector<size_t> result_shape = {shape[0]};
        Matrix<T> result(result_shape);
        for (size_t i = 0; i < shape[0]; ++i) {
            T min_val = at({i, 0});
            for (size_t j = 1; j < shape[1]; ++j) {
                min_val = std::min(min_val, at({i, j}));
            }
            result.data[i] = min_val;
        }
        return result;
    } else {
        throw std::invalid_argument("Invalid axis for 2D matrix");
    }
}

// Static factory methods
template<typename T>
Matrix<T> Matrix<T>::zeros(const std::vector<size_t>& shape) {
    return Matrix<T>(shape, T(0));
}

template<typename T>
Matrix<T> Matrix<T>::ones(const std::vector<size_t>& shape) {
    return Matrix<T>(shape, T(1));
}

template<typename T>
Matrix<T> Matrix<T>::eye(size_t n) {
    std::vector<size_t> result_shape = {n, n};
    Matrix<T> result(result_shape, T(0));
    for (size_t i = 0; i < n; ++i) {
        result.at({i, i}) = T(1);
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::random(const std::vector<size_t>& shape, T minVal, T maxVal) {
    Matrix<T> result(shape);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    if constexpr (std::is_integral_v<T>) {
        std::uniform_int_distribution<T> dis(minVal, maxVal);
        for (auto& element : result.data) {
            element = dis(gen);
        }
    } else {
        std::uniform_real_distribution<T> dis(minVal, maxVal);
        for (auto& element : result.data) {
            element = dis(gen);
        }
    }
    
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::arange(T start, T end, T step) {
    if (step == T(0)) {
        throw std::invalid_argument("Step cannot be zero");
    }
    
    std::vector<T> values;
    if (step > T(0)) {
        for (T val = start; val < end; val += step) {
            values.push_back(val);
        }
    } else {
        for (T val = start; val > end; val += step) {
            values.push_back(val);
        }
    }
    
    std::vector<size_t> result_shape = {values.size()};
    Matrix<T> result(result_shape);
    result.data = values;
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::linspace(T start, T stop, size_t num) {
    if (num == 0) {
        std::vector<size_t> empty_shape = {0};
        return Matrix<T>(empty_shape);
    }
    
    std::vector<size_t> result_shape = {num};
    Matrix<T> result(result_shape);
    if (num == 1) {
        result.data[0] = start;
        return result;
    }
    
    T step = (stop - start) / static_cast<T>(num - 1);
    for (size_t i = 0; i < num; ++i) {
        result.data[i] = start + static_cast<T>(i) * step;
    }
    
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::diag(const Matrix<T>& v, int k) {
    if (v.shape.size() != 1) {
        throw std::invalid_argument("Input must be 1D for diag");
    }
    
    size_t n = v.data.size() + std::abs(k);
    std::vector<size_t> result_shape = {n, n};
    Matrix<T> result(result_shape, T(0));
    
    for (size_t i = 0; i < v.data.size(); ++i) {
        if (k >= 0) {
            result.at({i, i + k}) = v.data[i];
        } else {
            result.at({i - k, i}) = v.data[i];
        }
    }
    
    return result;
}

// Array manipulation (simplified implementations)
template<typename T>
Matrix<T> Matrix<T>::concatenate(const Matrix<T>& other, size_t axis) const {
    if (shape.size() != other.shape.size()) {
        throw std::invalid_argument("Arrays must have same number of dimensions");
    }
    
    // Simple 2D case along axis 0 (rows)
    if (shape.size() == 2 && axis == 0) {
        if (shape[1] != other.shape[1]) {
            throw std::invalid_argument("Shapes not compatible for concatenation");
        }
        
        std::vector<size_t> result_shape = {shape[0] + other.shape[0], shape[1]};
        Matrix<T> result(result_shape);
        
        // Copy first matrix
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                result.at({i, j}) = at({i, j});
            }
        }
        
        // Copy second matrix
        for (size_t i = 0; i < other.shape[0]; ++i) {
            for (size_t j = 0; j < other.shape[1]; ++j) {
                result.at({shape[0] + i, j}) = other.at({i, j});
            }
        }
        
        return result;
    }
    
    throw std::runtime_error("concatenate not fully implemented for this case");
}

template<typename T>
Matrix<T> Matrix<T>::stack(const Matrix<T>& other, size_t axis) const {
    if (shape != other.shape) {
        throw std::invalid_argument("Arrays must have same shape for stacking");
    }
    
    std::vector<size_t> result_shape = shape;
    result_shape.insert(result_shape.begin() + axis, 2);
    
    Matrix<T> result(result_shape);
    
    // Simple 2D case for axis 0
    if (shape.size() == 2 && axis == 0) {
        // First matrix
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                result.at({0, i, j}) = at({i, j});
            }
        }
        // Second matrix
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                result.at({1, i, j}) = other.at({i, j});
            }
        }
        return result;
    }
    
    throw std::runtime_error("stack not fully implemented for this configuration");
}

template<typename T>
std::vector<Matrix<T>> Matrix<T>::split(size_t sections, size_t axis) const {
    if (shape.size() != 2 || axis >= shape.size()) {
        throw std::invalid_argument("split currently only supports 2D matrices");
    }
    
    size_t dim_size = shape[axis];
    if (dim_size % sections != 0) {
        throw std::invalid_argument("Array cannot be divided evenly into sections");
    }
    
    size_t section_size = dim_size / sections;
    std::vector<Matrix<T>> result;
    
    if (axis == 0) {
        // Split along rows
        for (size_t s = 0; s < sections; ++s) {
            std::vector<size_t> section_shape = {section_size, shape[1]};
            Matrix<T> section(section_shape);
            
            for (size_t i = 0; i < section_size; ++i) {
                for (size_t j = 0; j < shape[1]; ++j) {
                    section.at({i, j}) = at({s * section_size + i, j});
                }
            }
            result.push_back(section);
        }
    } else if (axis == 1) {
        // Split along columns
        for (size_t s = 0; s < sections; ++s) {
            std::vector<size_t> section_shape = {shape[0], section_size};
            Matrix<T> section(section_shape);
            
            for (size_t i = 0; i < shape[0]; ++i) {
                for (size_t j = 0; j < section_size; ++j) {
                    section.at({i, j}) = at({i, s * section_size + j});
                }
            }
            result.push_back(section);
        }
    }
    
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::swapaxes(size_t axis1, size_t axis2) const {
    if (axis1 >= shape.size() || axis2 >= shape.size()) {
        throw std::invalid_argument("Axis out of bounds");
    }
    
    if (axis1 == axis2) {
        return *this;
    }
    
    // For 2D matrices
    if (shape.size() == 2) {
        if ((axis1 == 0 && axis2 == 1) || (axis1 == 1 && axis2 == 0)) {
            return transpose();
        }
    }
    
    // General case - create new shape with swapped axes
    std::vector<size_t> new_shape = shape;
    std::swap(new_shape[axis1], new_shape[axis2]);
    
    Matrix<T> result(new_shape);
    
    // For 2D case
    if (shape.size() == 2) {
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                if (axis1 == 0 && axis2 == 1) {
                    result.at({j, i}) = at({i, j});
                } else {
                    result.at({i, j}) = at({i, j});
                }
            }
        }
    }
    
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::repeat(size_t repeats, size_t axis) const {
    if (axis == static_cast<size_t>(-1)) {
        // Repeat each element
        std::vector<size_t> result_shape = {data.size() * repeats};
        Matrix<T> result(result_shape);
        
        for (size_t i = 0; i < data.size(); ++i) {
            for (size_t r = 0; r < repeats; ++r) {
                result.data[i * repeats + r] = data[i];
            }
        }
        return result;
    }
    
    if (axis >= shape.size()) {
        throw std::invalid_argument("Axis out of bounds");
    }
    
    // For 2D matrices
    if (shape.size() == 2) {
        if (axis == 0) {
            // Repeat along rows
            std::vector<size_t> result_shape = {shape[0] * repeats, shape[1]};
            Matrix<T> result(result_shape);
            
            for (size_t i = 0; i < shape[0]; ++i) {
                for (size_t r = 0; r < repeats; ++r) {
                    for (size_t j = 0; j < shape[1]; ++j) {
                        result.at({i * repeats + r, j}) = at({i, j});
                    }
                }
            }
            return result;
        } else if (axis == 1) {
            // Repeat along columns
            std::vector<size_t> result_shape = {shape[0], shape[1] * repeats};
            Matrix<T> result(result_shape);
            
            for (size_t i = 0; i < shape[0]; ++i) {
                for (size_t j = 0; j < shape[1]; ++j) {
                    for (size_t r = 0; r < repeats; ++r) {
                        result.at({i, j * repeats + r}) = at({i, j});
                    }
                }
            }
            return result;
        }
    }
    
    throw std::runtime_error("repeat not fully implemented for this configuration");
}

// Broadcasting (simplified)
template<typename T>
bool Matrix<T>::broadcastable(const Matrix<T>& other) const {
    // Simple check - same shape for now
    return shape == other.shape;
}

template<typename T>
std::pair<Matrix<T>, Matrix<T>> Matrix<T>::broadcast(const Matrix<T>& other) const {
    // Simple broadcasting - if shapes are different, try to make them compatible
    if (shape == other.shape) {
        return {*this, other};
    }
    
    // Find the maximum number of dimensions
    size_t max_dims = std::max(shape.size(), other.shape.size());
    
    // Pad shapes with 1s at the beginning
    std::vector<size_t> shape1 = shape;
    std::vector<size_t> shape2 = other.shape;
    
    while (shape1.size() < max_dims) {
        shape1.insert(shape1.begin(), 1);
    }
    while (shape2.size() < max_dims) {
        shape2.insert(shape2.begin(), 1);
    }
    
    // Check if broadcasting is possible and find result shape
    std::vector<size_t> result_shape(max_dims);
    for (size_t i = 0; i < max_dims; ++i) {
        if (shape1[i] == shape2[i]) {
            result_shape[i] = shape1[i];
        } else if (shape1[i] == 1) {
            result_shape[i] = shape2[i];
        } else if (shape2[i] == 1) {
            result_shape[i] = shape1[i];
        } else {
            throw std::invalid_argument("Shapes are not broadcastable");
        }
    }
    
    // For simple cases, return reshaped versions
    Matrix<T> broadcasted1 = reshape(shape1);
    Matrix<T> broadcasted2 = other.reshape(shape2);
    
    return {broadcasted1, broadcasted2};
}

// Linear algebra (basic implementations)
template<typename T>
Matrix<T> Matrix<T>::dot(const Matrix<T>& other) const {
    return matmul(other);
}

template<typename T>
Matrix<T> Matrix<T>::inverse() const {
    if (shape.size() != 2 || shape[0] != shape[1]) {
        throw std::invalid_argument("Matrix must be square for inversion");
    }
    
    size_t n = shape[0];
    
    // Handle trivial cases
    if (n == 1) {
        if (std::abs(at({0, 0})) < 1e-10) {
            throw std::runtime_error("Matrix is singular");
        }
        Matrix<T> result(std::vector<size_t>{1, 1});
        result.at({0, 0}) = T(1) / at({0, 0});
        return result;
    }
    
    // Create augmented matrix [A | I]
    Matrix<T> augmented(std::vector<size_t>{n, 2 * n});
    
    // Copy original matrix to left side
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            augmented.at({i, j}) = at({i, j});
        }
    }
    
    // Create identity matrix on right side
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            augmented.at({i, n + j}) = (i == j) ? T(1) : T(0);
        }
    }
    
    // Gaussian elimination with partial pivoting
    for (size_t k = 0; k < n; ++k) {
        // Find pivot row
        size_t pivot_row = k;
        T max_val = std::abs(augmented.at({k, k}));
        
        for (size_t i = k + 1; i < n; ++i) {
            T val = std::abs(augmented.at({i, k}));
            if (val > max_val) {
                max_val = val;
                pivot_row = i;
            }
        }
        
        // Check for singularity
        if (std::abs(augmented.at({pivot_row, k})) < 1e-10) {
            throw std::runtime_error("Matrix is singular");
        }
        
        // Swap rows if needed
        if (pivot_row != k) {
            for (size_t j = 0; j < 2 * n; ++j) {
                T temp = augmented.at({k, j});
                augmented.at({k, j}) = augmented.at({pivot_row, j});
                augmented.at({pivot_row, j}) = temp;
            }
        }
        
        // Scale pivot row
        T pivot = augmented.at({k, k});
        for (size_t j = 0; j < 2 * n; ++j) {
            augmented.at({k, j}) /= pivot;
        }
        
        // Eliminate column
        for (size_t i = 0; i < n; ++i) {
            if (i != k) {
                T factor = augmented.at({i, k});
                for (size_t j = 0; j < 2 * n; ++j) {
                    augmented.at({i, j}) -= factor * augmented.at({k, j});
                }
            }
        }
    }
    
    // Extract inverse matrix from right side
    Matrix<T> result(std::vector<size_t>{n, n});
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result.at({i, j}) = augmented.at({i, n + j});
        }
    }
    
    return result;
}

template<typename T>
T Matrix<T>::determinant() const {
    if (shape.size() != 2 || shape[0] != shape[1]) {
        throw std::invalid_argument("Matrix must be square for determinant");
    }
    
    size_t n = shape[0];
    
    if (n == 1) {
        return at({0, 0});
    } else if (n == 2) {
        return at({0, 0}) * at({1, 1}) - at({0, 1}) * at({1, 0});
    }
    
    // For larger matrices, use LU decomposition
    Matrix<T> temp = *this; // Copy for in-place operations
    T det = T(1);
    
    // Gaussian elimination to get upper triangular matrix
    for (size_t k = 0; k < n; ++k) {
        // Find pivot
        size_t pivot_row = k;
        T max_val = std::abs(temp.at({k, k}));
        
        for (size_t i = k + 1; i < n; ++i) {
            T val = std::abs(temp.at({i, k}));
            if (val > max_val) {
                max_val = val;
                pivot_row = i;
            }
        }
        
        // If pivot is zero, determinant is zero
        if (std::abs(temp.at({pivot_row, k})) < 1e-10) {
            return T(0);
        }
        
        // Swap rows if needed (affects sign of determinant)
        if (pivot_row != k) {
            for (size_t j = k; j < n; ++j) {
                T swap_temp = temp.at({k, j});
                temp.at({k, j}) = temp.at({pivot_row, j});
                temp.at({pivot_row, j}) = swap_temp;
            }
            det = -det; // Row swap changes sign
        }
        
        // Update determinant with diagonal element
        det *= temp.at({k, k});
        
        // Eliminate below pivot
        for (size_t i = k + 1; i < n; ++i) {
            T factor = temp.at({i, k}) / temp.at({k, k});
            for (size_t j = k + 1; j < n; ++j) {
                temp.at({i, j}) -= factor * temp.at({k, j});
            }
        }
    }
    
    return det;
}

template<typename T>
std::pair<Matrix<T>, Matrix<T>> Matrix<T>::eigenDecomposition() const {
    if (!isSquare()) {
        throw std::invalid_argument("Eigendecomposition requires square matrix");
    }
    
    // Simple 2x2 case
    if (shape[0] == 2) {
        T a = at({0, 0});
        T b = at({0, 1});
        T c = at({1, 0});
        T d = at({1, 1});
        
        // Characteristic polynomial: λ² - (a+d)λ + (ad-bc) = 0
        T trace_val = a + d;
        T det = a * d - b * c;
        T discriminant = trace_val * trace_val - 4 * det;
        
        if (discriminant < 0) {
            throw std::runtime_error("Complex eigenvalues not supported");
        }
        
        T sqrt_disc = std::sqrt(discriminant);
        T lambda1 = (trace_val + sqrt_disc) / 2;
        T lambda2 = (trace_val - sqrt_disc) / 2;
        
        // Eigenvalues
        Matrix<T> eigenvalues({2});
        eigenvalues.at({0}) = lambda1;
        eigenvalues.at({1}) = lambda2;
        
        // Eigenvectors
        Matrix<T> eigenvectors({2, 2});
        
        // For lambda1
        if (std::abs(b) > 1e-10) {
            eigenvectors.at({0, 0}) = lambda1 - d;
            eigenvectors.at({1, 0}) = b;
        } else if (std::abs(c) > 1e-10) {
            eigenvectors.at({0, 0}) = c;
            eigenvectors.at({1, 0}) = lambda1 - a;
        } else {
            eigenvectors.at({0, 0}) = 1;
            eigenvectors.at({1, 0}) = 0;
        }
        
        // For lambda2
        if (std::abs(b) > 1e-10) {
            eigenvectors.at({0, 1}) = lambda2 - d;
            eigenvectors.at({1, 1}) = b;
        } else if (std::abs(c) > 1e-10) {
            eigenvectors.at({0, 1}) = c;
            eigenvectors.at({1, 1}) = lambda2 - a;
        } else {
            eigenvectors.at({0, 1}) = 0;
            eigenvectors.at({1, 1}) = 1;
        }
        
        // Normalize eigenvectors
        for (size_t j = 0; j < 2; ++j) {
            T norm = std::sqrt(eigenvectors.at({0, j}) * eigenvectors.at({0, j}) + 
                              eigenvectors.at({1, j}) * eigenvectors.at({1, j}));
            if (norm > 1e-10) {
                eigenvectors.at({0, j}) /= norm;
                eigenvectors.at({1, j}) /= norm;
            }
        }
        
        return {eigenvalues, eigenvectors};
    }
    
    throw std::runtime_error("Eigendecomposition not implemented for matrices larger than 2x2");
}

template<typename T>
std::vector<Matrix<T>> Matrix<T>::svd() const {
    if (shape.size() != 2) {
        throw std::invalid_argument("SVD requires 2D matrix");
    }
    
    // For square matrices, use simplified approach
    if (isSquare() && shape[0] <= 2) {
        // For small matrices, use eigendecomposition approach
        // A = U * Σ * V^T
        // A^T * A = V * Σ² * V^T (eigendecomposition of A^T * A)
        // A * A^T = U * Σ² * U^T (eigendecomposition of A * A^T)
        
        Matrix<T> AT = transpose();
        Matrix<T> ATA = AT.matmul(*this);
        Matrix<T> AAT = matmul(AT);
        
        auto [eigenvals_V, V] = ATA.eigenDecomposition();
        auto [eigenvals_U, U] = AAT.eigenDecomposition();
        
        // Singular values are sqrt of eigenvalues
        size_t min_dim = std::min(shape[0], shape[1]);
        std::vector<size_t> s_shape = {min_dim};
        Matrix<T> S(s_shape);
        for (size_t i = 0; i < S.size(); ++i) {
            S.at({i}) = std::sqrt(std::max(T(0), eigenvals_V.at({i})));
        }
        
        return {U, S, V.transpose()};
    }
    
    throw std::runtime_error("SVD not fully implemented for this matrix size");
}

template<typename T>
std::vector<Matrix<T>> Matrix<T>::LU_decomposition() const {
    if (!isSquare()) {
        throw std::invalid_argument("LU decomposition requires square matrix");
    }
    
    size_t n = shape[0];
    Matrix<T> L = Matrix<T>::eye(n);
    Matrix<T> U = *this;
    
    // Gaussian elimination with partial pivoting
    for (size_t k = 0; k < n - 1; ++k) {
        // Find pivot
        size_t pivot_row = k;
        for (size_t i = k + 1; i < n; ++i) {
            if (std::abs(U.at({i, k})) > std::abs(U.at({pivot_row, k}))) {
                pivot_row = i;
            }
        }
        
        // Swap rows if needed
        if (pivot_row != k) {
            for (size_t j = 0; j < n; ++j) {
                std::swap(U.data[k * n + j], U.data[pivot_row * n + j]);
            }
            for (size_t j = 0; j < k; ++j) {
                std::swap(L.data[k * n + j], L.data[pivot_row * n + j]);
            }
        }
        
        // Check for zero pivot
        if (std::abs(U.at({k, k})) < 1e-10) {
            throw std::runtime_error("Matrix is singular");
        }
        
        // Eliminate column
        for (size_t i = k + 1; i < n; ++i) {
            T factor = U.at({i, k}) / U.at({k, k});
            L.at({i, k}) = factor;
            
            for (size_t j = k; j < n; ++j) {
                U.at({i, j}) -= factor * U.at({k, j});
            }
        }
    }
    
    return {L, U};
}

// Statistics
template<typename T>
Matrix<T> Matrix<T>::normalize(T epsilon) const {
    T norm = std::sqrt(std::inner_product(data.begin(), data.end(), data.begin(), T(0)));
    if (norm < epsilon) {
        throw std::runtime_error("Cannot normalize zero vector");
    }
    return *this / norm;
}

template<typename T>
Matrix<T> Matrix<T>::standardize() const {
    T mean_val = mean();
    T std_val = standardDeviation();
    return (*this - mean_val) / std_val;
}

template<typename T>
T Matrix<T>::correlation(const Matrix<T>& A, const Matrix<T>& B) {
    if (A.shape != B.shape) {
        throw std::invalid_argument("Matrices must have same shape for correlation");
    }
    
    T mean_A = A.mean();
    T mean_B = B.mean();
    
    T numerator = T(0);
    T sum_A_sq = T(0);
    T sum_B_sq = T(0);
    
    for (size_t i = 0; i < A.data.size(); ++i) {
        T diff_A = A.data[i] - mean_A;
        T diff_B = B.data[i] - mean_B;
        numerator += diff_A * diff_B;
        sum_A_sq += diff_A * diff_A;
        sum_B_sq += diff_B * diff_B;
    }
    
    T denominator = std::sqrt(sum_A_sq * sum_B_sq);
    if (denominator == T(0)) {
        throw std::runtime_error("Cannot compute correlation for constant arrays");
    }
    
    return numerator / denominator;
}

template<typename T>
Matrix<T> Matrix<T>::covariance() const {
    if (shape.size() != 2) {
        throw std::invalid_argument("Covariance requires 2D matrix");
    }
    
    size_t n_features = shape[1];
    size_t n_samples = shape[0];
    
    if (n_samples <= 1) {
        throw std::runtime_error("Need at least 2 samples for covariance");
    }
    
    // Calculate means for each feature
    Matrix<T> means = mean(0);
    
    // Create covariance matrix
    Matrix<T> cov({n_features, n_features}, T(0));
    
    for (size_t i = 0; i < n_features; ++i) {
        for (size_t j = 0; j < n_features; ++j) {
            T sum_prod = T(0);
            
            for (size_t k = 0; k < n_samples; ++k) {
                T diff_i = at({k, i}) - means.at({i});
                T diff_j = at({k, j}) - means.at({j});
                sum_prod += diff_i * diff_j;
            }
            
            cov.at({i, j}) = sum_prod / static_cast<T>(n_samples - 1);
        }
    }
    
    return cov;
}

template<typename T>
T Matrix<T>::mean() const {
    if (data.empty()) {
        throw std::runtime_error("Cannot compute mean of empty matrix");
    }
    return sum() / static_cast<T>(data.size());
}

template<typename T>
T Matrix<T>::variance() const {
    if (data.size() <= 1) {
        throw std::runtime_error("Need at least 2 elements for variance");
    }
    
    T mean_val = mean();
    T sum_sq_diff = T(0);
    
    for (const auto& element : data) {
        T diff = element - mean_val;
        sum_sq_diff += diff * diff;
    }
    
    return sum_sq_diff / static_cast<T>(data.size() - 1);
}

template<typename T>
T Matrix<T>::median() const {
    if (data.empty()) {
        throw std::runtime_error("Cannot compute median of empty matrix");
    }
    
    std::vector<T> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    
    size_t n = sorted_data.size();
    if (n % 2 == 0) {
        return (sorted_data[n/2 - 1] + sorted_data[n/2]) / T(2);
    } else {
        return sorted_data[n/2];
    }
}

template<typename T>
T Matrix<T>::standardDeviation() const {
    return std::sqrt(variance());
}

template<typename T>
Matrix<T> Matrix<T>::mean(size_t axis) const {
    // Use the existing sum implementation and divide
    Matrix<T> sum_result = sum(axis);
    
    if (axis == 0) {
        return sum_result / static_cast<T>(shape[0]);
    } else if (axis == 1) {
        return sum_result / static_cast<T>(shape[1]);
    }
    
    throw std::invalid_argument("Invalid axis");
}

template<typename T>
Matrix<T> Matrix<T>::median(size_t axis) const {
    if (shape.size() != 2) {
        throw std::runtime_error("Axis-based median currently only supports 2D matrices");
    }
    
    if (axis == 0) {
        // Median along rows (result is 1D with shape[1] elements)
        std::vector<size_t> result_shape = {shape[1]};
        Matrix<T> result(result_shape);
        
        for (size_t j = 0; j < shape[1]; ++j) {
            std::vector<T> column_data;
            for (size_t i = 0; i < shape[0]; ++i) {
                column_data.push_back(at({i, j}));
            }
            
            std::sort(column_data.begin(), column_data.end());
            size_t n = column_data.size();
            
            if (n % 2 == 0) {
                result.at({j}) = (column_data[n/2 - 1] + column_data[n/2]) / T(2);
            } else {
                result.at({j}) = column_data[n/2];
            }
        }
        return result;
    } else if (axis == 1) {
        // Median along columns (result is 1D with shape[0] elements)
        std::vector<size_t> result_shape = {shape[0]};
        Matrix<T> result(result_shape);
        
        for (size_t i = 0; i < shape[0]; ++i) {
            std::vector<T> row_data;
            for (size_t j = 0; j < shape[1]; ++j) {
                row_data.push_back(at({i, j}));
            }
            
            std::sort(row_data.begin(), row_data.end());
            size_t n = row_data.size();
            
            if (n % 2 == 0) {
                result.at({i}) = (row_data[n/2 - 1] + row_data[n/2]) / T(2);
            } else {
                result.at({i}) = row_data[n/2];
            }
        }
        return result;
    } else {
        throw std::invalid_argument("Invalid axis for 2D matrix");
    }
}

template<typename T>
Matrix<T> Matrix<T>::variance(size_t axis) const {
    if (shape.size() != 2) {
        throw std::runtime_error("Axis-based variance currently only supports 2D matrices");
    }
    
    Matrix<T> mean_vals = mean(axis);
    
    if (axis == 0) {
        // Variance along rows
        std::vector<size_t> result_shape = {shape[1]};
        Matrix<T> result(result_shape);
        
        for (size_t j = 0; j < shape[1]; ++j) {
            T sum_sq_diff = T(0);
            for (size_t i = 0; i < shape[0]; ++i) {
                T diff = at({i, j}) - mean_vals.at({j});
                sum_sq_diff += diff * diff;
            }
            result.at({j}) = sum_sq_diff / static_cast<T>(shape[0] - 1);
        }
        return result;
    } else if (axis == 1) {
        // Variance along columns
        std::vector<size_t> result_shape = {shape[0]};
        Matrix<T> result(result_shape);
        
        for (size_t i = 0; i < shape[0]; ++i) {
            T sum_sq_diff = T(0);
            for (size_t j = 0; j < shape[1]; ++j) {
                T diff = at({i, j}) - mean_vals.at({i});
                sum_sq_diff += diff * diff;
            }
            result.at({i}) = sum_sq_diff / static_cast<T>(shape[1] - 1);
        }
        return result;
    } else {
        throw std::invalid_argument("Invalid axis for 2D matrix");
    }
}

template<typename T>
Matrix<T> Matrix<T>::standardDeviation(size_t axis) const {
    Matrix<T> var = variance(axis);
    Matrix<T> result(var.getShape());
    
    for (size_t i = 0; i < var.size(); ++i) {
        // Convert linear index to multi-dimensional indices
        std::vector<size_t> indices(var.getShape().size());
        size_t temp = i;
        for (int dim = static_cast<int>(var.getShape().size()) - 1; dim >= 0; --dim) {
            indices[dim] = temp % var.getShape()[dim];
            temp /= var.getShape()[dim];
        }
        result.at(indices) = std::sqrt(var.at(indices));
    }
    
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::percentile(T p) const {
    if (p < T(0) || p > T(100)) {
        throw std::invalid_argument("Percentile must be between 0 and 100");
    }
    
    std::vector<T> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    
    size_t n = sorted_data.size();
    if (n == 0) {
        throw std::runtime_error("Cannot compute percentile of empty matrix");
    }
    
    T index = (p / T(100)) * static_cast<T>(n - 1);
    size_t lower_index = static_cast<size_t>(std::floor(index));
    size_t upper_index = static_cast<size_t>(std::ceil(index));
    
    if (lower_index == upper_index) {
        Matrix<T> result({1});
        result.at({0}) = sorted_data[lower_index];
        return result;
    } else {
        T weight = index - static_cast<T>(lower_index);
        T interpolated_value = sorted_data[lower_index] * (T(1) - weight) + 
                              sorted_data[upper_index] * weight;
        Matrix<T> result({1});
        result.at({0}) = interpolated_value;
        return result;
    }
}

// Advanced statistics
template<typename T>
Matrix<T> Matrix<T>::histogram(size_t bins) const {
    if (bins == 0) {
        throw std::invalid_argument("Number of bins must be positive");
    }
    
    if (data.empty()) {
        return Matrix<T>({bins}, T(0));
    }
    
    T min_val = *std::min_element(data.begin(), data.end());
    T max_val = *std::max_element(data.begin(), data.end());
    
    if (min_val == max_val) {
        Matrix<T> result({bins}, T(0));
        result.data[0] = static_cast<T>(data.size());
        return result;
    }
    
    T bin_width = (max_val - min_val) / static_cast<T>(bins);
    Matrix<T> result({bins}, T(0));
    
    for (const auto& value : data) {
        size_t bin_index = static_cast<size_t>((value - min_val) / bin_width);
        if (bin_index >= bins) {
            bin_index = bins - 1; // Handle edge case where value == max_val
        }
        result.at({bin_index}) += T(1);
    }
    
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::bincount() const {
    if constexpr (!std::is_integral_v<T>) {
        throw std::invalid_argument("bincount requires integer type");
    }
    
    if (data.empty()) {
        return Matrix<T>({0});
    }
    
    T max_val = *std::max_element(data.begin(), data.end());
    T min_val = *std::min_element(data.begin(), data.end());
    
    if (min_val < T(0)) {
        throw std::invalid_argument("bincount requires non-negative values");
    }
    
    size_t num_bins = static_cast<size_t>(max_val) + 1;
    Matrix<T> result({num_bins}, T(0));
    
    for (const auto& value : data) {
        if (value >= T(0)) {
            size_t index = static_cast<size_t>(value);
            if (index < num_bins) {
                result.at({index}) += T(1);
            }
        }
    }
    
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::digitize(const Matrix<T>& bins) const {
    if (bins.getShape().size() != 1) {
        throw std::invalid_argument("Bins must be 1D array");
    }
    
    Matrix<T> result(shape);
    
    // Check if bins are sorted - need to access bins data properly
    std::vector<T> sorted_bins;
    for (size_t i = 0; i < bins.size(); ++i) {
        sorted_bins.push_back(bins.at({i}));
    }
    std::sort(sorted_bins.begin(), sorted_bins.end());
    
    for (size_t i = 0; i < data.size(); ++i) {
        T value = data[i];
        
        // Find the appropriate bin using binary search
        auto it = std::upper_bound(sorted_bins.begin(), sorted_bins.end(), value);
        size_t bin_index = std::distance(sorted_bins.begin(), it);
        
        // Convert linear index to multi-dimensional indices
        std::vector<size_t> indices(shape.size());
        size_t temp = i;
        for (int dim = static_cast<int>(shape.size()) - 1; dim >= 0; --dim) {
            indices[dim] = temp % shape[dim];
            temp /= shape[dim];
        }
        result.at(indices) = static_cast<T>(bin_index);
    }
    
    return result;
}

// Dimensional operations
template<typename T>
Matrix<T> Matrix<T>::flatten() const {
    std::vector<size_t> result_shape = {data.size()};
    Matrix<T> result(result_shape);
    result.data = data;
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::expandDims(size_t axis) const {
    std::vector<size_t> new_shape = shape;
    if (axis > new_shape.size()) {
        throw std::invalid_argument("Invalid axis for expandDims");
    }
    
    new_shape.insert(new_shape.begin() + axis, 1);
    return reshape(new_shape);
}

template<typename T>
Matrix<T> Matrix<T>::squeeze() const {
    std::vector<size_t> new_shape;
    for (size_t dim : shape) {
        if (dim != 1) {
            new_shape.push_back(dim);
        }
    }
    
    if (new_shape.empty()) {
        new_shape.push_back(1);
    }
    
    return reshape(new_shape);
}

template<typename T>
Matrix<T> Matrix<T>::pad(const std::vector<std::pair<size_t,size_t>>& pad_width, const std::string& mode, T constant_value) const {
    if (pad_width.size() != shape.size()) {
        throw std::invalid_argument("pad_width must match number of dimensions");
    }
    
    // Calculate new shape
    std::vector<size_t> new_shape(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        new_shape[i] = shape[i] + pad_width[i].first + pad_width[i].second;
    }
    
    Matrix<T> result(new_shape);
    
    if (mode == "constant") {
        result.fill(constant_value);
    } else if (mode == "edge") {
        result.fill(T(0)); // Will be filled with edge values later
    } else {
        throw std::invalid_argument("Unsupported padding mode");
    }
    
    // For 2D case
    if (shape.size() == 2) {
        size_t pad_top = pad_width[0].first;
        size_t pad_left = pad_width[1].first;
        
        // Copy original data
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                result.at({i + pad_top, j + pad_left}) = at({i, j});
            }
        }
        
        if (mode == "edge") {
            // Fill edges with edge values
            // Top and bottom edges
            for (size_t i = 0; i < pad_top; ++i) {
                for (size_t j = pad_left; j < pad_left + shape[1]; ++j) {
                    result.at({i, j}) = at({0, j - pad_left});
                    result.at({new_shape[0] - 1 - i, j}) = at({shape[0] - 1, j - pad_left});
                }
            }
            
            // Left and right edges
            for (size_t i = 0; i < new_shape[0]; ++i) {
                for (size_t j = 0; j < pad_left; ++j) {
                    size_t orig_i = std::min(std::max(static_cast<int>(i) - static_cast<int>(pad_top), 0), 
                                           static_cast<int>(shape[0] - 1));
                    result.at({i, j}) = at({orig_i, 0});
                    result.at({i, new_shape[1] - 1 - j}) = at({orig_i, shape[1] - 1});
                }
            }
        }
    }
    
    return result;
}

// Sorting
template<typename T>
Matrix<T> Matrix<T>::sort(size_t axis) const {
    if (axis == static_cast<size_t>(-1)) {
        // Sort all elements
        Matrix<T> result = *this;
        std::sort(result.data.begin(), result.data.end());
        return result;
    }
    
    if (shape.size() != 2) {
        throw std::runtime_error("Axis-based sort currently only supports 2D matrices");
    }
    
    Matrix<T> result = *this;
    
    if (axis == 0) {
        // Sort each column
        for (size_t j = 0; j < shape[1]; ++j) {
            std::vector<T> column_data;
            for (size_t i = 0; i < shape[0]; ++i) {
                column_data.push_back(at({i, j}));
            }
            std::sort(column_data.begin(), column_data.end());
            for (size_t i = 0; i < shape[0]; ++i) {
                result.at({i, j}) = column_data[i];
            }
        }
    } else if (axis == 1) {
        // Sort each row
        for (size_t i = 0; i < shape[0]; ++i) {
            std::vector<T> row_data;
            for (size_t j = 0; j < shape[1]; ++j) {
                row_data.push_back(at({i, j}));
            }
            std::sort(row_data.begin(), row_data.end());
            for (size_t j = 0; j < shape[1]; ++j) {
                result.at({i, j}) = row_data[j];
            }
        }
    } else {
        throw std::invalid_argument("Invalid axis for 2D matrix");
    }
    
    return result;
}

template<typename T>
Matrix<size_t> Matrix<T>::argSort(size_t axis) const {
    if (axis == static_cast<size_t>(-1)) {
        // Sort all elements and return indices
        std::vector<size_t> indices(data.size());
        std::iota(indices.begin(), indices.end(), 0);
        
        std::sort(indices.begin(), indices.end(), 
                 [this](size_t a, size_t b) { return data[a] < data[b]; });
        
        Matrix<size_t> result(shape);
        for (size_t i = 0; i < indices.size(); ++i) {
            // Convert linear index to multi-dimensional indices
            std::vector<size_t> multi_idx(shape.size());
            size_t temp = i;
            for (int dim = static_cast<int>(shape.size()) - 1; dim >= 0; --dim) {
                multi_idx[dim] = temp % shape[dim];
                temp /= shape[dim];
            }
            result.at(multi_idx) = indices[i];
        }
        return result;
    }
    
    if (shape.size() != 2) {
        throw std::runtime_error("Axis-based argSort currently only supports 2D matrices");
    }
    
    if (axis == 0) {
        // ArgSort along rows
        std::vector<size_t> result_shape = {shape[1]};
        Matrix<size_t> result(result_shape);
        
        for (size_t j = 0; j < shape[1]; ++j) {
            std::vector<std::pair<T, size_t>> column_with_indices;
            for (size_t i = 0; i < shape[0]; ++i) {
                column_with_indices.push_back({at({i, j}), i});
            }
            
            std::sort(column_with_indices.begin(), column_with_indices.end());
            
            // Store the index of the minimum element
            result.at({j}) = column_with_indices[0].second;
        }
        return result;
    } else if (axis == 1) {
        // ArgSort along columns
        std::vector<size_t> result_shape = {shape[0]};
        Matrix<size_t> result(result_shape);
        
        for (size_t i = 0; i < shape[0]; ++i) {
            std::vector<std::pair<T, size_t>> row_with_indices;
            for (size_t j = 0; j < shape[1]; ++j) {
                row_with_indices.push_back({at({i, j}), j});
            }
            
            std::sort(row_with_indices.begin(), row_with_indices.end());
            
            // Store the index of the minimum element
            result.at({i}) = row_with_indices[0].second;
        }
        return result;
    }
    
    throw std::invalid_argument("Invalid axis for 2D matrix");
}

template<typename T>
Matrix<size_t> Matrix<T>::argMax(size_t axis) const {
    if (axis == static_cast<size_t>(-1)) {
        // Find index of maximum element in flattened array
        auto max_it = std::max_element(data.begin(), data.end());
        size_t max_index = std::distance(data.begin(), max_it);
        
        Matrix<size_t> result({1});
        result.at({0}) = max_index;
        return result;
    }
    
    if (shape.size() != 2) {
        throw std::runtime_error("Axis-based argMax currently only supports 2D matrices");
    }
    
    if (axis == 0) {
        // ArgMax along rows
        std::vector<size_t> result_shape = {shape[1]};
        Matrix<size_t> result(result_shape);
        
        for (size_t j = 0; j < shape[1]; ++j) {
            size_t max_idx = 0;
            T max_val = at({0, j});
            
            for (size_t i = 1; i < shape[0]; ++i) {
                if (at({i, j}) > max_val) {
                    max_val = at({i, j});
                    max_idx = i;
                }
            }
            result.at({j}) = max_idx;
        }
        return result;
    } else if (axis == 1) {
        // ArgMax along columns
        std::vector<size_t> result_shape = {shape[0]};
        Matrix<size_t> result(result_shape);
        
        for (size_t i = 0; i < shape[0]; ++i) {
            size_t max_idx = 0;
            T max_val = at({i, 0});
            
            for (size_t j = 1; j < shape[1]; ++j) {
                if (at({i, j}) > max_val) {
                    max_val = at({i, j});
                    max_idx = j;
                }
            }
            result.at({i}) = max_idx;
        }
        return result;
    }
    
    throw std::invalid_argument("Invalid axis for 2D matrix");
}

template<typename T>
Matrix<size_t> Matrix<T>::argMin(size_t axis) const {
    if (axis == static_cast<size_t>(-1)) {
        // Find index of minimum element in flattened array
        auto min_it = std::min_element(data.begin(), data.end());
        size_t min_index = std::distance(data.begin(), min_it);
        
        Matrix<size_t> result({1});
        result.at({0}) = min_index;
        return result;
    }
    
    if (shape.size() != 2) {
        throw std::runtime_error("Axis-based argMin currently only supports 2D matrices");
    }
    
    if (axis == 0) {
        // ArgMin along rows
        std::vector<size_t> result_shape = {shape[1]};
        Matrix<size_t> result(result_shape);
        
        for (size_t j = 0; j < shape[1]; ++j) {
            size_t min_idx = 0;
            T min_val = at({0, j});
            
            for (size_t i = 1; i < shape[0]; ++i) {
                if (at({i, j}) < min_val) {
                    min_val = at({i, j});
                    min_idx = i;
                }
            }
            result.at({j}) = min_idx;
        }
        return result;
    } else if (axis == 1) {
        // ArgMin along columns
        std::vector<size_t> result_shape = {shape[0]};
        Matrix<size_t> result(result_shape);
        
        for (size_t i = 0; i < shape[0]; ++i) {
            size_t min_idx = 0;
            T min_val = at({i, 0});
            
            for (size_t j = 1; j < shape[1]; ++j) {
                if (at({i, j}) < min_val) {
                    min_val = at({i, j});
                    min_idx = j;
                }
            }
            result.at({i}) = min_idx;
        }
        return result;
    }
    
    throw std::invalid_argument("Invalid axis for 2D matrix");
}

// Advanced operations
template<typename T>
Matrix<T> Matrix<T>::conv(const Matrix<T>& kernel, const std::string& mode) const {
    if (shape.size() != 2 || kernel.shape.size() != 2) {
        throw std::invalid_argument("Convolution currently only supports 2D matrices");
    }
    
    size_t input_rows = shape[0];
    size_t input_cols = shape[1];
    size_t kernel_rows = kernel.shape[0];
    size_t kernel_cols = kernel.shape[1];
    
    size_t output_rows, output_cols;
    
    if (mode == "valid") {
        if (input_rows < kernel_rows || input_cols < kernel_cols) {
            throw std::invalid_argument("Input smaller than kernel for valid convolution");
        }
        output_rows = input_rows - kernel_rows + 1;
        output_cols = input_cols - kernel_cols + 1;
    } else if (mode == "same") {
        output_rows = input_rows;
        output_cols = input_cols;
    } else {
        throw std::invalid_argument("Unsupported convolution mode");
    }
    
    Matrix<T> result({output_rows, output_cols}, T(0));
    
    // Flip kernel for convolution (correlation vs convolution)
    Matrix<T> flipped_kernel = kernel.flip(true).flip(false);
    
    const size_t parallel_th = 100; // Lower threshold due to computational intensity
    
    if (output_rows >= parallel_th) {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t rows_per_thread = output_rows / num_threads;
        
        auto worker = [&](size_t start_row, size_t end_row) {
            for (size_t i = start_row; i < end_row; ++i) {
                for (size_t j = 0; j < output_cols; ++j) {
                    T sum = T(0);
                    
                    for (size_t ki = 0; ki < kernel_rows; ++ki) {
                        for (size_t kj = 0; kj < kernel_cols; ++kj) {
                            int input_i, input_j;
                            
                            if (mode == "valid") {
                                input_i = static_cast<int>(i + ki);
                                input_j = static_cast<int>(j + kj);
                            } else { // same
                                input_i = static_cast<int>(i + ki) - static_cast<int>(kernel_rows / 2);
                                input_j = static_cast<int>(j + kj) - static_cast<int>(kernel_cols / 2);
                            }
                            
                            if (input_i >= 0 && input_i < static_cast<int>(input_rows) &&
                                input_j >= 0 && input_j < static_cast<int>(input_cols)) {
                                sum += at({static_cast<size_t>(input_i), static_cast<size_t>(input_j)}) * 
                                       flipped_kernel.at({ki, kj});
                            }
                        }
                    }
                    
                    result.at({i, j}) = sum;
                }
            }
        };
        
        for (size_t t = 0; t < num_threads - 1; ++t) {
            threads.emplace_back(worker, t * rows_per_thread, (t + 1) * rows_per_thread);
        }
        worker((num_threads - 1) * rows_per_thread, output_rows);
        
        for (auto& thread : threads) thread.join();
    } else {
        for (size_t i = 0; i < output_rows; ++i) {
            for (size_t j = 0; j < output_cols; ++j) {
                T sum = T(0);
                
                for (size_t ki = 0; ki < kernel_rows; ++ki) {
                    for (size_t kj = 0; kj < kernel_cols; ++kj) {
                        int input_i, input_j;
                        
                        if (mode == "valid") {
                            input_i = static_cast<int>(i + ki);
                            input_j = static_cast<int>(j + kj);
                        } else { // same
                            input_i = static_cast<int>(i + ki) - static_cast<int>(kernel_rows / 2);
                            input_j = static_cast<int>(j + kj) - static_cast<int>(kernel_cols / 2);
                        }
                        
                        if (input_i >= 0 && input_i < static_cast<int>(input_rows) &&
                            input_j >= 0 && input_j < static_cast<int>(input_cols)) {
                            sum += at({static_cast<size_t>(input_i), static_cast<size_t>(input_j)}) * 
                                   flipped_kernel.at({ki, kj});
                        }
                    }
                }
                
                result.at({i, j}) = sum;
            }
        }
    }
    
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::trapz(const Matrix<T>& x, size_t axis) const {
    if (shape != x.shape) {
        throw std::invalid_argument("x and y arrays must have same shape");
    }
    
    if (shape.size() == 1) {
        if (data.size() < 2) {
            return Matrix<T>({1}, T(0));
        }
        
        T integral = T(0);
        for (size_t i = 0; i < data.size() - 1; ++i) {
            T dx = x.data[i + 1] - x.data[i];
            T avg_y = (data[i] + data[i + 1]) / T(2);
            integral += dx * avg_y;
        }
        
        Matrix<T> result({1});
        result.data[0] = integral;
        return result;
    }
    
    if (shape.size() == 2) {
        if (axis == 0) {
            // Integrate along rows
            std::vector<size_t> result_shape = {shape[1]};
            Matrix<T> result(result_shape, T(0));
            
            for (size_t j = 0; j < shape[1]; ++j) {
                T integral = T(0);
                for (size_t i = 0; i < shape[0] - 1; ++i) {
                    T dx = x.at({i + 1, j}) - x.at({i, j});
                    T avg_y = (at({i, j}) + at({i + 1, j})) / T(2);
                    integral += dx * avg_y;
                }
                result.data[j] = integral;
            }
            return result;
        } else if (axis == 1) {
            // Integrate along columns
            std::vector<size_t> result_shape = {shape[0]};
            Matrix<T> result(result_shape, T(0));
            
            for (size_t i = 0; i < shape[0]; ++i) {
                T integral = T(0);
                for (size_t j = 0; j < shape[1] - 1; ++j) {
                    T dx = x.at({i, j + 1}) - x.at({i, j});
                    T avg_y = (at({i, j}) + at({i, j + 1})) / T(2);
                    integral += dx * avg_y;
                }
                result.data[i] = integral;
            }
            return result;
        }
    }
    
    throw std::runtime_error("trapz not fully implemented for this configuration");
}

template<typename T>
Matrix<T> Matrix<T>::cumsum(size_t axis) const {
    if (shape.size() == 1 && axis == 0) {
        Matrix<T> result(shape);
        result.at({0}) = at({0});
        for (size_t i = 1; i < data.size(); ++i) {
            result.at({i}) = result.at({i-1}) + at({i});
        }
        return result;
    }
    
    if (shape.size() == 2) {
        if (axis == 0) {
            // Cumulative sum along rows
            Matrix<T> result(shape);
            
            // Copy first row
            for (size_t j = 0; j < shape[1]; ++j) {
                result.at({0, j}) = at({0, j});
            }
            
            // Calculate cumsum for remaining rows
            for (size_t i = 1; i < shape[0]; ++i) {
                for (size_t j = 0; j < shape[1]; ++j) {
                    result.at({i, j}) = result.at({i-1, j}) + at({i, j});
                }
            }
            return result;
        } else if (axis == 1) {
            // Cumulative sum along columns
            Matrix<T> result(shape);
            
            // Copy first column
            for (size_t i = 0; i < shape[0]; ++i) {
                result.at({i, 0}) = at({i, 0});
            }
            
            // Calculate cumsum for remaining columns
            for (size_t i = 0; i < shape[0]; ++i) {
                for (size_t j = 1; j < shape[1]; ++j) {
                    result.at({i, j}) = result.at({i, j-1}) + at({i, j});
                }
            }
            return result;
        }
    }
    
    throw std::runtime_error("cumsum not supported for this configuration");
}

template<typename T>
Matrix<T> Matrix<T>::cumprod(size_t axis) const {
    if (shape.size() == 1 && axis == 0) {
        Matrix<T> result(shape);
        result.at({0}) = at({0});
        for (size_t i = 1; i < data.size(); ++i) {
            result.at({i}) = result.at({i-1}) * at({i});
        }
        return result;
    }
    
    if (shape.size() == 2) {
        if (axis == 0) {
            // Cumulative product along rows
            Matrix<T> result(shape);
            
            // Copy first row
            for (size_t j = 0; j < shape[1]; ++j) {
                result.at({0, j}) = at({0, j});
            }
            
            // Calculate cumprod for remaining rows
            for (size_t i = 1; i < shape[0]; ++i) {
                for (size_t j = 0; j < shape[1]; ++j) {
                    result.at({i, j}) = result.at({i-1, j}) * at({i, j});
                }
            }
            return result;
        } else if (axis == 1) {
            // Cumulative product along columns
            Matrix<T> result(shape);
            
            // Copy first column
            for (size_t i = 0; i < shape[0]; ++i) {
                result.at({i, 0}) = at({i, 0});
            }
            
            // Calculate cumprod for remaining columns
            for (size_t i = 0; i < shape[0]; ++i) {
                for (size_t j = 1; j < shape[1]; ++j) {
                    result.at({i, j}) = result.at({i, j-1}) * at({i, j});
                }
            }
            return result;
        }
    }
    
    throw std::runtime_error("cumprod not supported for this configuration");
}

// Performance optimization
template<typename T>
void Matrix<T>::optimizeMemoryLayout() {
    // Could implement cache-friendly memory layout optimizations
    data.shrink_to_fit();
}

// Utilities
template<typename T>
bool Matrix<T>::isSquare() const {
    return shape.size() == 2 && shape[0] == shape[1];
}

template<typename T>
bool Matrix<T>::isSymmetric() const {
    if (!isSquare()) return false;
    
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            if (at({i, j}) != at({j, i})) {
                return false;
            }
        }
    }
    return true;
}

template<typename T>
bool Matrix<T>::isDiagonal() const {
    if (!isSquare()) return false;
    
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            if (i != j && at({i, j}) != T(0)) {
                return false;
            }
        }
    }
    return true;
}

template<typename T>
size_t Matrix<T>::rank() const {
    if (shape.size() != 2) {
        throw std::invalid_argument("Rank calculation requires 2D matrix");
    }
    
    // Use LU decomposition to find rank
    try {
        auto LU = LU_decomposition();
        Matrix<T> U = LU[1];
        
        size_t rank_count = 0;
        size_t min_dim = std::min(shape[0], shape[1]);
        
        for (size_t i = 0; i < min_dim; ++i) {
            if (std::abs(U.at({i, i})) > 1e-10) {
                rank_count++;
            }
        }
        
        return rank_count;
    } catch (const std::runtime_error&) {
        // If LU decomposition fails, use simpler method
        // Count non-zero rows after Gaussian elimination
        Matrix<T> temp = *this;
        size_t rank_count = 0;
        
        for (size_t i = 0; i < std::min(shape[0], shape[1]); ++i) {
            // Find pivot
            size_t pivot_row = i;
            for (size_t k = i + 1; k < shape[0]; ++k) {
                if (std::abs(temp.at({k, i})) > std::abs(temp.at({pivot_row, i}))) {
                    pivot_row = k;
                }
            }
            
            // Swap rows
            if (pivot_row != i) {
                for (size_t j = 0; j < shape[1]; ++j) {
                    std::swap(temp.data[i * shape[1] + j], 
                             temp.data[pivot_row * shape[1] + j]);
                }
            }
            
            // Check if pivot is non-zero
            if (std::abs(temp.at({i, i})) > 1e-10) {
                rank_count++;
                
                // Eliminate column
                for (size_t k = i + 1; k < shape[0]; ++k) {
                    T factor = temp.at({k, i}) / temp.at({i, i});
                    for (size_t j = i; j < shape[1]; ++j) {
                        temp.at({k, j}) -= factor * temp.at({i, j});
                    }
                }
            }
        }
        
        return rank_count;
    }
}

template<typename T>
T Matrix<T>::trace() const {
    if (!isSquare()) {
        throw std::invalid_argument("Trace requires square matrix");
    }
    
    T sum_val = T(0);
    for (size_t i = 0; i < shape[0]; ++i) {
        sum_val += at({i, i});
    }
    return sum_val;
}

// Type conversion
template<typename T>
template<typename U>
Matrix<U> Matrix<T>::asType() const {
    Matrix<U> result(shape);
    
    // Create indices vector for accessing elements
    std::vector<size_t> indices(shape.size(), 0);
    
    for (size_t i = 0; i < data.size(); ++i) {
        // Convert linear index back to multi-dimensional indices
        size_t temp = i;
        for (int dim = static_cast<int>(shape.size()) - 1; dim >= 0; --dim) {
            indices[dim] = temp % shape[dim];
            temp /= shape[dim];
        }
        
        result.at(indices) = static_cast<U>(data[i]);
    }
    return result;
}

// Utility methods
template<typename T>
void Matrix<T>::fill(T value) {
    std::fill(data.begin(), data.end(), value);
}

template<typename T>
void Matrix<T>::print(std::ostream& os) const {
    if (shape.empty()) {
        os << "[]";
        return;
    }
    
    if (shape.size() == 1) {
        os << "[";
        for (size_t i = 0; i < data.size(); ++i) {
            if (i > 0) os << ", ";
            os << data[i];
        }
        os << "]";
    } else if (shape.size() == 2) {
        os << "[";
        for (size_t i = 0; i < shape[0]; ++i) {
            if (i > 0) os << " ";
            os << "[";
            for (size_t j = 0; j < shape[1]; ++j) {
                if (j > 0) os << ", ";
                os << std::setw(8) << at({i, j});
            }
            os << "]";
            if (i < shape[0] - 1) os << "\n";
        }
        os << "]";
    } else {
        os << "Matrix with shape: [";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) os << ", ";
            os << shape[i];
        }
        os << "]";
    }
}

// Explicit template instantiations
template class Matrix<int>;
template class Matrix<float>;
template class Matrix<double>;
// Note: Matrix<bool> is problematic due to std::vector<bool> specialization
// Note: Matrix<size_t> causes ambiguity with std::abs and other mathematical functions

// Explicit instantiation for asType method
template Matrix<float> Matrix<int>::asType<float>() const;
template Matrix<double> Matrix<int>::asType<double>() const;
template Matrix<int> Matrix<float>::asType<int>() const;
template Matrix<double> Matrix<float>::asType<double>() const;
template Matrix<int> Matrix<double>::asType<int>() const;
template Matrix<float> Matrix<double>::asType<float>() const;
