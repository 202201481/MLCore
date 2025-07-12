#include "MLCore.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <set>

// ============================================================================
// STANDARD SCALER IMPLEMENTATION
// ============================================================================

template<typename T>
StandardScaler<T>::StandardScaler() : fitted_(false) {}

template<typename T>
void StandardScaler<T>::fit(const Matrix<T>& X) {
    auto shape = X.getShape();
    if (shape.size() != 2) {
        throw std::invalid_argument("Input must be 2D matrix");
    }
    
    size_t n_samples = shape[0];
    size_t n_features = shape[1];
    
    // Calculate mean
    mean_ = Matrix<T>(std::vector<size_t>{1, n_features});
    for (size_t j = 0; j < n_features; ++j) {
        T sum = 0;
        for (size_t i = 0; i < n_samples; ++i) {
            sum += X.at({i, j});
        }
        mean_.at({0, j}) = sum / n_samples;
    }
    
    // Calculate standard deviation
    std_ = Matrix<T>(std::vector<size_t>{1, n_features});
    for (size_t j = 0; j < n_features; ++j) {
        T sum_sq = 0;
        for (size_t i = 0; i < n_samples; ++i) {
            T diff = X.at({i, j}) - mean_.at({0, j});
            sum_sq += diff * diff;
        }
        std_.at({0, j}) = std::sqrt(sum_sq / n_samples);
        if (std_.at({0, j}) == 0) std_.at({0, j}) = 1; // Avoid division by zero
    }
    
    fitted_ = true;
}

template<typename T>
Matrix<T> StandardScaler<T>::transform(const Matrix<T>& X) const {
    if (!fitted_) {
        throw std::runtime_error("StandardScaler must be fitted before transform");
    }
    
    auto shape = X.getShape();
    size_t n_samples = shape[0];
    size_t n_features = shape[1];
    Matrix<T> result(std::vector<size_t>{n_samples, n_features});
    
    for (size_t i = 0; i < n_samples; ++i) {
        for (size_t j = 0; j < n_features; ++j) {
            result.at({i, j}) = (X.at({i, j}) - mean_.at({0, j})) / std_.at({0, j});
        }
    }
    
    return result;
}

template<typename T>
Matrix<T> StandardScaler<T>::fit_transform(const Matrix<T>& X) {
    fit(X);
    return transform(X);
}

template<typename T>
Matrix<T> StandardScaler<T>::inverse_transform(const Matrix<T>& X) const {
    if (!fitted_) {
        throw std::runtime_error("StandardScaler must be fitted before inverse_transform");
    }
    
    auto shape = X.getShape();
    size_t n_samples = shape[0];
    size_t n_features = shape[1];
    Matrix<T> result(std::vector<size_t>{n_samples, n_features});
    
    for (size_t i = 0; i < n_samples; ++i) {
        for (size_t j = 0; j < n_features; ++j) {
            result.at({i, j}) = X.at({i, j}) * std_.at({0, j}) + mean_.at({0, j});
        }
    }
    
    return result;
}

// ============================================================================
// MINMAX SCALER IMPLEMENTATION
// ============================================================================

template<typename T>
MinMaxScaler<T>::MinMaxScaler(T min_val, T max_val) 
    : feature_range_min_(min_val), feature_range_max_(max_val), fitted_(false) {}

template<typename T>
void MinMaxScaler<T>::fit(const Matrix<T>& X) {
    auto shape = X.getShape();
    size_t n_samples = shape[0];
    size_t n_features = shape[1];
    
    min_ = Matrix<T>(std::vector<size_t>{1, n_features});
    max_ = Matrix<T>(std::vector<size_t>{1, n_features});
    
    for (size_t j = 0; j < n_features; ++j) {
        T min_val = X.at({0, j});
        T max_val = X.at({0, j});
        
        for (size_t i = 1; i < n_samples; ++i) {
            if (X.at({i, j}) < min_val) min_val = X.at({i, j});
            if (X.at({i, j}) > max_val) max_val = X.at({i, j});
        }
        
        min_.at({0, j}) = min_val;
        max_.at({0, j}) = max_val;
    }
    
    fitted_ = true;
}

template<typename T>
Matrix<T> MinMaxScaler<T>::transform(const Matrix<T>& X) const {
    if (!fitted_) {
        throw std::runtime_error("MinMaxScaler must be fitted before transform");
    }
    
    auto shape = X.getShape();
    size_t n_samples = shape[0];
    size_t n_features = shape[1];
    Matrix<T> result(std::vector<size_t>{n_samples, n_features});
    
    for (size_t i = 0; i < n_samples; ++i) {
        for (size_t j = 0; j < n_features; ++j) {
            T range = max_.at({0, j}) - min_.at({0, j});
            if (range == 0) {
                result.at({i, j}) = feature_range_min_;
            } else {
                result.at({i, j}) = feature_range_min_ + 
                                  (feature_range_max_ - feature_range_min_) * 
                                  (X.at({i, j}) - min_.at({0, j})) / range;
            }
        }
    }
    
    return result;
}

template<typename T>
Matrix<T> MinMaxScaler<T>::fit_transform(const Matrix<T>& X) {
    fit(X);
    return transform(X);
}

template<typename T>
Matrix<T> MinMaxScaler<T>::inverse_transform(const Matrix<T>& X) const {
    if (!fitted_) {
        throw std::runtime_error("MinMaxScaler must be fitted before inverse_transform");
    }
    
    auto shape = X.getShape();
    size_t n_samples = shape[0];
    size_t n_features = shape[1];
    Matrix<T> result(std::vector<size_t>{n_samples, n_features});
    
    for (size_t i = 0; i < n_samples; ++i) {
        for (size_t j = 0; j < n_features; ++j) {
            T range = max_.at({0, j}) - min_.at({0, j});
            result.at({i, j}) = min_.at({0, j}) + 
                              (X.at({i, j}) - feature_range_min_) * range / 
                              (feature_range_max_ - feature_range_min_);
        }
    }
    
    return result;
}

// ============================================================================
// NORMALIZER IMPLEMENTATION
// ============================================================================

template<typename T>
Matrix<T> Normalizer<T>::normalize(const Matrix<T>& X, Norm norm) {
    auto shape = X.getShape();
    size_t n_samples = shape[0];
    size_t n_features = shape[1];
    Matrix<T> result(std::vector<size_t>{n_samples, n_features});
    
    for (size_t i = 0; i < n_samples; ++i) {
        T norm_val = 0;
        
        // Calculate norm
        switch (norm) {
            case L1:
                for (size_t j = 0; j < n_features; ++j) {
                    norm_val += std::abs(X.at({i, j}));
                }
                break;
            case L2:
                for (size_t j = 0; j < n_features; ++j) {
                    norm_val += X.at({i, j}) * X.at({i, j});
                }
                norm_val = std::sqrt(norm_val);
                break;
            case MAX:
                for (size_t j = 0; j < n_features; ++j) {
                    norm_val = std::max(norm_val, std::abs(X.at({i, j})));
                }
                break;
        }
        
        // Normalize
        if (norm_val == 0) norm_val = 1; // Avoid division by zero
        for (size_t j = 0; j < n_features; ++j) {
            result.at({i, j}) = X.at({i, j}) / norm_val;
        }
    }
    
    return result;
}

// ============================================================================
// ONE HOT ENCODER IMPLEMENTATION
// ============================================================================

template<typename T>
OneHotEncoder<T>::OneHotEncoder() : fitted_(false) {}

template<typename T>
void OneHotEncoder<T>::fit(const Matrix<T>& X) {
    auto shape = X.getShape();
    size_t n_samples = shape[0];
    size_t n_features = shape[1];
    
    categories_.clear();
    
    for (size_t j = 0; j < n_features; ++j) {
        std::set<T> unique_values;
        for (size_t i = 0; i < n_samples; ++i) {
            unique_values.insert(X.at({i, j}));
        }
        
        categories_[j] = std::vector<T>(unique_values.begin(), unique_values.end());
        std::sort(categories_[j].begin(), categories_[j].end());
    }
    
    fitted_ = true;
}

template<typename T>
Matrix<T> OneHotEncoder<T>::transform(const Matrix<T>& X) const {
    if (!fitted_) {
        throw std::runtime_error("OneHotEncoder must be fitted before transform");
    }
    
    auto shape = X.getShape();
    size_t n_samples = shape[0];
    size_t n_features = shape[1];
    
    // Calculate total output features
    size_t total_features = 0;
    for (const auto& pair : categories_) {
        total_features += pair.second.size();
    }
    
    Matrix<T> result(std::vector<size_t>{n_samples, total_features});
    result.fill(0);
    
    size_t col_offset = 0;
    for (size_t j = 0; j < n_features; ++j) {
        const auto& categories = categories_.at(j);
        
        for (size_t i = 0; i < n_samples; ++i) {
            auto it = std::find(categories.begin(), categories.end(), X.at({i, j}));
            if (it != categories.end()) {
                size_t category_idx = std::distance(categories.begin(), it);
                result.at({i, col_offset + category_idx}) = 1;
            }
        }
        
        col_offset += categories.size();
    }
    
    return result;
}

template<typename T>
Matrix<T> OneHotEncoder<T>::fit_transform(const Matrix<T>& X) {
    fit(X);
    return transform(X);
}

template<typename T>
Matrix<T> OneHotEncoder<T>::inverse_transform(const Matrix<T>& X) const {
    if (!fitted_) {
        throw std::runtime_error("OneHotEncoder must be fitted before inverse_transform");
    }
    
    auto shape = X.getShape();
    size_t n_samples = shape[0];
    size_t n_features = categories_.size();
    Matrix<T> result(std::vector<size_t>{n_samples, n_features});
    
    size_t col_offset = 0;
    for (size_t j = 0; j < n_features; ++j) {
        const auto& categories = categories_.at(j);
        
        for (size_t i = 0; i < n_samples; ++i) {
            bool found = false;
            for (size_t k = 0; k < categories.size(); ++k) {
                if (X.at({i, col_offset + k}) == 1) {
                    result.at({i, j}) = categories[k];
                    found = true;
                    break;
                }
            }
            if (!found) {
                result.at({i, j}) = categories[0]; // Default to first category
            }
        }
        
        col_offset += categories.size();
    }
    
    return result;
}

// ============================================================================
// LINEAR REGRESSION IMPLEMENTATION
// ============================================================================

template<typename T>
LinearRegression<T>::LinearRegression() : fitted_(false) {}

template<typename T>
void LinearRegression<T>::fit(const Matrix<T>& X, const Matrix<T>& y) {
    auto X_shape = X.getShape();
    size_t n_samples = X_shape[0];
    size_t n_features = X_shape[1];
    
    // Add bias column to X
    Matrix<T> X_with_bias(std::vector<size_t>{n_samples, n_features + 1});
    for (size_t i = 0; i < n_samples; ++i) {
        X_with_bias.at({i, 0}) = 1; // Bias term
        for (size_t j = 0; j < n_features; ++j) {
            X_with_bias.at({i, j + 1}) = X.at({i, j});
        }
    }
    
    // Normal equation: theta = (X^T X)^-1 X^T y
    Matrix<T> X_T = X_with_bias.transpose();
    Matrix<T> XTX = X_T.matmul(X_with_bias);
    Matrix<T> XTX_inv = XTX.inverse();
    Matrix<T> XTy = X_T.matmul(y);
    Matrix<T> theta = XTX_inv.matmul(XTy);
    
    intercept_ = theta.at({0, 0});
    coef_ = Matrix<T>(std::vector<size_t>{n_features, 1});
    for (size_t i = 0; i < n_features; ++i) {
        coef_.at({i, 0}) = theta.at({i + 1, 0});
    }
    
    fitted_ = true;
}

template<typename T>
Matrix<T> LinearRegression<T>::predict(const Matrix<T>& X) const {
    if (!fitted_) {
        throw std::runtime_error("LinearRegression must be fitted before predict");
    }
    
    auto shape = X.getShape();
    size_t n_samples = shape[0];
    size_t n_features = shape[1];
    Matrix<T> predictions(std::vector<size_t>{n_samples, 1});
    
    for (size_t i = 0; i < n_samples; ++i) {
        T pred = intercept_;
        for (size_t j = 0; j < n_features; ++j) {
            pred += X.at({i, j}) * coef_.at({j, 0});
        }
        predictions.at({i, 0}) = pred;
    }
    
    return predictions;
}

template<typename T>
T LinearRegression<T>::score(const Matrix<T>& X, const Matrix<T>& y) const {
    Matrix<T> y_pred = predict(X);
    return r2_score(y, y_pred);
}

// ============================================================================
// RIDGE REGRESSION IMPLEMENTATION
// ============================================================================

template<typename T>
Ridge<T>::Ridge(T alpha) : alpha_(alpha), fitted_(false) {}

template<typename T>
void Ridge<T>::fit(const Matrix<T>& X, const Matrix<T>& y) {
    auto X_shape = X.getShape();
    size_t n_samples = X_shape[0];
    size_t n_features = X_shape[1];
    
    // Add bias column to X
    Matrix<T> X_with_bias(std::vector<size_t>{n_samples, n_features + 1});
    for (size_t i = 0; i < n_samples; ++i) {
        X_with_bias.at({i, 0}) = 1; // Bias term
        for (size_t j = 0; j < n_features; ++j) {
            X_with_bias.at({i, j + 1}) = X.at({i, j});
        }
    }
    
    // Ridge equation: theta = (X^T X + alpha * I)^-1 X^T y
    Matrix<T> X_T = X_with_bias.transpose();
    Matrix<T> XTX = X_T.matmul(X_with_bias);
    
    // Add regularization term (don't regularize bias term)
    Matrix<T> I = Matrix<T>::eye(n_features + 1);
    I.at({0, 0}) = 0; // Don't regularize bias term
    XTX = XTX + I * alpha_;
    
    Matrix<T> XTX_inv = XTX.inverse();
    Matrix<T> XTy = X_T.matmul(y);
    Matrix<T> theta = XTX_inv.matmul(XTy);
    
    intercept_ = theta.at({0, 0});
    coef_ = Matrix<T>(std::vector<size_t>{n_features, 1});
    for (size_t i = 0; i < n_features; ++i) {
        coef_.at({i, 0}) = theta.at({i + 1, 0});
    }
    
    fitted_ = true;
}

template<typename T>
Matrix<T> Ridge<T>::predict(const Matrix<T>& X) const {
    if (!fitted_) {
        throw std::runtime_error("Ridge must be fitted before predict");
    }
    
    auto shape = X.getShape();
    size_t n_samples = shape[0];
    size_t n_features = shape[1];
    Matrix<T> predictions(std::vector<size_t>{n_samples, 1});
    
    for (size_t i = 0; i < n_samples; ++i) {
        T pred = intercept_;
        for (size_t j = 0; j < n_features; ++j) {
            pred += X.at({i, j}) * coef_.at({j, 0});
        }
        predictions.at({i, 0}) = pred;
    }
    
    return predictions;
}

template<typename T>
T Ridge<T>::score(const Matrix<T>& X, const Matrix<T>& y) const {
    Matrix<T> y_pred = predict(X);
    return r2_score(y, y_pred);
}

// ============================================================================
// LASSO REGRESSION IMPLEMENTATION
// ============================================================================

template<typename T>
Lasso<T>::Lasso(T alpha, size_t max_iter, T tol) 
    : alpha_(alpha), max_iter_(max_iter), tol_(tol), fitted_(false) {}

template<typename T>
void Lasso<T>::fit(const Matrix<T>& X, const Matrix<T>& y) {
    auto X_shape = X.getShape();
    size_t n_samples = X_shape[0];
    size_t n_features = X_shape[1];
    
    // Coordinate descent algorithm for Lasso
    coef_ = Matrix<T>(std::vector<size_t>{n_features, 1});
    coef_.fill(0);
    intercept_ = 0;
    
    // Center y
    T y_mean = 0;
    for (size_t i = 0; i < n_samples; ++i) {
        y_mean += y.at({i, 0});
    }
    y_mean /= n_samples;
    intercept_ = y_mean;
    
    // Coordinate descent iterations
    for (size_t iter = 0; iter < max_iter_; ++iter) {
        Matrix<T> coef_old = coef_;
        
        for (size_t j = 0; j < n_features; ++j) {
            // Calculate residual
            T residual_sum = 0;
            for (size_t i = 0; i < n_samples; ++i) {
                T pred = intercept_;
                for (size_t k = 0; k < n_features; ++k) {
                    if (k != j) pred += X.at({i, k}) * coef_.at({k, 0});
                }
                residual_sum += X.at({i, j}) * (y.at({i, 0}) - pred);
            }
            
            // Calculate sum of squares
            T sum_sq = 0;
            for (size_t i = 0; i < n_samples; ++i) {
                sum_sq += X.at({i, j}) * X.at({i, j});
            }
            
            // Soft thresholding
            if (sum_sq == 0) {
                coef_.at({j, 0}) = 0;
            } else {
                T z = residual_sum / n_samples;
                if (z > alpha_) {
                    coef_.at({j, 0}) = (z - alpha_) / (sum_sq / n_samples);
                } else if (z < -alpha_) {
                    coef_.at({j, 0}) = (z + alpha_) / (sum_sq / n_samples);
                } else {
                    coef_.at({j, 0}) = 0;
                }
            }
        }
        
        // Check convergence
        T diff = 0;
        for (size_t j = 0; j < n_features; ++j) {
            diff += std::abs(coef_.at({j, 0}) - coef_old.at({j, 0}));
        }
        if (diff < tol_) break;
    }
    
    fitted_ = true;
}

template<typename T>
Matrix<T> Lasso<T>::predict(const Matrix<T>& X) const {
    if (!fitted_) {
        throw std::runtime_error("Lasso must be fitted before predict");
    }
    
    auto shape = X.getShape();
    size_t n_samples = shape[0];
    size_t n_features = shape[1];
    Matrix<T> predictions(std::vector<size_t>{n_samples, 1});
    
    for (size_t i = 0; i < n_samples; ++i) {
        T pred = intercept_;
        for (size_t j = 0; j < n_features; ++j) {
            pred += X.at({i, j}) * coef_.at({j, 0});
        }
        predictions.at({i, 0}) = pred;
    }
    
    return predictions;
}

template<typename T>
T Lasso<T>::score(const Matrix<T>& X, const Matrix<T>& y) const {
    Matrix<T> y_pred = predict(X);
    return r2_score(y, y_pred);
}

// ============================================================================
// LOGISTIC REGRESSION IMPLEMENTATION
// ============================================================================

template<typename T>
LogisticRegression<T>::LogisticRegression(T learning_rate, size_t max_iter, T tol)
    : learning_rate_(learning_rate), max_iter_(max_iter), tol_(tol), fitted_(false) {}

template<typename T>
T LogisticRegression<T>::sigmoid(T z) const {
    return 1.0 / (1.0 + std::exp(-z));
}

template<typename T>
void LogisticRegression<T>::fit(const Matrix<T>& X, const Matrix<T>& y) {
    auto X_shape = X.getShape();
    size_t n_samples = X_shape[0];
    size_t n_features = X_shape[1];
    
    // Initialize coefficients
    coef_ = Matrix<T>(std::vector<size_t>{n_features, 1});
    coef_.fill(0);
    intercept_ = 0;
    
    // Gradient descent
    for (size_t iter = 0; iter < max_iter_; ++iter) {
        // Forward pass
        Matrix<T> predictions(std::vector<size_t>{n_samples, 1});
        for (size_t i = 0; i < n_samples; ++i) {
            T z = intercept_;
            for (size_t j = 0; j < n_features; ++j) {
                z += X.at({i, j}) * coef_.at({j, 0});
            }
            predictions.at({i, 0}) = sigmoid(z);
        }
        
        // Calculate gradients
        T grad_intercept = 0;
        Matrix<T> grad_coef(std::vector<size_t>{n_features, 1});
        grad_coef.fill(0);
        
        for (size_t i = 0; i < n_samples; ++i) {
            T error = predictions.at({i, 0}) - y.at({i, 0});
            grad_intercept += error;
            for (size_t j = 0; j < n_features; ++j) {
                grad_coef.at({j, 0}) += error * X.at({i, j});
            }
        }
        
        grad_intercept /= n_samples;
        for (size_t j = 0; j < n_features; ++j) {
            grad_coef.at({j, 0}) /= n_samples;
        }
        
        // Update parameters
        T old_intercept = intercept_;
        Matrix<T> old_coef = coef_;
        
        intercept_ -= learning_rate_ * grad_intercept;
        for (size_t j = 0; j < n_features; ++j) {
            coef_.at({j, 0}) -= learning_rate_ * grad_coef.at({j, 0});
        }
        
        // Check convergence
        T param_change = std::abs(intercept_ - old_intercept);
        for (size_t j = 0; j < n_features; ++j) {
            param_change += std::abs(coef_.at({j, 0}) - old_coef.at({j, 0}));
        }
        
        if (param_change < tol_) break;
    }
    
    fitted_ = true;
}

template<typename T>
Matrix<T> LogisticRegression<T>::predict_proba(const Matrix<T>& X) const {
    if (!fitted_) {
        throw std::runtime_error("LogisticRegression must be fitted before predict_proba");
    }
    
    auto shape = X.getShape();
    size_t n_samples = shape[0];
    size_t n_features = shape[1];
    Matrix<T> probabilities(std::vector<size_t>{n_samples, 1});
    
    for (size_t i = 0; i < n_samples; ++i) {
        T z = intercept_;
        for (size_t j = 0; j < n_features; ++j) {
            z += X.at({i, j}) * coef_.at({j, 0});
        }
        probabilities.at({i, 0}) = sigmoid(z);
    }
    
    return probabilities;
}

template<typename T>
Matrix<T> LogisticRegression<T>::predict(const Matrix<T>& X) const {
    Matrix<T> proba = predict_proba(X);
    auto shape = proba.getShape();
    Matrix<T> predictions(std::vector<size_t>{shape[0], 1});
    
    for (size_t i = 0; i < shape[0]; ++i) {
        predictions.at({i, 0}) = proba.at({i, 0}) >= 0.5 ? 1 : 0;
    }
    
    return predictions;
}

template<typename T>
T LogisticRegression<T>::score(const Matrix<T>& X, const Matrix<T>& y) const {
    Matrix<T> y_pred = predict(X);
    return accuracy_score(y, y_pred);
}

// ============================================================================
// KMEANS IMPLEMENTATION
// ============================================================================

template<typename T>
KMeans<T>::KMeans(size_t n_clusters, size_t max_iter, T tol, int random_state)
    : n_clusters_(n_clusters), max_iter_(max_iter), tol_(tol), random_state_(random_state), fitted_(false) {}

template<typename T>
void KMeans<T>::fit(const Matrix<T>& X) {
    auto X_shape = X.getShape();
    size_t n_samples = X_shape[0];
    size_t n_features = X_shape[1];
    
    // Initialize centroids randomly
    std::mt19937 gen(random_state_);
    std::uniform_int_distribution<size_t> dis(0, n_samples - 1);
    
    cluster_centers_ = Matrix<T>(std::vector<size_t>{n_clusters_, n_features});
    for (size_t k = 0; k < n_clusters_; ++k) {
        size_t random_idx = dis(gen);
        for (size_t j = 0; j < n_features; ++j) {
            cluster_centers_.at({k, j}) = X.at({random_idx, j});
        }
    }
    
    labels_ = std::vector<size_t>(n_samples);
    
    // K-means iterations
    for (size_t iter = 0; iter < max_iter_; ++iter) {
        Matrix<T> old_centers = cluster_centers_;
        
        // Assign points to closest centroids
        for (size_t i = 0; i < n_samples; ++i) {
            T min_dist = std::numeric_limits<T>::max();
            size_t best_cluster = 0;
            
            for (size_t k = 0; k < n_clusters_; ++k) {
                T dist = 0;
                for (size_t j = 0; j < n_features; ++j) {
                    T diff = X.at({i, j}) - cluster_centers_.at({k, j});
                    dist += diff * diff;
                }
                
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = k;
                }
            }
            
            labels_[i] = best_cluster;
        }
        
        // Update centroids
        for (size_t k = 0; k < n_clusters_; ++k) {
            Matrix<T> centroid_sum(std::vector<size_t>{1, n_features});
            centroid_sum.fill(0);
            size_t count = 0;
            
            for (size_t i = 0; i < n_samples; ++i) {
                if (labels_[i] == k) {
                    for (size_t j = 0; j < n_features; ++j) {
                        centroid_sum.at({0, j}) += X.at({i, j});
                    }
                    count++;
                }
            }
            
            if (count > 0) {
                for (size_t j = 0; j < n_features; ++j) {
                    cluster_centers_.at({k, j}) = centroid_sum.at({0, j}) / count;
                }
            }
        }
        
        // Check convergence
        T center_shift = 0;
        for (size_t k = 0; k < n_clusters_; ++k) {
            for (size_t j = 0; j < n_features; ++j) {
                T diff = cluster_centers_.at({k, j}) - old_centers.at({k, j});
                center_shift += diff * diff;
            }
        }
        
        if (std::sqrt(center_shift) < tol_) break;
    }
    
    fitted_ = true;
}

template<typename T>
std::vector<size_t> KMeans<T>::predict(const Matrix<T>& X) const {
    if (!fitted_) {
        throw std::runtime_error("KMeans must be fitted before predict");
    }
    
    auto X_shape = X.getShape();
    size_t n_samples = X_shape[0];
    size_t n_features = X_shape[1];
    std::vector<size_t> predictions(n_samples);
    
    for (size_t i = 0; i < n_samples; ++i) {
        T min_dist = std::numeric_limits<T>::max();
        size_t best_cluster = 0;
        
        for (size_t k = 0; k < n_clusters_; ++k) {
            T dist = 0;
            for (size_t j = 0; j < n_features; ++j) {
                T diff = X.at({i, j}) - cluster_centers_.at({k, j});
                dist += diff * diff;
            }
            
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = k;
            }
        }
        
        predictions[i] = best_cluster;
    }
    
    return predictions;
}

template<typename T>
std::vector<size_t> KMeans<T>::fit_predict(const Matrix<T>& X) {
    fit(X);
    return labels_;
}

// ============================================================================
// METRICS IMPLEMENTATIONS
// ============================================================================

template<typename T>
T mean_squared_error(const Matrix<T>& y_true, const Matrix<T>& y_pred) {
    auto shape = y_true.getShape();
    size_t n_samples = shape[0];
    
    T mse = 0;
    for (size_t i = 0; i < n_samples; ++i) {
        T diff = y_true.at({i, 0}) - y_pred.at({i, 0});
        mse += diff * diff;
    }
    
    return mse / n_samples;
}

template<typename T>
T r2_score(const Matrix<T>& y_true, const Matrix<T>& y_pred) {
    auto shape = y_true.getShape();
    size_t n_samples = shape[0];
    
    // Calculate mean of y_true
    T y_mean = 0;
    for (size_t i = 0; i < n_samples; ++i) {
        y_mean += y_true.at({i, 0});
    }
    y_mean /= n_samples;
    
    // Calculate SS_res and SS_tot
    T ss_res = 0;
    T ss_tot = 0;
    for (size_t i = 0; i < n_samples; ++i) {
        T res = y_true.at({i, 0}) - y_pred.at({i, 0});
        T tot = y_true.at({i, 0}) - y_mean;
        ss_res += res * res;
        ss_tot += tot * tot;
    }
    
    if (ss_tot == 0) return 1.0; // Perfect prediction when y is constant
    return 1.0 - (ss_res / ss_tot);
}

template<typename T>
T accuracy_score(const Matrix<T>& y_true, const Matrix<T>& y_pred) {
    auto shape = y_true.getShape();
    size_t n_samples = shape[0];
    
    size_t correct = 0;
    for (size_t i = 0; i < n_samples; ++i) {
        if (y_true.at({i, 0}) == y_pred.at({i, 0})) {
            correct++;
        }
    }
    
    return static_cast<T>(correct) / n_samples;
}

// ============================================================================
// EXPLICIT TEMPLATE INSTANTIATIONS
// ============================================================================

// Preprocessing classes
template class StandardScaler<float>;
template class StandardScaler<double>;
template class MinMaxScaler<float>;
template class MinMaxScaler<double>;
template class OneHotEncoder<float>;
template class OneHotEncoder<double>;
template class OneHotEncoder<int>;

// Linear models
template class LinearRegression<float>;
template class LinearRegression<double>;
template class Ridge<float>;
template class Ridge<double>;
template class Lasso<float>;
template class Lasso<double>;
template class LogisticRegression<float>;
template class LogisticRegression<double>;

// Clustering
template class KMeans<float>;
template class KMeans<double>;

// Metrics
template float mean_squared_error(const Matrix<float>&, const Matrix<float>&);
template double mean_squared_error(const Matrix<double>&, const Matrix<double>&);
template float r2_score(const Matrix<float>&, const Matrix<float>&);
template double r2_score(const Matrix<double>&, const Matrix<double>&);
template float accuracy_score(const Matrix<float>&, const Matrix<float>&);
template double accuracy_score(const Matrix<double>&, const Matrix<double>&);

// Normalization functions
template Matrix<float> Normalizer<float>::normalize(const Matrix<float>&, Norm);
template Matrix<double> Normalizer<double>::normalize(const Matrix<double>&, Norm);
