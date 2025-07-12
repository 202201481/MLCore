#ifndef MLCORE_HPP
#define MLCORE_HPP

#include "matrix.hpp"
#include "DataAnalytics.hpp"
#include <vector>
#include <map>
#include <string>
#include <functional>
#include <random>
#include <memory>

// ============================================================================
// PREPROCESSING CLASSES
// ============================================================================

template<typename T>
class StandardScaler {
private:
    Matrix<T> mean_;
    Matrix<T> std_;
    bool fitted_;
    
public:
    StandardScaler();
    void fit(const Matrix<T>& X);
    Matrix<T> transform(const Matrix<T>& X) const;
    Matrix<T> fit_transform(const Matrix<T>& X);
    Matrix<T> inverse_transform(const Matrix<T>& X) const;
};

template<typename T>
class MinMaxScaler {
private:
    Matrix<T> min_;
    Matrix<T> max_;
    T feature_range_min_;
    T feature_range_max_;
    bool fitted_;
    
public:
    MinMaxScaler(T min_val = 0, T max_val = 1);
    void fit(const Matrix<T>& X);
    Matrix<T> transform(const Matrix<T>& X) const;
    Matrix<T> fit_transform(const Matrix<T>& X);
    Matrix<T> inverse_transform(const Matrix<T>& X) const;
};

template<typename T>
class Normalizer {
public:
    enum Norm { L1, L2, MAX };
    
    static Matrix<T> normalize(const Matrix<T>& X, Norm norm = L2);
};

template<typename T>
class OneHotEncoder {
private:
    std::map<size_t, std::vector<T>> categories_;
    bool fitted_;
    
public:
    OneHotEncoder();
    void fit(const Matrix<T>& X);
    Matrix<T> transform(const Matrix<T>& X) const;
    Matrix<T> fit_transform(const Matrix<T>& X);
    Matrix<T> inverse_transform(const Matrix<T>& X) const;
};

// ============================================================================
// LINEAR REGRESSION CLASSES
// ============================================================================

template<typename T>
class LinearRegression {
private:
    Matrix<T> coef_;
    T intercept_;
    bool fitted_;
    
public:
    LinearRegression();
    void fit(const Matrix<T>& X, const Matrix<T>& y);
    Matrix<T> predict(const Matrix<T>& X) const;
    T score(const Matrix<T>& X, const Matrix<T>& y) const;
    Matrix<T> get_coef() const { return coef_; }
    T get_intercept() const { return intercept_; }
};

template<typename T>
class Ridge {
private:
    Matrix<T> coef_;
    T intercept_;
    T alpha_;
    bool fitted_;
    
public:
    Ridge(T alpha = 1.0);
    void fit(const Matrix<T>& X, const Matrix<T>& y);
    Matrix<T> predict(const Matrix<T>& X) const;
    T score(const Matrix<T>& X, const Matrix<T>& y) const;
    Matrix<T> get_coef() const { return coef_; }
    T get_intercept() const { return intercept_; }
};

template<typename T>
class Lasso {
private:
    Matrix<T> coef_;
    T intercept_;
    T alpha_;
    size_t max_iter_;
    T tol_;
    bool fitted_;
    
public:
    Lasso(T alpha = 1.0, size_t max_iter = 1000, T tol = 1e-4);
    void fit(const Matrix<T>& X, const Matrix<T>& y);
    Matrix<T> predict(const Matrix<T>& X) const;
    T score(const Matrix<T>& X, const Matrix<T>& y) const;
    Matrix<T> get_coef() const { return coef_; }
    T get_intercept() const { return intercept_; }
};

template<typename T>
class LogisticRegression {
private:
    Matrix<T> coef_;
    T intercept_;
    size_t max_iter_;
    T tol_;
    T learning_rate_;
    bool fitted_;
    
    T sigmoid(T z) const;
    Matrix<T> sigmoid(const Matrix<T>& z) const;
    
public:
    LogisticRegression(T learning_rate = 0.01, size_t max_iter = 1000, T tol = 1e-6);
    void fit(const Matrix<T>& X, const Matrix<T>& y);
    Matrix<T> predict(const Matrix<T>& X) const;
    Matrix<T> predict_proba(const Matrix<T>& X) const;
    T score(const Matrix<T>& X, const Matrix<T>& y) const;
    Matrix<T> get_coef() const { return coef_; }
    T get_intercept() const { return intercept_; }
};

// ============================================================================
// CLUSTERING CLASSES
// ============================================================================

template<typename T>
class KMeans {
private:
    size_t n_clusters_;
    size_t max_iter_;
    T tol_;
    int random_state_;
    Matrix<T> cluster_centers_;
    std::vector<size_t> labels_;
    T inertia_;
    bool fitted_;
    
    void init_centroids(const Matrix<T>& X);
    std::vector<size_t> assign_clusters(const Matrix<T>& X) const;
    void update_centroids(const Matrix<T>& X, const std::vector<size_t>& labels);
    T compute_inertia(const Matrix<T>& X, const std::vector<size_t>& labels) const;
    
public:
    KMeans(size_t n_clusters = 8, size_t max_iter = 300, T tol = 1e-4, int random_state = -1);
    void fit(const Matrix<T>& X);
    std::vector<size_t> predict(const Matrix<T>& X) const;
    std::vector<size_t> fit_predict(const Matrix<T>& X);
    Matrix<T> get_cluster_centers() const { return cluster_centers_; }
    std::vector<size_t> get_labels() const { return labels_; }
    T get_inertia() const { return inertia_; }
};

// ============================================================================
// OPTIMIZATION CLASSES
// ============================================================================

template<typename T>
class GradientDescent {
private:
    T learning_rate_;
    size_t max_iter_;
    T tol_;
    
public:
    GradientDescent(T learning_rate = 0.01, size_t max_iter = 1000, T tol = 1e-6);
    
    Matrix<T> minimize(std::function<T(const Matrix<T>&)> cost_function,
                      std::function<Matrix<T>(const Matrix<T>&)> gradient_function,
                      const Matrix<T>& initial_params);
};

template<typename T>
class StochasticGradientDescent {
private:
    T learning_rate_;
    size_t max_iter_;
    size_t batch_size_;
    T tol_;
    
public:
    StochasticGradientDescent(T learning_rate = 0.01, size_t max_iter = 1000, 
                             size_t batch_size = 32, T tol = 1e-6);
    
    Matrix<T> minimize(const Matrix<T>& X, const Matrix<T>& y,
                      std::function<T(const Matrix<T>&, const Matrix<T>&, const Matrix<T>&)> cost_function,
                      std::function<Matrix<T>(const Matrix<T>&, const Matrix<T>&, const Matrix<T>&)> gradient_function,
                      const Matrix<T>& initial_params);
};

template<typename T>
class AdamOptimizer {
private:
    T learning_rate_;
    T beta1_;
    T beta2_;
    T epsilon_;
    size_t max_iter_;
    T tol_;
    
public:
    AdamOptimizer(T learning_rate = 0.001, T beta1 = 0.9, T beta2 = 0.999, 
                  T epsilon = 1e-8, size_t max_iter = 1000, T tol = 1e-6);
    
    Matrix<T> minimize(std::function<T(const Matrix<T>&)> cost_function,
                      std::function<Matrix<T>(const Matrix<T>&)> gradient_function,
                      const Matrix<T>& initial_params);
};

// ============================================================================
// DIMENSIONALITY REDUCTION CLASSES
// ============================================================================

template<typename T>
class PCA {
private:
    size_t n_components_;
    Matrix<T> components_;
    Matrix<T> mean_;
    std::vector<T> explained_variance_;
    std::vector<T> explained_variance_ratio_;
    bool fitted_;
    
public:
    PCA(size_t n_components = 2);
    void fit(const Matrix<T>& X);
    Matrix<T> transform(const Matrix<T>& X) const;
    Matrix<T> fit_transform(const Matrix<T>& X);
    Matrix<T> inverse_transform(const Matrix<T>& X) const;
    
    Matrix<T> get_components() const { return components_; }
    std::vector<T> get_explained_variance() const { return explained_variance_; }
    std::vector<T> get_explained_variance_ratio() const { return explained_variance_ratio_; }
};

// ============================================================================
// K-NEAREST NEIGHBORS CLASSES
// ============================================================================

template<typename T>
class KNeighborsClassifier {
private:
    size_t n_neighbors_;
    Matrix<T> X_train_;
    Matrix<T> y_train_;
    bool fitted_;
    
    T distance(const Matrix<T>& x1, const Matrix<T>& x2) const;
    std::vector<size_t> find_k_nearest(const Matrix<T>& x) const;
    
public:
    KNeighborsClassifier(size_t n_neighbors = 5);
    void fit(const Matrix<T>& X, const Matrix<T>& y);
    Matrix<T> predict(const Matrix<T>& X) const;
    T score(const Matrix<T>& X, const Matrix<T>& y) const;
};

template<typename T>
class KNeighborsRegressor {
private:
    size_t n_neighbors_;
    Matrix<T> X_train_;
    Matrix<T> y_train_;
    bool fitted_;
    
    T distance(const Matrix<T>& x1, const Matrix<T>& x2) const;
    std::vector<size_t> find_k_nearest(const Matrix<T>& x) const;
    
public:
    KNeighborsRegressor(size_t n_neighbors = 5);
    void fit(const Matrix<T>& X, const Matrix<T>& y);
    Matrix<T> predict(const Matrix<T>& X) const;
    T score(const Matrix<T>& X, const Matrix<T>& y) const;
};

// ============================================================================
// UTILITY FUNCTIONS AND STRUCTURES
// ============================================================================

template<typename T>
struct TrainTestSplit {
    Matrix<T> X_train, X_test;
    Matrix<T> y_train, y_test;
};

template<typename T>
struct CrossValidationResult {
    std::vector<T> test_scores;
    std::vector<T> train_scores;
    T mean_test_score;
    T std_test_score;
    T mean_train_score;
    T std_train_score;
};

// ============================================================================
// PREPROCESSING FUNCTIONS
// ============================================================================

template<typename T>
TrainTestSplit<T> train_test_split(const Matrix<T>& X, const Matrix<T>& y, 
                                  T test_size = 0.2, int random_state = -1);

template<typename T>
Matrix<T> polynomial_features(const Matrix<T>& X, size_t degree = 2, bool include_bias = true);

template<typename T>
std::vector<size_t> select_k_best(const Matrix<T>& X, const Matrix<T>& y, size_t k);

// ============================================================================
// METRICS FUNCTIONS
// ============================================================================

// Regression Metrics
template<typename T>
T mean_squared_error(const Matrix<T>& y_true, const Matrix<T>& y_pred);

template<typename T>
T root_mean_squared_error(const Matrix<T>& y_true, const Matrix<T>& y_pred);

template<typename T>
T mean_absolute_error(const Matrix<T>& y_true, const Matrix<T>& y_pred);

template<typename T>
T r2_score(const Matrix<T>& y_true, const Matrix<T>& y_pred);

// Classification Metrics
template<typename T>
T accuracy_score(const std::vector<T>& y_true, const std::vector<T>& y_pred);

template<typename T>
T precision_score(const std::vector<T>& y_true, const std::vector<T>& y_pred);

template<typename T>
T recall_score(const std::vector<T>& y_true, const std::vector<T>& y_pred);

template<typename T>
T f1_score(const std::vector<T>& y_true, const std::vector<T>& y_pred);

template<typename T>
Matrix<T> confusion_matrix(const std::vector<T>& y_true, const std::vector<T>& y_pred);

// Clustering Metrics
template<typename T>
T silhouette_score(const Matrix<T>& X, const std::vector<size_t>& labels);

template<typename T>
T adjusted_rand_score(const std::vector<size_t>& labels_true, const std::vector<size_t>& labels_pred);

// Model Selection
template<typename T>
CrossValidationResult<T> cross_val_score(
    std::function<void(const Matrix<T>&, const Matrix<T>&)> fit_func,
    std::function<T(const Matrix<T>&, const Matrix<T>&)> score_func,
    const Matrix<T>& X, const Matrix<T>& y, size_t cv = 5);

#endif // MLCORE_HPP
