#ifndef MATRIX_HPP
#define MATRIX_HPP

#include<vector>
#include<iostream>
#include<stdexcept>
#include<memory>
#include<random>
#include<thread>
#include<numeric>
#include<functional>
#include<initializer_list>

template<typename T>
class Matrix{
private:
    std::vector<T> data;
    std::vector<size_t> shape;  // the shape/dimension of the array
    std::vector<size_t> strides; // no of elelment to step on 

    size_t calcOffset(const std::vector<size_t>& indices) const; // the const at the end indicates that this function won;t modify any vaiblees
    void calcStrides();
public:
    //Constructors
    Matrix();
    explicit Matrix(const std::vector<size_t>& shape); //simple empty vector
    Matrix(const std::vector<size_t>& shape,T value); //filled vector with value
    Matrix(std::initializer_list<T> list); // to inilize like this Matrix<int> m = {1,2,3,4};
    Matrix(const Matrix<T>& A); // copy constructor
    Matrix(Matrix<T>&& A) noexcept; //move constrtutor, noexcep means this will throw no error so compiler optiizes it to use over copy one

    //assignment operators
    Matrix<T>& operator=(const Matrix<T>& A);
    Matrix<T>& operator=(Matrix<T>&& A) noexcept;
    static Matrix<T> fromFile(const std::string& filename);


    // element access methods
    T& at(const std::vector<size_t>& indices);
    const T& at(const std::vector<size_t>& indices) const;
    T& operator [](const std::vector<size_t>& indices);
    const T& operator[](const std::vector<size_t>& indices) const;

    // basic getters
    size_t ndim() const {return shape.size();}
    size_t size() const {return data.size();}
    const std::vector<size_t>& getShape() const {return shape;}
    const std::vector<size_t>& getStrides() const {return strides;}
    bool any(size_t axis = -1) const;
    bool all(size_t axis = -1) const;


    // Matrix operations
    Matrix<T> reshape(const std::vector<size_t>& newShape) const;
    Matrix<T> transpose(const std::vector<size_t>& axes = {}) const;
    Matrix<T> slice(const std::vector<std::pair<size_t,size_t>>& ranges) const;
    Matrix<T> rotate90(int k = 1) const; // k*90 degrees
    Matrix<T> flip(bool horizontal = true) const;


    //element wise operations
    Matrix<T> operator+(const Matrix& A) const;
    Matrix<T> operator-(const Matrix& A) const;
    Matrix<T> operator*(const Matrix& A) const;
    Matrix<T> operator/(const Matrix& A) const;
    Matrix<T> operator+(T scaler) const;
    Matrix<T> operator-(T scaler) const;
    Matrix<T> operator*(T scaler) const;
    Matrix<T> operator/(T scaler) const;
    friend Matrix<T> operator+(T scaler,const Matrix<T>& matrix){ return matrix+scaler;}
    friend Matrix<T> operator-(T scaler,const Matrix<T>& matrix){ return (-matrix)+scaler;}
    friend Matrix<T> operator*(T scaler,const Matrix<T>& matrix){ return matrix * scaler;}
    Matrix<T> operator+() const;
    Matrix<T> operator-() const;

    //Maths element wise
    Matrix<T> exp() const;
    Matrix<T> log() const;
    Matrix<T> sqrt() const;
    Matrix<T> pow(T expo) const;
    Matrix<T> round() const;
    Matrix<T> ceil() const;
    Matrix<T> floor() const;
    Matrix<T> abs() const;
    Matrix<T> sin() const;
    Matrix<T> cos() const;
    Matrix<T> tan() const;
    Matrix<T> arcsin() const;
    Matrix<T> arccos() const;
    Matrix<T> arctan() const;
    Matrix<T> diff(size_t n= 1,size_t axis = -1) const;
    Matrix<T> gradient(const std::vector<T>& spacing = {1.0}) const;


    Matrix<T> matmul(const Matrix& A) const; // matrix mul
    bool operator==(const Matrix& A) const;
    bool operator!=(const Matrix& A) const;
    Matrix<bool> isnan() const;
    Matrix<bool> isinf() const;
    Matrix<bool> isfinite() const;


    //basic stats
    T sum() const;

    T max() const;
    T min() const;
    Matrix<T> sum(size_t axis) const;
    Matrix<T> max(size_t axis) const;
    Matrix<T> min(size_t axis) const;


    // impotant static methods
    static Matrix<T> zeros(const std::vector<size_t>& shape);
    static Matrix<T> ones(const std::vector<size_t>& shape);
    static Matrix<T> eye(size_t n);
    static Matrix<T> random(const std::vector<size_t>& shape,T minVal,T maxVal);
    static Matrix<T> arange(T start,T end,T step = 1);
    static Matrix<T> linspace(T start,T stop,size_t num);
    static Matrix<T> diag(const Matrix<T>& v,int k=0);


    // arrayy manupulation 
    Matrix<T> concatenate(const Matrix<T>& A,size_t axis) const;
    Matrix<T> stack(const Matrix<T>& A,size_t axis) const;
    std::vector<Matrix<T>> split(size_t sections, size_t axis) const;
    Matrix<T> swapaxes(size_t axis1,size_t axis2) const;
    Matrix<T> repeat(size_t repeats,size_t axis = 0) const;




    //broadcasting
    bool broadcastable(const Matrix<T>& A) const;
    std::pair<Matrix<T> ,Matrix<T>> broadcast(const Matrix<T>& A) const;


    // Linear algebra
    Matrix<T> dot(const Matrix<T>& A) const;
    Matrix<T> inverse() const;
    T determinant() const ;
    std::pair<Matrix<T>,Matrix<T>> eigenDecomposition() const;
    std::vector<Matrix<T>> svd() const; //singular value decomposition
    std::vector<Matrix<T>> LU_decomposition() const;


    // stats
    Matrix<T> normalize(T epsilion = 1e-8) const;
    Matrix<T> standardize() const;
    static T correlation(const Matrix<T>& A,const Matrix<T>& B);
    Matrix<T> covariance() const; // gives the variance co variance matrix
    T mean() const;
    T variance() const;
    T median() const;
    T standardDeviation() const;
    Matrix<T> mean(size_t axis) const;
    Matrix<T> median(size_t axis ) const;
    Matrix<T> variance(size_t axis) const;
    Matrix<T> standardDeviation(size_t axis) const;
    Matrix<T> percentile(T p) const;

    //adv stats
    Matrix<T> histogram(size_t bins = 10) const;
    Matrix<T> bincount() const;
    Matrix<T> digitize(const Matrix<T>& bins) const;



    // dimensionally operations
    Matrix<T> flatten() const; // to conver to 1d
    Matrix<T> expandDims(size_t axis) const; // to add new axes
    Matrix<T> squeeze() const; // remove single dimesninal entries
    Matrix<T> pad(const std::vector<std::pair<size_t,size_t>>& pad_width, const std::string& mode = "constant",T constant_value = 0) const;


    //sorting
    Matrix<T> sort(size_t axis = -1) const;
    Matrix<size_t> argSort(size_t axis = -1) const;
    Matrix<size_t> argMax(size_t axis = -1) const;
    Matrix<size_t> argMin(size_t axis  = -1) const;


    // adv operations
    Matrix<T> conv(const Matrix<T>& kernel , const std::string& mode = "full") const; // return the convolution 
    Matrix<T> trapz(const Matrix<T>& x = Matrix<T>(), size_t axis = -1) const;
    Matrix<T> cumsum(size_t axis = 0) const;
    Matrix<T> cumprod(size_t axis = 0) const;


    // performance opti
    void optimizeMemoryLayout();


    // basic utilities
    bool isSquare() const;
    bool isSymmetric() const;
    bool isDiagonal() const;
    size_t rank() const;
    T trace() const;


    // data type conversion
    template<typename U> 
    Matrix<U> asType() const;
    
    void fill(T value);
    bool empty() const{ return data.empty();}
    void print(std::ostream& os = std::cout) const;

    friend std::ostream& operator<<(std::ostream& os,const Matrix<T>& matrix){
        matrix.print(os);
        return os;
    }

};

#endif