#include "matrix.hpp"
template<typename T>
Matrix<T>::Matrix() {}

template<typename T>
Matrix<T>::Matrix(const std::vector<size_t>& shape) : shape(shape){
    size_t total_size = 1;
    for(size_t dim : shape)
        total_size *= dim;
    data.resize(total_size);
    calcStrides();
}

template<typename T>
Matrix<T>::Matrix(const std::vector<size_t>& shape,T value) : Matrix(shape){
    std::fill(data.begin(),data.end(),value);
}

template<typename T>
Matrix<T>::Matrix(std::initializer_list<T> list) {
    shape = {list.size()};
    data = std::vector<T>(list);
    calcStrides();
}

template<typename T>
Matrix<T>::Matrix(const Matrix<T>& A): data(A.data) , shape(A.shape),strides(A.strides){}

template<typename T>
Matrix<T>::Matrix(Matrix<T>&& A) noexcept: data(std::move(A.data)) , shape(std::move(A.shape)),strides(std::move(A.strides)){}

template<typename T> 
void Matrix<T>::calcStrides(){
    strides.resize(shape.size());
    if(shape.empty()) return;

    strides[shape.size() -1] = 1;
    for(int i= shape.size() - 2;i>=0;i--)
    {
        strides[i] = strides[i+1] * shape[i+1];
    }
    return ;

}

template<typename T>
size_t Matrix<T>::calcOffset(const std::vector<size_t>& indices) const{
    if(indices.size() != shape.size())
        throw std::invalid_argument("Invalid number of indices");
    size_t offset =0;
    for(size_t i=0;i<indices.size();i++)
    {
        if(indices[i] >= shape[i])
            throw std::out_of_range("Index out of Bounds");
        offset += indices[i]*strides[i];
    }
    return offset;
}

template<typename T>
T& Matrix<T>::at(const std::vector<size_t>& indices)
{
    return data[calcOffset(indices)];
}

template<typename T>
const T& Matrix<T>::at(const std::vector<size_t>& indices) const
{
    return data[calcOffset(indices)];
}

template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& A){
    if(this != A){
        data = A.data;
        shape = A.shape;
        strides = A.strides;
    }
    return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::operator=(Matrix<T>&& A) noexcept {
    if(this != A){
        data = std::move(A.data);
        shape = std::move(A.shape);
        strides = std::move(A.strides);
    }
    return *this;
}

template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& A) const{
    if(this->shape !=A.shape)
        throw std::invalid_argument("Shapes don't match for addition");
    Matrix<T> ans(shape);
    std::transform(this->data.begin(),this->data.end(),A.data.begin(),A.data.end(),std::plus<T>());
    return ans;
}

template<typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& A) const{
    if(this->shape !=A.shape)
        throw std::invalid_argument("Shapes don't match for Subtraction");
    Matrix<T> ans(shape);
    std::transform(this->data.begin(),this->data.end(),A.data.begin(),A.data.end(),std::minus<T>());
    return ans;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& A) const{
    if(this->shape !=A.shape)
        throw std::invalid_argument("Shapes don't match for element wise multiplication");
    Matrix<T> ans(shape);
    std::transform(this->data.begin(),this->data.end(),A.data.begin(),A.data.end(),std::multiplies<T>());
    return ans;
}

template<typename T>
Matrix<T> Matrix<T>::matmul(const Matrix<T>& A) const{
    if(this->shape.size() !=2 || A.shape.size() !=2 || this->shape[1] ! A.shape[0])
        throw std::invalid_argument("Invalid shapes for matrix multiplication");

    size_t m  =this->shape[0];
    size_t k = this->shape[1];
    size_t n = A.shape[1];
    Matrix ans({m,n},T(0));
    
    // we will use the parralelle algo for large matrices
    const size_t min_sz_for_prl = 50;
    if(m*n*k >= min_sz_for_prl*min_sz_for_prl*min_sz_for_prl)
    {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads; // vector to store the threads

        auto worker= [&](size_t start_row,size_t end_row)
        {
            for(size_t i = start_row;i<end_row;i++)
            {
                for(size_t j=0;j<n;j++)
                {
                    T sum = T(0);
                    for(size_t p=0;p<k;p++)
                        sum += this->at({i,p}) * A.at({p,j});
                    ans.at({i,j}) = sum;
                }
                
            }
        };
        size_t rows_per_thread = m/num_threads;
        for(size_t t=0;t<num_threads -1 ;t++)
        {
            threads.emplace_back(worker,t* rows_per_thread,(t+1)*rows_per_thread);
        }

        worker((num_threads-1)*rows_per_thread,m);
        for(auto& thread:threads)
            thread.join();

    }
    else
    {
        // Sequential multiplication for smaller matrices
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                T sum = T(0);
                for (size_t p = 0; p < k; ++p) {
                    sum += this->at({i, p}) * A.at({p, j});
                }
                ans.at({i, j}) = sum;
            }
        }
    }
    return ans;
}

template<typename T>
Matrix<T> Matrix<T>::operator+(T scaler) const{
    Matrix<T> ans(this->shape);
    std::transform(this->data.begin(),this->data.end(),ans.data.begin(),[scaler](const T& val) {return val+scaler;});
    return ans;
}

template<typename T>
Matrix<T> Matrix<T>::operator-(T scaler) const{
    Matrix<T> ans(this->shape);
    std::transform(this->data.begin(),this->data.end(),ans.data.begin(),[scaler](const T& val) {return val-scaler;});
    return ans;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(T scaler) const{
    Matrix<T> ans(this->shape);
    std::transform(this->data.begin(),this->data.end(),ans.data.begin(),[scaler](const T& val) {return val*scaler;});
    return ans;
}

template<typename T>
Matrix<T> Matrix<T>::operator/(T scaler) const{
    if(scaler == T(0))
        throw std::invalid_argument("Division by zero errors");
    Matrix<T> ans(this->shape);
    std::transform(this->data.begin(),this->data.end(),ans.data.begin(),[scaler](const T& val) {return val+scaler;});
    return ans;
}
template<typename T>
Matrix<T> Matrix<T>::operator-() const{
    Matrix<T> ans(this->shape);
    std::transform(this->data.begin(),this->data.end(),ans.data.begin(),std::negate<T>());
    return ans;
}

template<typename T>
Matrix<T> Matrix<T>::operator+() const{
    return *this;
}

template<typename T>
T& Matrix<T>::operator[](const std::vector<size_t>& indices){
    return data[calcOffset(indices)];
}

template<typename T>
const T& Matrix<T>::operator[](const std::vector<size_t>& indices) const{
    return data[calcOffset(indices)];
}

template<typename T>
bool Matrix<T>::any(size_t axis) const{
    if(axis == static_cast<size_t>(-1))
    {
        // using parrallel algo for large matrices
        const size_t min_sz_for_par = 5000;
        if(this->data.size()>=min_sz_for_par)
        {
            size_t num_threads = std::thread::hardware_concurrency();
            std::vector<std::thread> threads;
            std::vector<bool> ans(num_threads,false);
            size_t chunk_size = this->data.size()/num_threads;
            auto worker = [this](size_t start,size_t end,bool& result)
            {
                result = std::any_of(data.begin()+start,data.begin()+end,[](const T& val){return static_cast<bool>(val);});
            };
            //launching the threads
            for(size_t i=0;i<num_threads-1;i++)
            {
                threads.emplace_back(worker, i*chunk_size,(i+1)*chunk_size,std::ref(ans[i]));
            }
            worker((num_threads-1)*chunk_size,this->data.size(),ans[num_threads-1]);
            for(auto& thread:threads)
            {
                thread.join();
            }
            return std::any_of(ans.begin(),ans.end(),[](bool b){return b;});

        }
        else{
            //continue with the serial implementation
            return std::any_of(this->data.begin(),this->data.end(),[](const T& val){return static_cast<bool>(val);});
        }

    }
    // For specific axis
if(axis >= shape.size())
    throw std::invalid_argument("Axis out of bounds");

std::vector<size_t> ans_shape = shape;
ans_shape.erase(ans_shape.begin() + axis);
Matrix<bool> ans(ans_shape, false);

auto process_slice = [&](const std::vector<size_t>& base_indices) {
    size_t slice_size = shape[axis];
    for(size_t i = 0; i < slice_size; i++) {
        auto indices = base_indices;
        indices.insert(indices.begin() + axis, i);
        if(static_cast<bool>(at(indices)))
            return true;
    }
    return false;
};

// Process slices either in parallel or sequentially
const size_t min_slices_for_parallel = 500;
size_t total_slices = ans.size();

if(total_slices >= min_slices_for_parallel) {
    // Parallel processing for large matrices
    size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    size_t slices_per_thread = total_slices / num_threads;

    auto worker = [&](size_t start_slice, size_t end_slice) {
        std::vector<size_t> indices(shape.size() - 1);
        for(size_t slice = start_slice; slice < end_slice; slice++) {
            // Convert linear index to multi-dimensional indices
            size_t temp = slice;
            for(size_t i = 0; i < indices.size(); i++) {
                indices[i] = temp / ans.strides[i];
                temp %= ans.strides[i];
            }
            ans.at(indices) = process_slice(indices);
        }
    };

    // Launch threads
    for(size_t i = 0; i < num_threads - 1; i++) {
        threads.emplace_back(worker, i * slices_per_thread, 
                           (i + 1) * slices_per_thread);
    }
    worker((num_threads - 1) * slices_per_thread, total_slices);

    for(auto& thread : threads)
        thread.join();
} else {
    // Sequential processing
    std::vector<size_t> indices(shape.size() - 1);
    for(size_t i = 0; i < ans.size(); i++) {
        size_t temp = i;
        for(size_t j = 0; j < indices.size(); j++) {
            indices[j] = temp / ans.strides[j];
            temp %= ans.strides[j];
        }
        ans.at(indices) = process_slice(indices);
    }
}

return ans;

}

template<typename T>
bool Matrix<T>::all(size_t axis) const{

    if(axis == static_cast<size_t>(-1))
    {
        const size_t min_sz_for_par = 5000;
        //using parallel algo for large matrices
        if(this->data.size() >= min_sz_for_par)
        {
            size_t num_threads = std::thread::hardware_concurrency();
            std::vector<std::thread> threads;
            std::vector<bool> ans(num_threads,true);
            size_t chunk_size = this->data.size() / num_threads;

            auto worket = [this](size_t start,size_t end, bool& result)
            {
                result = std::all_of(this->data.begin()+start,this->data.begin()+end,[](const T& val){
                    return static_cast<bool>(val);
                });

            };
            for(size_t i=0;i<num_threads-1;i++)
            {
                threads.emplace_back(worker,i*chunk_size,(i+1)*chunk_size,std::ref(ans[i]));
            }
            worker((num_threads-1)*chunk_size,this->data.size(),ans[num_threads-1]);

            for(auto& thread:threads)
            {
                thread.join();
            }
            // if all threads return true then return true
            return std::all_of(ans.begin(),ans.end(),[](bool b){
                return b;
            });
        }
        return std::all_of(data.begin(), 
                          data.end(),
                          [](const T& val) { 
                              return static_cast<bool>(val); 
                          });


    }

    if(axis >= this->shape.size())
        throw std::invalid_argument("Axis out of bounds");
        std::vector<size_t> result_shape = shape;
    result_shape.erase(result_shape.begin() + axis);
    Matrix<bool> result(result_shape, true);
    
    // Lambda function to check a slice along the specified axis
    auto check_slice = [&](const std::vector<size_t>& base_indices) {
        size_t slice_size = shape[axis];
        for (size_t i = 0; i < slice_size; i++) {
            auto indices = base_indices;
            indices.insert(indices.begin() + axis, i);
            if (!static_cast<bool>(this->at(indices))) {
                return false;  // Found a false value
            }
        }
        return true;  // All values were true
    };
        // Process each position in the result matrix
    const size_t MIN_SLICES_FOR_PARALLEL = 1000;
    size_t total_slices = result.size();
    
    if (total_slices >= MIN_SLICES_FOR_PARALLEL) {
        // Parallel processing for large matrices
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t slices_per_thread = total_slices / num_threads;
        
        auto worker = [&](size_t start_slice, size_t end_slice) {
            std::vector<size_t> indices(shape.size() - 1);
            for (size_t slice = start_slice; slice < end_slice; slice++) {
                // Convert linear index to multi-dimensional indices
                size_t temp = slice;
                for (size_t j = 0; j < indices.size(); j++) {
                    indices[j] = temp / result.strides[j];
                    temp %= result.strides[j];
                }
                result.at(indices) = check_slice(indices);
            }
        };
        
        // Launch threads
        for (size_t i = 0; i < num_threads - 1; ++i) {
            threads.emplace_back(worker, 
                               i * slices_per_thread,
                               (i + 1) * slices_per_thread);
        }
        
        // Process remaining slices in current thread
        worker((num_threads - 1) * slices_per_thread, total_slices);
        
        // Join threads
        for (auto& thread : threads) {
            thread.join();
        }
    } else {
        // Sequential processing for small matrices
        std::vector<size_t> indices(shape.size() - 1);
        for (size_t i = 0; i < result.size(); i++) {
            size_t temp = i;
            for (size_t j = 0; j < indices.size(); j++) {
                indices[j] = temp / result.strides[j];
                temp %= result.strides[j];
            }
            result.at(indices) = check_slice(indices);
        }
    }
    
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::reshape(const std::vector<size_t>& newShape) const{
    // first we check if reshape is possible or not
    size_t new_size = 1 //  for that we need to calculate the total size 
    for(size_t dim:newShape)
        new_size *= dim;
    if(new_size != this->data.size())
        throw std::invalid_argument("Total size mismatch for reshape");

    Matrix<T> ans;
    ans.data = this->data;
    ans.shape = newShape;
    ans.calcStrides();
    return ans;
    
}

template<typename T>
Matrix<T> Matrix<T>::transpose(const std::vector<size_t>& axes) const{
    if(this->shape.empty())     return *this;
    std::vector<size_t> permutation;
    if(axs.empty())
    {
        //default transpose just reverse the dimensions
        for(size_t i=0;i< this->size();i++)
        {
            permutation[i] = this->shape.size() - i- 1;
        }
    }
    else{
        //custom permutation
        if(axes.size() != this->shape.size())
            throw std::invalid_argument("Axes permutation must match maxtix dimensions");
        std::vector<bool> used(this->shape.size(),false);
        for(size_t axis:axes)
        {
            if(axis>= this->shape.size() || used[axis])
                throw std::invalid_argument("Invalid axes permuation");
            used[axis] = true;
        }
        permutation = axes;

    }
    // creating ans matrix and new shape and strides
    std::vector<size_t> newShape(this->shape.size());
    for(size_t i=0;i< this->shape.size();i++)
        newShape[i] = shape[permutation[i]];
    Matrix<T> ans(newShape);

    auto linear_to_inx = [](size_t linear_idx,const std::vector<size_t>& shape,std::vector<size_t>& strides)
    {
        std::vector<size_t> indices(this->shape.size());
        for(size_t i=0;i<this->shape.size();i++)
            indices[i] = (linear_idx / strides[i])  % this->shape[i];
        return indices;
    };

    const size_t min_sz_for_par = 5000;
    if(this->data.size() >= min_sz_for_par)
    {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t chunk_size = this->data.size() / num_threads;

        auto worker = [&](size_t start,size_t end)
        {
            for(size_t i=start;i<end;i++)
            {
                auto src_indices = linear_to_inx(i,this->shape,this->strides);
                std::vector<size_t> dst_indices(this->shape.size());
                for(size_t j=0;j<this->shape.size();j++)
                    dst_indices[j] = src_indices[permutation[i]];
                ans.at(dst_indices) = this->data[i];

    
            }
        };
        for (size_t i = 0; i < num_threads - 1; ++i) {
            threads.emplace_back(worker, i * chunk_size, (i + 1) * chunk_size);
        }
        worker((num_threads - 1) * chunk_size, this->data.size());
        
        // Join threads
        for (auto& thread : threads) {
            thread.join();
    }
    }
    else
    {
        for (size_t i = 0; i < this->data.size(); ++i) {
            auto src_indices = linear_to_indices(i, this->shape, this->strides);
            std::vector<size_t> dst_indices(this->shape.size());
            for (size_t j = 0; j < shape.size(); ++j) {
                dst_indices[j] = src_indices[permutation[j]];
            }
            ans.at(dst_indices) = this->data[i];
        }        
    }
    return ans;
        
}

template<typename T>
Matrix<T> Matrix<T>:: exp() const{
    Matrix<T> ans(this->shape);
    const size_t min_sz_for_par = 5000;
    if(this->data.size() >= min_sz_for_par)
    {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t chunks_size = this->data.size() / num_threads;
        auto worker = [this,&ans] (size_t start,size_t end)
        {
            std:transform(data.begin()+start,data.begin()+end,and.data.begin()+start,[](const T& x){return std::exp(x);});
        };
        for(size_t i=0;i<num_threads -1;i++)
        {
            threads.emplace_back(worker,i*chunks_size,(i+1)*chunks_size);
        }
        worker((num_threads-1)*chunks_size,this->data.size());

        for(auto& thread:threads)
            thread.join();
        
    }
    else    
        std::transform(this->data.begin,this->data.end(),ans.data.begin(),[](const T& x){return std::exp(x);});
    return ans;
}

template<typename T>
Matrix<T> Matrix<T>:: log() const{
    Matrix<T> ans(this->shape);
    auto operation  = [](const T& x)
    {
        if(x <=T(0))
            throw std::domain_error("Log of non positive number is not possible");
        return std::log(x);
    };
    const size_t min_sz_for_par = 5000;
    if(this->data.size() >= min_sz_for_par)
    {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t chunks_size = this->data.size() / num_threads;
        auto worker = [this,&ans] (size_t start,size_t end)
        {
            std:transform(data.begin()+start,data.begin()+end,and.data.begin()+start,operation);
        };
        for(size_t i=0;i<num_threads -1;i++)
        {
            threads.emplace_back(worker,i*chunks_size,(i+1)*chunks_size);
        }
        worker((num_threads-1)*chunks_size,this->data.size());

        for(auto& thread:threads)
            thread.join();
        
    }
    else    
        std::transform(this->data.begin,this->data.end(),ans.data.begin(),operation);
    return ans;
}

template<typename T>
Matrix<T> Matrix<T>:: sqrt() const{
    Matrix<T> ans(this->shape);
    auto operation  = [](const T& x)
    {
        if(x <=T(0))
            throw std::domain_error("Sqrt of negative number is not possible");
        return std::sqrt(x);
    };
    const size_t min_sz_for_par = 5000;
    if(this->data.size() >= min_sz_for_par)
    {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t chunks_size = this->data.size() / num_threads;
        auto worker = [this,&ans] (size_t start,size_t end)
        {
            std:transform(data.begin()+start,data.begin()+end,and.data.begin()+start,operation);
        };
        for(size_t i=0;i<num_threads -1;i++)
        {
            threads.emplace_back(worker,i*chunks_size,(i+1)*chunks_size);
        }
        worker((num_threads-1)*chunks_size,this->data.size());

        for(auto& thread:threads)
            thread.join();
        
    }
    else    
        std::transform(this->data.begin,this->data.end(),ans.data.begin(),operation);
    return ans;
}

template<typename T>
Matrix<T> Matrix<T>:: pow(T expo) const{
    Matrix<T> ans(this->shape);
    auto operation  = [](const T& x)
    {

        return std::pow(x,expo);
    };
    const size_t min_sz_for_par = 5000;
    if(this->data.size() >= min_sz_for_par)
    {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t chunks_size = this->data.size() / num_threads;
        auto worker = [this,&ans] (size_t start,size_t end)
        {
            std:transform(data.begin()+start,data.begin()+end,and.data.begin()+start,operation);
        };
        for(size_t i=0;i<num_threads -1;i++)
        {
            threads.emplace_back(worker,i*chunks_size,(i+1)*chunks_size);
        }
        worker((num_threads-1)*chunks_size,this->data.size());

        for(auto& thread:threads)
            thread.join();
        
    }
    else    
        std::transform(this->data.begin,this->data.end(),ans.data.begin(),operation);
    return ans;
}

template<typename T>
Matrix<T> Matrix<T>:: round() const{
    Matrix<T> ans(this->shape);
    auto operation  = [](const T& x)
    {

        return std::round(x);
    };
    const size_t min_sz_for_par = 5000;
    if(this->data.size() >= min_sz_for_par)
    {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t chunks_size = this->data.size() / num_threads;
        auto worker = [this,&ans] (size_t start,size_t end)
        {
            std:transform(data.begin()+start,data.begin()+end,and.data.begin()+start,operation);
        };
        for(size_t i=0;i<num_threads -1;i++)
        {
            threads.emplace_back(worker,i*chunks_size,(i+1)*chunks_size);
        }
        worker((num_threads-1)*chunks_size,this->data.size());

        for(auto& thread:threads)
            thread.join();
        
    }
    else    
        std::transform(this->data.begin,this->data.end(),ans.data.begin(),operation);
    return ans;
}

template<typename T>
Matrix<T> Matrix<T>:: ceil() const{
    Matrix<T> ans(this->shape);
    auto operation  = [](const T& x)
    {

        return std::ceil(x);
    };
    const size_t min_sz_for_par = 5000;
    if(this->data.size() >= min_sz_for_par)
    {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t chunks_size = this->data.size() / num_threads;
        auto worker = [this,&ans] (size_t start,size_t end)
        {
            std:transform(data.begin()+start,data.begin()+end,and.data.begin()+start,operation);
        };
        for(size_t i=0;i<num_threads -1;i++)
        {
            threads.emplace_back(worker,i*chunks_size,(i+1)*chunks_size);
        }
        worker((num_threads-1)*chunks_size,this->data.size());

        for(auto& thread:threads)
            thread.join();
        
    }
    else    
        std::transform(this->data.begin,this->data.end(),ans.data.begin(),operation);
    return ans;
}

template<typename T>
Matrix<T> Matrix<T>:: floor() const{
    Matrix<T> ans(this->shape);
    auto operation  = [](const T& x)
    {

        return std::floor(x);
    };
    const size_t min_sz_for_par = 5000;
    if(this->data.size() >= min_sz_for_par)
    {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t chunks_size = this->data.size() / num_threads;
        auto worker = [this,&ans] (size_t start,size_t end)
        {
            std:transform(data.begin()+start,data.begin()+end,and.data.begin()+start,operation);
        };
        for(size_t i=0;i<num_threads -1;i++)
        {
            threads.emplace_back(worker,i*chunks_size,(i+1)*chunks_size);
        }
        worker((num_threads-1)*chunks_size,this->data.size());

        for(auto& thread:threads)
            thread.join();
        
    }
    else    
        std::transform(this->data.begin,this->data.end(),ans.data.begin(),operation);
    return ans;
}

template<typename T>
Matrix<T> Matrix<T>:: abs() const{
    Matrix<T> ans(this->shape);
    auto operation  = [](const T& x)
    {
        return std::abs(x);
    };
    const size_t min_sz_for_par = 5000;
    if(this->data.size() >= min_sz_for_par)
    {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t chunks_size = this->data.size() / num_threads;
        auto worker = [this,&ans] (size_t start,size_t end)
        {
            std:transform(data.begin()+start,data.begin()+end,and.data.begin()+start,operation);
        };
        for(size_t i=0;i<num_threads -1;i++)
        {
            threads.emplace_back(worker,i*chunks_size,(i+1)*chunks_size);
        }
        worker((num_threads-1)*chunks_size,this->data.size());

        for(auto& thread:threads)
            thread.join();
        
    }
    else    
        std::transform(this->data.begin,this->data.end(),ans.data.begin(),operation);
    return ans;
}

template<typename T>
Matrix<T> Matrix<T>:: sin() const{
    Matrix<T> ans(this->shape);
    auto operation  = [](const T& x)
    {

        return std::sin(x);
    };
    const size_t min_sz_for_par = 5000;
    if(this->data.size() >= min_sz_for_par)
    {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t chunks_size = this->data.size() / num_threads;
        auto worker = [this,&ans] (size_t start,size_t end)
        {
            std:transform(data.begin()+start,data.begin()+end,and.data.begin()+start,operation);
        };
        for(size_t i=0;i<num_threads -1;i++)
        {
            threads.emplace_back(worker,i*chunks_size,(i+1)*chunks_size);
        }
        worker((num_threads-1)*chunks_size,this->data.size());

        for(auto& thread:threads)
            thread.join();
        
    }
    else    
        std::transform(this->data.begin,this->data.end(),ans.data.begin(),operation);
    return ans;
}

template<typename T>
Matrix<T> Matrix<T>:: cos() const{
    Matrix<T> ans(this->shape);
    auto operation  = [](const T& x)
    {

        return std::cos(x);
    };
    const size_t min_sz_for_par = 5000;
    if(this->data.size() >= min_sz_for_par)
    {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t chunks_size = this->data.size() / num_threads;
        auto worker = [this,&ans] (size_t start,size_t end)
        {
            std:transform(data.begin()+start,data.begin()+end,and.data.begin()+start,operation);
        };
        for(size_t i=0;i<num_threads -1;i++)
        {
            threads.emplace_back(worker,i*chunks_size,(i+1)*chunks_size);
        }
        worker((num_threads-1)*chunks_size,this->data.size());

        for(auto& thread:threads)
            thread.join();
        
    }
    else    
        std::transform(this->data.begin,this->data.end(),ans.data.begin(),operation);
    return ans;
}

template<typename T>
Matrix<T> Matrix<T>:: tan() const{
    Matrix<T> ans(this->shape);
    auto operation  = [](const T& x)
    {

        return std::tan(x);
    };
    const size_t min_sz_for_par = 5000;
    if(this->data.size() >= min_sz_for_par)
    {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t chunks_size = this->data.size() / num_threads;
        auto worker = [this,&ans] (size_t start,size_t end)
        {
            std:transform(data.begin()+start,data.begin()+end,and.data.begin()+start,operation);
        };
        for(size_t i=0;i<num_threads -1;i++)
        {
            threads.emplace_back(worker,i*chunks_size,(i+1)*chunks_size);
        }
        worker((num_threads-1)*chunks_size,this->data.size());

        for(auto& thread:threads)
            thread.join();
        
    }
    else    
        std::transform(this->data.begin,this->data.end(),ans.data.begin(),operation);
    return ans;
}

template<typename T>
Matrix<T> Matrix<T>:: arcsin() const{
    Matrix<T> ans(this->shape);
    auto operation  = [](const T& x)
    {
        if(x < T(-1) || x>T(1))
            throw std::domain_error("Values are not in range for Arc sin");
        return std::asin(x);
    };
    const size_t min_sz_for_par = 5000;
    if(this->data.size() >= min_sz_for_par)
    {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t chunks_size = this->data.size() / num_threads;
        auto worker = [this,&ans] (size_t start,size_t end)
        {
            std:transform(data.begin()+start,data.begin()+end,and.data.begin()+start,operation);
        };
        for(size_t i=0;i<num_threads -1;i++)
        {
            threads.emplace_back(worker,i*chunks_size,(i+1)*chunks_size);
        }
        worker((num_threads-1)*chunks_size,this->data.size());

        for(auto& thread:threads)
            thread.join();
        
    }
    else    
        std::transform(this->data.begin,this->data.end(),ans.data.begin(),operation);
    return ans;
}

template<typename T>
Matrix<T> Matrix<T>:: arccos() const{
    Matrix<T> ans(this->shape);
    auto operation  = [](const T& x)
    {
        if(x<T(-1) || x>T(1))
            throw std::domain_error("values not in range for arc cos")
        return std::acos(x);
    };
    const size_t min_sz_for_par = 5000;
    if(this->data.size() >= min_sz_for_par)
    {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t chunks_size = this->data.size() / num_threads;
        auto worker = [this,&ans] (size_t start,size_t end)
        {
            std:transform(data.begin()+start,data.begin()+end,and.data.begin()+start,operation);
        };
        for(size_t i=0;i<num_threads -1;i++)
        {
            threads.emplace_back(worker,i*chunks_size,(i+1)*chunks_size);
        }
        worker((num_threads-1)*chunks_size,this->data.size());

        for(auto& thread:threads)
            thread.join();
        
    }
    else    
        std::transform(this->data.begin,this->data.end(),ans.data.begin(),operation);
    return ans;
}

template<typename T>
Matrix<T> Matrix<T>:: arctan() const{
    Matrix<T> ans(this->shape);
    auto operation  = [](const T& x)
    {
        return std::atan(x);
    };
    const size_t min_sz_for_par = 5000;
    if(this->data.size() >= min_sz_for_par)
    {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t chunks_size = this->data.size() / num_threads;
        auto worker = [this,&ans] (size_t start,size_t end)
        {
            std:transform(data.begin()+start,data.begin()+end,and.data.begin()+start,operation);
        };
        for(size_t i=0;i<num_threads -1;i++)
        {
            threads.emplace_back(worker,i*chunks_size,(i+1)*chunks_size);
        }
        worker((num_threads-1)*chunks_size,this->data.size());

        for(auto& thread:threads)
            thread.join();
        
    }
    else    
        std::transform(this->data.begin,this->data.end(),ans.data.begin(),operation);
    return ans;
}

template<typename T>
Matrix<T> Matrix<T>::diff(size_t n,size_t axis) const{ 
    // to calculate nth order differenetion along axis
    if(axis >= this->shape.size())
        throw std::invalid_argument("Axis out of bounds");
    if(this->shape[axis] <=n)
        throw std::invalid_argument("Not enough elements along the axis for diff");

    std::vector<size_t> newShape = this->shape;
    newShape[axis] -=n;
    Matrix<T> ans(newShape);
    auto fn = [&](const std::vector<size_t>& base_indices)
    {
        std::vector<T> values(shape[axis]);
        for(size_t i=0;<shape[axis];i++)
        {
            auto indices = base_indices;
            indices.insert(indices.begin()+axis,i);
            values[i] = this->at(indices);
        }
        for(size_t k=0;k<n;k++)
        {
            for(size_t i=0;i<values.size();i++)
                values[i] = values[i+1] - values[i];
            values.pop_back();
        }
        return values;
    };
    std::vector<size_t> indices(this->shape.size()-1);
    for(size_t i=0;i<ans.size();i++)
    {
        size_t temp = i;
        for(size_t j=0;j<indices.size();j++)
        {
            indices[j] = temp / ans.strides[j];
            temp %= ans.strides[j];
        }
        auto diff_values = process_slice(indices);
        for(size_t j=0;j<diff_values.size();j++)
        {
            auto result_indices = indices;
            result_indices.insert(result_indices.begin() + axis,j);
            ans.at(result_indices) = diff_values[j];
        }
    }
    return ans;
}

template<typename T>
Matrix<T> Matrix<T>::gradient(const std::vector<T>& spacing) const{

    if(this->shape.empty())
        throw std::invalid_argument("Empty matrix for gradient");
    std::vector<T> newSpec;
    if(spacing.empty())
        newSpec = std::vector<T>(this->shape.size(),T(1));
    else if(spacing.size() != this->shape.size())
        throw std::invalid_argument("Spacing size must matcht the matrix dimensions for gradient");
    else
        newSpec = spacing;

    std::vector<Matrix<T>> gradients;
    for(size_t i=0;i<this->shape.size();i++)
    {
        auto diff_result = diff(1,i);
        diff_result = diff_result / spacing[i];
        gradients.push_back(diff_result);
    }
    // no need for this diff and gradient method i feel let's see in future
    return gradients[0];
}

template<typename T>
bool Matrix<T>::operator==(const Matrix& A) const{
    if(this->shape != A.shape())
        return false;
    
    const size_t min_sz_for_parallel = 5000;
    if(this->data.size()>=min_sz_for_parallel)
    {
        //employing parallel algo for larger objects
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        std::vector<bool> ans(num_threads,true);
        size_t chunk_size = this->data.size() / num_threads;
        auto worker = [this, &A](size_t start,size_t end,bool& result){
            result  = std::equal(this->data.begin(),this->data.end(),A.data.begin(),[](const T& a, const T& b){
                if constexpr (std::is_floating_point_v<T>){
                    const T epsilon = std::numeric_limits<T>::epsilon()*100;
                    return std::abs(a-b)<=epsilon * std::max({T(1),std::abs(a),std::abs(b)});
                }
                else
                    return a == b;
            });
        };
        for(size_t i=0;i<num_threads-1;i++)
            threads.emplace_back(worker,i*chunk_size,(i+1)*chunk_size,std::ref(ans[i]));
        worker((num_threads-1)*chunk_size,this->data.size(),ans[num_threads-1]);
        for(auto& thread:threads)   thread.join();
        return std::all_of(ans.begin(),ans.end(),[](bool x){return x;});
    }
    return std::equal(this->data.begin(),this->data.end(),A.data.begin(),
    [](const T& a ,const T& b){
        if constexpr (std::is_floating_point_v<T>)
        {
            const T epsilon = std::numeric_limits<T>::epsilon() * 100;
            return std::abs(a - b) <= epsilon * std::max({T(1), std::abs(a), std::abs(b)});
        }
        return a==b;
    });
}

template<typename T>
bool Matrix<T>::operator!=(const Matrix& A) const{
    return !(*this == A);
}

template<typename T>
Matrix<bool> Matrix<T>::isnan() const {
    Matrix<bool> result(this->shape);
    
    const size_t MIN_SIZE_FOR_PARALLEL = 5000;
    if (this->data.size() >= MIN_SIZE_FOR_PARALLEL) {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t chunk_size = this->data.size() / num_threads;

        auto worker = [this, &result](size_t start, size_t end) {
            std::transform(
                this->data.begin() + start,
                this->data.begin() + end,
                result.data.begin() + start,
                [](const T& x) { 
                    if constexpr (std::is_floating_point_v<T>) {
                        return std::isnan(x);
                    }
                    return false;
                }
            );
        };
        // Launch threads
        for (size_t i = 0; i < num_threads - 1; ++i) {
            threads.emplace_back(worker, i * chunk_size, (i + 1) * chunk_size);
        }
        worker((num_threads - 1) * chunk_size, this->data.size());

        // Join threads
        for (auto& thread : threads) {
            thread.join();
        }
    } else {
        std::transform(this->data.begin(), this->data.end(), result.data.begin(),
            [](const T& x) { 
                if constexpr (std::is_floating_point_v<T>) {
                    return std::isnan(x);
                }
                return false;
            });
    }
    return result;
}

template<typename T>
Matrix<bool> Matrix<T>::isinf() const {
    Matrix<bool> result(shape);
    
    auto check_inf = [](const T& x) {
        if constexpr (std::is_floating_point_v<T>) {
            return std::isinf(x);
        }
        return false;
    };

    const size_t MIN_SIZE_FOR_PARALLEL = 5000;
    if (data.size() >= MIN_SIZE_FOR_PARALLEL) {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t chunk_size = data.size() / num_threads;

        auto worker = [this, &result, &check_inf](size_t start, size_t end) {
            std::transform(
                data.begin() + start,
                data.begin() + end,
                result.data.begin() + start,
                check_inf
            );
        };

        // Launch threads
        for (size_t i = 0; i < num_threads - 1; ++i) {
            threads.emplace_back(worker, i * chunk_size, (i + 1) * chunk_size);
        }
        worker((num_threads - 1) * chunk_size, data.size());

        // Join threads
        for (auto& thread : threads) {
            thread.join();
        }
    } else {
        std::transform(data.begin(), data.end(), result.data.begin(), check_inf);
    }
    
    return result;
}

template<typename T>
Matrix<bool> Matrix<T>::isfinite() const {
    Matrix<bool> result(shape);
    
    auto check_finite = [](const T& x) {
        if constexpr (std::is_floating_point_v<T>) {
            return std::isfinite(x);
        }
        return true;
    };

    const size_t MIN_SIZE_FOR_PARALLEL = 5000;
    if (data.size() >= MIN_SIZE_FOR_PARALLEL) {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t chunk_size = data.size() / num_threads;

        auto worker = [this, &result, &check_finite](size_t start, size_t end) {
            std::transform(
                data.begin() + start,
                data.begin() + end,
                result.data.begin() + start,
                check_finite
            );
        };

        // Launch threads
        for (size_t i = 0; i < num_threads - 1; ++i) {
            threads.emplace_back(worker, i * chunk_size, (i + 1) * chunk_size);
        }
        worker((num_threads - 1) * chunk_size, data.size());

        // Join threads
        for (auto& thread : threads) {
            thread.join();
        }
    } else {
        std::transform(data.begin(), data.end(), result.data.begin(), check_finite);
    }
    return result;
}

template<typename T>
T Matrix<T>::sum() const{
    const size_t min_sz_for_parallel = 5000;
    if(this->data.size() >= min_sz_for_parallel)
    {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        std::vector<T> partial_sums(num_threads);
        size_t chunk_size = this->data.size()/num_threads;
        auto worker = [this](size_t start,size_t end,T& result){
            result = std::accumulate(this->data.begin()+start,this->data.begin()+end,T(0));
        };
        for(size_t i=0;i<num_threads-1;i++)
            threads.emplace_back(worker,i*chunk_size,(i+1)*chunk_size,std::ref(partial_sums[i]));
        
        worker((num_threads-1)*chunk_size,this->data.size(),partial_sums[num_threads-1]);
        for(auto& thread:threads)
            thread.join();
        return std::accumulate(partial_sums.begin(),partial_sums.end(),T(0));
    }
    return std::accumulate(this->data.begin(),this->data.end(),T(0));
}

template<typename T>
T Matrix<T>::mean() const{
    if(this->data.empty())
        throw std::runtime_error("Cannot calculate mean of empty matrix object");
    return sum() / static_cast<T>(this->data.size());
}

template<typename T>
T Matrix<T>::max() const{
    if(this->data.empty())
        throw std::runtime_error("Cannot find max of empty matrix object");

    const size_t min_sz_for_parallel = 5000;
    if(this->data.size() >= min_sz_for_parallel)
    {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        std::vector<T> partial_maxes(num_threads);
        size_t chunk_size = this->data.size()/num_threads;
        auto worker = [this](size_t start,size_t end,T& result){
            result = *std::max_element(this->data.begin()+start,this->data.begin()+end);
        };
        for(size_t i=0;i<num_threads-1;i++)
            threads.emplace_back(worker,i*chunk_size,(i+1)*chunk_size,std::ref(partial_maxes[i]));
        
        worker((num_threads-1)*chunk_size,this->data.size(),partial_maxes[num_threads-1]);
        for(auto& thread:threads)
            thread.join();
        return *std::max_element(partial_maxes.begin(),partial_maxes.end());
    }
    return *std::max_element(this->data.begin(),this->data.end());
}

template<typename T>
T Matrix<T>::min() const{
    if(this->data.empty())
        throw std::runtime_error("Cannot find min of empty matrix object");

    const size_t min_sz_for_parallel = 5000;
    if(this->data.size() >= min_sz_for_parallel)
    {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        std::vector<T> partial_mins(num_threads);
        size_t chunk_size = this->data.size()/num_threads;
        auto worker = [this](size_t start,size_t end,T& result){
            result = *std::min_element(this->data.begin()+start,this->data.begin()+end);
        };
        for(size_t i=0;i<num_threads-1;i++)
            threads.emplace_back(worker,i*chunk_size,(i+1)*chunk_size,std::ref(partial_mins[i]));
        
        worker((num_threads-1)*chunk_size,this->data.size(),partial_mins[num_threads-1]);
        for(auto& thread:threads)
            thread.join();
        return *std::min_element(partial_mins.begin(),partial_mins.end());
    }
    return *std::min_element(this->data.begin(),this->data.end());
}

template<typename T>
Matrix<T> Matrix<T>::sum(size_t axis) const{
    if(axis >= this->shape.size())
        throw std::invalid_argument("Axis out of bounds for sum");
    std::vector<size_t> ans_shape = this->shape;
    ans_shape.erase(ans_shape.begin()+axis);
    Matrix<T> ans(ans_shape,T(0));

    auto process_slice = [&](const std::vector<size_t>& base_indices)
    {
        T sum = T(0);
        for(size_t i=0;i<this->shape[axis];i++)
        {
            auto indices = base_indices;
            indices.insert(indices.begin()+axis,i);
            sum += this->at(indices);
        }
        return sum;
    };
    const size_t total_slices =  ans.size();
    const size_t min_slices_for_parallel = 500;
    if(total_slices >= min_slices_for_parallel)
    {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t slices_per_thread = total_slices / num_threads;
        auto worker = [&](size_t start_slice,size_t end_slice)
        {
            std::vector<size_t> indices(this->shape.size()-1);
            for(size_t slice = start_slice;slice<end_slice;slice++)
            {
                size_t temp = slice;
                for(size_t j=0;j<indices.size();j++)
                {
                    indices[j] = temp / ans.strides[j];
                    temp %= ans.strides[j];
                }
                ans.at(indices) = process_slice(indices);
            }
        };
        for(size_t i=0;i<num_threads-1;i++)
            threads.emplace_back(worker,i*slices_per_thread,(i+1)*slices_per_thread);
        worker((num_threads-1)*slices_per_thread,total_slices);
        for(auto& thread:threads)   thread.join();

    }
    else
    {
        std::vector<size_t> indices(this->shape() -1 );
        for(size_t i=0;i<ans.size();i++)
        {
            size_t temp = i;
            for(size_t j=0;j<indices.size();j++)
            {
                indices[j] = temp;
                temp %= ans.strides[j];
            }
            ans.at(indices) = process_slice(indices);
        }
    }
    return ans;

}

template<typename T>
Matrix<T> Matrix<T>::mean(size_t axis) const{
    Matrix<T> ans = sum(axis);
    ans = ans/static_cast<T>(this->shape[axis]);
    return ans;
}

template<typename T>
Matrix<T> Matrix<T>::max(size_t axis) const {
    if (axis >= shape.size()) 
        throw std::invalid_argument("Axis out of bounds");

    std::vector<size_t> result_shape = shape;
    result_shape.erase(result_shape.begin() + axis);
    Matrix<T> result(result_shape);
    // Initialize result with lowest possible value
    std::fill(result.data.begin(), result.data.end(), 
              std::numeric_limits<T>::lowest());
    const size_t total_slices = result.size();
    const size_t MIN_SLICES_FOR_PARALLEL = 1000;

    auto process_slice = [&](const std::vector<size_t>& base_indices) {
        T max_val = std::numeric_limits<T>::lowest();
        for (size_t i = 0; i < shape[axis]; ++i) {
            auto indices = base_indices;
            indices.insert(indices.begin() + axis, i);
            max_val = std::max(max_val, this->at(indices));
        }
        return max_val;
    };

    if (total_slices >= MIN_SLICES_FOR_PARALLEL) {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t slices_per_thread = total_slices / num_threads;
        auto worker = [&](size_t start_slice, size_t end_slice) {
            std::vector<size_t> indices(shape.size() - 1);
            for (size_t slice = start_slice; slice < end_slice; ++slice) {
                size_t temp = slice;
                for (size_t j = 0; j < indices.size(); ++j) {
                    indices[j] = temp / result.strides[j];
                    temp %= result.strides[j];
                }
                result.at(indices) = process_slice(indices);
            }
        };

        for (size_t i = 0; i < num_threads - 1; ++i) 
            threads.emplace_back(worker, i * slices_per_thread, (i + 1) * slices_per_thread);

        worker((num_threads - 1) * slices_per_thread, total_slices);
        for (auto& thread : threads) 
            thread.join();
        }
     else {
        // Sequential processing for small matrices
        std::vector<size_t> indices(shape.size() - 1);
        for (size_t i = 0; i < result.size(); ++i) {
            size_t temp = i;
            for (size_t j = 0; j < indices.size(); ++j) {
                indices[j] = temp / result.strides[j];
                temp %= result.strides[j];
            }
            result.at(indices) = process_slice(indices);
        }
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::min(size_t axis) const {
    if (axis >= shape.size()) {
        throw std::invalid_argument("Axis out of bounds");
    }

    std::vector<size_t> result_shape = shape;
    result_shape.erase(result_shape.begin() + axis);
    Matrix<T> result(result_shape);

    // Initialize result with highest possible value
    std::fill(result.data.begin(), result.data.end(), 
              std::numeric_limits<T>::max());

    const size_t total_slices = result.size();
    const size_t MIN_SLICES_FOR_PARALLEL = 1000;

    auto process_slice = [&](const std::vector<size_t>& base_indices) {
        T min_val = std::numeric_limits<T>::max();
        for (size_t i = 0; i < shape[axis]; ++i) {
            auto indices = base_indices;
            indices.insert(indices.begin() + axis, i);
            min_val = std::min(min_val, this->at(indices));
        }
        return min_val;
    };

    if (total_slices >= MIN_SLICES_FOR_PARALLEL) {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t slices_per_thread = total_slices / num_threads;

        auto worker = [&](size_t start_slice, size_t end_slice) {
            std::vector<size_t> indices(shape.size() - 1);
            for (size_t slice = start_slice; slice < end_slice; ++slice) {
                // Convert linear index to multi-dimensional indices
                size_t temp = slice;
                for (size_t j = 0; j < indices.size(); ++j) {
                    indices[j] = temp / result.strides[j];
                    temp %= result.strides[j];
                }
                result.at(indices) = process_slice(indices);
            }
        };

        // Launch threads
        for (size_t i = 0; i < num_threads - 1; ++i) {
            threads.emplace_back(worker, 
                               i * slices_per_thread, 
                               (i + 1) * slices_per_thread);
        }
        
        // Process remaining slices in current thread
        worker((num_threads - 1) * slices_per_thread, total_slices);

        // Join threads
        for (auto& thread : threads) {
            thread.join();
        }
    } else {
        // Sequential processing for small matrices
        std::vector<size_t> indices(shape.size() - 1);
        for (size_t i = 0; i < result.size(); ++i) {
            size_t temp = i;
            for (size_t j = 0; j < indices.size(); ++j) {
                indices[j] = temp / result.strides[j];
                temp %= result.strides[j];
            }
            result.at(indices) = process_slice(indices);
        }
    }

    return result;
}

template<typename T>
Matrix<T> Matrix<T>::zeros(const std::vector<size_t>& shape)
{
    Matrix<T> ans(shape);
    std::fill(ans.data.begin(),ans.data.end(),T(0));
    return ans;
}

template<typename T>
Matrix<T> Matrix<T>::ones(const std::vector<size_t>& shape)
{
    Matrix<T> ans(shape);
    std::fill(ans.data.begin(),ans.data.end(),T(1));
    return ans;
}

template<typename T>
Matrix<T> Matrix<T>::eye(size_t n){
    Matrix<T> ans({n,n},T(0));
    for(size_t i=0;i<n;i++)
        ans.at({i,i}) = T(1);
    return ans;
}

template<typename T>
Matrix<T> Matrix<T>::random(const std::vector<size_t>& shape, T minVal,T maxVal){
    Matrix<T> ans(shape);
    const size_t min_sz_for_parallel = 5000;
    std::random_device rd;
    std::mt19937 gen(rd());
    if constexpr(std::is_integral_v<T>)
    {
        std::uniform_int_distribution<T> dist(minVal,maxVal);
        if(ans.data.size() >=min_sz_for_parallel)
        {
            size_t num_threads = std::thread::hardware_concurrency();
            std::vector<std::thread> threads;
            size_t chunk_size = ans.data.size() / num_threads;
            auto worker = [&](size_t start,size_t end)
            {
                std::mt19937 local_gen(rd());
                for(size_t i=start;i<end;i++)
                    ans.data[i] = dist(local_gen);
            };
            for(size_t i=0;i<num_threads-1;i++)
                threads.emplace_back(worker,i*chunk_size,(i+1)*chunk_size);
            worker((num_threads-1)*chunk_size,ans.data.size());
            for(auto& thread:threads)   thread.join();
            
        }
        else
            for(auto& val:ans.data)
                val = dist(gen);
    }
    else
    {
        std::uniform_real_distribution<T> dist(minVal,maxVal);
        if(ans.data.size() >=min_sz_for_parallel)
        {
            size_t num_threads = std::thread::hardware_concurrency();
            std::vector<std::thread> threads;
            size_t chunk_size = ans.data.size() / num_threads;
            auto worker = [&](size_t start,size_t end)
            {
                std::mt19937 local_gen(rd());
                for(size_t i=start;i<end;i++)
                    ans.data[i] = dist(local_gen);
            };
            for(size_t i=0;i<num_threads-1;i++)
                threads.emplace_back(worker,i*chunk_size,(i+1)*chunk_size);
            worker((num_threads-1)*chunk_size,ans.data.size());
            for(auto& thread:threads)   thread.join();
            
        }
        else
            for(auto& val:ans.data)
                val = dist(gen);        
    }
    return ans;
}

template<typename T>
Matrix<T> Matrix<T>::arange(T start,T end,T step){
    if(step == T(0))
        throw std::invalid_argument("Step size cannot be zero");
    size_t size = static_cast<size_t>((end-start)/ step);
    Matrix<T> ans({size});
    const size_t min_sz_for_parallel = 5000;
    if(size >= min_sz_for_parallel)
    {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t chunk_size = size/num_threads;
        auto worker = [&](size_t start_idx,size_t end_idx)
        {
            for(size_t i=start_idx;i<end_idx;i++)
                ans.data[i] = start + step*static_cast<T>(i);
        };
        for(size_t i=0;i<num_threads-1;i++)
            threads.emplace_back(worker,i*chunk_size,(i+1)*chunk_size);
        worker((num_threads-1)*chunk_size,size);
        for(auto& thread:threads)   thread.join();

    }
    else
    {
        for(size_t i=0;i<size;i++)
            ans.data[i] = start + step*static_cast<T>(i);
        
    }
    return ans;
}

template<typename T> 
Matrix<T> Matrix<T>::linspace(T start, T stop, size_t num)
{
    if(num<2)
        throw std::invalid_argument("Number of points must be at least 2");
    Matrix<T> ans({num});
    T step = (stop - start) / static_cast<T>(num-1);
    const size_t min_sz_for_parallel = 5000;
    if(num>=min_sz_for_parallel)
    {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t chunk_size = num / num_threads;
        auto worker = [&](size_t start_idx,size_t end_idx)
        {
            for(size_t i=start_idx;i<end_idx;i++)
                ans.data[i] = start + step* static_cast<T>(i);
        };
        for(size_t i=0;i<num_threads-1;i++)
            threads.emplace_back(worker,i*chunk_size,(i+1)*chunk_size);
        worker((num_threads-1)*chunk_size,num);
        for(auto& thread:threads)
            thread.join();
    }
    else
    {
        for(size_t i=0;i<num;i++)
            ans.data[i] = start + step*static_cast<T>(i);
    }
    ans.data[num-1] = stop;
    return ans;
}

template<typename T>
Matrix<T> Matrix<T>::diag(const Matrix<T>& v,int k)
{
    if(v.shape.size() !=1)
        throw std::invalid_argument("Input matrix must be 1D for diag");
    size_t size = v.data.size();
    size_t n = size+ std::abs(k);
    Matrix<T> ans({n,n},T(0));
    if(k>=0)
    {
        //superdiagonal or the main diagonal, ie top left corner side
        for(size_t i=0;i<size;i++)
            ans.at({i,i+k}) = v.data[i];
    }
    else
        // subdiagonal
        for(size_t i=0;i<size;i++)
            ans.at({i-k,i})=v.data[i];
    return ans;
}

template<typename T>
Matrix<T> Matrix<T>::dot(const Matrix<T>& A) const{
    if(this->shape.size() != 2 || A.shape.size()!=2 || this->shape[1] != A.shape[0])
        throw std::invalid_argument("Invalid Matrix dimensions for dot product");
    size_t M = this->shape[0];
    size_t K  = this->shape[1];
    size_t N = A.shape[1];

    Matrix<T> ans({M,N},T(0));
    const size_t BLOCK_SIZE = 32;
    const size_t min_sz_for_parallel = 128;
    if(M >= min_sz_for_parallel)
    {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        auto worker = [&](size_t start_row,size_t end_row){
            for(size_t i= start_row;i<end_row;i+= BLOCK_SIZE){
                for(size_t j=0;j<N;j+=BLOCK_SIZE){
                    for(size_t k=0;k<K; k+=BLOCK_SIZE){
                        for(size_t ii=i;ii<std::min(i+BLOCK_SIZE,end_row);ii++){
                            for(size_t jj=j;j<std::min(j+BLOCK_SIZE,N);jj++){
                                T sum = T(0);
                                for(size_t kk=k;kk<std::min(k+BLOCK_SIZE,K);kk++)
                                    sum+= this->at({ii,kk}) * A.at({kk,jj});
                                ans.at{ii,jj} += sum;
                            }
                        }
                    }
                }
            }
        };
        size_t rows_per_thread = M / num_threads;
        for(size_t i=0;i<num_threads-1;i++)
            threads.emplace_back(worker,i*rows_per_thread,(i+1)*rows_per_thread);
        worker((num_threads-1)*rows_per_thread,M);
        for(auto& thread: threads)  thread.join();
    }
    else
    {
        for(size_t i= 0;i<M;i+= BLOCK_SIZE){
            for(size_t j=0;j<N;j+=BLOCK_SIZE){
                for(size_t k=0;k<K; k+=BLOCK_SIZE){
                    for(size_t ii=i;ii<std::min(i+BLOCK_SIZE,M);ii++){
                        for(size_t jj=j;j<std::min(j+BLOCK_SIZE,N);jj++){
                            T sum = T(0);
                            for(size_t kk=k;kk<std::min(k+BLOCK_SIZE,K);kk++)
                                sum+= this->at({ii,kk}) * A.at({kk,jj});
                            ans.at{ii,jj} += sum;
                        }
                    }
                }
            }
        }        
    }
    return ans;
}

template<typename T>
T Matrix<T>::determinant() const{
    if(this->shape.size()!= 2 || shape[0]!=shape[1])
        throw std::invalid_argument("Matrix must be a Square for determinant");
    size_t n = this->shape[0];
    if(n==1)    return this->at({0,0});
    else if(n==2)
        return this->at({0,0}) * this->at({1,1}) - this->at({0,1})*this->at({1,0});
    auto LU = LU_decomposition();
    const auto& U = LU[1];
    T det = U.at({0,0});
    for(size_t i=1;i<n;i++)
        det*= U.at({i,j});
    return det;
}

template<typename T>
std::vector<Matrix<T>> Matrix<T>::LU_decomposition() const{
    if (shape.size() != 2 || shape[0] != shape[1]) {
        throw std::invalid_argument("Matrix must be square for LU decomposition");
    }
    size_t n = this->shape[0];
    Matrix<T> L({n,n},T(0));
    Matrix<T> U({n,n}),T(0));
    for(size_t i=0;i<n;i++)
    {
        // upper triangular matrix
        for(size_t k=i;k<n;k++)
        {
            T sum = T(0);
            for(size_t j=0;j<i;j++)
            {
                sum += L.at({i,j}) * U.at({j,k});
            }
            U.at({i,k}) = at({i,k}) - sum;
        }
        L.at({i,i}) = T(1);
        for(size_t k=i+1;k<n;k++)
        {
            T sum = T(0);
            for(size_t j=0;j<i;j++)
                sum += L.at({k,j}) * U.at({j,i});
            if(std::abs(U.at({i,i})) < std::numeric_limits<T>::epsilon())
                throw std::runtime_error("Matrix is singular so LU is not possible");
            L.at({k,i}) = (at({k,i})-sum)/U.at({i,i});
        }
    }
    return {L,U};
}

template<typename T>
Matrix<T> Matrix<T>::inverse() const{
    if (shape.size() != 2 || shape[0] != shape[1]) {
        throw std::invalid_argument("Matrix must be square for inverse calculation");
    }
    size_t n = shape[0];
    auto [L,U] = LU_decomposition();
    Matrix<T> ans({n,n});
    const size_t min_sz_for_parallel = 128;


    if(n >= min_sz_for_parallel)
    {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        auto worker = [&](size_t start_col,size_t end_col){
            for(size_t j=start_col;j<end_col;j++){
                        std::vector<T> b(n,T(0));
        b[j] = T(1);
        Matrix<T> ej({n,1},b);
        Matrix<T> y({n,1},T(0));
        for(size_t i=0;i<n;i++)
        {
            T sum = T(0);
            for(size_t k=0;k<i;k++)
                sum += L.at({i,k}) * y.at({k,0});
            y.at({i,0}) = (ej.at({i,k})-sum) / L.at({i,i});
        }
        Matrix<T> x({n,1,T(0)});
        for(size_t i=n;i>0;i--)
        {
            T sum = T(0);
            for(size_t k=i+1;k<n;k++)
                sum += U.at({i,k}) * x.at({k,0});
            if(std::abs(U.at({i,i}))< std::numeric_limits<T>::epsilon()){
                throw std::runtime_error("Matrix is singular");
            }
            x.at({i,0}) = (y.at({i,0})-sum) / U.at({i,i});
        }

        for(size_t i=0;i<n;i++)
            ans.at({i,j}) = x.at({i,0});
            }
            size_t cols_per_thread = n / num_threads;
            for(size_t i=0;i<num_threads-1;i++)
                threads.emplace_back(worker,i*cols_per_thread,(i+1)*cols_per_thread);
            worker((num_threads-1)*cols_per_thread,n);
            for(auto& thread:threads)   thread.join();
        }
    }
    else
    {
for(size_t j=0;j<n;j++)
    {
        std::vector<T> b(n,T(0));
        b[j] = T(1);
        Matrix<T> ej({n,1},b);
        Matrix<T> y({n,1},T(0));
        for(size_t i=0;i<n;i++)
        {
            T sum = T(0);
            for(size_t k=0;k<i;k++)
                sum += L.at({i,k}) * y.at({k,0});
            y.at({i,0}) = (ej.at({i,k})-sum) / L.at({i,i});
        }
        Matrix<T> x({n,1,T(0)});
        for(size_t i=n;i>0;i--)
        {
            T sum = T(0);
            for(size_t k=i+1;k<n;k++)
                sum += U.at({i,k}) * x.at({k,0});
            if(std::abs(U.at({i,i}))< std::numeric_limits<T>::epsilon()){
                throw std::runtime_error("Matrix is singular");
            }
            x.at({i,0}) = (y.at({i,0})-sum) / U.at({i,i});
        }

        for(size_t i=0;i<n;i++)
            ans.at({i,j}) = x.at({i,0});
    }
    }
    
    return ans;
}

template<typename T>
Matrix<T> Matrix<T>::normalize(T epsilon) const{
    T min_val = this->min();
    T max_val = this->max();
    T range = max_val - min_val;
    if(std::abs(range) < epsilon)
        throw std::runtime_error("Range is too small for normalization");
    Matrix<T> ans(this->shape);
    const size_t min_sz_for_parallel = 5000;
    if(this->data.size() >= min_sz_for_parallel){
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t chunk_size = this->data.size() / num_threads;
        auto worker = [&](size_t start,size_t end)
        {
            for(size_t i = start,i<end;i++)
                ans.data[i] = (this->data[i] - min_val) / range;
        };
                for (size_t i = 0; i < num_threads - 1; ++i) {
            threads.emplace_back(worker, i * chunk_size, (i + 1) * chunk_size);
        }
        worker((num_threads - 1) * chunk_size, this->data.size());

        for (auto& thread : threads) {
            thread.join();
        }
    }
    else
    {
        std::transform(data.begin(), data.end(), ans.data.begin(),
                      [min_val, range](const T& x) { 
                          return (x - min_val) / range; 
                      });        
    }
    return ans;
}

template<typename T>
T Matrix<T>::correlation(const Matrix<T>& A,const Matrix<T>& B){
    if(A.shape != B.shape)
        throw std::invalid_argument("Matrices must have same shape for correlation ");
    T mean1 = A.mean();
    T mean2 = B.mean();

    T numerator = T(0);
    T denom1 = T(0);
    T denom2 = T(0);
    const size_t min_sz_for_parallel = 5000;
    if(A.data.size() >= min_sz_for_parallel)
    {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        std::vector<std::tuple<T, T, T>> partial_sums(num_threads);
        size_t chunk_size = A.data.size() / num_threads;
        auto worker = [&](size_t start,size_t end,std::tuple<T,T,T>& result){
            T local_num =T(0),local_d1 = T(0),local_d2 = T(0);
            for(size_t i=start;i<end;i++)
            {
                T diff1 = A.data[i] - mean1;
                T diff2 = B.data[i] - mean2;
                local_num += diff1 * diff2;
                local_d1 += diff1*diff1;
                local_d2 += diff2*diff2;
            }
            result = {local_num,local_d1,local_d2};
        };
        for(size_t i=0;i<num_threads-1;i++)
            threads.emplace_back(worker,i*chunk_size,(i+1)*chunk_size,std::ref(partial_sums[i]));
        worker((num_threads-1)*chunk_size,A.data.size(),partial_sums[num_threads-1]);
        for(auto& thread:threads)
            thread.join();
        for(const auto& [num,d1,d2]:partial_sums){
            numerator += num;
            denom1+= d1;
            denom2 += d2;
        }


    }
    else
    {
        for(size_t i=0;i<A.data.size();i++)
        {
            T diff1 = A.data[i] - mean1;
            T diff2 = B.data[i] - mean2;
            numerator += diff1*diff2;
            denom1 += diff1*diff1;
            denom2 += diff2*diff2;
        }

    }
    T denominator = std::sqrt(denom1*denom2);
    if(denominator < std::numeric_limits<T>::epsilon())
        throw std::runtime_error("cannot compute correlation : zero variance");
    return numerator/denominator;
}

