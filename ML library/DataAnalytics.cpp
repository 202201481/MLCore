#include "DataAnalytics.hpp"
template<typename T>
DataFrame<T>::DataFrame() : num_rows(0), next_index(0){}

template<typename T>
DataFrame<T>::DataFrame(const std::map<std::string,std::vector<T>>& input_data){
    if(input_data.empty())
        throw std::invalid_argument("Cannot create datafrom from empty data");

    num_rows = input_data.begin()->second.size();
    next_index = 0;
    for(const auto& [column_name,column_data] : input_data){
        if(column_data.size() != num_rows){
            throw std::invalid_argument("Column '"  + column_name +"' has inconsistent lenght. " + "Expected: "+std::to_string(num_rows) + ", Got: " +std::to_string(column_data.size()));
        }
        if(column_name.empty())
            throw std::invalid_argument("Column names cannot be empty");
        if(column_to_index.find(column_name) != column_to_index.end()) {
            throw std::invalid_argument("Duplicate column name found: " + column_name);
        }
        for(const auto& value:column_data){
            if constexpr (std::is_arithmetic_v<T>){
                if (!std::isfinite(static_cast<double>(value))){
                    throw std::invalid_argument("Invalid numeric value in column: "+column_name);
                }
            }
        }

        data[column_name] = column_data;
        column_to_index[column_name] = next_index;
        index_to_column[next_index] = column_name;
        next_index++;
    }
    //reserve memeory for future use
    for(auto& [_,vec]:data){
        vec.reserve(vec.size()*1.5);
    }


}

template<typename T>
DataFrame<T>::DataFrame(const std::vector<std::string>& columns) : num_rows(0),next_index(0) {
    if(columns.empty())
        throw std::invalid_argument("cannot create dataframe with empty columns");
    for(const auto& column : columns) {
        if(column.empty()) {
            throw std::invalid_argument("Column names cannot be empty");
        }
        if(column_to_index.find(column) != column_to_index.end()) {
            throw std::invalid_argument("Duplicate column name found: " + column);
        }

        data[column] = std::vector<T>();
        data[column].reserve(100);  // Initial capacity
        column_to_index[column] = next_index;
        index_to_column[next_index] = column;
        next_index++;
    }

}

template<typename T>
void DataFrame<T>::addColumn(const std::string& name, const std::vector<T>& values){
    if(name.empty()) {
        throw std::invalid_argument("Column name cannot be empty");
    }
    if(column_to_index.find(name) != column_to_index.end()) {
        throw std::invalid_argument("Column '" + name + "' already exists");
    }
    if(num_rows > 0 && values.size() != num_rows) {
        throw std::invalid_argument("Column length mismatch. Expected: " + std::to_string(num_rows) +", Got: " + std::to_string(values.size()));
    }
    data[name] = values;
    column_to_index[name] = next_index;
    index_to_column[next_index] = name;
    next_index++;

    if(num_rows == 0) {
        num_rows = values.size();
    }
}

template<typename T>
void DataFrame<T>::addRow(const std::vector<T>& values){
    if(values.size() != data.size()) 
        throw std::invalid_argument("Row length mismatch. Expected: " + std::to_string(data.size()) +", Got: " + std::to_string(values.size()));

    // Add values in column order using index mapping
    for(size_t i = 0; i < next_index; ++i) {
        const std::string& col_name = index_to_column[i];
        data[col_name].push_back(values[i]);
    }
    num_rows++;
}

template<typename T>
void DataFrame<T>::removeColumn(const std::string& name){
    auto col_it = column_to_index.find(name);
    if(col_it == column_to_index.end()) 
        throw std::invalid_argument("Column '" + name + "' not found");
    size_t idx = col_it->second;
    data.erase(name);
    column_to_index.erase(name);
    index_to_column.erase(idx);

    // Update indices for columns after the removed one
    for(auto& [col, index] : column_to_index) {
        if(index > idx) {
            index--;
            index_to_column[index] = col;
        }
    }
    next_index--;
    if(data.empty()) 
        num_rows = 0;
}

template<typename T>
void DataFrame<T>::removeRow(size_t index){
    if(index >= num_rows) 
        throw std::out_of_range("Row index " + std::to_string(index) + " out of range. DataFrame has " + std::to_string(num_rows) + " rows");

    for(size_t i = 0; i < next_index; ++i) {
        const std::string& col_name = index_to_column[i];
        data[col_name].erase(data[col_name].begin() + index);
    }
    num_rows--;
}

template<typename T>
void DataFrame<T>::renameColumn(const std::string& old_name,const std::string& new_name){
    if(old_name == new_name)
        return;
    if(new_name.empty()) 
        throw std::invalid_argument("New column name cannot be empty");

    if(column_to_index.find(new_name) != column_to_index.end()) 
        throw std::invalid_argument("Column '" + new_name + "' already exists");

    auto old_it = column_to_index.find(old_name);
    if(old_it == column_to_index.end()) 
        throw std::invalid_argument("Column '" + old_name + "' not found");

    size_t idx = old_it->second;
    data[new_name] = std::move(data[old_name]);
    data.erase(old_name);
    column_to_index.erase(old_name);
    column_to_index[new_name] = idx;
    index_to_column[idx] = new_name;
}

template<typename T>
std::vector<T>& DataFrame<T>::operator[](const std::string& column){
    auto it = column_to_index.find(column);
    if(it == column_to_index.end())
        throw std::out_of_range("Column '" + column + "' not found");
    return data[column];
}
template<typename T>
const std::vector<T>& DataFrame<T>::operator[](const std::string& column) const {
    auto it = column_to_index.find(column);
    if (it == column_to_index.end()) 
        throw std::out_of_range("Column '" + column + "' not found");
    return data.at(column);
}

template<typename T>
DataFrame<T> DataFrame<T>::head(size_t n) const {
    if (num_rows == 0) return DataFrame<T>();

    n = std::min(n, num_rows);
    std::map<std::string, std::vector<T>> result_data;

    const size_t PARALLEL_THRESHOLD = 5000;
    if (n >= PARALLEL_THRESHOLD && next_index >= 4) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < next_index; ++i) {
            const std::string& col_name = index_to_column.at(i);
            const auto& src = data.at(col_name);
            std::vector<T> temp(n);
            std::copy_n(src.begin(), n, temp.begin());
            #pragma omp critical
            result_data[col_name] = std::move(temp);
        }
    } else {
        for (size_t i = 0; i < next_index; ++i) {
            const std::string& col_name = index_to_column.at(i);
            const auto& src = data.at(col_name);
            result_data[col_name].assign(src.begin(), src.begin() + n);
        }
    }

    return DataFrame<T>(result_data);
}

template<typename T>
DataFrame<T> DataFrame<T>::tail(size_t n) const {
    if (num_rows == 0) return DataFrame<T>();

    n = std::min(n, num_rows);
    size_t start = num_rows - n;
    std::map<std::string, std::vector<T>> result_data;

    const size_t PARALLEL_THRESHOLD = 5000;
    if (n >= PARALLEL_THRESHOLD && next_index >= 4) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < next_index; ++i) {
            const std::string& col_name = index_to_column.at(i);
            const auto& src = data.at(col_name);
            std::vector<T> temp(n);
            std::copy_n(src.begin() + start, n, temp.begin());
            #pragma omp critical
            result_data[col_name] = std::move(temp);
        }
    } else {
        for (size_t i = 0; i < next_index; ++i) {
            const std::string& col_name = index_to_column.at(i);
            const auto& src = data.at(col_name);
            result_data[col_name].assign(src.begin() + start, src.end());
        }
    }

    return DataFrame<T>(result_data);
}

template<typename T>
DataFrame<T> DataFrame<T>::select(const std::vector<std::string>& columns) const {
    if (columns.empty()) {
        throw std::invalid_argument("No columns specified for selection");
    }

    std::map<std::string, std::vector<T>> result_data;
    for (const auto& col : columns) {
        auto it = column_to_index.find(col);
        if (it == column_to_index.end()) {
            throw std::invalid_argument("Column not found: " + col);
        }
        result_data[col] = data.at(col);
    }

    return DataFrame<T>(result_data);
}

template<typename T>
DataFrame<T> DataFrame<T>::slice(size_t start,size_t end) const{
    if(start >= num_rows || end > num_rows || start >= end)
        throw std::invalid_argument("Invalid slice range [" + std::to_string(start) + ", " + std::to_string(end) + "]");

    std::map<std::string,std::vector<T>> result_data;
    for(size_t i=0;i<next_index;i++){
        const std::string& col_name = index_to_column.at(i);
        const auto& src = this->data.at(col_name);
        result_data[col_name] = std::vector<T>(src.begin()+start,src.begin()+end);
    }
    return DataFrame<T>(result_data);
}

template<typename T>
DataFrame<T> DataFrame<T>::filter(std::function<bool(const std::vector<T>&)> predicate) const{
    if(num_rows == 0)   return DataFrame<T>();
    if(!predicate)
        throw std::invalid_argument("Invalid predicate function");
    
    // evaluate predicates and mark the rows to keep
    std::vector<bool> keep_rows(num_rows);
    size_t result_size = 0;
    std::vector<T> row_data(next_index);

    const size_t parallel_th  = 10000;
    if(num_rows >= parallel_th){
        #pragma omp parallel
        {
            std::vector<T> thread_row(next_index);
            #pragma omp for reduction(+:result_size)
            for(size_t i=0;i<num_rows;++i){
                for(size_t j=0;j<next_index;j++)
                    thread_row[j] = data.at(index_to_column.at(j))[i];
                keep_rows[i] = predicate(thread_row);
                result_size += keep_rows[i] ? 1 : 0;
            }
        }
    }
    else
    {
        for(size_t i=0;i<num_rows;i++){
            for(size_t j=0;j<next_index;j++){
                row_data[j] = data.at(index_to_column.at(j))[i];
            }
            keep_rows[i] = predicate(row_data);
            result_size += keep_rows[i] ? 1:0;
        }
    }
    
    // Create result DataFrame with filtered rows
    std::map<std::string, std::vector<T>> result_data;
    for(size_t i = 0; i < next_index; i++) {
        const std::string& col_name = index_to_column.at(i);
        result_data[col_name].reserve(result_size);
    }
    
    for(size_t i = 0; i < num_rows; i++) {
        if(keep_rows[i]) {
            for(size_t j = 0; j < next_index; j++) {
                const std::string& col_name = index_to_column.at(j);
                result_data[col_name].push_back(data.at(col_name)[i]);
            }
        }
    }
    
    return DataFrame<T>(result_data);
}

template<typename T>
void DataFrame<T>::dropNA() {
    // Remove this implementation for string types to avoid arithmetic issues
    return;
}

template<typename T>
void DataFrame<T>::fillNA(const T& value) {
    for(auto& [col_name, col_data] : data) {
        for(auto& val : col_data) {
            if constexpr (std::is_floating_point_v<T>) {
                if(std::isnan(val)) val = value;
            }
        }
    }
}

template<typename T>
std::map<std::string, size_t> DataFrame<T>::isNA() const {
    std::map<std::string, size_t> na_counts;
    for(size_t i = 0; i < next_index; i++) {
        const std::string& col_name = index_to_column.at(i);
        size_t count = 0;
        for(const auto& val : data.at(col_name)) {
            if constexpr (std::is_floating_point_v<T>) {
                if(std::isnan(val)) count++;
            }
        }
        na_counts[col_name] = count;
    }
    return na_counts;
}

template<typename T>
std::map<std::string, T> DataFrame<T>::mean() const {
    std::map<std::string, T> means;
    
    // Skip string types for arithmetic operations
    if constexpr (!std::is_arithmetic_v<T>) {
        return means;
    }
    
    for(size_t i = 0; i < next_index; i++) {
        const std::string& col_name = index_to_column.at(i);
        const auto& col_data = data.at(col_name);
        T sum = T{};
        size_t valid_count = 0;
        
        for(const auto& val : col_data) {
            if constexpr (std::is_floating_point_v<T>) {
                if(!std::isnan(val)) {
                    sum += val;
                    valid_count++;
                }
            } else {
                sum += val;
                valid_count++;
            }
        }
        
        means[col_name] = valid_count > 0 ? sum / static_cast<T>(valid_count) : T{};
    }
    return means;
}

template<typename T>
std::map<std::string, T> DataFrame<T>::median() const {
    std::map<std::string, T> medians;
    
    // Skip string types for arithmetic operations
    if constexpr (!std::is_arithmetic_v<T>) {
        return medians;
    }
    
    for(size_t i = 0; i < next_index; i++) {
        const std::string& col_name = index_to_column.at(i);
        auto col_data = data.at(col_name);
        
        // Remove NaN values
        if constexpr (std::is_floating_point_v<T>) {
            col_data.erase(std::remove_if(col_data.begin(), col_data.end(), 
                [](const T& val) { return std::isnan(val); }), col_data.end());
        }
        
        if(col_data.empty()) {
            medians[col_name] = T{};
            continue;
        }
        
        std::sort(col_data.begin(), col_data.end());
        size_t size = col_data.size();
        
        if(size % 2 == 0) {
            medians[col_name] = (col_data[size/2 - 1] + col_data[size/2]) / static_cast<T>(2);
        } else {
            medians[col_name] = col_data[size/2];
        }
    }
    return medians;
}

template<typename T>
std::map<std::string, T> DataFrame<T>::stddev() const {
    std::map<std::string, T> stddevs;
    
    // Skip string types for arithmetic operations  
    if constexpr (!std::is_arithmetic_v<T>) {
        return stddevs;
    }
    
    auto means = mean();
    
    for(size_t i = 0; i < next_index; i++) {
        const std::string& col_name = index_to_column.at(i);
        const auto& col_data = data.at(col_name);
        T mean_val = means[col_name];
        T sum_sq_diff = T{};
        size_t valid_count = 0;
        
        for(const auto& val : col_data) {
            if constexpr (std::is_floating_point_v<T>) {
                if(!std::isnan(val)) {
                    T diff = val - mean_val;
                    sum_sq_diff += diff * diff;
                    valid_count++;
                }
            } else {
                T diff = val - mean_val;
                sum_sq_diff += diff * diff;
                valid_count++;
            }
        }
        
        stddevs[col_name] = valid_count > 1 ? 
            std::sqrt(sum_sq_diff / static_cast<T>(valid_count - 1)) : T{};
    }
    return stddevs;
}

template<typename T>
std::map<std::string, T> DataFrame<T>::variance() const {
    auto stddevs = stddev();
    std::map<std::string, T> variances;
    for(const auto& [col, stddev_val] : stddevs) {
        variances[col] = stddev_val * stddev_val;
    }
    return variances;
}

template<typename T>
std::map<std::string, T> DataFrame<T>::min() const {
    std::map<std::string, T> mins;
    for(size_t i = 0; i < next_index; i++) {
        const std::string& col_name = index_to_column.at(i);
        const auto& col_data = data.at(col_name);
        
        if(col_data.empty()) {
            mins[col_name] = T{};
            continue;
        }
        
        T min_val = col_data[0];
        for(const auto& val : col_data) {
            if constexpr (std::is_floating_point_v<T>) {
                if(!std::isnan(val) && val < min_val) min_val = val;
            } else {
                if(val < min_val) min_val = val;
            }
        }
        mins[col_name] = min_val;
    }
    return mins;
}

template<typename T>
std::map<std::string, T> DataFrame<T>::max() const {
    std::map<std::string, T> maxs;
    for(size_t i = 0; i < next_index; i++) {
        const std::string& col_name = index_to_column.at(i);
        const auto& col_data = data.at(col_name);
        
        if(col_data.empty()) {
            maxs[col_name] = T{};
            continue;
        }
        
        T max_val = col_data[0];
        for(const auto& val : col_data) {
            if constexpr (std::is_floating_point_v<T>) {
                if(!std::isnan(val) && val > max_val) max_val = val;
            } else {
                if(val > max_val) max_val = val;
            }
        }
        maxs[col_name] = max_val;
    }
    return maxs;
}

template<typename T>
Matrix<T> DataFrame<T>::toMatrix() const {
    if(num_rows == 0 || next_index == 0) {
        return Matrix<T>(std::vector<size_t>{0, 0});
    }
    
    Matrix<T> result(std::vector<size_t>{num_rows, next_index});
    
    for(size_t i = 0; i < num_rows; i++) {
        for(size_t j = 0; j < next_index; j++) {
            const std::string& col_name = index_to_column.at(j);
            result.at({i, j}) = data.at(col_name)[i];
        }
    }
    
    return result;
}

template<typename T>
DataFrame<T> DataFrame<T>::fromMatrix(const Matrix<T>& matrix, const std::vector<std::string>& column_names) {
    auto shape = matrix.getShape();
    if(shape.size() != 2) {
        throw std::invalid_argument("Matrix must be 2D");
    }
    
    size_t rows = shape[0];
    size_t cols = shape[1];
    
    if(column_names.size() != cols) {
        throw std::invalid_argument("Number of column names must match matrix columns");
    }
    
    std::map<std::string, std::vector<T>> data_map;
    for(size_t j = 0; j < cols; j++) {
        const std::string& col_name = column_names[j];
        data_map[col_name].reserve(rows);
        for(size_t i = 0; i < rows; i++) {
            data_map[col_name].push_back(matrix.at({i, j}));
        }
    }
    
    return DataFrame<T>(data_map);
}

template<typename T>
DataFrame<T> DataFrame<T>::sort(const std::string& column, bool ascending) const {
    auto it = column_to_index.find(column);
    if(it == column_to_index.end()) {
        throw std::invalid_argument("Column not found: " + column);
    }
    
    // Create indices vector
    std::vector<size_t> indices(num_rows);
    std::iota(indices.begin(), indices.end(), 0);
    
    // Sort indices based on column values
    const auto& sort_col = data.at(column);
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        if(ascending) return sort_col[a] < sort_col[b];
        else return sort_col[a] > sort_col[b];
    });
    
    // Build result
    std::map<std::string, std::vector<T>> result_data;
    for(size_t i = 0; i < next_index; i++) {
        const std::string& col_name = index_to_column.at(i);
        result_data[col_name].reserve(num_rows);
        const auto& src_col = data.at(col_name);
        for(size_t idx : indices) {
            result_data[col_name].push_back(src_col[idx]);
        }
    }
    
    return DataFrame<T>(result_data);
}

template<typename T>
void DataFrame<T>::info() const {
    std::cout << "DataFrame Info:\n";
    std::cout << "Rows: " << num_rows << "\n";
    std::cout << "Columns: " << next_index << "\n";
    std::cout << "Column Names: ";
    for(size_t i = 0; i < next_index; i++) {
        std::cout << index_to_column.at(i);
        if(i < next_index - 1) std::cout << ", ";
    }
    std::cout << "\n";
}

// Template instantiations
template class DataFrame<int>;
template class DataFrame<float>;
template class DataFrame<double>;