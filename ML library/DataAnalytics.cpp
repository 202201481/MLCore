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
            for(size_t i=0;i<num_rows,i++){
                for(size_t j=0;j<next_index;j++)
                    thread_row[j] = data.at(index_to_column.at(j))[i];
                keep_rows[i] = predicate(thread_row);
                result_size += keep_rows[i] ? 1 ; 0;
            }
        }
    }
    else
    {
        for(size_t i=0;i<num_rows;i++){
            for(size_t j=0;j<next_index;j++){
                row_data[j] = data.at(index_to_column.at(j))[i];
            }
            keep_rows[j] = predicate(row_data);
            result_size += keep_rows[i] ? 1:0;
        }
    }
    
}