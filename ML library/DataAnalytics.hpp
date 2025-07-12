#ifndef DATA_ANALYTICS_HPP
#define DATA_ANALYTICS_HPP

#include<string>
#include<vector>
#include<map>
#include<unordered_map>
#include<unordered_set>
#include<algorithm>
#include<memory>
#include "matrix.hpp"

template<typename T>
class DataFrame{
private:
    std::unordered_map<std::string,std::vector<T>> data;
    // bi derectional index mapping
    std::unordered_map<size_t,std::string> index_to_column;
    std::unordered_map<std::string,size_t> column_to_index;
    size_t num_rows;
    size_t next_index;

public:
    //constructors
    DataFrame();
    DataFrame(const std::map<std::string,std::vector<T>>& data);
    DataFrame(const std::vector<std::string>& columns);

    //basic operations
    void addColumn(const std::string& name,const std::vector<T>& values);
    void addRow(const std::vector<T>& values);
    void removeColumn(const std::string& name);
    void removeRow(size_t index);
    void renameColumn(const std::string& old_name,const std::string& new_name);

    // access methods
    std::vector<T>& operator[](const std::string& column);
    const std::vector<T>& operator[](const std::string& columns) const;  //might also implement [][] operator
    DataFrame<T> head(size_t n = 5) const;
    DataFrame<T> tail(size_t n = 5) const;

    //selection and filtering
    DataFrame<T> select(const std::vector<std::string>& columns) const;
    DataFrame<T> filter(std::function<bool(const std::vector<T>&)> predicate) const;
    DataFrame<T> slice(size_t start,size_t end) const;

    //stats
    std::map<std::string,T> mean() const;
    std::map<std::string,T> median() const;
    std::map<std::string,T> stddev() const;
    std::map<std::string,T> variance() const;
    std::map<std::string,T> min() const;
    std::map<std::string,T> max() const;
    std::map<std::string,std::map<T,size_t>> valueCounts() const;

    //grouping and aggregation and data manipulations
    DataFrame<T> groupBy(const std::string& column, std::function<T(const std::vector<T>&)> aff_func) const;
    DataFrame<T> pivot(const std::string& index,const std::string& columns,const std::string& values) const;
    DataFrame<T> sort(const std::string& column,bool ascending = true) const;
    DataFrame<T> merge(const DataFrame<T>& other,const std::string& on,const std::string& how = "inner") const;
    DataFrame<T> join(const DataFrame<T>& other, const std::string& left_on,const std::string& right_on,const std::string& how = "inner") const; 
    
    //missing value handling
    void fillNA(const T& value);
    void dropNA();
    std::map<std::string,size_t> isNA() const;

    //data transformations
    DataFrame<T> apply(const std::string& column,std::function<T(const T&)> func) const;
    DataFrame<T> applyAll(std::function<T(const T&)> func) const;

    //Matrix operations
    Matrix<T> toMatrix() const;
    static DataFrame<T> fromMatrix(const Matrix<T>& matrix,const std::vector<std::string>& column_names);

    //IO operations
    void readCSV(const std::string& filename);
    void writeCSV(const std::string& filename) const;
    void readJSON(const std::string& filename);
    void writeJSON(const std::string& filename) const;

    size_t rows() const {return num_rows;}
    size_t cols() const {return next_index;}
    std::vector<std::string> columns() const {
        std::vector<std::string> result;
        result.reserve(next_index);
        for(size_t i = 0; i < next_index; i++) {
            result.push_back(index_to_column.at(i));
        }
        return result;
    }
    void info() const;
    std::string describe() const;

    void interpolate(const std::string& method = "linear");
    DataFrame<T> normalize(const std::string& method = "minmax") const;
    DataFrame<T> standardize() const;
    DataFrame<T> oneHotEncode(const std::string& column) const;

    DataFrame<T> unique() const;
    DataFrame<T> drop_duplicates(const std::vector<std::string>& subset = {}) const;
    static DataFrame<T> concat(const std::vector<DataFrame<T>>& frames,const std::string& axis = "0");
    
};


#endif