#pragma once


#include <vector>
#include "distribution.hpp"


namespace mkt {

template<typename T>
class DMatrix {
 public:

  // CONSTRUCTORS / DESTRUCTOR
  DMatrix(int pid, int number_of_rows, int number_of_columns, int number_of_rows_local, 
          int number_of_columns_local, int size, int size_local, T init_value, 
          int partitions_in_row, int partitions_in_col, int partition_x_pos, int partition_y_pos, 
          int row_offset, int col_offset, Distribution d = DIST);



  T get_global(int row, int col) const;
  void set_global(int row, int col, const T& value);

  T get_local(int row, int col) const;
  void set_local(int row, int col, const T& value);

  int get_size() const;
  int get_size_local() const;

  int get_row_offset() const;
  int get_col_offset() const;

  int get_number_of_rows_local() const;
  int get_number_of_cols_local() const;

 private:

  // position of processor in data parallel group of processors; zero-base
  int _pid;
  // number of rows
  int _number_of_rows;
  // number of columns
  int _number_of_columns;
  // number of local rows
  int _number_of_rows_local;
  // number of local columns
  int _number_of_columns_local;

  int _size;
  int _size_local;

  // number of (local) partitions per row
  int _partitions_in_row;
  // number of (local) partitions per column
  int _partitions_in_col;

  /*
  *               |
  *     X=0, Y=0  | X=0, Y=1
  * X   __________|_________
  *               |
  *     X=1, Y=0  | X=1, Y=1
  *               |
  * 
  *             Y
  * 
  * */


  // X position of processor in data parallel group of processors
  int _partition_x_pos;
  // Y position of processor in data parallel group of processors
  int _partition_y_pos;

  // first row index in local partition; assuming division mode: block
  int _row_offset;
  // first column index in local partition; assuming division mode: block
  int _col_offset;

  // checks whether data is copy distributed among all processes
  Distribution _dist;

  std::vector<T> _data;
};

}  // namespace mkt

template<typename T>
mkt::DMatrix<T>::DMatrix(int pid, int number_of_rows, int number_of_columns, int number_of_rows_local, 
                         int number_of_columns_local, int size, int size_local, T init_value, 
                         int partitions_in_row, int partitions_in_col, int partition_x_pos, int partition_y_pos, 
                         int row_offset, int col_offset, Distribution d)
    : _pid(pid),
      _number_of_rows(number_of_rows),
      _number_of_columns(number_of_columns),
      _number_of_rows_local(number_of_rows_local),
      _number_of_columns_local(number_of_columns_local),
      _size(size),
      _size_local(size_local),
      _partitions_in_row(partitions_in_row),
      _partitions_in_col(partitions_in_col),
      _partition_x_pos(partition_y_pos),
      _partition_y_pos(partition_y_pos),
      _row_offset(row_offset),
      _col_offset(col_offset),
      _dist(Distribution::DIST),
      _data(size_local, init_value) {
}

template<typename T>
T mkt::DMatrix<T>::get_local(int row, int col) const {
  return _data[row * _number_of_columns_local + col];
}

template<typename T>
void mkt::DMatrix<T>::set_local(int row, int col, const T& v) {
  _data[row * _number_of_columns_local + col] = v;
}

template<typename T>
T mkt::DMatrix<T>::get_global(int row, int col) const {
  // TODO
}

template<typename T>
void mkt::DMatrix<T>::set_global(int row, int col, const T& v) {
  // TODO
}

template<typename T>
int mkt::DMatrix<T>::get_size() const {
  return _size;
}

template<typename T>
int mkt::DMatrix<T>::get_size_local() const {
  return _size_local;
}

template<typename T>
int mkt::DMatrix<T>::get_row_offset() const {
  return _row_offset;
}

template<typename T>
int mkt::DMatrix<T>::get_col_offset() const {
  return _col_offset;
}

template<typename T>
int mkt::DMatrix<T>::get_number_of_rows_local() const {
  return _number_of_rows_local;
}

template<typename T>
int mkt::DMatrix<T>::get_number_of_columns_local() const {
  return number_of_columns_local;
}

