#pragma once
#include <string>
#include "matmult-n-4-g-1.hpp"

namespace mkt {
enum Distribution {DIST, COPY};

template<typename T>
class DMatrix {
 public:

  // CONSTRUCTORS / DESTRUCTOR
  DMatrix(int pid, int number_of_rows, int number_of_columns, int number_of_rows_local, 
          int number_of_columns_local, int size, int size_local, T init_value, 
          int partitions_in_row, int partitions_in_column, int partition_x_pos, int partition_y_pos, 
          int row_offset, int column_offset, Distribution d = DIST);

// Getter and Setter

  T get_global(int row, int column) const;
  void set_global(int row, int column, const T& value);

  T get_local(int row, int column) const;
  T get_local(int index) const;
  void set_local(int row, int column, const T& value);
  void set_local(int index, const T& value);

  T& operator[](int local_index);
  const T& operator[](int local_index) const;

  int get_size() const;
  int get_size_local() const;

  int get_row_offset() const;
  int get_column_offset() const;

  int get_number_of_rows() const;
  int get_number_of_columns() const;

  int get_number_of_rows_local() const;
  int get_number_of_columns_local() const;
  
  int get_partitions_in_row() const;
  int get_partitions_in_column() const;
  
  int get_partition_x_pos() const;
  int get_partition_y_pos() const;
  
  Distribution get_distribution() const;

  T* get_data();
  const T* get_data() const;
  
  auto begin() noexcept;
  auto end() noexcept;
  
  auto begin() const noexcept;
  auto end() const noexcept;

 private:

  //
  // Attributes
  //

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
  int _partitions_in_column;

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
  int _column_offset;

  // checks whether data is copy distributed among all processes
  Distribution _dist;

  std::vector<T> _data;
};
template<typename T, typename R, typename Functor>
void map(const mkt::DMatrix<T>& in, mkt::DMatrix<R>& out, const Functor& f);

template<typename T, typename R, typename Functor>
void map_index(const mkt::DMatrix<T>& in, mkt::DMatrix<R>& out, const Functor& f);

template<typename T, typename R, typename Functor>
void map_local_index(const mkt::DMatrix<T>& in, mkt::DMatrix<R>& out, const Functor& f);

template<typename T, typename Functor>
void map_in_place(mkt::DMatrix<T>& m, const Functor& f);

template<typename T, typename Functor>
void map_index_in_place(mkt::DMatrix<T>& m, const Functor& f);

template<typename T, typename Functor>
void map_local_index_in_place(mkt::DMatrix<T>& m, const Functor& f);

template<typename T, typename Functor>
void fold(const mkt::DMatrix<T>& m, T& out, const T identity, const Functor& f);

template<typename T, typename Functor>
void fold_copy(const mkt::DMatrix<T>& m, T& out, const T identity, const Functor& f);

template<typename T, typename R, typename MapFunctor, typename FoldFunctor>
void map_fold(const mkt::DMatrix<T>& m, T& out, const MapFunctor& f_map, const R& identity, const FoldFunctor& f_fold);

template<typename T, typename R, typename MapFunctor, typename FoldFunctor>
void map_fold_copy(const mkt::DMatrix<T>& m, T& out, const MapFunctor& f_map, const R& identity, const FoldFunctor& f_fold);


template<typename T>
void print(std::ostringstream& stream, const T& a);


	
template<typename T>
void gather(const mkt::DMatrix<T>& in, mkt::DMatrix<T>& out, const MPI_Datatype& dt);
	
template<typename T>
void scatter(const mkt::DMatrix<T>& in, mkt::DMatrix<T>& out);
template<typename T, typename Functor>
void shift_partitions_horizontally(mkt::DMatrix<T>& m, const Functor& f);

template<typename T, typename Functor>
void shift_partitions_vertically(mkt::DMatrix<T>& m, const Functor& f);

} // namespace mkt


		
template<typename T>
mkt::DMatrix<T>::DMatrix(int pid, int number_of_rows, int number_of_columns, int number_of_rows_local, 
                         int number_of_columns_local, int size, int size_local, T init_value, 
                         int partitions_in_row, int partitions_in_column, int partition_x_pos, int partition_y_pos, 
                         int row_offset, int column_offset, Distribution d)
    : _pid(pid),
      _number_of_rows(number_of_rows),
      _number_of_columns(number_of_columns),
      _number_of_rows_local(number_of_rows_local),
      _number_of_columns_local(number_of_columns_local),
      _size(size),
      _size_local(size_local),
      _partitions_in_row(partitions_in_row),
      _partitions_in_column(partitions_in_column),
      _partition_x_pos(partition_x_pos),
      _partition_y_pos(partition_y_pos),
      _row_offset(row_offset),
      _column_offset(column_offset),
      _dist(d),
      _data(size_local, init_value) {
}

template<typename T>
T mkt::DMatrix<T>::get_local(int row, int column) const {
  return _data[row * _number_of_columns_local + column];
}

template<typename T>
T mkt::DMatrix<T>::get_local(int index) const {
  return _data[index];
}

template<typename T>
void mkt::DMatrix<T>::set_local(int row, int column, const T& v) {
  _data[row * _number_of_columns_local + column] = v;
}

template<typename T>
void mkt::DMatrix<T>::set_local(int index, const T& v) {
  _data[index] = v;
}

template<typename T>
T mkt::DMatrix<T>::get_global(int row, int column) const {
  // TODO
}

template<typename T>
void mkt::DMatrix<T>::set_global(int row, int column, const T& v) {
  // TODO
}

template<typename T>
T& mkt::DMatrix<T>::operator[](int local_index) {
  return _data[local_index];
}

template<typename T>
const T& mkt::DMatrix<T>::operator[](int local_index) const {
  return _data[local_index];
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
int mkt::DMatrix<T>::get_column_offset() const {
  return _column_offset;
}

template<typename T>
int mkt::DMatrix<T>::get_number_of_rows() const {
  return _number_of_rows;
}

template<typename T>
int mkt::DMatrix<T>::get_number_of_columns() const {
  return _number_of_columns;
}

template<typename T>
int mkt::DMatrix<T>::get_number_of_rows_local() const {
  return _number_of_rows_local;
}

template<typename T>
int mkt::DMatrix<T>::get_number_of_columns_local() const {
  return _number_of_columns_local;
}

template<typename T>
int mkt::DMatrix<T>::get_partitions_in_row() const {
  return _partitions_in_row;
}

template<typename T>
int mkt::DMatrix<T>::get_partitions_in_column() const {
  return _partitions_in_column;
}

template<typename T>
int mkt::DMatrix<T>::get_partition_x_pos() const {
  return _partition_x_pos;
}

template<typename T>
int mkt::DMatrix<T>::get_partition_y_pos() const {
  return _partition_y_pos;
}

template<typename T>
mkt::Distribution mkt::DMatrix<T>::get_distribution() const {
  return _dist;
}

template<typename T>
const T* mkt::DMatrix<T>::get_data() const {
  return _data.data();
}

template<typename T>
T* mkt::DMatrix<T>::get_data() {
  return _data.data();
}

template<typename T>
auto mkt::DMatrix<T>::begin() noexcept{
  return _data.begin();
}

template<typename T>
auto mkt::DMatrix<T>::end() noexcept{
  return _data.end();
}

template<typename T>
auto mkt::DMatrix<T>::begin() const noexcept{
  return _data.begin();
}

template<typename T>
auto mkt::DMatrix<T>::end() const noexcept{
  return _data.end();
}		
template<typename T, typename R, typename Functor>
void mkt::map(const mkt::DMatrix<T>& in, mkt::DMatrix<R>& out, const Functor& f) {
#pragma omp parallel for simd
  for (int i = 0; i < in.get_size_local(); ++i) {
      out[i] = f(in[i]);
  }
}

template<typename T, typename R, typename Functor>
void mkt::map_index(const mkt::DMatrix<T>& in, mkt::DMatrix<R>& out, const Functor& f) {
  int row_offset = in.get_row_offset();
  int column_offset = in.get_column_offset();
  int rows_local = in.get_number_of_rows_local();
  int columns_local = in.get_number_of_columns_local();
  
#pragma omp parallel for
  for (int i = 0; i < rows_local; ++i) {
  	#pragma omp simd
  	for (int j = 0; j < columns_local; ++j) {
      out[i * columns_local + j] = f(i + row_offset, j + column_offset, in[i * columns_local + j]);
    }
  }
}

template<typename T, typename R, typename Functor>
void mkt::map_local_index(const mkt::DMatrix<T>& in, mkt::DMatrix<R>& out, const Functor& f) {
  int rows_local = in.get_number_of_rows_local();
  int columns_local = in.get_number_of_columns_local();
#pragma omp parallel for
  for (int i = 0; i < rows_local; ++i) {
  	#pragma omp simd
  	for (int j = 0; j < columns_local; ++j) {
      out[i * columns_local + j] = f(i, j, in[i * columns_local + j]);
    }
  }
}

template<typename T, typename Functor>
void mkt::map_in_place(mkt::DMatrix<T>& m, const Functor& f){
#pragma omp parallel for simd
  for (int i = 0; i < m.get_size_local(); ++i) {
    f(m[i]);
  }
}

template<typename T, typename Functor>
void mkt::map_index_in_place(mkt::DMatrix<T>& m, const Functor& f){
  int row_offset = m.get_row_offset();
  int column_offset = m.get_column_offset();
  int rows_local = m.get_number_of_rows_local();
  int columns_local = m.get_number_of_columns_local();
  
  #pragma omp parallel for
  for (int i = 0; i < rows_local; ++i) {
  	#pragma omp simd
  	for (int j = 0; j < columns_local; ++j) {
      f(i + row_offset, j + column_offset, m[i * columns_local + j]);
    }
  }
}

template<typename T, typename Functor>
void mkt::map_local_index_in_place(mkt::DMatrix<T>& m, const Functor& f){
  int number_of_rows_local = m.get_number_of_rows_local();
  int number_of_columns_local = m.get_number_of_columns_local();

  #pragma omp parallel for
  for (int i = 0; i < number_of_rows_local; ++i) {
    #pragma omp simd
    for (int j = 0; j < number_of_columns_local; ++j) {
      f(i, j, m[i * number_of_columns_local + j]);
    }
  }
}


template<typename T>
void mkt::print(std::ostringstream& stream, const T& a) {
	if(std::is_fundamental<T>::value){
		stream << a;
	}
}



template<>
void mkt::gather<float>(const mkt::DMatrix<float>& in, mkt::DMatrix<float>& out, const MPI_Datatype& dt){
	MPI_Allgatherv(in.get_data(), 67108864, MPI_FLOAT, out.get_data(), (std::array<int, 4>{1, 1, 1, 1}).data(), (std::array<int, 4>{0, 1, 16384, 16385}).data(), dt, MPI_COMM_WORLD);
}
	
template<typename T>
void mkt::scatter(const mkt::DMatrix<T>& in, mkt::DMatrix<T>& out){
	int row_offset = out.get_row_offset();
	int column_offset = out.get_column_offset();
	#pragma omp parallel for
	for(int i = 0; i < out.get_number_of_rows_local(); ++i){
	  #pragma omp simd
	  for(int j = 0; j < out.get_number_of_columns_local(); ++j){
	    out.set_local(i, j, in.get_local(i + row_offset, j + column_offset));
	  }
	}
}
