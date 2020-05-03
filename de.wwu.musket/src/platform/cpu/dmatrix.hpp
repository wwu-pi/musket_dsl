#pragma once

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

//   DMatrix(const DMatrix<T>& cs);
//   ~DMatrix();

  // ASSIGNMENT OPERATOR
//   DMatrix<T>& operator=(const DMatrix<T>& rhs);

  // SKELETONS / COMPUTATION
  // SKELETONS / COMPUTATION / MAP

  // template<typename Functor>
  // void mapInPlace(const Functor& f);

  // template<typename Functor>
  // void mapIndexInPlace(const Functor& f);

  // template<typename Functor>
  // void mapLocalIndexInPlace(const Functor& f);

  // template<typename R, typename MapFunctor>
  // msl::DMatrix<R> map(MapFunctor& f);

  // SKELETONS / COMPUTATION / ZIP

  // template<typename T2, typename ZipFunctor>
  // void zipInPlace(DMatrix<T2>& b, ZipFunctor& f);

  // template<typename T2, typename ZipIndexFunctor>
  // void zipIndexInPlace(DMatrix<T2>& b, ZipIndexFunctor& f);

  // template<typename R, typename T2, typename ZipFunctor>
  // DMatrix<R> zip(DMatrix<T2>& b, ZipFunctor& f);

  // template<typename R, typename T2, typename ZipIndexFunctor>
  // DMatrix<R> zipIndex(DMatrix<T2>& b, ZipIndexFunctor& f);

  // SKELETONS / COMPUTATION / FOLD

  //template<typename Functor>
  //T fold(T identity, const Functor& f);

  //
  // SKELETONS / COMMUNICATION
  //

  // SKELETONS / COMMUNICATION / BROADCAST PARTITION

  // void broadcastPartition(int blockRow, int blockCol);

  // SKELETONS / COMMUNICATION / GATHER SCATTER

  // void gather();
  // void scatter();


  // template<class F1, class F2>
  // void permutePartition(const Fct2<int, int, int, F1>& newRow,
  //                       const Fct2<int, int, int, F2>& newCol);


  // SKELETONS / COMMUNICATION / SHIFT

  // template<typename F>
  // void shiftPartitionsVertically();

  // template<typename F>
  // void shiftPartitionsHorizontally();

  // void transposeLocalPartition();

   /**
   * \brief Prints the distributed array to standard output. Optionally, the user
   * may pass a description that will be printed with the output.
   *
   * @param descr The description string.
   */
  // void show(const std::string& descr = std::string());

  /**
   * \brief Each process prints its local partition of the distributed array.
   */
  // void printLocal();

// Getter and Setter

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

// template<typename T>
// msl::DMatrix<T>::DMatrix(const DMatrix<T>& cs)
//     : n(cs.n),
//       m(cs.m),
//       dist(cs.dist) {
//   }

// template<typename T>
// msl::DMatrix<T>::~DMatrix() {

// }

// template<typename T>
// msl::DMatrix<T>& msl::DMatrix<T>::operator=(const DMatrix<T>& rhs) {
//   if (this != &rhs) {
//     n = rhs.n;
//     m = rhs.m;
//     dist = rhs.dist;
//     init(rhs.blocksInCol, rhs.blocksInRow);

//     bool create_new_local_partition = false;

//     if (nLocal != rhs.nLocal || mLocal != rhs.mLocal) {
//       create_new_local_partition = true;
//     }

//     T* new_localPartition;
// }

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
int mkt::DMatrix<T>::get_number_of_cols_local() const {
  return _number_of_cols_local;
}

// template<typename T> 
// template<typename Functor>
// void mkt::DMatrix<T>::mapInPlace(const Functor& f) {
// #pragma omp parallel for
//   for (int i = 0; i < _number_of_rows_local; ++i) {
// #pragma omp simd
//     for (int j = 0; j < _number_of_columns_local; ++j) {
//       set_local(i, j, f(get_local(i, j)));
//     }
//   }
// }

// template<typename T> 
// template<typename Functor>
// void mkt::DMatrix<T>::mapIndexInPlace(const Functor& f) {
// #pragma omp parallel for
//   for (int i = 0; i < _number_of_rows_local; ++i) {
// #pragma omp simd
//     for (int j = 0; j < _number_of_columns_local; ++j) {
//       set_local(i, j, f(i + _row_offset, j + _col_offset, get_local(i, j)));
//     }
//   }
// }

// template<typename T> 
// template<typename Functor>
// void mkt::DMatrix<T>::mapLocalIndexInPlace(const Functor& f) {
// #pragma omp parallel for
//   for (int i = 0; i < _number_of_rows_local; ++i) {
// #pragma omp simd
//     for (int j = 0; j < _number_of_columns_local; ++j) {
//       set_local(i, j, f(i, j, get_local(i, j)));
//     }
//   }
// }

// template<typename T> 
// template<typename Functor>
// T mkt::DMatrix<T>::fold(T identity, const Functor& f) {

//   T result = identity;

//   #pragma omp parallel for simd reduction()
// 	for(int i = 0; i < _size_local; ++i){
// 		result = f(result, _data[i]);
// 	}

//   return result;
// }
