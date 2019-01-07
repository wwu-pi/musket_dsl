#pragma once

#include "../dmatrix.hpp"
#include "../skeletons.hpp"

// Matrix

template<typename T, typename R, typename Functor>
void mkt::map(const DMatrix<T>& in, DMatrix<R>& out, const Functor& f) {
#pragma omp parallel for
  for (int i = 0; i < _number_of_rows_local; ++i) {
#pragma omp simd
    for (int j = 0; j < _number_of_columns_local; ++j) {
      out.set_local(i, j, f(in.get_local(i, j)));
    }
  }
}

template<typename T, typename R, typename Functor>
void mkt::mapIndex(const DMatrix<T>& in, DMatrix<R>& out, const Functor& f) {
    int row_offset = in.get_row_offset();
    int col_offset = in.get_col_offset();
#pragma omp parallel for
  for (int i = 0; i < _number_of_rows_local; ++i) {
#pragma omp simd
    for (int j = 0; j < _number_of_columns_local; ++j) {
      out.set_local(i, j, f(i + row_offset, j + col_offset, in.get_local(i, j)));
    }
  }
}

template<typename T, typename R, typename Functor>
void mkt::mapLocalIndex(const DMatrix<T>& in, DMatrix<R>& out, const Functor& f) {
#pragma omp parallel for
  for (int i = 0; i < _number_of_rows_local; ++i) {
#pragma omp simd
    for (int j = 0; j < _number_of_columns_local; ++j) {
      out.set_local(i, j, f(i, j, in.get_local(i, j)));
    }
  }
}

template<typename T, typename Functor>
void mkt::mapInPlace(DMatrix<T>& m, const Functor& f){
#pragma omp parallel for
  for (int i = 0; i < _number_of_rows_local; ++i) {
#pragma omp simd
    for (int j = 0; j < _number_of_columns_local; ++j) {
      out.set_local(i, j, f(in.get_local(i, j)));
    }
  }
}

template<typename T, typename Functor>
void mkt::mapIndexInPlace(DMatrix<T>& m, const Functor& f){
    int row_offset = in.get_row_offset();
    int col_offset = in.get_col_offset();
#pragma omp parallel for
  for (int i = 0; i < _number_of_rows_local; ++i) {
#pragma omp simd
    for (int j = 0; j < _number_of_columns_local; ++j) {
      out.set_local(i, j, f(i + row_offset, j + col_offset, in.get_local(i, j)));
    }
  }
}

template<typename T, typename Functor>
void mkt::mapLocalIndexInPlace(DMatrix<T>& m, const Functor& f){
#pragma omp parallel for
  for (int i = 0; i < _number_of_rows_local; ++i) {
#pragma omp simd
    for (int j = 0; j < _number_of_columns_local; ++j) {
      out.set_local(i, j, f(i, j, in.get_local(i, j)));
    }
  }
}

template<typename T, typename Functor>
T mkt::fold(const DMatrix<T>& m, T identity, const Functor& f){
    int row_offset = in.get_row_offset();
    int col_offset = in.get_col_offset();
#pragma omp parallel for
  for (int i = 0; i < _number_of_rows_local; ++i) {
#pragma omp simd
    for (int j = 0; j < _number_of_columns_local; ++j) {
      out.set_local(i, j, f(i + row_offset, j + col_offset, in.get_local(i, j)));
    }
  }
}

template<typename T, typename Functor>
T mkt::mapFold(const DMatrix<T>& m, const Functor& f_map, const T identity, const Functor& f_fold){
#pragma omp parallel for
  for (int i = 0; i < _number_of_rows_local; ++i) {
#pragma omp simd
    for (int j = 0; j < _number_of_columns_local; ++j) {
      out.set_local(i, j, f(i, j, in.get_local(i, j)));
    }
  }
}
