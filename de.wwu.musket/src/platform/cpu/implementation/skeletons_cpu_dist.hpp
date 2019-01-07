#pragma once

#include "../dmatrix.hpp"
#include "../skeletons.hpp"

// Matrix

template<typename T, typename R, typename Functor>
void mkt::map(const DMatrix<T>& in, DMatrix<R>& out, const Functor& f) {
#pragma omp parallel for
  for (int i = 0; i < in._number_of_rows_local; ++i) {
#pragma omp simd
    for (int j = 0; j < in._number_of_columns_local; ++j) {
      out.set_local(i, j, f(in.get_local(i, j)));
    }
  }
}

template<typename T, typename R, typename Functor>
void mkt::map_index(const DMatrix<T>& in, DMatrix<R>& out, const Functor& f) {
    int row_offset = in.get_row_offset();
    int col_offset = in.get_col_offset();
#pragma omp parallel for
  for (int i = 0; i < in.get_number_of_rows_local; ++i) {
#pragma omp simd
    for (int j = 0; j < in.get_number_of_columns_local; ++j) {
      out.set_local(i, j, f(i + row_offset, j + col_offset, in.get_local(i, j)));
    }
  }
}

template<typename T, typename R, typename Functor>
void mkt::map_local_index(const DMatrix<T>& in, DMatrix<R>& out, const Functor& f) {
#pragma omp parallel for
  for (int i = 0; i < in.get_number_of_rows_local; ++i) {
#pragma omp simd
    for (int j = 0; j < in.get_number_of_columns_local; ++j) {
      out.set_local(i, j, f(i, j, in.get_local(i, j)));
    }
  }
}

template<typename T, typename Functor>
void mkt::map_in_place(DMatrix<T>& m, const Functor& f){
#pragma omp parallel for
  for (int i = 0; i < m.get_number_of_rows_local; ++i) {
#pragma omp simd
    for (int j = 0; j < m.get_number_of_columns_local; ++j) {
      m.set_local(i, j, f(m.get_local(i, j)));
    }
  }
}

template<typename T, typename Functor>
void mkt::map_index_in_place(DMatrix<T>& m, const Functor& f){
    int row_offset = in.get_row_offset();
    int col_offset = in.get_col_offset();
#pragma omp parallel for
  for (int i = 0; i < m.get_number_of_rows_local; ++i) {
#pragma omp simd
    for (int j = 0; j < m.get_number_of_columns_local; ++j) {
      m.set_local(i, j, f(i + row_offset, j + col_offset, m.get_local(i, j)));
    }
  }
}

template<typename T, typename Functor>
void mkt::map_local_index_in_place(DMatrix<T>& m, const Functor& f){
#pragma omp parallel for
  for (int i = 0; i < m.get_number_of_rows_local; ++i) {
#pragma omp simd
    for (int j = 0; j < m.get_number_of_columns_local; ++j) {
      m.set_local(i, j, f(i, j, m.get_local(i, j)));
    }
  }
}

template<typename T, typename R, typename Functor, int C, int P>
R mkt::fold(const DMatrix<T>& m, const R identity, const Functor& f){
  std::array<R, P> global_results;
  std::array<R, C> local_results;
  int elems_per_thread = m.get_size_local() / C;
  if (elemsPerThread == 0) {
    elemsPerThread = 1;
    nThreads = localsize;
  }

  // step 1: local fold
  for (int i = 0; i < nThreads; i++) {
    localResults[i] = getLocal(i * elemsPerThread / mLocal,
                               i * elemsPerThread % mLocal);
  }

#pragma omp parallel
  {
    int id = omp_get_thread_num();
    for (int i = 1; i < elemsPerThread; i++) {
      localResults[id] = f(localResults[id],
                           localPartition[id * elemsPerThread + i]);
    }
  }

  // if nThreads does not divide localsize fold up the remaining elements
  if (localsize % nThreads > 0) {
    for (int i = elemsPerThread * nThreads; i < localsize; i++) {
      localResults[0] = f(localResults[0], localPartition[i]);
    }
  }

  // fold local results
  for (int i = 1; i < nThreads; i++) {
    localResults[0] = f(localResults[0], localResults[i]);
  }

  // step 2: global folding
  msl::allgather(localResults, globalResults, 1);

  T result = globalResults[0];
  for (int i = 1; i < np; i++) {
    result = f(result, globalResults[i]);
  }

  delete[] localResults;
  delete[] globalResults;

  return result;
}

template<typename T, typename Functor>
T mkt::map_fold(const DMatrix<T>& m, const Functor& f_map, const T identity, const Functor& f_fold){
#pragma omp parallel for
  for (int i = 0; i < _number_of_rows_local; ++i) {
#pragma omp simd
    for (int j = 0; j < _number_of_columns_local; ++j) {
      out.set_local(i, j, f(i, j, in.get_local(i, j)));
    }
  }
}
