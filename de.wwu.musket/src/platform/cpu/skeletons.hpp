#pragma once

#include "dmatrix.hpp"

namespace mkt{

// Matrix

template<typename T, typename R, typename Functor>
void map(const DMatrix<T>& in, DMatrix<R>& out, const Functor& f);

template<typename T, typename R, typename Functor>
void map_index(const DMatrix<T>& in, DMatrix<R>& out, const Functor& f);

template<typename T, typename R, typename Functor>
void map_local_index(const DMatrix<T>& in, DMatrix<R>& out, const Functor& f);

template<typename T, typename Functor>
void map_in_place(DMatrix<T>& m, const Functor& f);

template<typename T, typename Functor>
void map_index_in_place(DMatrix<T>& m, const Functor& f);

template<typename T, typename Functor>
void map_local_index_in_place(DMatrix<T>& m, const Functor& f);

template<typename T, typename Functor>
T fold(const DMatrix<T>& m, T identity, const Functor& f);

template<typename T, typename Functor>
T map_fold(const DMatrix<T>& m, const Functor& f_map, const T identity, const Functor& f_fold);
}
