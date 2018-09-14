#pragma once


#include <vector>
#include "distribution.hpp"


namespace mkt {

template<typename T>
class DArray {
 public:

  // CONSTRUCTORS / DESTRUCTOR
  DArray(int pid, int size, int size_local, T init_value, 
          int partitions, int partition_pos, int offset, Distribution d = DIST);



  T get_global(int index) const;
  void set_global(int index, const T& value);

  T get_local(int index) const;
  void set_local(int index, const T& value);

  int get_size() const;
  int get_size_local() const;

  int get_offset() const;

 private:

  // position of processor in data parallel group of processors; zero-base
  int _pid;

  int _size;
  int _size_local;

  // number of (local) partitions
  int _partitions;

  /*
  *     X=0 | X=1 | ....
 */

  // position of processor in data parallel group of processors
  int _partition_pos;

  // first index in local partition
  int _offset;

  // checks whether data is copy distributed among all processes
  Distribution _dist;

  std::vector<T> _data;
};

}  // namespace mkt

template<typename T>
mkt::DArray<T>::DArray(int pid, int size, int size_local, T init_value, 
                         int partitions, int partition_pos, int offset, 
                         Distribution d)
    : _pid(pid),
      _size(size),
      _size_local(size_local),
      _partitions(partitions),
      _partition_pos(partition_pos),
      _offset(offset),
      _dist(Distribution::DIST),
      _data(size_local, init_value) {
}

template<typename T>
T mkt::DArray<T>::get_local(int index) const {
  return _data[index];
}

template<typename T>
void mkt::DArray<T>::set_local(int index, const T& v) {
  _data[index] = v;
}

template<typename T>
T mkt::DArray<T>::get_global(int index) const {
  // TODO
}

template<typename T>
void mkt::DArray<T>::set_global(int index, const T& v) {
  // TODO
}

template<typename T>
int mkt::DArray<T>::get_size() const {
  return _size;
}

template<typename T>
int mkt::DArray<T>::get_size_local() const {
  return _size_local;
}

template<typename T>
int mkt::DArray<T>::get_offset() const {
  return _offset;
}

