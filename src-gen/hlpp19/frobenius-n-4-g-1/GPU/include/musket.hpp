#pragma once
#include <string>
#include "frobenius-n-4-g-1.hpp"

namespace mkt {
enum Distribution {DIST, COPY};

template<typename T>
class DMatrix {
 public:

  // CONSTRUCTORS / DESTRUCTOR
  DMatrix(int pid, int number_of_rows, int number_of_columns, int number_of_rows_local, 
          int number_of_columns_local, int size, int size_local, T init_value, 
          int partitions_in_row, int partitions_in_column, int partition_x_pos, int partition_y_pos, 
          int row_offset, int column_offset, Distribution d = DIST, Distribution device_dist = DIST);
  ~DMatrix();
  
  void update_self();
  void update_devices();
  void map_pointer();

// Getter and Setter

  T get_global(int row, int column) const;
  void set_global(int row, int column, const T& value);

  T get_local(int row, int column);
  T get_local(int index);
  void set_local(int row, int column, const T& value);
  void set_local(int index, const T& value);

  T& get_local_host_data(int row, int column);
  const T& get_local_host_data(int row, int column) const;
  T& operator[](int local_index);
  const T& operator[](int local_index) const;
  
  T* get_local_device_data(int row, int column);

  int get_size() const;
  int get_size_local() const;
  int get_size_gpu() const;
  int get_rows_gpu() const;

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
  Distribution get_device_distribution() const;

  T* get_data();
  const T* get_data() const;
  
  T* get_device_pointer(int gpu) const;
  
  auto begin() noexcept;
  auto end() noexcept;
  
  auto begin() const noexcept;
  auto end() const noexcept;

 private:

int get_gpu_by_local_index(int local_index) const;
int get_gpu_by_global_index(int global_index) const;

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
  int _size_gpu;
  int _rows_gpu;

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
  Distribution _device_dist;

  std::vector<T> _data;
  std::array<T*, 1> _host_data;
  std::array<T*, 1> _gpu_data;
};
template<typename T, typename R, typename Functor>
void map(const mkt::DMatrix<T>& in, mkt::DMatrix<R>& out, Functor f);

template<typename T, typename R, typename Functor>
void map_index(const mkt::DMatrix<T>& in, mkt::DMatrix<R>& out, Functor f);

template<typename T, typename R, typename Functor>
void map_local_index(const mkt::DMatrix<T>& in, mkt::DMatrix<R>& out, Functor f);

template<typename T, typename Functor>
void map_in_place(mkt::DMatrix<T>& m, Functor f);

template<typename T, typename Functor>
void map_index_in_place(mkt::DMatrix<T>& m, Functor f);

template<typename T, typename Functor>
void map_local_index_in_place(mkt::DMatrix<T>& m, Functor f);

template<typename T, typename Functor>
void fold(const mkt::DMatrix<T>& m, T& out, const T identity, const Functor f);

template<typename T, typename Functor>
void fold_copy(const mkt::DMatrix<T>& m, T& out, const T identity, const Functor f);

template<typename T, typename R, typename MapFunctor, typename FoldFunctor>
void map_fold(const mkt::DMatrix<T>& m, T& out, const MapFunctor f_map, const R identity, const FoldFunctor f_fold);

template<typename T, typename R, typename MapFunctor, typename FoldFunctor>
void map_fold_copy(const mkt::DMatrix<T>& m, T& out, const MapFunctor f_map, const R identity, const FoldFunctor f_fold);
template<typename T>
class DeviceMatrix {
 public:

  // CONSTRUCTORS / DESTRUCTOR
  DeviceMatrix(const DMatrix<T>& dm);
  DeviceMatrix(const DeviceMatrix<T>& dm);
  ~DeviceMatrix();
  
  void init(int device_id);
  
  
// Getter and Setter

  const T& get_data_device(int device_index) const;
  const T& get_data_device(int device_row, int device_column) const;

  const T& get_data_local(int local_row, int local_column) const;
  

 private:

  //
  // Attributes
  //

  int _size;
  int _size_local;
  int _size_device;
  int _rows_device;
  int _columns_device;

  int _row_offset;
  int _column_offset;
  
  int _device_row_offset;
  int _device_column_offset;

  Distribution _dist;
  Distribution _device_dist;

  T* _device_data;

  std::array<T*, 1> _gpu_data;
};


template<typename T>
void print(std::ostringstream& stream, const T& a);


	
template<typename T>
void gather(mkt::DMatrix<T>& in, mkt::DMatrix<T>& out, const MPI_Datatype& dt);
	
template<typename T>
void scatter(mkt::DMatrix<T>& in, mkt::DMatrix<T>& out);


template<typename T>
T reduce_plus(mkt::DMatrix<T>& m);

template<typename T>
T reduce_multiply(mkt::DMatrix<T>& m);
		
template<typename T>
T reduce_max(mkt::DMatrix<T>& m);
				
template<typename T>
T reduce_min(mkt::DMatrix<T>& m);

template<typename T>
T reduce_plus_copy(mkt::DMatrix<T>& m);

template<typename T>
T reduce_multiply_copy(mkt::DMatrix<T>& m);
		
template<typename T>
T reduce_max_copy(mkt::DMatrix<T>& m);
				
template<typename T>
T reduce_min_copy(mkt::DMatrix<T>& m);


} // namespace mkt


		
template<typename T>
mkt::DMatrix<T>::DMatrix(int pid, int number_of_rows, int number_of_columns, int number_of_rows_local, 
                         int number_of_columns_local, int size, int size_local, T init_value, 
                         int partitions_in_row, int partitions_in_column, int partition_x_pos, int partition_y_pos, 
                         int row_offset, int column_offset, Distribution d, Distribution device_dist)
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
      _device_dist(device_dist),
      _data(size_local, init_value) {
	if(device_dist == mkt::Distribution::DIST){
    	_size_gpu = size_local / 1; // assume even distribution for now
    	_rows_gpu = number_of_rows_local / 1;
    }else if(device_dist == mkt::Distribution::COPY){
    	_size_gpu = size_local;
    }
    
    #pragma omp parallel for
	for(int gpu = 0; gpu < 1; ++gpu){
		acc_set_device_num(gpu, acc_device_not_host);
		// allocate memory
		T* devptr = static_cast<T*>(acc_malloc(_size_gpu * sizeof(T)));
		
		// store pointer to device memory and host memory
		_gpu_data[gpu] = devptr;
		if(device_dist == mkt::Distribution::DIST){
	    	_host_data[gpu] = _data.data() + gpu * _size_gpu;
	    }else if(device_dist == mkt::Distribution::COPY){
	    	_host_data[gpu] = _data.data(); // all gpus have complete data, thus point to the beginning of host vector
	    }
		
		// init values
		#pragma acc parallel loop deviceptr(devptr) async(0)
		for(int i = 0; i < _size_gpu; ++i){
		  devptr[i] = init_value;
		}
	}
	this->map_pointer();		      	
}

template<typename T>
mkt::DMatrix<T>::~DMatrix(){
	// free device memory
	#pragma omp parallel for
	for(size_t gpu = 0; gpu < 1; ++gpu){
		acc_set_device_num(gpu, acc_device_not_host);
		acc_free(_gpu_data[gpu]);
	}
}
		
template<typename T>
void mkt::DMatrix<T>::update_self() {
  	#pragma omp parallel for
	for(size_t gpu = 0; gpu < 1; ++gpu){
		acc_set_device_num(gpu, acc_device_not_host);
		acc_update_self_async(_host_data[gpu], _size_gpu * sizeof(T), 0);
		#pragma acc wait
	}
}

template<typename T>
void mkt::DMatrix<T>::update_devices() {
  	#pragma omp parallel for
	for(int gpu = 0; gpu < 1; ++gpu){
		acc_set_device_num(gpu, acc_device_not_host);
		acc_update_device_async(_host_data[gpu], _size_gpu * sizeof(T), 0);
		#pragma acc wait
	}
}

template<typename T>
void mkt::DMatrix<T>::map_pointer() {
	for(size_t gpu = 0; gpu < 1; ++gpu){
		acc_set_device_num(gpu, acc_device_not_host);
		acc_map_data( _host_data[gpu], _gpu_data[gpu], _size_gpu * sizeof(T));
	}
}

template<typename T>
T mkt::DMatrix<T>::get_local(int row, int column) {
	int index = row * _number_of_columns_local + column;
	int gpu = get_gpu_by_local_index(index);
	acc_set_device_num(gpu, acc_device_not_host);
	T* host_pointer = _data.data() + index;
	T* gpu_pointer = _gpu_data[gpu] + (index % _size_gpu );
	acc_memcpy_from_device_async(host_pointer, gpu_pointer, sizeof(T), 0);
	#pragma acc wait
    return _data[index];
}

template<typename T>
T mkt::DMatrix<T>::get_local(int index) {
	int gpu = get_gpu_by_local_index(index);
	acc_set_device_num(gpu, acc_device_not_host);
	T* host_pointer = _data.data() + index;
	T* gpu_pointer = _gpu_data[gpu] + (index % _size_gpu );
	acc_memcpy_from_device_async(host_pointer, gpu_pointer, sizeof(T), 0);
	#pragma acc wait
    return _data[index];
}

template<typename T>
void mkt::DMatrix<T>::set_local(int row, int column, const T& v) {
	int index = row * _number_of_columns_local + column;
	_data[index] = v;
	T* host_pointer = _data.data() + index;
	if(_device_dist == mkt::Distribution::COPY){
		#pragma omp parallel for
		for(int gpu = 0; gpu < 1; ++gpu){
			acc_set_device_num(gpu, acc_device_not_host);
			T* gpu_pointer = _gpu_data[gpu] + index;
			acc_memcpy_to_device_async(host_pointer, gpu_pointer, sizeof(T), 0 );
		}
	}else if(_device_dist == mkt::Distribution::DIST){
		int gpu = get_gpu_by_local_index(index);
		acc_set_device_num(gpu, acc_device_not_host);
		T* gpu_pointer = _gpu_data[gpu] + (index % _size_gpu );
		acc_memcpy_to_device_async(host_pointer, gpu_pointer, sizeof(T), 0 );
	}
	#pragma acc wait
}

template<typename T>
void mkt::DMatrix<T>::set_local(int index, const T& v) {
	_data[index] = v;
	T* host_pointer = _data.data() + index;
	if(_device_dist == mkt::Distribution::COPY){
		#pragma omp parallel for
		for(int gpu = 0; gpu < 1; ++gpu){
			acc_set_device_num(gpu, acc_device_not_host);
			T* gpu_pointer = _gpu_data[gpu] + index;
			acc_memcpy_to_device_async(host_pointer, gpu_pointer, sizeof(T), 0 );
		}
	}else if(_device_dist == mkt::Distribution::DIST){
		int gpu = get_gpu_by_local_index(index);
		acc_set_device_num(gpu, acc_device_not_host);
		T* gpu_pointer = _gpu_data[gpu] + (index % _size_gpu );
		acc_memcpy_to_device_async(host_pointer, gpu_pointer, sizeof(T), 0 );
	}
	#pragma acc wait
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
T& mkt::DMatrix<T>::get_local_host_data(int row, int column) {
  return _data[row * _number_of_columns_local + column];
}

template<typename T>
const T& mkt::DMatrix<T>::get_local_host_data(int row, int column) const {
  return _data[row * _number_of_columns_local + column];
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
T* mkt::DMatrix<T>::get_local_device_data(int local_row, int local_column) {
  //TODO
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
int mkt::DMatrix<T>::get_size_gpu() const {
  return _size_gpu;
}

template<typename T>
int mkt::DMatrix<T>::get_rows_gpu() const {
  return _rows_gpu;
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
mkt::Distribution mkt::DMatrix<T>::get_device_distribution() const {
  return _device_dist;
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

template<typename T>
T* mkt::DMatrix<T>::get_device_pointer(int gpu) const{
  return _gpu_data[gpu];
}

template<typename T>
int mkt::DMatrix<T>::get_gpu_by_local_index(int local_index) const {
	if(_device_dist == mkt::Distribution::COPY){
		return 0;
	}else if(_device_dist == mkt::Distribution::DIST){
		return local_index / _size_gpu;
	}
	else{
		return -1;
	}
}

template<typename T>
int mkt::DMatrix<T>::get_gpu_by_global_index(int global_index) const {
	// TODO
	return -1;
}
template<typename T, typename R, typename Functor>
void mkt::map(const mkt::DMatrix<T>& in, mkt::DMatrix<R>& out, Functor f) {
	//#pragma omp parallel for firstprivate(f)
	for(int gpu = 0; gpu < 1; ++gpu){
		acc_set_device_num(gpu, acc_device_not_host);
		f.init(gpu);
		T* in_devptr = in.get_device_pointer(gpu);
		R* out_devptr = out.get_device_pointer(gpu);
		const int gpu_elements = in.get_size_gpu();
		#pragma acc parallel loop deviceptr(in_devptr, out_devptr) firstprivate(f) async(0)
		for (int i = 0; i < gpu_elements; ++i) {
			f.set_id(__pgi_gangidx(), __pgi_workeridx(),__pgi_vectoridx());
			out_devptr[i] = f(in_devptr[i]);
		}
	}
}

template<typename T, typename R, typename Functor>
void mkt::map_index(const mkt::DMatrix<T>& in, mkt::DMatrix<R>& out, Functor f) {
	int row_offset = in.get_row_offset();
	int column_offset = in.get_column_offset();
	int columns_local = in.get_number_of_columns_local();

  	int gpu_elements = in.get_size_gpu();
  	int rows_on_gpu = in.get_rows_gpu();
	//#pragma omp parallel for firstprivate(f)
	for(int gpu = 0; gpu < 1; ++gpu){
		acc_set_device_num(gpu, acc_device_not_host);
		f.init(gpu);
		T* in_devptr = in.get_device_pointer(gpu);
		R* out_devptr = out.get_device_pointer(gpu);
		
		if(in.get_device_distribution() == mkt::Distribution::DIST){
			row_offset += gpu * rows_on_gpu;
		}
		
		#pragma acc parallel loop deviceptr(in_devptr, out_devptr) firstprivate(f) async(0)
		for (int i = 0; i < gpu_elements; ++i) {
			f.set_id(__pgi_gangidx(), __pgi_workeridx(),__pgi_vectoridx());
			int row_index = row_offset + (i / columns_local);
			int column_index = column_offset + (i % columns_local);
			out_devptr[i] = f(row_index, column_index, in_devptr[i]);
		}
	}
}

template<typename T, typename R, typename Functor>
void mkt::map_local_index(const mkt::DMatrix<T>& in, mkt::DMatrix<R>& out, Functor f) {
	int columns_local = in.get_number_of_columns_local();

  	int gpu_elements = in.get_size_gpu();
  	int rows_on_gpu = in.get_rows_gpu();
  	
	//#pragma omp parallel for firstprivate(f)
	for(int gpu = 0; gpu < 1; ++gpu){
		acc_set_device_num(gpu, acc_device_not_host);
		f.init(gpu);
		T* in_devptr = in.get_device_pointer(gpu);
		R* out_devptr = out.get_device_pointer(gpu);
		
		int row_offset = 0;
		if(in.get_device_distribution() == mkt::Distribution::DIST){
			row_offset = gpu * rows_on_gpu;
		}
		
		#pragma acc parallel loop deviceptr(in_devptr, out_devptr) firstprivate(f) async(0)
		for (int i = 0; i < gpu_elements; ++i) {
			f.set_id(__pgi_gangidx(), __pgi_workeridx(),__pgi_vectoridx());
			int row_index = row_offset + (i / columns_local);
			int column_index = i % columns_local;
			out_devptr[i] = f(row_index, column_index, in_devptr[i]);
		}
	}
}

template<typename T, typename Functor>
void mkt::map_in_place(mkt::DMatrix<T>& m, Functor f) {
	//#pragma omp parallel for firstprivate(f)
	for(int gpu = 0; gpu < 1; ++gpu){
		acc_set_device_num(gpu, acc_device_not_host);
		f.init(gpu);
		T* devptr = m.get_device_pointer(gpu);
		const int gpu_elements = m.get_size_gpu();
		#pragma acc parallel loop deviceptr(devptr) firstprivate(f) async(0)
		for (int i = 0; i < gpu_elements; ++i) {
			f.set_id(__pgi_gangidx(), __pgi_workeridx(),__pgi_vectoridx());
			f(devptr[i]);
		}
	}
}

template<typename T, typename Functor>
void mkt::map_index_in_place(mkt::DMatrix<T>& m, Functor f){
	int row_offset = m.get_row_offset();
	int column_offset = m.get_column_offset();
	int columns_local = m.get_number_of_columns_local();

  	int gpu_elements = m.get_size_gpu();
  	int rows_on_gpu = m.get_rows_gpu();
	//#pragma omp parallel for firstprivate(f)
	for(int gpu = 0; gpu < 1; ++gpu){
		acc_set_device_num(gpu, acc_device_not_host);
		f.init(gpu);
		T* devptr = m.get_device_pointer(gpu);
		
		if(m.get_device_distribution() == mkt::Distribution::DIST){
			row_offset += gpu * rows_on_gpu;
		}
		
		#pragma acc parallel loop deviceptr(devptr) firstprivate(f) async(0)
		for (int i = 0; i < gpu_elements; ++i) {
			f.set_id(__pgi_gangidx(), __pgi_workeridx(),__pgi_vectoridx());
			int row_index = row_offset + (i / columns_local);
			int column_index = column_offset + (i % columns_local);
			f(row_index, column_index, devptr[i]);
		}
	}
}	

template<typename T, typename Functor>
void mkt::map_local_index_in_place(mkt::DMatrix<T>& m, Functor f){
	int columns_local = m.get_number_of_columns_local();

  	int gpu_elements = m.get_size_gpu();
  	int rows_on_gpu = m.get_rows_gpu();
  	
	//#pragma omp parallel for shared(f)
	for(int gpu = 0; gpu < 1; ++gpu){
		acc_set_device_num(gpu, acc_device_not_host);
		f.init(gpu);
		
		T* devptr = m.get_device_pointer(gpu);

		int row_offset = 0;
		if(m.get_device_distribution() == mkt::Distribution::DIST){
			row_offset = gpu * rows_on_gpu;
		}
		
		#pragma acc parallel loop deviceptr(devptr) firstprivate(f) async(0)
		for (int i = 0; i < gpu_elements; ++i) {
			f.set_id(__pgi_gangidx(), __pgi_workeridx(),__pgi_vectoridx());
			int row_index = row_offset + (i / columns_local);
			int column_index = i % columns_local;
			f(row_index, column_index, devptr[i]);
		}
	}
}
		
template<typename T>
mkt::DeviceMatrix<T>::DeviceMatrix(const DMatrix<T>& dm)
    : _size(dm.get_size()),
      _size_local(dm.get_size_local()),
      _size_device(dm.get_size_gpu()),
      _rows_device(dm.get_rows_gpu()),
      _columns_device(dm.get_number_of_columns_local()),
      _row_offset(dm.get_row_offset()),
      _column_offset(dm.get_column_offset()),
      _dist(dm.get_distribution()),
      _device_dist(dm.get_device_distribution()) 
{
	for(int i = 0; i < 1; ++i){
		_gpu_data[i] = dm.get_device_pointer(i);
	}
}

template<typename T>
mkt::DeviceMatrix<T>::DeviceMatrix(const DeviceMatrix<T>& dm)
    : _size(dm._size),
      _size_local(dm._size_local),
      _size_device(dm._size_device),
      _rows_device(dm._rows_device),
      _columns_device(dm._columns_device),
      _row_offset(dm._row_offset),
      _column_offset(dm._column_offset),
      _dist(dm._dist),
      _device_dist(dm._device_dist) 
{
	_device_data = dm._device_data;
	for(int i = 0; i < 1; ++i){
		_gpu_data[i] = dm._gpu_data[i];
	}
}

template<typename T>
mkt::DeviceMatrix<T>::~DeviceMatrix(){
}

template<typename T>
void mkt::DeviceMatrix<T>::init(int gpu) {
	if(_device_dist == Distribution::COPY){
		_device_row_offset = 0;
		_device_column_offset = 0;
	} else {
		_device_row_offset = _rows_device * gpu;
		_device_column_offset = 0;
	}
	    
	_device_data = _gpu_data[gpu];
}

template<typename T>
const T& mkt::DeviceMatrix<T>::get_data_device(int device_index) const {
  return _device_data[device_index];
}

template<typename T>
const T& mkt::DeviceMatrix<T>::get_data_device(int device_row, int device_column) const {
  return this->get_data_device(device_row * _columns_device + device_column);
}

template<typename T>
const T& mkt::DeviceMatrix<T>::get_data_local(int local_row, int local_column) const {
  return this->get_data_device(local_row - _device_row_offset, local_column - _device_column_offset);
}



template<typename T>
void mkt::print(std::ostringstream& stream, const T& a) {
	if(std::is_fundamental<T>::value){
		stream << a;
	}
}



template<>
void mkt::gather<double>(mkt::DMatrix<double>& in, mkt::DMatrix<double>& out, const MPI_Datatype& dt){
	in.update_self();
	MPI_Allgatherv(in.get_data(), 268435456, MPI_DOUBLE, out.get_data(), (std::array<int, 4>{1, 1, 1, 1}).data(), (std::array<int, 4>{0, 1, 32768, 32769}).data(), dt, MPI_COMM_WORLD);
	out.update_devices();
}
	
template<typename T>
void mkt::scatter(mkt::DMatrix<T>& in, mkt::DMatrix<T>& out){
	in.update_self();
	int row_offset = out.get_row_offset();
	int column_offset = out.get_column_offset();
	#pragma omp parallel for
	for(int i = 0; i < out.get_number_of_rows_local(); ++i){
	  #pragma omp simd
	  for(int j = 0; j < out.get_number_of_columns_local(); ++j){
	    out.set_local(i, j, in.get_local(i + row_offset, j + column_offset));
	  }
	}
	out.update_devices();
}
