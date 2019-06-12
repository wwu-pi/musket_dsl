#pragma once
#include <string>
#include "frobenius-n-1-g-1.cuh"
#include "kernel.cuh"

namespace mkt {
enum Distribution {DIST, COPY};

// Musket variables
cudaStream_t cuda_streams[4];

// Musket functions
void init();		
void sync_streams();


template<typename T>
class DMatrix {
 public:

  // CONSTRUCTORS / DESTRUCTOR
  DMatrix(int pid, size_t number_of_rows, size_t number_of_columns, size_t number_of_rows_local, 
          size_t number_of_columns_local, size_t size, size_t size_local, T init_value, 
          int partitions_in_row, int partitions_in_column, int partition_x_pos, int partition_y_pos, 
          size_t row_offset, size_t column_offset, mkt::Distribution d = DIST, mkt::Distribution device_dist = DIST);
  ~DMatrix();
  
  void update_self();
  void update_devices();

// Getter and Setter

  T get_global(size_t row, size_t column) const;
  void set_global(size_t row, size_t column, const T& value);

  T get_local(size_t row, size_t column);
  T get_local(size_t index);
  void set_local(size_t row, size_t column, const T& value);
  void set_local(size_t index, const T& value);

  T get_local_host_data(size_t row, size_t column) const;
  T get_local_host_data(size_t index) const;
  
  void set_local_host_data(size_t row, size_t column, T v);
  void set_local_host_data(size_t index, T v);
  
  T& operator[](size_t local_index);
  const T& operator[](size_t local_index) const;
  
  T* get_local_device_data(size_t row, size_t column);

  size_t get_size() const;
  size_t get_size_local() const;
  size_t get_size_gpu() const;
  size_t get_rows_gpu() const;
  size_t get_bytes_gpu() const;

  size_t get_row_offset() const;
  size_t get_column_offset() const;

  size_t get_number_of_rows() const;
  size_t get_number_of_columns() const;

  size_t get_number_of_rows_local() const;
  size_t get_number_of_columns_local() const;
  
  int get_partitions_in_row() const;
  int get_partitions_in_column() const;
  
  int get_partition_x_pos() const;
  int get_partition_y_pos() const;
  
  mkt::Distribution get_distribution() const;
  mkt::Distribution get_device_distribution() const;

  T* get_data();
  const T* get_data() const;
  
  T* get_device_pointer(int gpu) const;
  
  auto begin() noexcept;
  auto end() noexcept;
  
  auto begin() const noexcept;
  auto end() const noexcept;

 private:

int get_gpu_by_local_index(size_t local_index) const;
int get_gpu_by_global_index(size_t global_index) const;

int get_pid_by_global_index(size_t global_row_index, size_t global_column_index) const;
bool is_local(size_t global_row_index, size_t global_column_index) const;

  //
  // Attributes
  //

  // position of processor in data parallel group of processors; zero-base
  int _pid;
  // number of rows
  size_t _number_of_rows;
  // number of columns
  size_t _number_of_columns;
  // number of local rows
  size_t _number_of_rows_local;
  // number of local columns
  size_t _number_of_columns_local;

  size_t _size;
  size_t _size_local;
  size_t _size_gpu;
  size_t _rows_gpu;
  size_t _bytes_gpu;

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
  size_t _row_offset;
  // first column index in local partition; assuming division mode: block
  size_t _column_offset;

  // checks whether data is copy distributed among all processes
  mkt::Distribution _dist;
  mkt::Distribution _device_dist;

  T* _data;
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

  __device__ const T& get_data_device(size_t device_index) const;
  __device__ const T& get_data_device(size_t device_row, size_t device_column) const;

  __device__ const T& get_data_local(size_t local_row, size_t local_column) const;
  

 private:

  //
  // Attributes
  //

  size_t _size;
  size_t _size_local;
  size_t _size_device;
  size_t _rows_device;
  size_t _columns_device;
  size_t _bytes_device;

  size_t _row_offset;
  size_t _column_offset;
  
  size_t _device_row_offset;
  size_t _device_column_offset;

  mkt::Distribution _dist;
  mkt::Distribution _device_dist;

  T* _device_data;

  std::array<T*, 1> _gpu_data;
};


template<typename T>
void print(std::ostringstream& stream, const T& a);


	
template<typename T>
void gather(mkt::DMatrix<T>& in, mkt::DMatrix<T>& out);
	
template<typename T>
void scatter(mkt::DMatrix<T>& in, mkt::DMatrix<T>& out);



template<typename T, typename R, typename Functor>
R map_reduce_plus(mkt::DMatrix<T>& m, Functor f);

template<typename T, typename R, typename Functor>
R map_reduce_multiply(mkt::DMatrix<T>& m, Functor f);
		
template<typename T, typename R, typename Functor>
R map_reduce_max(mkt::DMatrix<T>& m, Functor f);
				
template<typename T, typename R, typename Functor>
R map_reduce_min(mkt::DMatrix<T>& m, Functor f);


} // namespace mkt


void mkt::init(){

	for(int gpu = 0; gpu < 1; ++gpu){
		cudaSetDevice(gpu);
		cudaStreamCreate(&cuda_streams[gpu]);
	}
}

void mkt::sync_streams(){
	for(int gpu = 0; gpu < 1; ++gpu){
		cudaSetDevice(gpu);
		cudaStreamSynchronize(cuda_streams[gpu]);
	}
}



		
template<typename T>
mkt::DMatrix<T>::DMatrix(int pid, size_t number_of_rows, size_t number_of_columns, size_t number_of_rows_local, 
                         size_t number_of_columns_local, size_t size, size_t size_local, T init_value, 
                         int partitions_in_row, int partitions_in_column, int partition_x_pos, int partition_y_pos, 
                         size_t row_offset, size_t column_offset, mkt::Distribution d, mkt::Distribution device_dist)
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
      _device_dist(device_dist) {
	if(device_dist == mkt::Distribution::DIST){
    	_size_gpu = size_local / 1; // assume even distribution for now
    	_rows_gpu = number_of_rows_local / 1;
    }else if(device_dist == mkt::Distribution::COPY){
    	_size_gpu = size_local;
    	_rows_gpu = number_of_rows_local;
    }
    _bytes_gpu = sizeof(T) * _size_gpu;
	cudaMallocHost((void**)&_data, _size_local * sizeof(T));
    
	for(int gpu = 0; gpu < 1; ++gpu){
		cudaSetDevice(gpu);
		
		// allocate memory
		T* devptr;
		cudaMalloc((void**)&devptr, _bytes_gpu);
		
		// store pointer to device memory and host memory
		_gpu_data[gpu] = devptr;
		if(device_dist == mkt::Distribution::DIST){
	    	_host_data[gpu] = _data + gpu * _size_gpu;
	    }else if(device_dist == mkt::Distribution::COPY){
	    	_host_data[gpu] = _data; // all gpus have complete data, thus point to the beginning of host vector
	    }
		
		
	}
	for(size_t i = 0; i < _size_local; ++i){
	  _data[i] = init_value;
	}
	update_devices();
}

template<typename T>
mkt::DMatrix<T>::~DMatrix(){
	cudaFreeHost(_data);
	// free device memory
	#pragma omp parallel for
	for(int gpu = 0; gpu < 1; ++gpu){
		cudaSetDevice(gpu);
		cudaFree(_gpu_data[gpu]);
	}
}
		
template<typename T>
void mkt::DMatrix<T>::update_self() {
	if(_device_dist == mkt::Distribution::DIST){
		for(int gpu = 0; gpu < 1; ++gpu){
			cudaSetDevice(gpu);
			cudaMemcpyAsync(_host_data[gpu], _gpu_data[gpu], _bytes_gpu, cudaMemcpyDeviceToHost, mkt::cuda_streams[gpu]);
		}
	}else{
		cudaSetDevice(0);
		cudaMemcpyAsync(_host_data[0], _gpu_data[0], _bytes_gpu, cudaMemcpyDeviceToHost, mkt::cuda_streams[0]);
	}
	mkt::sync_streams();
}

template<typename T>
void mkt::DMatrix<T>::update_devices() {
	for(int gpu = 0; gpu < 1; ++gpu){
		cudaSetDevice(gpu);
		cudaMemcpyAsync(_gpu_data[gpu], _host_data[gpu], _bytes_gpu, cudaMemcpyHostToDevice, mkt::cuda_streams[gpu]);
	}
}

template<typename T>
T mkt::DMatrix<T>::get_local(size_t row, size_t column) {
	size_t index = row * _number_of_columns_local + column;
	return get_local(index);
}

template<typename T>
T mkt::DMatrix<T>::get_local(size_t index) {
	int gpu = get_gpu_by_local_index(index);
	cudaSetDevice(gpu);
	T* host_pointer = _data + index;
	T* gpu_pointer = _gpu_data[gpu] + (index % _size_gpu);
	cudaMemcpyAsync(host_pointer, gpu_pointer, sizeof(T), cudaMemcpyDeviceToHost, mkt::cuda_streams[gpu]);
	mkt::sync_streams();
    return _data[index];
}

template<typename T>
void mkt::DMatrix<T>::set_local(size_t row, size_t column, const T& v) {
	size_t index = row * _number_of_columns_local + column;
	set_local(index, v);
}

template<typename T>
void mkt::DMatrix<T>::set_local(size_t index, const T& v) {
	mkt::sync_streams();
	_data[index] = v;
	T* host_pointer = _data + index;
	if(_device_dist == mkt::Distribution::COPY){
		for(int gpu = 0; gpu < 1; ++gpu){
			cudaSetDevice(gpu);
			T* gpu_pointer = _gpu_data[gpu] + index;
			cudaMemcpyAsync(gpu_pointer, host_pointer, sizeof(T), cudaMemcpyHostToDevice, mkt::cuda_streams[gpu]);
		}
	}else if(_device_dist == mkt::Distribution::DIST){
		int gpu = get_gpu_by_local_index(index);
		cudaSetDevice(gpu);
		T* gpu_pointer = _gpu_data[gpu] + (index % _size_gpu);
		cudaMemcpyAsync(gpu_pointer, host_pointer, sizeof(T), cudaMemcpyHostToDevice, mkt::cuda_streams[gpu]);
	}
}

template<typename T>
T mkt::DMatrix<T>::get_global(size_t row, size_t column) const {
  // TODO
}

template<typename T>
void mkt::DMatrix<T>::set_global(size_t row, size_t column, const T& v) {
  // TODO
}

template<typename T>
T mkt::DMatrix<T>::get_local_host_data(size_t row, size_t column) const {
  return get_local_host_data(row * _number_of_columns_local + column);
}

template<typename T>
T mkt::DMatrix<T>::get_local_host_data(size_t local_index) const {
  return _data[local_index];
}

template<typename T>
void mkt::DMatrix<T>::set_local_host_data(size_t row, size_t column, T v) {
	set_local_host_data(row * _number_of_columns_local + column, v);
}

template<typename T>
void mkt::DMatrix<T>::set_local_host_data(size_t index, T v) {
	_data[index] = v;
}

template<typename T>
T& mkt::DMatrix<T>::operator[](size_t local_index) {
  return _data[local_index];
}

template<typename T>
const T& mkt::DMatrix<T>::operator[](size_t local_index) const {
  return _data[local_index];
}

template<typename T>
size_t mkt::DMatrix<T>::get_size() const {
  return _size;
}

template<typename T>
size_t mkt::DMatrix<T>::get_size_local() const {
  return _size_local;
}

template<typename T>
size_t mkt::DMatrix<T>::get_size_gpu() const {
  return _size_gpu;
}

template<typename T>
size_t mkt::DMatrix<T>::get_rows_gpu() const {
  return _rows_gpu;
}

template<typename T>
size_t mkt::DMatrix<T>::get_bytes_gpu() const {
  return _bytes_gpu;
}


template<typename T>
size_t mkt::DMatrix<T>::get_row_offset() const {
  return _row_offset;
}

template<typename T>
size_t mkt::DMatrix<T>::get_column_offset() const {
  return _column_offset;
}

template<typename T>
size_t mkt::DMatrix<T>::get_number_of_rows() const {
  return _number_of_rows;
}

template<typename T>
size_t mkt::DMatrix<T>::get_number_of_columns() const {
  return _number_of_columns;
}

template<typename T>
size_t mkt::DMatrix<T>::get_number_of_rows_local() const {
  return _number_of_rows_local;
}

template<typename T>
size_t mkt::DMatrix<T>::get_number_of_columns_local() const {
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
  return _data;
}

template<typename T>
T* mkt::DMatrix<T>::get_data() {
  return _data;
}

template<typename T>
auto mkt::DMatrix<T>::begin() noexcept{
  return _data;
}

template<typename T>
auto mkt::DMatrix<T>::end() noexcept{
  return _data + _size_local;
}

template<typename T>
auto mkt::DMatrix<T>::begin() const noexcept{
  return _data;
}

template<typename T>
auto mkt::DMatrix<T>::end() const noexcept{
  return _data + _size_local;
}		

template<typename T>
T* mkt::DMatrix<T>::get_device_pointer(int gpu) const{
  return _gpu_data[gpu];
}

template<typename T>
int mkt::DMatrix<T>::get_pid_by_global_index(size_t global_row, size_t global_column) const {
	size_t row_pos = global_row / _number_of_rows_local; // assumes even distribution
	size_t col_pos = global_column / _number_of_columns_local;
	return row_pos * _partitions_in_row + col_pos;
}

template<typename T>
bool mkt::DMatrix<T>::is_local(size_t global_row, size_t global_column) const {
	int pid = get_pid_by_global_index(global_row, global_column);
	return (pid == _pid);
}

template<typename T>
int mkt::DMatrix<T>::get_gpu_by_local_index(size_t local_index) const {
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
int mkt::DMatrix<T>::get_gpu_by_global_index(size_t global_index) const {
	// TODO
	return -1;
}
template<typename T, typename R, typename Functor>
void mkt::map(const mkt::DMatrix<T>& in, mkt::DMatrix<R>& out, Functor f) {
	
	size_t smem_bytes = f.get_smem_bytes();
	const size_t gpu_elements = in.get_size_gpu();
	for(int gpu = 0; gpu < 1; ++gpu){
		cudaSetDevice(gpu);
		f.init(gpu);
		T* in_devptr = in.get_device_pointer(gpu);
		R* out_devptr = out.get_device_pointer(gpu);
		
		dim3 dimBlock(1024);
	    dim3 dimGrid((gpu_elements + dimBlock.x - 1) / dimBlock.x);
	    mkt::kernel::map<<<dimGrid, dimBlock, smem_bytes, mkt::cuda_streams[gpu]>>>(in_devptr, out_devptr, gpu_elements, f);
	}
}

template<typename T, typename R, typename Functor>
void mkt::map_index(const mkt::DMatrix<T>& in, mkt::DMatrix<R>& out, Functor f) {
	size_t row_offset = in.get_row_offset();
	size_t column_offset = in.get_column_offset();
	size_t columns_local = in.get_number_of_columns_local();

	size_t smem_bytes = f.get_smem_bytes();

  	size_t gpu_elements = in.get_size_gpu();
  	size_t rows_on_gpu = in.get_rows_gpu();
	for(int gpu = 0; gpu < 1; ++gpu){
		cudaSetDevice(gpu);
		f.init(gpu);
		T* in_devptr = in.get_device_pointer(gpu);
		R* out_devptr = out.get_device_pointer(gpu);
		
		size_t gpu_row_offset = row_offset;
		size_t gpu_column_offset = column_offset;
		
		if(in.get_device_distribution() == mkt::Distribution::DIST){
			gpu_row_offset += gpu * rows_on_gpu;
		}
		
		dim3 dimBlock(32, 32);
	    dim3 dimGrid((columns_local + dimBlock.x - 1) / dimBlock.x, (rows_on_gpu + dimBlock.y - 1) / dimBlock.y);
	    mkt::kernel::map_index<<<dimGrid, dimBlock, smem_bytes, mkt::cuda_streams[gpu]>>>(in_devptr, out_devptr, rows_on_gpu, columns_local, gpu_row_offset, gpu_column_offset, f);
	}
}

template<typename T, typename R, typename Functor>
void mkt::map_local_index(const mkt::DMatrix<T>& in, mkt::DMatrix<R>& out, Functor f) {
	size_t columns_local = in.get_number_of_columns_local();

	size_t smem_bytes = f.get_smem_bytes();

  	size_t gpu_elements = in.get_size_gpu();
  	size_t rows_on_gpu = in.get_rows_gpu();
	for(int gpu = 0; gpu < 1; ++gpu){
		cudaSetDevice(gpu);
		f.init(gpu);
		T* in_devptr = in.get_device_pointer(gpu);
		R* out_devptr = out.get_device_pointer(gpu);
		
		size_t gpu_row_offset = 0;
		size_t gpu_column_offset = 0;
		
		if(in.get_device_distribution() == mkt::Distribution::DIST){
			gpu_row_offset += gpu * rows_on_gpu;
		}
		
		dim3 dimBlock(32, 32);
	    dim3 dimGrid((columns_local + dimBlock.x - 1) / dimBlock.x, (rows_on_gpu + dimBlock.y - 1) / dimBlock.y);
	    mkt::kernel::map_index<<<dimGrid, dimBlock, smem_bytes, mkt::cuda_streams[gpu]>>>(in_devptr, out_devptr, rows_on_gpu, columns_local, gpu_row_offset, gpu_column_offset, f);
	}
}

template<typename T, typename Functor>
void mkt::map_in_place(mkt::DMatrix<T>& m, Functor f) {
	size_t smem_bytes = f.get_smem_bytes();
	const size_t gpu_elements = m.get_size_gpu();
	for(int gpu = 0; gpu < 1; ++gpu){
		cudaSetDevice(gpu);
		f.init(gpu);
		T* m_devptr = m.get_device_pointer(gpu);
		
		dim3 dimBlock(1024);
	    dim3 dimGrid((gpu_elements + dimBlock.x - 1) / dimBlock.x);
	    mkt::kernel::map_in_place<<<dimGrid, dimBlock, smem_bytes, mkt::cuda_streams[gpu]>>>(m_devptr, gpu_elements, f);
	}
}

template<typename T, typename Functor>
void mkt::map_index_in_place(mkt::DMatrix<T>& m, Functor f){
	size_t row_offset = m.get_row_offset();
	size_t column_offset = m.get_column_offset();
	size_t columns_local = m.get_number_of_columns_local();

	size_t smem_bytes = f.get_smem_bytes();

  	size_t gpu_elements = m.get_size_gpu();
  	size_t rows_on_gpu = m.get_rows_gpu();
	for(int gpu = 0; gpu < 1; ++gpu){
		cudaSetDevice(gpu);
		f.init(gpu);
		T* m_devptr = m.get_device_pointer(gpu);
		
		size_t gpu_row_offset = row_offset;
		size_t gpu_column_offset = column_offset;
		
		if(m.get_device_distribution() == mkt::Distribution::DIST){
			gpu_row_offset += gpu * rows_on_gpu;
		}
		
		dim3 dimBlock(32, 32);
	    dim3 dimGrid((columns_local + dimBlock.x - 1) / dimBlock.x, (rows_on_gpu + dimBlock.y - 1) / dimBlock.y);
	    mkt::kernel::map_index_in_place<<<dimGrid, dimBlock, smem_bytes, mkt::cuda_streams[gpu]>>>(m_devptr, rows_on_gpu, columns_local, gpu_row_offset, gpu_column_offset, f);
	}
}	

template<typename T, typename Functor>
void mkt::map_local_index_in_place(mkt::DMatrix<T>& m, Functor f){
	size_t columns_local = m.get_number_of_columns_local();

	size_t smem_bytes = f.get_smem_bytes();

  	size_t gpu_elements = m.get_size_gpu();
  	size_t rows_on_gpu = m.get_rows_gpu();
	for(int gpu = 0; gpu < 1; ++gpu){
		cudaSetDevice(gpu);
		f.init(gpu);
		T* m_devptr = m.get_device_pointer(gpu);
		
		size_t gpu_row_offset = 0;
		size_t gpu_column_offset = 0;
		
		if(m.get_device_distribution() == mkt::Distribution::DIST){
			gpu_row_offset += gpu * rows_on_gpu;
		}
		
		dim3 dimBlock(32, 32);
	    dim3 dimGrid((columns_local + dimBlock.x - 1) / dimBlock.x, (rows_on_gpu + dimBlock.y - 1) / dimBlock.y);
	    mkt::kernel::map_index_in_place<<<dimGrid, dimBlock, smem_bytes, mkt::cuda_streams[gpu]>>>(m_devptr, rows_on_gpu, columns_local, gpu_row_offset, gpu_column_offset, f);
	}
}
		
template<typename T>
mkt::DeviceMatrix<T>::DeviceMatrix(const DMatrix<T>& dm)
    : _size(dm.get_size()),
      _size_local(dm.get_size_local()),
      _size_device(dm.get_size_gpu()),
      _rows_device(dm.get_rows_gpu()),
      _columns_device(dm.get_number_of_columns_local()),
      _bytes_device(dm.get_bytes_gpu()),
      _row_offset(dm.get_row_offset()),
      _column_offset(dm.get_column_offset()),
      _device_row_offset(0),
      _device_column_offset(0),
      _dist(dm.get_distribution()),
      _device_dist(dm.get_device_distribution()) 
{
	_device_data = nullptr;
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
      _bytes_device(dm._bytes_device),
      _row_offset(dm._row_offset),
      _column_offset(dm._column_offset),
      _device_row_offset(dm._device_row_offset),
      _device_column_offset(dm._device_column_offset),
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
	if(_device_dist == mkt::Distribution::COPY){
		_device_row_offset = 0;
		_device_column_offset = 0;
	} else {
		_device_row_offset = _rows_device * gpu;
		_device_column_offset = 0;
	}
	    
	_device_data = _gpu_data[gpu];
}

template<typename T>
__device__ const T& mkt::DeviceMatrix<T>::get_data_device(size_t device_index) const {
  return _device_data[device_index];
}

template<typename T>
__device__ const T& mkt::DeviceMatrix<T>::get_data_device(size_t device_row, size_t device_column) const {
  return this->get_data_device(device_row * _columns_device + device_column);
}

template<typename T>
__device__ const T& mkt::DeviceMatrix<T>::get_data_local(size_t local_row, size_t local_column) const {
  return this->get_data_device(local_row - _device_row_offset, local_column - _device_column_offset);
}



template<typename T>
void mkt::print(std::ostringstream& stream, const T& a) {
	if(std::is_fundamental<T>::value){
		stream << a;
	}
}



template<>
void mkt::gather<double>(mkt::DMatrix<double>& in, mkt::DMatrix<double>& out){
	in.update_self();
	std::copy(in.get_data(), in.get_data() + in.get_size_local(), out.get_data());
	out.update_devices();
}
	
template<typename T>
void mkt::scatter(mkt::DMatrix<T>& in, mkt::DMatrix<T>& out){
	in.update_self();
	std::copy(in.get_data(), in.get_data() + in.get_size(), out.get_data());
	out.update_devices();
}
