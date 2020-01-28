package de.wwu.musket.generator.cuda.lib

import org.apache.log4j.LogManager
import org.apache.log4j.Logger
import org.eclipse.emf.ecore.resource.Resource
import org.eclipse.xtext.generator.IFileSystemAccess2
import org.eclipse.xtext.generator.IGeneratorContext

import static extension de.wwu.musket.generator.cuda.DataGenerator.*
import static extension de.wwu.musket.generator.cuda.StructGenerator.*
import static extension de.wwu.musket.generator.extensions.ModelElementAccess.*
import de.wwu.musket.generator.cuda.Config

class GPUArray {
	private static final Logger logger = LogManager.getLogger(GPUArray)

	def static generateGPUArrayDeclaration() '''		
		template<typename T>
		class GPUArray {
		 public:
		
		  // CONSTRUCTORS / DESTRUCTOR
		  GPUArray(int pid, size_t size, size_t size_local, int init, T init_value, int partitions, int partition_pos, size_t offset, mkt::Distribution d = DIST, mkt::Distribution device_dist = DIST, int view);
		  ~ GPUArray();
		  
		  template<std::size_t N> 
		  void operator=(const std::array<T, N>& a);

		   void update_devices();
		   
		// Getter and Setter
		
		  T get_global(size_t index);
		  void set_global(size_t index, const T& value);
		
		  T get_local(size_t index);
		  void set_local(size_t index, const T& value);
		
		  T& operator[](size_t local_index);
		  const T& operator[](size_t local_index) const;
		
		  size_t get_size() const;
		  size_t get_size_local() const;
		  size_t get_size_gpu() const;
		  size_t get_bytes_gpu() const;
		
		  size_t get_offset() const;
				
		  mkt::Distribution get_distribution() const;
		  mkt::Distribution get_device_distribution() const;

		  T* get_device_pointer(int gpu) const;
				
		 private:
		
		  int get_gpu_by_local_index(size_t local_index) const;
		  int get_gpu_by_global_index(size_t global_index) const;
			
		  int get_pid_by_global_index(size_t global_index) const;
		  bool is_local(size_t global_index) const;
		
		  //
		  // Attributes
		  //
		
		  // position of processor in data parallel group of processors; zero-base
		  int _pid;
		
		  size_t _size;
		  size_t _size_local;
		  size_t _size_gpu;
		  size_t _bytes_gpu;
		
		  // number of (local) partitions in array
		  int _partitions;
		
		  // position of processor in data parallel group of processors
		  int _partition_pos;
		
		  // first index in local partition
		  size_t _offset;
		
		  // checks whether data is copy distributed among all processes
		  mkt::Distribution _dist;
		  mkt::Distribution _device_dist;
		
		  std::array<T*, «Config.gpus»> _gpu_data;
		};
	'''
	
	def static generateGPUArrayDefinition() '''
		template<typename T>
		mkt::GPUArray<T>::GPUArray(int pid, size_t size, size_t size_local, int init, T init_value, int partitions, int partition_pos, size_t offset, mkt::Distribution d, mkt::Distribution device_dist)
		    : _pid(pid),
		      _size(size),
		      _size_local(size_local),
		      _size_gpu(0), 
		      _partitions(partitions),
		      _partition_pos(partition_pos),
		      _offset(offset),
		      _dist(d),
		      _device_dist(device_dist) {
		    
		    if(device_dist == mkt::Distribution::DIST){
		    	_size_gpu = size_local / «Config.gpus»; // assume even distribution for now
		    }else if(device_dist == mkt::Distribution::COPY){
		    	_size_gpu = size_local;
		    }
		    _bytes_gpu = _size_gpu * sizeof(T);
		    
			for(int gpu = 0; gpu < «Config.gpus»; ++gpu){
				cudaSetDevice(gpu);
				
				// allocate memory
				T* devptr;
				cudaMalloc((void**)&devptr, _size_gpu * sizeof(T));
				
				// store pointer to device memory and host memory
				_gpu_data[gpu] = devptr;
				if(init == 1){
					for(size_t i = 0; i< _size_local; ++i){
						// TODO kernel dafür
						_gpu_data[gpu] = init_value;
					}	
				}
			}
		}
		
		template<typename T>
		mkt::GPUArray<T>::~GPUArray(){
			for(int gpu = 0; gpu < «Config.gpus»; ++gpu){
				cudaSetDevice(gpu);
				cudaFree(_gpu_data[gpu]);
			}
		}
				
		template<typename T>
		template<std::size_t N>
		void mkt::GPUArray<T>::operator=(const std::array<T, N>& a) {
		  mkt::sync_streams();
		  // TODO kernel dafür
		  for(size_t element = 0; element < _size_local; ++element){
			_gpu_data[element] = a[element];
		  }
		}	
		
		template<typename T>
		void mkt::GPUArray<T>::update_devices() {
			//TODO: MPI sync zwischen GPUs?
		}
				
		template<typename T>
		T mkt::GPUArray<T>::get_local(size_t index) {
			// was wollen wir hier? Bei einem Host call müssten wir erst eine Variable auf dem Host erstellen 
			// Wenn wir die Daten auf der GPU brauchen reicht ein gpu_pointer (weil local)
			/*int gpu = get_gpu_by_local_index(index);
			cudaSetDevice(gpu);
			T* host_pointer = _data + index;
			T* gpu_pointer = _gpu_data[gpu] + (index % _size_gpu );
			cudaMemcpyAsync(host_pointer, gpu_pointer, sizeof(T), cudaMemcpyDeviceToHost, mkt::cuda_streams[gpu]);
			mkt::sync_streams();
		    return _data[index];*/
		}
		
		template<typename T>
		void mkt::GPUArray<T>::set_local(size_t index, const T& v) {
			// Siehe get_local
			/*mkt::sync_streams();
			_data[index] = v;
			T* host_pointer = _data + index;
			if(_device_dist == mkt::Distribution::COPY){
				for(int gpu = 0; gpu < «Config.gpus»; ++gpu){
					cudaSetDevice(gpu);
					T* gpu_pointer = _gpu_data[gpu] + index;
					cudaMemcpyAsync(gpu_pointer, host_pointer, sizeof(T), cudaMemcpyHostToDevice, mkt::cuda_streams[gpu]);
				}
			}else if(_device_dist == mkt::Distribution::DIST){
				int gpu = get_gpu_by_local_index(index);
				cudaSetDevice(gpu);
				T* gpu_pointer = _gpu_data[gpu] + (index % _size_gpu );
				cudaMemcpyAsync(gpu_pointer, host_pointer, sizeof(T), cudaMemcpyHostToDevice, mkt::cuda_streams[gpu]);
			}*/
		}

		template<typename T>
		int mkt::GPUArray<T>::get_pid_by_global_index(size_t global_index) const {
		  return global_index / _size_local;
		}
		
		template<typename T>
		bool mkt::GPUArray<T>::is_local(size_t global_index) const {
			int pid = get_pid_by_global_index(global_index);
		  return (pid == _pid);
		}

		template<typename T>
		T mkt::GPUArray<T>::get_global(size_t index) {
		  	// TODO Bei einem Host call müssten wir erst eine Variable auf dem Host erstellen 
		  	// Wenn wir die Daten auf der GPU brauchen reicht ein gpu_pointer (weil local)
		  	// T result;
		  	// if(is_local(global_index)){
		  	// 	size_t local_index = global_index - _offset;
		  	// 	result = get_local(local_index);
		  	// 	//MPI_Bcast(&result, 1, MPI_Datatype datatype, _pid, )
		  	// }else{
		  	// 	int root = get_pid_by_global_index(global_index);
		  	// 	//MPI_Bcast(&result, 1, MPI_Datatype datatype, root, )
		  	// }
		  	// return result;
		}
		
		template<typename T>
		void mkt::GPUArray<T>::set_global(size_t index, const T& v) {
		  // TODO siehe get_global/get_local
		}
		
		template<typename T>
		T& mkt::GPUArray<T>::operator[](size_t local_index) {
			// todo cp one element of type T and return
		}
		
		template<typename T>
		const T& mkt::GPUArray<T>::operator[](size_t local_index) const {
			// todo cp one element of type T and return
		}
		
		template<typename T>
		size_t mkt::GPUArray<T>::get_size() const {
		  return _size;
		}
		
		template<typename T>
		size_t mkt::GPUArray<T>::get_size_local() const {
		  return _size_local;
		}
		
		template<typename T>
		size_t mkt::GPUArray<T>::get_size_gpu() const {
		  return _size_gpu;
		}
		
		template<typename T>
		size_t mkt::GPUArray<T>::get_bytes_gpu() const {
		  return _bytes_gpu;
		}
		
		template<typename T>
		size_t mkt::GPUArray<T>::get_offset() const {
		  return _offset;
		}
		
		template<typename T>
		mkt::Distribution mkt::GPUArray<T>::get_distribution() const {
		  return _dist;
		}
		
		template<typename T>
		mkt::Distribution mkt::GPUArray<T>::get_device_distribution() const {
		  return _device_dist;
		}
		
		template<typename T>
		T* mkt::GPUArray<T>::get_device_pointer(int gpu) const{
		  return _gpu_data[gpu];
		}
		
		template<typename T>
		int mkt::GPUArray<T>::get_gpu_by_local_index(size_t local_index) const {
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
		int mkt::GPUArray<T>::get_gpu_by_global_index(size_t global_index) const {
			// TODO
			return -1;
		}
	'''
	
	
	def static generateGPUArrayDeclarations() '''
		template<typename T, typename R, typename Functor>
		void map(const mkt::GPUArray<T>& in, mkt::GPUArray<R>& out, Functor f);
		
		template<typename T, typename R, typename Functor>
		void map_index(const mkt::GPUArray<T>& in, mkt::GPUArray<R>& out, Functor f);
		
		template<typename T, typename R, typename Functor>
		void map_local_index(const mkt::GPUArray<T>& in, mkt::GPUArray<R>& out, Functor f);
		
		template<typename T, typename Functor>
		void map_in_place(mkt::GPUArray<T>& a, Functor f);
		
		template<typename T, typename Functor>
		void map_index_in_place(mkt::GPUArray<T>& a, Functor f);
		
		template<typename T, typename Functor>
		void map_local_index_in_place(mkt::GPUArray<T>& a, Functor f);
		
		template<typename T, typename Functor>
		void fold(const mkt::GPUArray<T>& a, T& out, const T identity, const Functor f);
		
		template<typename T, typename Functor>
		void fold_copy(const mkt::GPUArray<T>& a, T& out, const T identity, const Functor f);
		
		template<typename T, typename R, typename MapFunctor, typename FoldFunctor>
		void map_fold(const mkt::GPUArray<T>& a, R& out, const MapFunctor& f_map, const R identity, const FoldFunctor f_fold);
		
		template<typename T, typename R, typename MapFunctor, typename FoldFunctor>
		void map_fold_copy(const mkt::GPUArray<T>& a, R& out, const MapFunctor& f_map, const R identity, const FoldFunctor f_fold);

		template<typename T, typename R, typename I, typename MapFunctor, typename FoldFunctor>
		void map_fold(const mkt::GPUArray<T>& a, mkt::GPUArray<R>& out, const MapFunctor& f_map, const I identity, const FoldFunctor f_fold);

		template<typename T, typename R, typename I, typename MapFunctor, typename FoldFunctor>
		void map_fold_copy(const mkt::GPUArray<T>& a, mkt::GPUArray<R>& out, const MapFunctor f_map, const I identity, const FoldFunctor f_fold);
	'''
	
	def static generateGPUArraySkeletonDefinitions() '''
		template<typename T, typename R, typename Functor>
		void mkt::map(const mkt::GPUArray<T>& in, mkt::GPUArray<R>& out, Functor f) {
			size_t gpu_elements = in.get_size_gpu();
						  
			for(int gpu = 0; gpu < «Config.gpus»; ++gpu){
				cudaSetDevice(gpu);
				f.init(gpu);
				T* in_devptr = in.get_device_pointer(gpu);
				R* out_devptr = out.get_device_pointer(gpu);

				size_t smem_bytes = f.get_smem_bytes();
				
				dim3 dimBlock(«Config.threads»);
				dim3 dimGrid((gpu_elements+dimBlock.x-1)/dimBlock.x);
				mkt::kernel::map<<<dimGrid, dimBlock, smem_bytes, mkt::cuda_streams[gpu]>>>(in_devptr, out_devptr, gpu_elements, f);
			}
		}
		
		template<typename T, typename R, typename Functor>
		void mkt::map_index(const mkt::GPUArray<T>& in, mkt::GPUArray<R>& out, Functor f) {
			size_t offset = in.get_offset();
			size_t gpu_elements = in.get_size_gpu();
						  
			for(int gpu = 0; gpu < «Config.gpus»; ++gpu){
				cudaSetDevice(gpu);
				f.init(gpu);
				T* in_devptr = in.get_device_pointer(gpu);
				R* out_devptr = out.get_device_pointer(gpu);
				
				size_t gpu_offset = offset;
				if(in.get_device_distribution() == mkt::Distribution::DIST){
					gpu_offset += gpu * gpu_elements;
				}

				size_t smem_bytes = f.get_smem_bytes();
				
				dim3 dimBlock(«Config.threads»);
				dim3 dimGrid((gpu_elements+dimBlock.x-1)/dimBlock.x);
				mkt::kernel::map_index<<<dimGrid, dimBlock, smem_bytes, mkt::cuda_streams[gpu]>>>(in_devptr, out_devptr, gpu_elements, gpu_offset, f);
			}
		}
		
		template<typename T, typename R, typename Functor>
		void mkt::map_local_index(const mkt::GPUArray<T>& in, mkt::GPUArray<R>& out, Functor f) {
			size_t gpu_elements = in.get_size_gpu();
						  
			for(int gpu = 0; gpu < «Config.gpus»; ++gpu){
				cudaSetDevice(gpu);
				f.init(gpu);
				T* in_devptr = in.get_device_pointer(gpu);
				R* out_devptr = out.get_device_pointer(gpu);
				
				size_t gpu_offset = 0;
				if(in.get_device_distribution() == mkt::Distribution::DIST){
					gpu_offset += gpu * gpu_elements;
				}

				size_t smem_bytes = f.get_smem_bytes();
				
				dim3 dimBlock(«Config.threads»);
				dim3 dimGrid((gpu_elements+dimBlock.x-1)/dimBlock.x);
				mkt::kernel::map_index<<<dimGrid, dimBlock, smem_bytes, mkt::cuda_streams[gpu]>>>(in_devptr, out_devptr, gpu_elements, gpu_offset, f);
			}
		}
		
		template<typename T, typename Functor>
		void mkt::map_in_place(mkt::GPUArray<T>& a, Functor f){
			size_t gpu_elements = a.get_size_gpu();
						  
			for(int gpu = 0; gpu < «Config.gpus»; ++gpu){
				cudaSetDevice(gpu);
				f.init(gpu);
				T* devptr = a.get_device_pointer(gpu);

				size_t smem_bytes = f.get_smem_bytes();
				
				dim3 dimBlock(«Config.threads»);
				dim3 dimGrid((gpu_elements+dimBlock.x-1)/dimBlock.x);
				mkt::kernel::map_in_place<<<dimGrid, dimBlock, smem_bytes, mkt::cuda_streams[gpu]>>>(devptr, gpu_elements, f);
			}
		}
		
		template<typename T, typename Functor>
		void mkt::map_index_in_place(mkt::GPUArray<T>& a, Functor f){
			size_t offset = a.get_offset();
			size_t gpu_elements = a.get_size_gpu();
						  
			for(int gpu = 0; gpu < «Config.gpus»; ++gpu){
				cudaSetDevice(gpu);
				f.init(gpu);
				T* devptr = a.get_device_pointer(gpu);
				
				size_t gpu_offset = offset;
				if(a.get_device_distribution() == mkt::Distribution::DIST){
					gpu_offset += gpu * gpu_elements;
				}

				size_t smem_bytes = f.get_smem_bytes();
				
				dim3 dimBlock(«Config.threads»);
				dim3 dimGrid((gpu_elements+dimBlock.x-1)/dimBlock.x);
				mkt::kernel::map_index_in_place<<<dimGrid, dimBlock, smem_bytes, mkt::cuda_streams[gpu]>>>(devptr, gpu_elements, gpu_offset, f);
			}
		}
		
		template<typename T, typename Functor>
		void mkt::map_local_index_in_place(mkt::GPUArray<T>& a, Functor f){
			size_t gpu_elements = a.get_size_gpu();
						  
			for(int gpu = 0; gpu < «Config.gpus»; ++gpu){
				cudaSetDevice(gpu);
				f.init(gpu);
				T* devptr = a.get_device_pointer(gpu);
				
				size_t gpu_offset = 0;
				if(a.get_device_distribution() == mkt::Distribution::DIST){
					gpu_offset = gpu * gpu_elements;
				}

				size_t smem_bytes = f.get_smem_bytes();
				
				dim3 dimBlock(«Config.threads»);
				dim3 dimGrid((gpu_elements+dimBlock.x-1)/dimBlock.x);
				mkt::kernel::map_index_in_place<<<dimGrid, dimBlock, smem_bytes, mkt::cuda_streams[gpu]>>>(devptr, gpu_elements, gpu_offset, f);
			}
		}
	'''
}