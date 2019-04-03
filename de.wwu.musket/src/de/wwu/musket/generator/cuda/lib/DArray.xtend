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

class DArray {
	private static final Logger logger = LogManager.getLogger(DArray)

	def static generateDArrayDeclaration() '''		
		template<typename T>
		class DArray {
		 public:
		
		  // CONSTRUCTORS / DESTRUCTOR
		  DArray(int pid, size_t size, size_t size_local, T init_value, int partitions, int partition_pos, size_t offset, mkt::Distribution d = DIST, Distribution device_dist = DIST);
		  ~ DArray();
		  
		  template<std::size_t N> 
		  void operator=(const std::array<T, N>& a);
		  
		   void update_self();
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
				
		  Distribution get_distribution() const;
		  Distribution get_device_distribution() const;
				
		  T* get_data();
		  const T* get_data() const;
		  
		  T* get_device_pointer(int gpu) const;
				
		 private:
		
		  int get_gpu_by_local_index(size_t local_index) const;
		  int get_gpu_by_global_index(size_t global_index) const;
		
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
		  Distribution _dist;
		  Distribution _device_dist;
		
		  T* _data;
		  std::array<T*, «Config.gpus»> _host_data;
		  std::array<T*, «Config.gpus»> _gpu_data;
		};
	'''
	
	def static generateDArrayDefinition() '''
		template<typename T>
		mkt::DArray<T>::DArray(int pid, size_t size, size_t size_local, T init_value, int partitions, int partition_pos, size_t offset, Distribution d, Distribution device_dist)
		    : _pid(pid),
		      _size(size),
		      _size_local(size_local),
		      _size_gpu(0), 
		      _partitions(partitions),
		      _partition_pos(partition_pos),
		      _offset(offset),
		      _dist(d),
		      _device_dist(device_dist),
		      _data(size_local, init_value) {
		    
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
				cudaMallocHost((void**)&_data, _size_local * sizeof(T));
				
				
				// store pointer to device memory and host memory
				_gpu_data[gpu] = devptr;
				if(device_dist == mkt::Distribution::DIST){
			    	_host_data[gpu] = _data + gpu * _size_gpu;
			    }else if(device_dist == mkt::Distribution::COPY){
			    	_host_data[gpu] = _data; // all gpus have complete data, thus point to the beginning of host vector
			    }		
			}
			
			//init data
			for(size_t i = 0; i< _size_local; ++i){
				_data[i] = init_value;
			}
			update_devices();
		}
		
		template<typename T>
		mkt::DArray<T>::~DArray(){
			cudaFreeHost(_data);
			for(int gpu = 0; gpu < 4; ++gpu){
				cudaSetDevice(gpu);
				cudaFree(_gpu_data[gpu]);
			}
		}
				
		template<typename T>
		template<std::size_t N>
		void mkt::DArray<T>::operator=(const std::array<T, N>& a) {
		  mkt::sync_streams();
		  for(size_t element = 0; element < _size_local; ++element){
			_data[element] = a[element];
		  }
		  update_devices();
		}	
				
		template<typename T>
		void mkt::DArray<T>::update_self() {
			if(_device_dist == Distribution::DIST){
				for(int gpu = 0; gpu < «Config.gpus»; ++gpu){
					cudaSetDevice(gpu);
					cudaMemcpyAsync(_host_data[gpu], _gpu_data[gpu], _bytes_gpu, cudaMemcpyDeviceToHost, mkt::cuda_streams[gpu]);
				}
			}else{
				cudaSetDevice(0);
				cudaMemcpyAsync(_host_data[0], _gpu_data[0], _bytes_gpu, cudaMemcpyDeviceToHost, mkt::cuda_streams[0]);
			}	
		}
		
		template<typename T>
		void mkt::DArray<T>::update_devices() {
			for(int gpu = 0; gpu < «Config.gpus»; ++gpu){
				cudaSetDevice(gpu);
				cudaMemcpyAsync(_gpu_data[gpu], _host_data[gpu], _bytes_gpu, cudaMemcpyHostToDevice, mkt::cuda_streams[gpu]);
			}
		}
				
		template<typename T>
		T mkt::DArray<T>::get_local(size_t index) {
			int gpu = get_gpu_by_local_index(index);
			cudaSetDevice(gpu);
			T* host_pointer = _data + index;
			T* gpu_pointer = _gpu_data[gpu] + (index % _size_gpu );
			cudaMemcpyAsync(host_pointer, gpu_pointer, sizeof(T), cudaMemcpyDeviceToHost, mkt::cuda_streams[gpu]);
			mkt::sync_streams();
		    return _data[index];
		}
		
		template<typename T>
		void mkt::DArray<T>::set_local(size_t index, const T& v) {
			mkt::sync_streams();
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
			}
		}

		template<typename T>
		T mkt::DArray<T>::get_host_local(int index) {
			return _data[index];
		}
		
		template<typename T>
		void mkt::DArray<T>::set_host_local(int index, T v) {
			_data[index] = v;
		}

		template<typename T>
		int mkt::DArray<T>::get_pid_by_global_index(size_t global_index) const {
		  return global_index / _size_local;
		}
		
		template<typename T>
		bool mkt::DArray<T>::is_local(size_t global_index) const {
			int pid = get_pid_by_global_index(global_index);
		  return (pid == _pid);
		}

		template<typename T>
		T mkt::DArray<T>::get_global(size_t index) {
		  	// TODO
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
		void mkt::DArray<T>::set_global(size_t index, const T& v) {
		  // TODO
		}
		
		template<typename T>
		T& mkt::DArray<T>::operator[](size_t local_index) {
		  	return _data[local_index];
		}
		
		template<typename T>
		const T& mkt::DArray<T>::operator[](size_t local_index) const {
		  	return _data[local_index];
		}
		
		template<typename T>
		size_t mkt::DArray<T>::get_size() const {
		  return _size;
		}
		
		template<typename T>
		size_t mkt::DArray<T>::get_size_local() const {
		  return _size_local;
		}
		
		template<typename T>
		size_t mkt::DArray<T>::get_size_gpu() const {
		  return _size_gpu;
		}
		
		template<typename T>
		size_t mkt::DArray<T>::get_bytes_gpu() const {
		  return _bytes_gpu;
		}
		
		template<typename T>
		size_t mkt::DArray<T>::get_offset() const {
		  return _offset;
		}
		
		template<typename T>
		mkt::Distribution mkt::DArray<T>::get_distribution() const {
		  return _dist;
		}
		
		template<typename T>
		mkt::Distribution mkt::DArray<T>::get_device_distribution() const {
		  return _device_dist;
		}
		
		template<typename T>
		const T* mkt::DArray<T>::get_data() const {
		  return _data;
		}
		
		template<typename T>
		T* mkt::DArray<T>::get_data() {
		  return _data;
		}
		
		template<typename T>
		T* mkt::DArray<T>::get_device_pointer(int gpu) const{
		  return _gpu_data[gpu];
		}
		
		template<typename T>
		int mkt::DArray<T>::get_gpu_by_local_index(size_t local_index) const {
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
		int mkt::DArray<T>::get_gpu_by_global_index(size_t global_index) const {
			// TODO
			return -1;
		}
	'''
	
	
	def static generateDArraySkeletonDeclarations() '''
		template<typename T, typename R, typename Functor>
		void map(const mkt::DArray<T>& in, mkt::DArray<R>& out, Functor f);
		
		template<typename T, typename R, typename Functor>
		void map_index(const mkt::DArray<T>& in, mkt::DArray<R>& out, Functor f);
		
		template<typename T, typename R, typename Functor>
		void map_local_index(const mkt::DArray<T>& in, mkt::DArray<R>& out, Functor f);
		
		template<typename T, typename Functor>
		void map_in_place(mkt::DArray<T>& a, Functor f);
		
		template<typename T, typename Functor>
		void map_index_in_place(mkt::DArray<T>& a, Functor f);
		
		template<typename T, typename Functor>
		void map_local_index_in_place(mkt::DArray<T>& a, Functor f);
		
		template<typename T, typename Functor>
		void fold(const mkt::DArray<T>& a, T& out, const T identity, const Functor f);
		
		template<typename T, typename Functor>
		void fold_copy(const mkt::DArray<T>& a, T& out, const T identity, const Functor f);
		
		template<typename T, typename R, typename MapFunctor, typename FoldFunctor>
		void map_fold(const mkt::DArray<T>& a, R& out, const MapFunctor& f_map, const R identity, const FoldFunctor f_fold);
		
		template<typename T, typename R, typename MapFunctor, typename FoldFunctor>
		void map_fold_copy(const mkt::DArray<T>& a, R& out, const MapFunctor& f_map, const R identity, const FoldFunctor f_fold);

		template<typename T, typename R, typename I, typename MapFunctor, typename FoldFunctor>
		void map_fold(const mkt::DArray<T>& a, mkt::DArray<R>& out, const MapFunctor& f_map, const I identity, const FoldFunctor f_fold);

		template<typename T, typename R, typename I, typename MapFunctor, typename FoldFunctor>
		void map_fold_copy(const mkt::DArray<T>& a, mkt::DArray<R>& out, const MapFunctor f_map, const I identity, const FoldFunctor f_fold);
	'''
	
	def static generateDArraySkeletonDefinitions() '''
		template<typename T, typename R, typename Functor>
		void mkt::map(const mkt::DArray<T>& in, mkt::DArray<R>& out, Functor f) {
«««			//«IF Config.cores > 1»#pragma omp parallel for«ENDIF»
«««			for(int gpu = 0; gpu < «Config.gpus»; ++gpu){
«««				acc_set_device_num(gpu, acc_device_not_host);
«««				f.init(gpu);
«««				T* in_devptr = in.get_device_pointer(gpu);
«««				R* out_devptr = out.get_device_pointer(gpu);
«««				const size_t gpu_elements = in.get_size_gpu();
«««				#pragma acc parallel loop deviceptr(in_devptr, out_devptr) firstprivate(f) async(0)
«««				for(size_t i = 0; i < gpu_elements; ++i) {
«««					f.set_id(__pgi_gangidx(), __pgi_workeridx(),__pgi_vectoridx());
«««					out_devptr[i] = f(in_devptr[i]);
«««				}
«««			}
		}
		
		template<typename T, typename R, typename Functor>
		void mkt::map_index(const mkt::DArray<T>& in, mkt::DArray<R>& out, Functor f) {
«««			size_t offset = in.get_offset();
«««			size_t gpu_elements = in.get_size_gpu();
«««			
«««			//«IF Config.cores > 1»#pragma omp parallel for«ENDIF»
«««			for(int gpu = 0; gpu < «Config.gpus»; ++gpu){
«««				acc_set_device_num(gpu, acc_device_not_host);
«««				f.init(gpu);
«««				T* in_devptr = in.get_device_pointer(gpu);
«««				R* out_devptr = out.get_device_pointer(gpu);
«««				
«««				size_t gpu_offset = offset;
«««				if(in.get_device_distribution() == mkt::Distribution::DIST){
«««					gpu_offset += gpu * gpu_elements;
«««				}
«««				
«««				#pragma acc parallel loop deviceptr(in_devptr, out_devptr) firstprivate(f) async(0)
«««				for(size_t i = 0; i < gpu_elements; ++i) {
«««					f.set_id(__pgi_gangidx(), __pgi_workeridx(),__pgi_vectoridx());
«««					out_devptr[i] = f(i + gpu_offset, in_devptr[i]);
«««				}
«««			}
		}
		
		template<typename T, typename R, typename Functor>
		void mkt::map_local_index(const mkt::DArray<T>& in, mkt::DArray<R>& out, Functor f) {
«««			size_t gpu_elements = in.get_size_gpu();
«««			
«««			//«IF Config.cores > 1»#pragma omp parallel for«ENDIF»
«««			for(int gpu = 0; gpu < «Config.gpus»; ++gpu){
«««				acc_set_device_num(gpu, acc_device_not_host);
«««				f.init(gpu);
«««				T* in_devptr = in.get_device_pointer(gpu);
«««				R* out_devptr = out.get_device_pointer(gpu);
«««				
«««				size_t gpu_offset = 0;
«««				if(in.get_device_distribution() == mkt::Distribution::DIST){
«««					gpu_offset = gpu * gpu_elements;
«««				}
«««				#pragma acc parallel loop deviceptr(in_devptr, out_devptr) firstprivate(f) async(0)
«««				for(size_t i = 0; i < gpu_elements; ++i) {
«««					f.set_id(__pgi_gangidx(), __pgi_workeridx(),__pgi_vectoridx());
«««					out_devptr[i] = f(i + gpu_offset, in_devptr[i]);
«««				}
«««			}
		}
		
		template<typename T, typename Functor>
		void mkt::map_in_place(mkt::DArray<T>& a, Functor f){
«««			size_t gpu_elements = a.get_size_gpu();
«««			
«««		  //«IF Config.cores > 1»#pragma omp parallel for«ENDIF»
«««		  for(int gpu = 0; gpu < «Config.gpus»; ++gpu){
«««			acc_set_device_num(gpu, acc_device_not_host);
«««			f.init(gpu);
«««			T* devptr = a.get_device_pointer(gpu);
«««			
«««			#pragma acc parallel loop deviceptr(devptr) firstprivate(f) async(0)
«««		  	for(size_t i = 0; i < gpu_elements; ++i) {
«««		  		f.set_id(__pgi_gangidx(), __pgi_workeridx(),__pgi_vectoridx());
«««		    	f(devptr[i]);
«««		  	}
«««		  }
		}
		
		template<typename T, typename Functor>
		void mkt::map_index_in_place(mkt::DArray<T>& a, Functor f){
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
				dim3 dimGrid((gpu_elements+dimBlock.x)/dimBlock.x);
				mkt::kernel::mapIndexInPlaceKernel<<<dimGrid, dimBlock, smem_bytes, mkt::cuda_streams[gpu]>>>(devptr, gpu_elements, gpu_offset, f);
			}
		}
		
		template<typename T, typename Functor>
		void mkt::map_local_index_in_place(mkt::DArray<T>& a, Functor f){
«««			size_t gpu_elements = a.get_size_gpu();
«««			
«««		  //«IF Config.cores > 1»#pragma omp parallel for«ENDIF»
«««		  for(int gpu = 0; gpu < «Config.gpus»; ++gpu){
«««			acc_set_device_num(gpu, acc_device_not_host);
«««			f.init(gpu);
«««			T* devptr = a.get_device_pointer(gpu);				
«««			
«««			size_t gpu_offset = 0;
«««			if(a.get_device_distribution() == mkt::Distribution::DIST){
«««				gpu_offset = gpu * gpu_elements;
«««			}
«««			#pragma acc parallel loop deviceptr(devptr) firstprivate(f) async(0)
«««		  	for(size_t i = 0; i < gpu_elements; ++i) {
«««		  		f.set_id(__pgi_gangidx(), __pgi_workeridx(),__pgi_vectoridx());
«««		    	f(i + gpu_offset, devptr[i]);
«««		  	}
«««		  }
		}
	'''
}