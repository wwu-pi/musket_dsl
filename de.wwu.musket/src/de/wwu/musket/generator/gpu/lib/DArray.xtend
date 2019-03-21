package de.wwu.musket.generator.gpu.lib

import org.apache.log4j.LogManager
import org.apache.log4j.Logger
import org.eclipse.emf.ecore.resource.Resource
import org.eclipse.xtext.generator.IFileSystemAccess2
import org.eclipse.xtext.generator.IGeneratorContext

import static extension de.wwu.musket.generator.gpu.DataGenerator.*
import static extension de.wwu.musket.generator.gpu.StructGenerator.*
import static extension de.wwu.musket.generator.extensions.ModelElementAccess.*
import de.wwu.musket.generator.gpu.Config

class DArray {
	private static final Logger logger = LogManager.getLogger(DArray)

	def static generateDArrayDeclaration() '''		
		template<typename T>
		class DArray {
		 public:
		
		  // CONSTRUCTORS / DESTRUCTOR
		  DArray(int pid, int size, int size_local, T init_value, int partitions, int partition_pos, int offset, mkt::Distribution d = DIST, Distribution device_dist = DIST);
		  ~ DArray(); 
		   
		   void update_self();
		   void update_devices();
		   void map_pointer();
		   
		// Getter and Setter
		
		  T get_global(int index);
		  void set_global(int index, const T& value);
		
		  T get_local(int index);
		  void set_local(int index, const T& value);
		
		  T& operator[](int local_index);
		  const T& operator[](int local_index) const;
		
		  int get_size() const;
		  int get_size_local() const;
		  int get_size_gpu() const;
		
		  int get_offset() const;
				
		  Distribution get_distribution() const;
		  Distribution get_device_distribution() const;
				
		  T* get_data();
		  const T* get_data() const;
		  
		  T* get_device_pointer(int gpu) const;
				
		 private:
		
		  int get_gpu_by_local_index(int local_index) const;
		  int get_gpu_by_global_index(int global_index) const;
		
		  //
		  // Attributes
		  //
		
		  // position of processor in data parallel group of processors; zero-base
		  int _pid;
		
		  int _size;
		  int _size_local;
		  int _size_gpu;
		
		  // number of (local) partitions in array
		  int _partitions;
		
		  // position of processor in data parallel group of processors
		  int _partition_pos;
		
		  // first index in local partition
		  int _offset;
		
		  // checks whether data is copy distributed among all processes
		  Distribution _dist;
		  Distribution _device_dist;
		
		  std::vector<T> _data;
		  std::array<T*, «Config.gpus»> _host_data;
		  std::array<T*, «Config.gpus»> _gpu_data;
		};
	'''
	
	def static generateDArrayDefinition() '''
		template<typename T>
		mkt::DArray<T>::DArray(int pid, int size, int size_local, T init_value, int partitions, int partition_pos, int offset, Distribution d, Distribution device_dist)
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
		    
		    #pragma omp parallel for
			for(int gpu = 0; gpu < «Config.gpus»; ++gpu){
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
		mkt::DArray<T>::~DArray(){
			// free device memory
			#pragma omp parallel for
			for(size_t gpu = 0; gpu < «Config.gpus»; ++gpu){
				acc_set_device_num(gpu, acc_device_not_host);
				acc_free(_gpu_data[gpu]);
			}
		}
				
		template<typename T>
		void mkt::DArray<T>::update_self() {
		  	#pragma omp parallel for
			for(size_t gpu = 0; gpu < «Config.gpus»; ++gpu){
				acc_set_device_num(gpu, acc_device_not_host);
				acc_update_self_async(_host_data[gpu], _size_gpu * sizeof(T), 0);
				//acc_memcpy_from_device_async(acc_hostptr(_gpu_data[gpu]), _gpu_data[gpu], _size_gpu * sizeof(T), 0);
				#pragma acc wait
			}
		}
		
		template<typename T>
		void mkt::DArray<T>::update_devices() {
		  	#pragma omp parallel for
			for(int gpu = 0; gpu < «Config.gpus»; ++gpu){
				acc_set_device_num(gpu, acc_device_not_host);
				acc_update_device_async(_host_data[gpu], _size_gpu * sizeof(T), 0);
				#pragma acc wait
			}
		}
		
		template<typename T>
		void mkt::DArray<T>::map_pointer() {
			for(size_t gpu = 0; gpu < «Config.gpus»; ++gpu){
				acc_set_device_num(gpu, acc_device_not_host);
				acc_map_data( _host_data[gpu], _gpu_data[gpu], _size_gpu * sizeof(T));
				//acc_map_data( _data.data() + gpu * _size_gpu , _gpu_data[gpu], _size_gpu * sizeof(T));
			}
		}
				
		template<typename T>
		T mkt::DArray<T>::get_local(int index) {
			int gpu = get_gpu_by_local_index(index);
			acc_set_device_num(gpu, acc_device_not_host);
			T* host_pointer = _data.data() + index;
			T* gpu_pointer = _gpu_data[gpu] + (index % _size_gpu );
			acc_memcpy_from_device_async(host_pointer, gpu_pointer, sizeof(T), 0);
			#pragma acc wait
		    return _data[index];
		}
		
		template<typename T>
		void mkt::DArray<T>::set_local(int index, const T& v) {
			_data[index] = v;
			T* host_pointer = _data.data() + index;
			if(_device_dist == mkt::Distribution::COPY){
				#pragma omp parallel for
				for(int gpu = 0; gpu < «Config.gpus»; ++gpu){
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
		T mkt::DArray<T>::get_global(int index) {
		  // TODO
		}
		
		template<typename T>
		void mkt::DArray<T>::set_global(int index, const T& v) {
		  // TODO
		}
		
		template<typename T>
		T& mkt::DArray<T>::operator[](int local_index) {
		  	return _data[local_index];
		}
		
		template<typename T>
		const T& mkt::DArray<T>::operator[](int local_index) const {
		  	return _data[local_index];
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
		int mkt::DArray<T>::get_size_gpu() const {
		  return _size_gpu;
		}
		
		template<typename T>
		int mkt::DArray<T>::get_offset() const {
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
		  return _data.data();
		}
		
		template<typename T>
		T* mkt::DArray<T>::get_data() {
		  return _data.data();
		}
		
		template<typename T>
		T* mkt::DArray<T>::get_device_pointer(int gpu) const{
		  return _gpu_data[gpu];
		}
		
		template<typename T>
		int mkt::DArray<T>::get_gpu_by_local_index(int local_index) const {
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
		int mkt::DArray<T>::get_gpu_by_global_index(int global_index) const {
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
			//«IF Config.cores > 1»#pragma omp parallel for«ENDIF»
			for(int gpu = 0; gpu < «Config.gpus»; ++gpu){
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
		void mkt::map_index(const mkt::DArray<T>& in, mkt::DArray<R>& out, Functor f) {
			int offset = in.get_offset();
			int gpu_elements = in.get_size_gpu();
			
			//«IF Config.cores > 1»#pragma omp parallel for«ENDIF»
			for(int gpu = 0; gpu < «Config.gpus»; ++gpu){
				acc_set_device_num(gpu, acc_device_not_host);
				f.init(gpu);
				T* in_devptr = in.get_device_pointer(gpu);
				R* out_devptr = out.get_device_pointer(gpu);
				
				int gpu_offset = offset;
				if(in.get_device_distribution() == mkt::Distribution::DIST){
					gpu_offset += gpu * gpu_elements;
				}
				
				#pragma acc parallel loop deviceptr(in_devptr, out_devptr) firstprivate(f) async(0)
				for (int i = 0; i < gpu_elements; ++i) {
					f.set_id(__pgi_gangidx(), __pgi_workeridx(),__pgi_vectoridx());
					out_devptr[i] = f(i + gpu_offset, in_devptr[i]);
				}
			}
		}
		
		template<typename T, typename R, typename Functor>
		void mkt::map_local_index(const mkt::DArray<T>& in, mkt::DArray<R>& out, Functor f) {
			int gpu_elements = in.get_size_gpu();
			
			//«IF Config.cores > 1»#pragma omp parallel for«ENDIF»
			for(int gpu = 0; gpu < «Config.gpus»; ++gpu){
				acc_set_device_num(gpu, acc_device_not_host);
				f.init(gpu);
				T* in_devptr = in.get_device_pointer(gpu);
				R* out_devptr = out.get_device_pointer(gpu);
				
				int gpu_offset = 0;
				if(in.get_device_distribution() == mkt::Distribution::DIST){
					gpu_offset = gpu * gpu_elements;
				}
				#pragma acc parallel loop deviceptr(in_devptr, out_devptr) firstprivate(f) async(0)
				for (int i = 0; i < gpu_elements; ++i) {
					f.set_id(__pgi_gangidx(), __pgi_workeridx(),__pgi_vectoridx());
					out_devptr[i] = f(i + gpu_offset, in_devptr[i]);
				}
			}
		}
		
		template<typename T, typename Functor>
		void mkt::map_in_place(mkt::DArray<T>& a, Functor f){
			int gpu_elements = a.get_size_gpu();
			
		  //«IF Config.cores > 1»#pragma omp parallel for«ENDIF»
		  for(int gpu = 0; gpu < «Config.gpus»; ++gpu){
			acc_set_device_num(gpu, acc_device_not_host);
			f.init(gpu);
			T* devptr = a.get_device_pointer(gpu);
			
			#pragma acc parallel loop deviceptr(devptr) firstprivate(f) async(0)
		  	for (int i = 0; i < gpu_elements; ++i) {
		  		f.set_id(__pgi_gangidx(), __pgi_workeridx(),__pgi_vectoridx());
		    	f(i, devptr[i]);
		  	}
		  }
		}
		
		template<typename T, typename Functor>
		void mkt::map_index_in_place(mkt::DArray<T>& a, Functor f){
			int offset = a.get_offset();
			int gpu_elements = a.get_size_gpu();
					  
		  	//«IF Config.cores > 1»#pragma omp parallel for«ENDIF»
		  	for(int gpu = 0; gpu < «Config.gpus»; ++gpu){
				acc_set_device_num(gpu, acc_device_not_host);
				f.init(gpu);
				T* devptr = a.get_device_pointer(gpu);
				
				int gpu_offset = offset;
				if(a.get_device_distribution() == mkt::Distribution::DIST){
					gpu_offset += gpu * gpu_elements;
				}
				#pragma acc parallel loop deviceptr(devptr) firstprivate(f) async(0)
			  	for (int i = 0; i < gpu_elements; ++i) {
			  		f.set_id(__pgi_gangidx(), __pgi_workeridx(),__pgi_vectoridx());
			    	f(i + gpu_offset, devptr[i]);
			  	}
		  	}
		}
		
		template<typename T, typename Functor>
		void mkt::map_local_index_in_place(mkt::DArray<T>& a, Functor f){
			int gpu_elements = a.get_size_gpu();
			
		  //«IF Config.cores > 1»#pragma omp parallel for«ENDIF»
		  for(int gpu = 0; gpu < «Config.gpus»; ++gpu){
			acc_set_device_num(gpu, acc_device_not_host);
			f.init(gpu);
			T* devptr = a.get_device_pointer(gpu);				
			
			int gpu_offset = 0;
			if(a.get_device_distribution() == mkt::Distribution::DIST){
				gpu_offset = gpu * gpu_elements;
			}
			#pragma acc parallel loop deviceptr(devptr) firstprivate(f) async(0)
		  	for (int i = 0; i < gpu_elements; ++i) {
		  		f.set_id(__pgi_gangidx(), __pgi_workeridx(),__pgi_vectoridx());
		    	f(i + gpu_offset, devptr[i]);
		  	}
		  }
		}
	'''
}