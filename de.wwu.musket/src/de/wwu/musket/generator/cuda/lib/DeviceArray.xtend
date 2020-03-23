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

class DeviceArray {
	private static final Logger logger = LogManager.getLogger(DeviceArray)

	def static generateDeviceArrayDeclaration() '''		
		template<typename T>
		class DeviceArray {
		 public:
		
		  // CONSTRUCTORS / DESTRUCTOR
		  DeviceArray(const DArray<T>& da);
		  DeviceArray(const DeviceArray<T>& da);
		  ~DeviceArray();
		  
		  void init(int device_id);
		  
		  
		// Getter and Setter
			size_t get_bytes_device() const;
		
		__device__ const T& get_data_device(size_t device_index) const;
		__device__ const T& get_data_local(size_t local_index) const;
		__device__ T get_global(size_t local_index);
		__device__ T set_global(size_t local_index, T value);
		
		 private:
		
		  //
		  // Attributes
		  //

		  size_t _size;
		  size_t _size_local;
		  size_t _size_device;
		  size_t _bytes_device;

		  size_t _offset;
		  
		  size_t _device_offset;

		  mkt::Distribution _dist;
		  mkt::Distribution _device_dist;

		  T* _device_data;
		
		  std::array<T*, «Config.gpus»> _gpu_data;
		
		  };
	'''
	
	def static generateDeviceArrayDefinition() '''
		template<typename T>
		mkt::DeviceArray<T>::DeviceArray(const DArray<T>& da)
		    : _size(da.get_size()),
		      _size_local(da.get_size_local()),
		      _size_device(da.get_size_gpu()),
		      _bytes_device(da.get_bytes_gpu()),
		      _offset(da.get_offset()),
		      _device_offset(0),
		      _dist(da.get_distribution()),
		      _device_dist(da.get_device_distribution()) 
		{
			_device_data = nullptr;
			for(int i = 0; i < «Config.gpus»; ++i){
				_gpu_data[i] = da.get_device_pointer(i);
			}
		}
		
		template<typename T>
		mkt::DeviceArray<T>::DeviceArray(const DeviceArray<T>& da)
		    : _size(da._size),
		      _size_local(da._size_local),
		      _size_device(da._size_device),
		      _bytes_device(da._bytes_device),
		      _offset(da._offset),
		      _device_offset(da._device_offset),
		      _dist(da._dist),
		      _device_dist(da._device_dist) 
		{
			_device_data = da._device_data;
			for(int i = 0; i < «Config.gpus»; ++i){
				_gpu_data[i] = da._gpu_data[i];
			}
		}

		
		template<typename T>
		mkt::DeviceArray<T>::~DeviceArray(){
		}
		
		template<typename T>
		void mkt::DeviceArray<T>::init(int gpu) {
			if(_device_dist == mkt::Distribution::COPY){
				_device_offset = 0;
			} else {
				_device_offset = _size_device * gpu;
			}
			    
			_device_data = _gpu_data[gpu];
		}
		
		template<typename T>
		__device__ T mkt::DeviceArray<T>::get_global(size_t index) {
			«IF Config.gpus == 1»
			// One GPU is configured if the datastructure is up-to-date it can be returned. 
			return get_data_local(index);
			«ELSE»
			// Multiple GPUs are configured.
			// TODO if(is_local(global_index)){
			// 	size_t local_index = global_index - _offset;
			// 	result = get_local(local_index);
			// 	//MPI_Bcast(&result, 1, MPI_Datatype datatype, _pid, )
			// }else{
			// 	int root = get_pid_by_global_index(global_index);
			// 	//MPI_Bcast(&result, 1, MPI_Datatype datatype, root, )
			// }
			«ENDIF»
		}
		
								
		template<typename T>
		__device__ T mkt::DeviceArray<T>::set_global(size_t local_index, T value) {
		  _device_data[local_index] = value;
		}
		
		template<typename T>
		size_t mkt::DeviceArray<T>::get_bytes_device() const {
		  return _bytes_device;
		}
		
		template<typename T>
		__device__ const T& mkt::DeviceArray<T>::get_data_device(size_t device_index) const {
		  return _device_data[device_index];
		}
		
		
		template<typename T>
		__device__ const T& mkt::DeviceArray<T>::get_data_local(size_t local_index) const {
		  return this->get_data_device(local_index - _device_offset);
		}
	'''
		
	def static generateDeviceArraySkeletonDeclarations() '''
		template<typename T, typename R, typename Functor>
		void map(const mkt::DeviceArray<T>& in, mkt::DeviceArray<R>& out, Functor f);
		
		template<typename T, typename R, typename Functor>
		void map_index(const mkt::DeviceArray<T>& in, mkt::DeviceArray<R>& out, Functor f);
		
		template<typename T, typename R, typename Functor>
		void map_local_index(const mkt::DeviceArray<T>& in, mkt::DeviceArray<R>& out, Functor f);
		
		template<typename T, typename Functor>
		void map_in_place(mkt::DeviceArray<T>& a, Functor f);
		
		template<typename T, typename Functor>
		void map_index_in_place(mkt::DeviceArray<T>& a, Functor f);
		
		template<typename T, typename Functor>
		void map_local_index_in_place(mkt::DeviceArray<T>& a, Functor f);
		
		template<typename T, typename Functor>
		void fold(const mkt::DeviceArray<T>& a, T& out, const T identity, const Functor f);
		
		template<typename T, typename Functor>
		void fold_copy(const mkt::DeviceArray<T>& a, T& out, const T identity, const Functor f);
		
		template<typename T, typename R, typename MapFunctor, typename FoldFunctor>
		void map_fold(const mkt::DeviceArray<T>& a, R& out, const MapFunctor& f_map, const R identity, const FoldFunctor f_fold);
		
		template<typename T, typename R, typename MapFunctor, typename FoldFunctor>
		void map_fold_copy(const mkt::DeviceArray<T>& a, R& out, const MapFunctor& f_map, const R identity, const FoldFunctor f_fold);

		template<typename T, typename R, typename I, typename MapFunctor, typename FoldFunctor>
		void map_fold(const mkt::DeviceArray<T>& a, mkt::DeviceArray<R>& out, const MapFunctor& f_map, const I identity, const FoldFunctor f_fold);

		template<typename T, typename R, typename I, typename MapFunctor, typename FoldFunctor>
		void map_fold_copy(const mkt::DeviceArray<T>& a, mkt::DeviceArray<R>& out, const MapFunctor f_map, const I identity, const FoldFunctor f_fold);
	'''
	
	def static generateDeviceArraySkeletonDefinitions() '''
		template<typename T, typename R, typename Functor>
		void mkt::map(const mkt::DeviceArray<T>& in, mkt::DeviceArray<R>& out, Functor f) {
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
		void mkt::map_index(const mkt::DeviceArray<T>& in, mkt::DeviceArray<R>& out, Functor f) {
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
		void mkt::map_local_index(const mkt::DeviceArray<T>& in, mkt::DeviceArray<R>& out, Functor f) {
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
		void mkt::map_in_place(mkt::DeviceArray<T>& a, Functor f){
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
		void mkt::map_index_in_place(mkt::DeviceArray<T>& a, Functor f){
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
		void mkt::map_local_index_in_place(mkt::DeviceArray<T>& a, Functor f){
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