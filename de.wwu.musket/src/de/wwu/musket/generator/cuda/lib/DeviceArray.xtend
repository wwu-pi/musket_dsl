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
		
			const T& get_data_device(size_t device_index) const;
			const T& get_data_local(size_t local_index) const;
		
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

		  Distribution _dist;
		  Distribution _device_dist;

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
		      _dist(da.get_distribution()),
		      _device_dist(da.get_device_distribution()) 
		{
			for(int i = 0; i < «Config.gpus»; ++i){
				_gpu_data[i] = da.get_device_pointer(i);
			}
		}
		
		template<typename T>
		mkt::DeviceArray<T>::DeviceArray(const DeviceArray<T>& da)
		    : _size(da._size),
		      _size_local(da._size_local),
		      _size_device(da._size_device),
		      _offset(da._offset),
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
			if(_device_dist == Distribution::COPY){
				_device_offset = 0;
			} else {
				_device_offset = _size_device * gpu;
			}
			    
			_device_data = _gpu_data[gpu];
		}
		
		template<typename T>
		size_t mkt::DeviceArray<T>::get_bytes_device() const {
		  return _bytes_device;
		}
		
		template<typename T>
		const T& mkt::DeviceArray<T>::get_data_device(size_t device_index) const {
		  return _device_data[device_index];
		}
		
		
		template<typename T>
		const T& mkt::DeviceArray<T>::get_data_local(size_t local_index) const {
		  return this->get_data_device(local_index - _device_offset);
		}
	'''
	
	
}