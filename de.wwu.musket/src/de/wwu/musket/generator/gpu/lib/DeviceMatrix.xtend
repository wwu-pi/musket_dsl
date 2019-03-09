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

class DeviceMatrix {
	private static final Logger logger = LogManager.getLogger(DeviceMatrix)

	
	def static generateDeviceMatrixDeclaration() '''		
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
		  
«««		  int get_row_offset() const;
«««		  int get_column_offset() const;
«««		
«««		  int get_number_of_rows() const;
«««		  int get_number_of_columns() const;
«««		
«««		  int get_number_of_rows_local() const;
«««		  int get_number_of_columns_local() const;
«««		  
«««		  int get_partitions_in_row() const;
«««		  int get_partitions_in_column() const;
«««		  
«««		  int get_partition_x_pos() const;
«««		  int get_partition_y_pos() const;
«««		  
«««		  Distribution get_distribution() const;
«««		  Distribution get_device_distribution() const;
		
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
		
		  std::array<T*, «Config.gpus»> _gpu_data;
		};
	'''
	
	def static generateDeviceMatrixDefinition() '''
				
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
			for(int i = 0; i < «Config.gpus»; ++i){
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
			for(int i = 0; i < «Config.gpus»; ++i){
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

	'''
}