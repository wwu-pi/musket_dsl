package de.wwu.musket.generator.cpu.mpmd.lib

import org.apache.log4j.LogManager
import org.apache.log4j.Logger
import org.eclipse.emf.ecore.resource.Resource
import org.eclipse.xtext.generator.IFileSystemAccess2
import org.eclipse.xtext.generator.IGeneratorContext

import static extension de.wwu.musket.generator.cpu.mpmd.DataGenerator.*
import static extension de.wwu.musket.generator.cpu.mpmd.StructGenerator.*
import static extension de.wwu.musket.generator.extensions.ModelElementAccess.*
import de.wwu.musket.generator.cpu.mpmd.Config

class DArray {
	private static final Logger logger = LogManager.getLogger(DArray)

	/**
	 * Creates the DArray header file
	 */
	def static void generateDArrayHeaderFile(Resource resource, IFileSystemAccess2 fsa, IGeneratorContext context) {
		logger.info("Generate DArray file.")
		fsa.generateFile(Config.base_path + Config.include_path + 'darray' + Config.header_extension,
			headerFileContent(resource))
		logger.info("Generation of DArray file done.")
	}
	
	def static headerFileContent(Resource resource) '''
		#pragma once
		
		#include "distribution.hpp"
		
		namespace mkt {
		
		template<typename T>
		class DArray {
		 public:
		
		  // CONSTRUCTORS / DESTRUCTOR
		  DArray(int pid, int size, int size_local, T init_value, int partitions, int partition_pos, int offset, Distribution d = DIST);
		
		// Getter and Setter
		
		  T get_global(int index) const;
		  void set_global(int index, const T& value);
		
		  T get_local(int index) const;
		  void set_local(int index, const T& value);
		
		  int get_size() const;
		  int get_size_local() const;
		
		  int get_offset() const;
				
		 private:
		
		  //
		  // Attributes
		  //
		
		  // position of processor in data parallel group of processors; zero-base
		  int _pid;
		
		  int _size;
		  int _size_local;
		
		  // number of (local) partitions in array
		  int _partitions;
		
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
		mkt::DArray<T>::DArray(int pid, int size, int size_local, T init_value, int partitions, int partition_pos, int offset, Distribution d)
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
		T mkt::DArray<T>::get_global(int row, int col) const {
		  // TODO
		}
		
		template<typename T>
		void mkt::DArray<T>::set_global(int row, int col, const T& v) {
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
	'''
}