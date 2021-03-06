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

	def static generateDArrayDeclaration() '''		
		template<typename T>
		class DArray {
		 public:
		
		  // CONSTRUCTORS / DESTRUCTOR
		  DArray(int pid, int size, int size_local, T init_value, int partitions, int partition_pos, int offset, mkt::Distribution d = DIST);
		
		// Getter and Setter
		
		  T get_global(int index) const;
		  void set_global(int index, const T& value);
		
		  T get_local(int index) const;
		  void set_local(int index, const T& value);
		
		  T& operator[](int local_index);
		  const T& operator[](int local_index) const;
		
		  int get_size() const;
		  int get_size_local() const;
		
		  int get_offset() const;
				
		  Distribution get_distribution() const;
				
		  T* get_data();
		  const T* get_data() const;
				
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
	'''
	
	def static generateDArrayDefinition() '''
		template<typename T>
		mkt::DArray<T>::DArray(int pid, int size, int size_local, T init_value, int partitions, int partition_pos, int offset, Distribution d)
		    : _pid(pid),
		      _size(size),
		      _size_local(size_local),
		      _partitions(partitions),
		      _partition_pos(partition_pos),
		      _offset(offset),
		      _dist(d),
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
		int mkt::DArray<T>::get_offset() const {
		  return _offset;
		}
		
		template<typename T>
		mkt::Distribution mkt::DArray<T>::get_distribution() const {
		  return _dist;
		}
		
		template<typename T>
		const T* mkt::DArray<T>::get_data() const {
		  return _data.data();
		}
		
		template<typename T>
		T* mkt::DArray<T>::get_data() {
		  return _data.data();
		}
	'''
	
	
	def static generateDArraySkeletonDeclarations() '''
		template<typename T, typename R, typename Functor>
		void map(const mkt::DArray<T>& in, mkt::DArray<R>& out, const Functor& f);
		
		template<typename T, typename R, typename Functor>
		void map_index(const mkt::DArray<T>& in, mkt::DArray<R>& out, const Functor& f);
		
		template<typename T, typename R, typename Functor>
		void map_local_index(const mkt::DArray<T>& in, mkt::DArray<R>& out, const Functor& f);
		
		template<typename T, typename Functor>
		void map_in_place(mkt::DArray<T>& a, const Functor& f);
		
		template<typename T, typename Functor>
		void map_index_in_place(mkt::DArray<T>& a, const Functor& f);
		
		template<typename T, typename Functor>
		void map_local_index_in_place(mkt::DArray<T>& a, const Functor& f);
		
		template<typename T, typename Functor>
		void fold(const mkt::DArray<T>& a, T& out, const T identity, const Functor& f);
		
		template<typename T, typename Functor>
		void fold_copy(const mkt::DArray<T>& a, T& out, const T identity, const Functor& f);
		
		template<typename T, typename R, typename MapFunctor, typename FoldFunctor>
		void map_fold(const mkt::DArray<T>& a, R& out, const MapFunctor& f_map, const R& identity, const FoldFunctor& f_fold);
		
		template<typename T, typename R, typename MapFunctor, typename FoldFunctor>
		void map_fold_copy(const mkt::DArray<T>& a, R& out, const MapFunctor& f_map, const R& identity, const FoldFunctor& f_fold);

		template<typename T, typename R, typename I, typename MapFunctor, typename FoldFunctor>
		void map_fold(const mkt::DArray<T>& a, mkt::DArray<R>& out, const MapFunctor& f_map, const I& identity, const FoldFunctor& f_fold);
				
		template<typename T, typename R, typename I, typename MapFunctor, typename FoldFunctor>
		void map_fold_copy(const mkt::DArray<T>& a, mkt::DArray<R>& out, const MapFunctor& f_map, const I& identity, const FoldFunctor& f_fold);
	'''
	
	def static generateDArraySkeletonDefinitions() '''
		template<typename T, typename R, typename Functor>
		void mkt::map(const mkt::DArray<T>& in, mkt::DArray<R>& out, const Functor& f) {
		#pragma omp «IF Config.cores > 1»parallel for «ENDIF»simd
		  for (int i = 0; i < in.get_size_local(); ++i) {
		      out[i] = f(in[i]);
		  }
		}
		
		template<typename T, typename R, typename Functor>
		void mkt::map_index(const mkt::DArray<T>& in, mkt::DArray<R>& out, const Functor& f) {
		  int offset = in.get_offset();
		#pragma omp «IF Config.cores > 1»parallel for «ENDIF»simd
		  for (int i = 0; i < in.get_size_local(); ++i) {
		    out[i] = f(i + offset, in[i]);
		  }
		}
		
		template<typename T, typename R, typename Functor>
		void mkt::map_local_index(const mkt::DArray<T>& in, mkt::DArray<R>& out, const Functor& f) {
		#pragma omp «IF Config.cores > 1»parallel for «ENDIF»simd
		  for (int i = 0; i < in.get_size_local(); ++i) {
		      out[i] = f(i, in[i]);
		  }
		}
		
		template<typename T, typename Functor>
		void mkt::map_in_place(mkt::DArray<T>& a, const Functor& f){
		#pragma omp «IF Config.cores > 1»parallel for «ENDIF»simd
		  for (int i = 0; i < a.get_size_local(); ++i) {
		    f(a[i]);
		  }
		}
		
		template<typename T, typename Functor>
		void mkt::map_index_in_place(mkt::DArray<T>& a, const Functor& f){
		  int offset = a.get_offset();
		#pragma omp «IF Config.cores > 1»parallel for «ENDIF»simd
		  for (int i = 0; i < a.get_size_local(); ++i) {
		    f(i + offset, a[i]);
		  }
		}
		
		template<typename T, typename Functor>
		void mkt::map_local_index_in_place(mkt::DArray<T>& a, const Functor& f){
		#pragma omp «IF Config.cores > 1»parallel for «ENDIF»simd
		  for (int i = 0; i < a.get_size_local(); ++i) {
		    f(i, a[i]);
		  }
		}
	'''
}