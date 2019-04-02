package de.wwu.musket.generator.cuda

import de.wwu.musket.musket.SkeletonExpression


class ZipSkeletonGenerator {

	def static generateArrayFunctionDeclarations() '''
		template<typename T, typename R, typename Functor>
		void zip(const mkt::DArray<T>& in1, const mkt::DArray<T>& in2, mkt::DArray<R>& out, const Functor& f);
		
		template<typename T, typename R, typename Functor>
		void zip_index(const mkt::DArray<T>& in1, const mkt::DArray<T>& in2, mkt::DArray<R>& out, const Functor& f);
		
		template<typename T, typename R, typename Functor>
		void zip_local_index(const mkt::DArray<T>& in1, const mkt::DArray<T>& in2, mkt::DArray<R>& out, const Functor& f);
		
		template<typename T, typename Functor>
		void zip_in_place(mkt::DArray<T>& inout, const mkt::DArray<T>& in, const Functor& f);
		
		template<typename T, typename Functor>
		void zip_index_in_place(mkt::DArray<T>& inout, const mkt::DArray<T>& in, const Functor& f);
		
		template<typename T, typename Functor>
		void zip_local_index_in_place(mkt::DArray<T>& inout, const mkt::DArray<T>& in, const Functor& f);
	'''

	def static generateArrayFunctionDefinitions() '''
		template<typename T, typename R, typename Functor>
		void mkt::zip(const mkt::DArray<T>& in1, const mkt::DArray<T>& in2, mkt::DArray<R>& out, const Functor& f) {
		  #pragma omp parallel for simd
		  for (int i = 0; i < in1.get_size_local(); ++i) {
		      out.set_local(i, f(in1.get_local(i), in2.get_local(i)));
		  }
		}
		
		template<typename T, typename R, typename Functor>
		void mkt::zip_index(const mkt::DArray<T>& in1, const mkt::DArray<T>& in2, mkt::DArray<R>& out, const Functor& f) {
		  int offset = in1.get_offset();
		  #pragma omp parallel for simd
		  for (int i = 0; i < in1.get_size_local(); ++i) {
		    out.set_local(i, f(i + offset, in1.get_local(i), in2.get_local(i)));
		  }
		}
		
		template<typename T, typename R, typename Functor>
		void mkt::zip_local_index(const mkt::DArray<T>& in1, const mkt::DArray<T>& in2, mkt::DArray<R>& out, const Functor& f) {
		  #pragma omp parallel for simd
		  for (int i = 0; i < in1.get_size_local(); ++i) {
		      out.set_local(i, f(i, in1.get_local(i), in2.get_local(i)));
		  }
		}
		
		template<typename T, typename Functor>
		void mkt::zip_in_place(mkt::DArray<T>& inout, const mkt::DArray<T>& in, const Functor& f){
		#pragma omp parallel for simd
		  for (int i = 0; i < inout.get_size_local(); ++i) {
		    inout.set_local(i, f(inout.get_local(i), in.get_local(i)));
		  }
		}
		
		template<typename T, typename Functor>
		void mkt::zip_index_in_place(mkt::DArray<T>& inout, const mkt::DArray<T>& in, const Functor& f){
		  int offset = inout.get_offset();
		#pragma omp parallel for simd
		  for (int i = 0; i < inout.get_size_local(); ++i) {
		    inout.set_local(i, f(i + offset, inout.get_local(i), in.get_local(i)));
		  }
		}
		
		template<typename T, typename Functor>
		void mkt::zip_local_index_in_place(mkt::DArray<T>& inout, const mkt::DArray<T>& in, const Functor& f) {
		#pragma omp parallel for simd
		  for (int i = 0; i < inout.get_size_local(); ++i) {
		    inout.set_local(i, f(i, inout.get_local(i), in.get_local(i)));
		  }
		}
	'''
	
	def static generateMatrixFunctionDeclarations() '''
		template<typename T, typename R, typename Functor>
		void zip(const mkt::DMatrix<T>& in1, const mkt::DMatrix<T>& in2, mkt::DMatrix<R>& out, const Functor& f);
		
		template<typename T, typename R, typename Functor>
		void zip_index(const mkt::DMatrix<T>& in1, const mkt::DMatrix<T>& in2, mkt::DMatrix<R>& out, const Functor& f);
		
		template<typename T, typename R, typename Functor>
		void zip_local_index(const mkt::DMatrix<T>& in1, const mkt::DMatrix<T>& in2, mkt::DMatrix<R>& out, const Functor& f);
		
		template<typename T, typename Functor>
		void zip_in_place(mkt::DMatrix<T>& inout, const mkt::DMatrix<T>& in, const Functor& f);
		
		template<typename T, typename Functor>
		void zip_index_in_place(mkt::DMatrix<T>& inout, const mkt::DMatrix<T>& in, const Functor& f);
		
		template<typename T, typename Functor>
		void zip_local_index_in_place(mkt::DMatrix<T>& inout, const mkt::DMatrix<T>& in, const Functor& f);
	'''
	
	def static generateMatrixFunctionDefinitions() '''
		template<typename T, typename R, typename Functor>
		void mkt::zip(const mkt::DMatrix<T>& in1, const mkt::DMatrix<T>& in2, mkt::DMatrix<R>& out, const Functor& f) {
		#pragma omp parallel for simd
		  for (int i = 0; i < in1.get_size_local(); ++i) {
		      out.set_local(i, f(in1.get_local(i), in2.get_local(i)));
		  }
		}
		
		template<typename T, typename R, typename Functor>
		void mkt::zip_index(const mkt::DMatrix<T>& in1, const mkt::DMatrix<T>& in2, mkt::DMatrix<R>& out, const Functor& f){
		  int row_offset = in1.get_row_offset();
		  int column_offset = in1.get_column_offset();
		#pragma omp parallel for
		  for (int i = 0; i < in1.get_number_of_rows_local(); ++i) {
		  	#pragma omp simd
		  	for (int j = 0; j < in1.get_number_of_columns_local(); ++j) {
		      out.set_local(i, j, f(i + row_offset, j + column_offset, in1.get_local(i, j), in2.get_local(i, j)));
		    }
		  }
		}
		
		template<typename T, typename R, typename Functor>
		void mkt::zip_local_index(const mkt::DMatrix<T>& in1, const mkt::DMatrix<T>& in2, mkt::DMatrix<R>& out, const Functor& f) {
		#pragma omp parallel for
		  for (int i = 0; i < in1.get_number_of_rows_local(); ++i) {
		  	#pragma omp simd
		  	for (int j = 0; j < in1.get_number_of_columns_local(); ++j) {
		      out.set_local(i, j, f(i, j, in1.get_local(i, j), in2.get_local(i, j)));
		    }
		  }
		}
		
		template<typename T, typename Functor>
		void mkt::zip_in_place(mkt::DMatrix<T>& inout, const mkt::DMatrix<T>& in, const Functor& f){
		#pragma omp parallel for simd
		  for (int i = 0; i < inout.get_size_local(); ++i) {
		    inout.set_local(i, f(inout.get_local(i), in.get_local(i)));
		  }
		}
		
		template<typename T, typename Functor>
		void mkt::zip_index_in_place(mkt::DMatrix<T>& inout, const mkt::DMatrix<T>& in, const Functor& f){
		  int row_offset = inout.get_row_offset();
		  int column_offset = inout.get_column_offset();
		  #pragma omp parallel for
		  for (int i = 0; i < inout.get_number_of_rows_local(); ++i) {
		  	#pragma omp simd
		  	for (int j = 0; j < inout.get_number_of_columns_local(); ++j) {
		      inout.set_local(i, j, f(i + row_offset, j + column_offset, inout.get_local(i, j), in.get_local(i, j)));
		    }
		  }
		}
		
		template<typename T, typename Functor>
		void mkt::zip_local_index_in_place(mkt::DMatrix<T>& inout, const mkt::DMatrix<T>& in, const Functor& f){
		  #pragma omp parallel for
		  for (int i = 0; i < inout.get_number_of_rows_local(); ++i) {
		    #pragma omp simd
		    for (int j = 0; j < inout.get_number_of_columns_local(); ++j) {
		      inout.set_local(i, j, f(i, j, inout.get_local(i, j), in.get_local(i, j)));
		    }
		  }
		}
	'''
}
