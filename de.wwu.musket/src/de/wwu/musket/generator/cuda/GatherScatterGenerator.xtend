package de.wwu.musket.generator.cuda

import de.wwu.musket.musket.MatrixType
import de.wwu.musket.musket.CollectionObject
import static extension de.wwu.musket.generator.cuda.util.DataHelper.*
import static extension de.wwu.musket.util.TypeHelper.*
import static extension de.wwu.musket.util.MusketHelper.*
import static de.wwu.musket.generator.cuda.MPIRoutines.generateMPIAllgather
import static extension de.wwu.musket.generator.extensions.ModelElementAccess.*
import org.eclipse.emf.ecore.resource.Resource
import de.wwu.musket.musket.SkeletonExpression

class GatherScatterGenerator {
	def static generateGatherDeclarations(Resource resource) '''
		«IF resource.Arrays.size() > 0»
			template<typename T>
			void gather(mkt::DArray<T>& in, mkt::DArray<T>& out);
		«ENDIF»
			
		«IF resource.Matrices.size() > 0»
			template<typename T>
			void gather(mkt::DMatrix<T>& in, mkt::DMatrix<T>& out«IF Config.processes > 1», const MPI_Datatype& dt«ENDIF»);
		«ENDIF»		
	'''
	
	def static generateGatherDefinitions(Resource resource) {
		var result = ''''''
		var coTypePairs = newArrayList
		for(se : resource.SkeletonExpressions){
			val co = if(se.obj.calculateType.isArray) 'array' else 'matrix'
			val type = se.obj.calculateCollectionType.MPIType
			val pair = co -> type
			if(!coTypePairs.contains(pair)){
				result += generateGatherDefinition(se)
				coTypePairs.add(pair)
			}			
		}
		return result
	}
	
	def static generateGatherDefinition(SkeletonExpression se) '''
		«val co = se.obj»
		«val type = co.calculateCollectionType.cppType»
		«val dtype = if (co.calculateType.array) "DArray" else "DMatrix"»
		template<>
		void mkt::gather<«type»>(mkt::«dtype»<«type»>& in, mkt::«dtype»<«type»>& out«IF co.calculateType.matrix && Config.processes > 1», const MPI_Datatype& dt«ENDIF»){
			in.update_self();
			«IF Config.processes > 1»
				«IF co.calculateType.array»
					«generateMPIAllgather('in.get_data()', co.type.sizeLocal(0), co.calculateCollectionType, 'out.get_data()')»
				«ELSE»
					«var displacement = (co.collectionType as MatrixType).rowsLocal * (co.collectionType as MatrixType).blocksInRow»
					MPI_Allgatherv(in.get_data(), «se.obj.type.sizeLocal(0)», «se.obj.calculateCollectionType.MPIType», out.get_data(), (std::array<int, «Config.processes»>{«FOR i: 0 ..< Config.processes SEPARATOR ', '»1«ENDFOR»}).data(), (std::array<int, «Config.processes»>{«FOR i: 0 ..< Config.processes SEPARATOR ', '»«displacement * (co.collectionType as MatrixType).partitionPosition(i).key + (co.collectionType as MatrixType).partitionPosition(i).value»«ENDFOR»}).data(), dt, MPI_COMM_WORLD);
				«ENDIF»				
			«ELSE»
				std::copy(in.get_data(), in.get_data() + in.get_size_local(), out.get_data());
			«ENDIF»
			out.update_devices();
		}
	'''
			
	def static generateScatterDeclarations(Resource resource) '''
		«IF resource.Arrays.size() > 0»
			template<typename T>
			void scatter(mkt::DArray<T>& in, mkt::DArray<T>& out);
		«ENDIF»
			
		«IF resource.Matrices.size() > 0»
			template<typename T>
			void scatter(mkt::DMatrix<T>& in, mkt::DMatrix<T>& out);
		«ENDIF»	
	'''

	def static generateScatterDefinitions(Resource resource) '''
		«IF resource.Arrays.size() > 0»
			template<typename T>
			void mkt::scatter(mkt::DArray<T>& in, mkt::DArray<T>& out){
				in.update_self();
				«IF Config.processes > 1»
					int offset = out.get_offset();
					std::copy(in.get_data() + offset, in.get_data() + in.get_size_local(), out.get_data());
				«ELSE»
					std::copy(in.get_data(), in.get_data() + in.get_size(), out.get_data());
				«ENDIF»
				out.update_devices();
			}
		«ENDIF»
			
		«IF resource.Matrices.size() > 0»
			template<typename T>
			void mkt::scatter(mkt::DMatrix<T>& in, mkt::DMatrix<T>& out){
				in.update_self();
				«IF Config.processes > 1»
					int row_offset = out.get_row_offset();
					int column_offset = out.get_column_offset();
					for(unsigned int i = 0; i < out.get_number_of_rows_local(); ++i){
					  for(int j = 0; j < out.get_number_of_columns_local(); ++j){
					    out.set_host_local(i, j, in.get_host_local(i + row_offset, j + column_offset));
					  }
					}
				«ELSE»
					std::copy(in.get_data(), in.get_data() + in.get_size(), out.get_data());
				«ENDIF»
				out.update_devices();
			}
		«ENDIF»
	'''

}