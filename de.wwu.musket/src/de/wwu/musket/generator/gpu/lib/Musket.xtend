package de.wwu.musket.generator.gpu.lib

import org.apache.log4j.LogManager
import org.apache.log4j.Logger
import org.eclipse.emf.ecore.resource.Resource
import org.eclipse.xtext.generator.IFileSystemAccess2
import org.eclipse.xtext.generator.IGeneratorContext

import static extension de.wwu.musket.generator.gpu.DataGenerator.*
import static extension de.wwu.musket.generator.gpu.StructGenerator.*
import static extension de.wwu.musket.generator.extensions.ModelElementAccess.*
import static extension de.wwu.musket.generator.gpu.util.DataHelper.*
import static extension de.wwu.musket.util.MusketHelper.*
import static extension de.wwu.musket.util.TypeHelper.*
import static de.wwu.musket.generator.gpu.lib.DArray.*
import static de.wwu.musket.generator.gpu.lib.DMatrix.*
import de.wwu.musket.generator.gpu.Config
import java.util.List
import de.wwu.musket.musket.CollectionFunctionCall
import de.wwu.musket.musket.DistributionMode
import de.wwu.musket.musket.CollectionObject
import de.wwu.musket.musket.ArrayType
import de.wwu.musket.musket.Struct
import de.wwu.musket.musket.MatrixType
import static de.wwu.musket.generator.gpu.GatherScatterGenerator.*
import static de.wwu.musket.generator.gpu.ShiftSkeletonGenerator.*

class Musket {
	static final Logger logger = LogManager.getLogger(Musket)
	
	def static void generateMusketHeaderFile(Resource resource, IFileSystemAccess2 fsa, IGeneratorContext context) {
		logger.info("Generate Musket header file.")
		fsa.generateFile(Config.base_path + Config.include_path + 'musket' + Config.header_extension,
			headerFileContent(resource))
		logger.info("Generation of Musket header file done.")
	}
	
	def static headerFileContent(Resource resource) '''
		#pragma once
		#include <string>
		#include "«resource.ProjectName»«Config.header_extension»"
		
		namespace mkt {
		«generateDistEnum»
		«IF resource.Arrays.size() > 0»
			«generateDArrayDeclaration»
			«generateDArraySkeletonDeclarations»
		«ENDIF»
		
		«IF resource.Matrices.size() > 0»
			«generateDMatrixDeclaration»
			«generateDMatrixSkeletonDeclarations»
		«ENDIF»
		
		«val showCalls = resource.ShowCalls»
		«IF showCalls.size() > 0»			
			«IF resource.Arrays.size() > 0»
				«IF Config.processes > 1»
					template<typename T>
					void print_dist(const std::string& name, const mkt::DArray<T>& a);
				«ENDIF»
				template<typename T>
				void print(const std::string& name, const mkt::DArray<T>& a);				
			«ENDIF»
			«IF resource.Matrices.size() > 0»
				template<typename T>
				void print(const std::string& name, const mkt::DMatrix<T>& a);
				«IF Config.processes > 1»
					«generatePrintDistFunctionDeclarationsMatrix(showCalls)»
				«ENDIF»
			«ENDIF»			
		«ENDIF»
		
		template<typename T>
		void print(std::ostringstream& stream, const T& a);
		
		«IF resource.ShowLocalCalls.size() > 0»			
			«IF resource.Arrays.size() > 0»
				template<typename T>
				void print_local_partition(const std::string& name, const int pid, const mkt::DArray<T>& a);
			«ENDIF»
			«IF resource.Matrices.size() > 0»
				template<typename T>
				void print_local_partition(const std::string& name, const int pid, const mkt::DMatrix<T>& a);
			«ENDIF»
		«ENDIF»
		
		«generateGatherDeclarations(resource)»
		«generateScatterDeclarations(resource)»
		«IF resource.ShiftPartitionsHorizontallySkeletons.size() > 0 && Config.processes > 1»
			«generateShiftHorizontallySkeletonsFunctionDeclarations»
		«ENDIF»
		
		«IF resource.ShiftPartitionsVerticallySkeletons.size() > 0 && Config.processes > 1»
			«generateShiftVerticallySkeletonsFunctionDeclarations»
		«ENDIF»
		
		} // namespace mkt
		
		«IF resource.Arrays.size() > 0»
			«generateDArrayDefinition»
			«generateDArraySkeletonDefinitions»
		«ENDIF»
		
		«IF resource.Matrices.size() > 0»
			«generateDMatrixDefinition»	
			«generateDMatrixSkeletonDefinitions»
		«ENDIF»
		
		«IF resource.Structs.size > 0»
			«resource.generateStructPrintFunctions»
		«ENDIF»
		
		template<typename T>
		void mkt::print(std::ostringstream& stream, const T& a) {
			if(std::is_fundamental<T>::value){
				stream << a;
			}
		}
		
		«IF showCalls.size() > 0»			
			«IF resource.Arrays.size() > 0»
				«generatePrintCopyArray»
				«IF Config.processes > 1»
					«generatePrintDistFunctionsArray(showCalls)»
				«ENDIF»
			«ENDIF»
			«IF resource.Matrices.size() > 0»
				«generatePrintCopyMatrix»
				«IF Config.processes > 1»
					«generatePrintDistFunctionsMatrix(showCalls)»
				«ENDIF»
			«ENDIF»
		«ENDIF»
		
		«IF resource.ShowLocalCalls.size() > 0»			
			«IF resource.Arrays.size() > 0»
				«generatePrintLocalPartitionArrayDefinition»
			«ENDIF»
			«IF resource.Matrices.size() > 0»
				«generatePrintLocalPartitionMatrixDefinition»
			«ENDIF»
		«ENDIF»
		
		«generateGatherDefinitions(resource)»
		«generateScatterDefinitions(resource)»
	'''
		
	def static generatePrintCopyArray()'''
		template<typename T>
		void mkt::print(const std::string& name, const mkt::DArray<T>& a) {
		  std::ostringstream stream;
		  stream << name << ": " << std::endl << "[";
		  for (int i = 0; i < a.get_size() - 1; ++i) {
		  	mkt::print<T>(stream, a.get_local(i));
		  	stream << "; ";
		  }
		  mkt::print<T>(stream, a.get_local(a.get_size() - 1));
		  stream << "]" << std::endl << std::endl;
		  printf("%s", stream.str().c_str());
		}
	'''
	
	def static generatePrintCopyMatrix()'''
		template<typename T>
		void mkt::print(const std::string& name, const mkt::DMatrix<T>& m) {
		  std::ostringstream stream;
		  stream << name << ": " << std::endl;
		  for (int i = 0; i < m.get_number_of_rows_local(); ++i) {
		  	stream << "[";
		  	for (int j = 0; j < m.get_number_of_columns_local() - 1; ++j) {
		  	  mkt::print<T>(stream, m.get_local(i, j));
		  	  stream << "; ";
		  	}
		  	mkt::print<T>(stream, m.get_local(i, m.get_number_of_columns_local() - 1));
		  	stream << "]" << std::endl;
		  }		  
		  stream << std::endl;
		  printf("%s", stream.str().c_str());
		}
	'''
	
	def static generatePrintLocalPartitionArrayDefinition()'''
		template<typename T>
		void mkt::print_local_partition(const std::string& name, const int pid, const mkt::DArray<T>& a) {
		  std::ostringstream stream;
		  stream << name << " on process " << pid <<": " << std::endl << "[";
		  for (int i = 0; i < a.get_size_local() - 1; ++i) {
		  	mkt::print<T>(stream, a.get_local(i));
		  	stream << "; ";
		  }
		  mkt::print<T>(stream, a.get_local(a.get_size_local() - 1));
		  stream << "]" << std::endl << std::endl;
		  printf("%s", stream.str().c_str());
		}
	'''
	
	def static generatePrintLocalPartitionMatrixDefinition()'''
		template<typename T>
		void mkt::print_local_partition(const std::string& name, const int pid, const mkt::DMatrix<T>& m) {
		  std::ostringstream stream;
		  stream << name << " on process " << pid << ": " << std::endl;
		  for (int i = 0; i < m.get_number_of_rows_local(); ++i) {
		  	stream << "[";
		  	for (int j = 0; j < m.get_number_of_columns_local() - 1; ++j) {
		  	  mkt::print<T>(stream, m.get_local(i, j));
		  	  stream << "; ";
		  	}
		  	mkt::print<T>(stream, m.get_local(i, m.get_number_of_columns_local() - 1));
		  	stream << "]" << std::endl;
		  }		  
		  stream << std::endl;
		  printf("%s", stream.str().c_str());
		}
	'''
	
	def static generatePrintDistFunctionsArray(List<CollectionFunctionCall> showCalls){
		var result = ""
		var types = newArrayList()
		for (sc : showCalls){
			if(sc.^var.type.distributionMode == DistributionMode.DIST && sc.^var.calculateType.isArray && !types.contains(sc.^var.calculateCollectionType)){
				types.add(sc.^var.calculateCollectionType)
				result += generatePrintDistFunctionArray(sc.^var)
			}
		}
		return result
	}
	
	def static generatePrintDistFunctionArray(CollectionObject co)'''
		«var type = co.calculateCollectionType.cppType»
		«var mpiType = co.calculateCollectionType.MPIType»
		«var size = co.collectionType.size»
		«var sizeLocal = co.collectionType.sizeLocal(0)»
		template<>
		void mkt::print_dist<«type»>(const std::string& name, const mkt::DArray<«type»>& a) {
		  std::array<«type», «size»> a_copy;
		  MPI_Gather(a.get_data(), «sizeLocal», «mpiType», a_copy.data(), «sizeLocal», «mpiType», 0, MPI_COMM_WORLD);
		  std::ostringstream stream;
		  stream << name << ": " << std::endl << "[";
		  for (int i = 0; i < «size - 1»; ++i) {
		  	mkt::print<«type»>(stream, a_copy[i]);
		  	stream << "; ";
		  }
		  mkt::print<«type»>(stream, a_copy[«size - 1»]);
		  stream << "]" << std::endl << std::endl;
		  printf("%s", stream.str().c_str());
		}
	'''
	
	def static generatePrintDistFunctionDeclarationsMatrix(List<CollectionFunctionCall> showCalls){
		var result = ""
		var matrices = newArrayList()
		for (sc : showCalls){
			if(sc.^var.type.distributionMode == DistributionMode.DIST && sc.^var.calculateType.isMatrix && !matrices.contains(sc.^var)){
				matrices.add(sc.^var)
				result += generatePrintDistFunctionDeclarationMatrix(sc.^var)
			}
		}
		return result
	}
	
	def static generatePrintDistFunctionDeclarationMatrix(CollectionObject co)'''
		«var type = co.calculateCollectionType.cppType»
		«var name = co.name»
		void print_dist_«name»(const mkt::DMatrix<«type»>& m, const MPI_Datatype& dt);
	'''
	
	def static generatePrintDistFunctionsMatrix(List<CollectionFunctionCall> showCalls){
		var result = ""
		var matrices = newArrayList()
		for (sc : showCalls){
			if(sc.^var.type.distributionMode == DistributionMode.DIST && sc.^var.calculateType.isMatrix && !matrices.contains(sc.^var)){
				matrices.add(sc.^var)
				result += generatePrintDistFunctionMatrix(sc.^var)
			}
		}
		return result
	}
	
	def static generatePrintDistFunctionMatrix(CollectionObject co)'''
		«var type = co.calculateCollectionType.cppType»
		«var mpiType = co.calculateCollectionType.MPIType»
		«var size = co.collectionType.size»
		«var sizeLocal = co.collectionType.sizeLocal(0)»
		«var displacement = (co.collectionType as MatrixType).rowsLocal * (co.collectionType as MatrixType).blocksInRow»
		void mkt::print_dist_«co.name»(const mkt::DMatrix<«type»>& m, const MPI_Datatype& dt) {
		  std::array<«type», «size»> m_copy;
		  MPI_Gatherv(m.get_data(), «sizeLocal», «mpiType», m_copy.data(), (std::array<int, «Config.processes»>{«FOR i: 0 ..< Config.processes SEPARATOR ', '»1«ENDFOR»}).data(), (std::array<int, «Config.processes»>{«FOR i: 0 ..< Config.processes SEPARATOR ', '»«displacement * (co.collectionType as MatrixType).partitionPosition(i).key + (co.collectionType as MatrixType).partitionPosition(i).value»«ENDFOR»}).data(), dt, 0, MPI_COMM_WORLD);
		  std::ostringstream stream;
		  stream << "«co.name»" << ": " << std::endl;
		  for (int i = 0; i < m.get_number_of_rows(); ++i) {
		    stream << "[";
		    for (int j = 0; j < m.get_number_of_columns() - 1; ++j) {
		      mkt::print<«type»>(stream, m_copy[i * m.get_number_of_columns() + j]);
		      stream << "; ";
		    }
		    mkt::print<«type»>(stream, m_copy[i * m.get_number_of_columns() + m.get_number_of_columns() - 1]);
		    stream << "]" << std::endl;
		  }		  
		  stream << std::endl;
		  printf("%s", stream.str().c_str());
		}
	'''
	
	def static generateStructPrintFunctions(Resource resource){
		var result = ""
		for(struct : resource.Structs){
			result += struct.generateStructPrintFunction
		}
		return result
	}
	
	def static generateStructPrintFunction(Struct struct)'''
		«val type = struct.name.toFirstUpper»
		template<>
		void mkt::print<«type»>(std::ostringstream& stream, const «type»& a) {
		  stream << "[";
		  «FOR member : struct.attributes SEPARATOR '''stream << "; ";'''»
		  	«val memberType = member.calculateType»
		  	«IF memberType.isCollection»
		  		«val collectionType = memberType.calculateCollectionType»
		  		«val size = (member.collectionType as ArrayType).size.concreteValue»
		  		stream << "«member.name»: [";
		  		for(int i = 0; i < «size -1»; ++i){
		  		  mkt::print<«collectionType»>(stream, a.«member.name»[i]);
		  		  stream << ", ";
		  		}
		  		mkt::print<«collectionType»>(stream, a.«member.name»[«size - 1»]);
		  		stream << "]";
		  	«ELSE»
		  		stream << "«member.name»: ";
		  		mkt::print<«memberType.cppType»>(stream, a.«member.name»);
		  	«ENDIF»
		  «ENDFOR»
		  stream << "]";
		}
	'''
		
	def static generateForwardDeclarations(Resource resource) '''
		«IF resource.Arrays.size() > 0»
			template<typename T>
			class DArray;
		«ENDIF»
		
		«IF resource.Matrices.size() > 0»
			template<typename T>
			class DMatrix;
		«ENDIF»
	'''
	
	def static generateDistEnum() '''
		enum Distribution {DIST, COPY};
	'''
}