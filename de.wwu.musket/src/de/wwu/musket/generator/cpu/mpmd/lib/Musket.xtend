package de.wwu.musket.generator.cpu.mpmd.lib

import org.apache.log4j.LogManager
import org.apache.log4j.Logger
import org.eclipse.emf.ecore.resource.Resource
import org.eclipse.xtext.generator.IFileSystemAccess2
import org.eclipse.xtext.generator.IGeneratorContext

import static extension de.wwu.musket.generator.cpu.mpmd.DataGenerator.*
import static extension de.wwu.musket.generator.cpu.mpmd.StructGenerator.*
import static extension de.wwu.musket.generator.extensions.ModelElementAccess.*
import static extension de.wwu.musket.generator.cpu.mpmd.util.DataHelper.*
import static extension de.wwu.musket.util.MusketHelper.*
import static extension de.wwu.musket.util.TypeHelper.*
import static de.wwu.musket.generator.cpu.mpmd.lib.DArray.*
import static de.wwu.musket.generator.cpu.mpmd.lib.DMatrix.*
import de.wwu.musket.generator.cpu.mpmd.Config
import java.util.List
import de.wwu.musket.musket.CollectionFunctionCall
import de.wwu.musket.musket.DistributionMode
import de.wwu.musket.musket.CollectionObject
import de.wwu.musket.musket.ArrayType
import de.wwu.musket.musket.Struct

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
		
		namespace mkt {
		
		«generateDistEnum»
		
		«generateDArrayDeclaration»
		«generateDMatrixDeclaration»
		«generateDArraySkeletonDeclarations»
		
		«val showCalls = resource.ShowCalls»
		«IF showCalls.size() > 0»
			template<typename T>
			void print_dist(const std::string& name, const mkt::DArray<T>& a);
		«ENDIF»
		
		template<typename T>
		void print(const std::string& name, const mkt::DArray<T>& a);
		
		// for primitive values
		
		template<typename T>
		void print(std::ostringstream& stream, const T a);
		
		«IF resource.Structs.size > 0»
			// for structs
			template<typename T>
			void print(std::ostringstream& stream, const T& a);
		«ENDIF»
		
		} // namespace mkt
		
		«generateDArrayDefinition»
		«generateDMatrixDefinition»
		«generateDArraySkeletonDefinitions»
		
		«generatePrintDistFunctions(showCalls)»
			
		template<typename T>
		void mkt::print(const std::string& name, const mkt::DArray<T>& a) {
		  std::ostringstream stream;
		  stream << name << ": " << std::endl << "[";
		  for (int i = 0; i < a.get_size() - 1; ++i) {
		  	mkt::print(stream, a.get_local(i));
		  	stream << "; ";
		  }
		  mkt::print(stream, a.get_local(a.get_size() - 1));
		  stream << "]" << std::endl << std::endl;
		  printf("%s", stream.str().c_str());
		}

		//template<typename T>
		//void mkt::print(const std::string name, mkt::DMatrix<T> m) {
		  //std::ostringstream stream;
		  //stream << name << ": " << std::endl << "[";
		  //for (int i = 0; i < m.get_size() - 1; ++i) {
		  //	stream << m[i];
		  //	stream << "; ";
		  //}
		  //stream << temp1[m.get_size() - 1] << "]" << std::endl;
		  //stream << std::endl;
		  //printf("%s", stream.str().c_str());
		//}
		
		template<typename T>
		void mkt::print(std::ostringstream& stream, const T a) {
			if(std::is_fundamental<T>::value){
				stream << a;
			}
		}
		
		«IF resource.Structs.size > 0»
			«resource.generateStructPrintFunctions»
		«ENDIF»
	'''
	
	def static generatePrintDistFunctions(List<CollectionFunctionCall> showCalls){
		var result = ""
		var types = newArrayList()
		for (sc : showCalls){
			if(sc.^var.type.distributionMode == DistributionMode.DIST && !types.contains(sc.^var.calculateCollectionType)){
				types.add(sc.^var.calculateCollectionType)
				result += generatePrintDistFunction(sc.^var)
			}
		}
		return result
	}
	
	def static generatePrintDistFunction(CollectionObject co)'''
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
		  	mkt::print(stream, a_copy[i]);
		  	stream << "; ";
		  }
		  mkt::print(stream, a_copy[«size - 1»]);
		  stream << "]" << std::endl << std::endl;
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
		  «FOR member : struct.attributes SEPARATOR '''stream << ";";'''»
		  	«IF member.calculateType.isCollection»
		  		«val size = member.calculateType.size»
		  		stream << "«member.name»: [";
		  		for(int i = 0; i < «size -1»; ++i){
		  		  mkt::print(stream, a.«member.name»[i]);
		  		}
		  		mkt::print(stream, a.«member.name»[«size - 1»]);
		  		stream << "]";
		  	«ELSE»
		  		stream << "«member.name»: ";
		  		mkt::print(stream, a.«member.name»);
		  	«ENDIF»
		  «ENDFOR»
		  stream << "]";
		}
	'''
		
	def static generateForwardDeclarations() '''
		template<typename T>
		class DArray;
		
		template<typename T>
		class DMatrix;
	'''
	
	def static generateDistEnum() '''
		enum Distribution {DIST, COPY};
	'''
}