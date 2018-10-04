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

class Musket {
	static final Logger logger = LogManager.getLogger(DMatrix)
	
	def static void generateDArrayHeaderFile(Resource resource, IFileSystemAccess2 fsa, IGeneratorContext context) {
		logger.info("Generate Musket header file.")
		fsa.generateFile(Config.base_path + Config.include_path + 'musket' + Config.header_extension,
			headerFileContent(resource))
		logger.info("Generation of Musket header file done.")
	}
	
	def static headerFileContent(Resource resource) '''
		#pragma once
		
		template<typename T, bool D>
		void mkt::print(string name, mkt::DArray<T> a);
		
		template<typename T>
		void mkt::print<true>(string name, mkt::DArray<T> a) {
		  std::array<T, a.get_size()> temp;
		  MPI_Gather(a.get_data(), a.get_size_local(), MPI_Datatype send_datatype, temp.data(), a.get_size_local(),
		      MPI_Datatype recv_datatype,
		      int 0, world)
		  std::ostringstream stream;
		  s1 << name << ": " << std::endl << "[";
		  for (int i = 0; i < a.get_size() - 1; ++i) {
		  	mkt::print(stream, a[i]);
		  	s1 << "; ";
		  }
		  mkt::print(stream, a[a.get_size() - 1]);
		  s1 << "]" << std::endl << std::endl;
		  printf("%s", s1.str().c_str());
		}
		
		template<typename T>
		void mkt::print<false>(string name, mkt::DArray<T> a) {
		  std::ostringstream stream;
		  s1 << name << ": " << std::endl << "[";
		  for (int i = 0; i < a.get_size() - 1; ++i) {
		  	mkt::print(stream, a[i]);
		  	s1 << "; ";
		  }
		  mkt::print(stream, a[a.get_size() - 1]);
		  s1 << "]" << std::endl << std::endl;
		  printf("%s", s1.str().c_str());
		}

		//template<typename T>
		//void mkt::print(string name, mkt::DMatrix<T> m) {
		  //std::ostringstream stream;
		  //s1 << name << ": " << std::endl << "[";
		  //for (int i = 0; i < m.get_size() - 1; ++i) {
		  //	s1 << m[i];
		  //	s1 << "; ";
		  //}
		  //s1 << temp1[m.get_size() - 1] << "]" << std::endl;
		  //s1 << std::endl;
		  //printf("%s", s1.str().c_str());
		//}
		
		template<T>
		void mkt::print(std::ostringstream& stream, T a) {
			if(std::is_fundamental<T>::value){
				stream << a;
			}
		}
	'''
}