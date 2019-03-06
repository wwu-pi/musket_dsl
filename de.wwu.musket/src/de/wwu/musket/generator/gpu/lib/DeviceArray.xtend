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

class DeviceArray {
	private static final Logger logger = LogManager.getLogger(DeviceArray)

	def static generateDArrayDeclaration() '''		
		template<typename T>
		class DeviceArray {
		 public:
		
		  };
	'''
	
	def static generateDArrayDefinition() '''
		
	'''
	
	
}