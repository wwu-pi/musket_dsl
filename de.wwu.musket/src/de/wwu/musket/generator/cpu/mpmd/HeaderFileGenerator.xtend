package de.wwu.musket.generator.cpu.mpmd

import org.apache.log4j.LogManager
import org.apache.log4j.Logger
import org.eclipse.emf.ecore.resource.Resource
import org.eclipse.xtext.generator.IFileSystemAccess2
import org.eclipse.xtext.generator.IGeneratorContext

import static extension de.wwu.musket.generator.cpu.mpmd.DataGenerator.*
import static extension de.wwu.musket.generator.cpu.mpmd.StructGenerator.*
import static extension de.wwu.musket.generator.extensions.ModelElementAccess.*

/** 
 * Generates the header file.
 * <p>
 * Entry point is the function generateHeaderFile(Resource resource, IFileSystemAccess2 fsa, IGeneratorContext context).
 * The generated header file is located in the include folder, with the name project_name.hpp.
 * It inludes declarations of data strucutures and structs.
 */
class HeaderFileGenerator {

	private static final Logger logger = LogManager.getLogger(HeaderFileGenerator)

	/**
	 * Creates a new header file for the project.
	 */
	def static void generateHeaderFile(Resource resource, IFileSystemAccess2 fsa, IGeneratorContext context, int processId) {
		logger.info("Generate Header file.")
		fsa.generateFile(Config.base_path + Config.include_path + resource.ProjectName + '_' + processId + Config.header_extension,
			headerFileContent(resource))
		logger.info("Generation of header file done.")
	}

	/**
	 * Generates the content of the header file.
	 * 
	 * @param resource the resource object
	 * @return the content of the header file
	 */
	def static headerFileContent(Resource resource) '''
		#pragma once
		
		«FOR s : resource.Structs»
			«s.generateStructDeclaration»
		«ENDFOR»
		
		«FOR co : resource.CollectionObjects»
			«co.generateObjectDeclaration»
		«ENDFOR»		
	'''
}
