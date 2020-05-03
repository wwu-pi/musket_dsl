package de.wwu.musket.generator.cuda

import org.apache.log4j.LogManager
import org.apache.log4j.Logger
import org.eclipse.emf.ecore.resource.Resource
import org.eclipse.xtext.generator.IFileSystemAccess2
import org.eclipse.xtext.generator.IGeneratorContext

import static extension de.wwu.musket.generator.cuda.DataGenerator.*
import static extension de.wwu.musket.generator.cuda.FoldSkeletonGenerator.*
import static extension de.wwu.musket.generator.cuda.StructGenerator.*
import static extension de.wwu.musket.generator.extensions.ModelElementAccess.*

import static de.wwu.musket.generator.cuda.ShiftSkeletonGenerator.*
import de.wwu.musket.musket.MatrixType
import de.wwu.musket.musket.DistributionMode

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
	def static void generateHeaderFile(Resource resource, IFileSystemAccess2 fsa, IGeneratorContext context) {
		logger.info("Generate Header file.")
		fsa.generateFile(Config.base_path + Config.include_path + resource.ProjectName + Config.header_extension,
			headerFileContent(resource))
		logger.info("Generation of header file done.")
	}
	
	/**
	 * Creates a new header file for each executable.
	 */
	def static void generateProcessHeaderFiles(Resource resource, IFileSystemAccess2 fsa, IGeneratorContext context, int processId) {
		logger.info("Generate Header file.")
		fsa.generateFile(Config.base_path + Config.include_path + resource.ProjectName + '_' + processId + Config.header_extension,
			processHeaderFileContent(resource))
		logger.info("Generation of header file done.")
	}

	/**
	 * Generates the content of the header file.
	 * 
	 * TODO: fix/add forward declarations
	 * 
	 * @param resource the resource object
	 * @return the content of the header file
	 */
	def static headerFileContent(Resource resource) '''
		#pragma once
		
		«FOR s : resource.Structs»
			«s.generateStructDeclaration»
		«ENDFOR»
				
		«IF Config.processes > 1»
			«generateMPIStructTypeDeclarations(resource)»
			«generateMPIFoldOperatorDeclarations(resource)»
			
			«val dist_matrices = resource.Matrices.filter[it.type.distributionMode == DistributionMode.DIST]»
			«FOR m : dist_matrices»
				«generateMPIVectorTypeVariable(m.type as MatrixType)»
			«ENDFOR»
		«ENDIF»
	'''

	/**
	 * Generates the content of the header file.
	 * 
	 * @param resource the resource object
	 * @return the content of the header file
	 */
	def static processHeaderFileContent(Resource resource) '''
		#pragma once
		
		«FOR co : resource.CollectionObjects»
			«co.generateObjectDeclaration»
		«ENDFOR»		
	'''
}
