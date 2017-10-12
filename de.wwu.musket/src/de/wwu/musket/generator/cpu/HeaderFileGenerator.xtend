package de.wwu.musket.generator.cpu

import org.apache.log4j.LogManager
import org.apache.log4j.Logger
import org.eclipse.emf.ecore.resource.Resource
import org.eclipse.xtext.generator.IFileSystemAccess2
import org.eclipse.xtext.generator.IGeneratorContext

import static extension de.wwu.musket.generator.cpu.DataGenerator.*
import static extension de.wwu.musket.generator.cpu.FunctorGenerator.*

import static extension de.wwu.musket.generator.extensions.ModelElementAccess.*

class HeaderFileGenerator {

	private static final Logger logger = LogManager.getLogger(HeaderFileGenerator)

	def static void generateHeaderFile(Resource resource, IFileSystemAccess2 fsa, IGeneratorContext context) {
		logger.info("Generate Header file.")
		fsa.generateFile(Config.base_path + Config.include_path + resource.ProjectName + Config.header_extension,
			headerFileContent(resource))
		logger.info("Generation of header file done.")
	}

	def static headerFileContent(Resource resource) '''
		#pragma once
	
		«FOR d : resource.Data»
			«d.generateObjectDeclaration»
		«ENDFOR»
		
		«FOR f : resource.Functions»
			«f.generateFunctorDeclaration»
		«ENDFOR»
	'''
}
