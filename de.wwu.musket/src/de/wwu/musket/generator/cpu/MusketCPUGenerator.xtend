package de.wwu.musket.generator.cpu

import org.eclipse.emf.ecore.resource.Resource
import org.eclipse.xtext.generator.IFileSystemAccess2
import org.eclipse.xtext.generator.IGeneratorContext

class MusketCPUGenerator {
	def static void doGenerate(Resource resource, IFileSystemAccess2 fsa, IGeneratorContext context) {
		fsa.generateFile("relative/path/cpu/AllTheStates.txt", '''Some text''')
	}
}
