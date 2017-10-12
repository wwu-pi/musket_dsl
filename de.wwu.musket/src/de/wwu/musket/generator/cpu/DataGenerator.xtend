package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.Array
import de.wwu.musket.musket.Constant
import de.wwu.musket.musket.Variable

import static extension de.wwu.musket.generator.extensions.ObjectExtension.*

class DataGenerator {
// Generate declarations	
	// variables
	def static dispatch generateObjectDeclaration(Variable v) '''extern «v.CppPrimitiveTypeAsSting» «v.name»;'''

	// constants
	def static dispatch generateObjectDeclaration(Constant c) '''extern const «c.CppPrimitiveTypeAsSting» «c.name»;'''

	// Arrays objects
	def static dispatch generateObjectDeclaration(Array a) '''extern std::array<«a.CppPrimitiveTypeAsSting», «a.sizeLocal»> «a.name»;'''

	// Helper
	def static sizeLocal(Array a) {
		switch a.distributionMode {
			case DIST: a.size / Config.processes
			case COPY: a.size
			default: a.size
		}
	}
}
