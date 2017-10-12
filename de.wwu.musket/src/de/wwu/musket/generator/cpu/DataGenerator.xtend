package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.Array
import de.wwu.musket.musket.Constant
import de.wwu.musket.musket.Variable

import static extension de.wwu.musket.generator.extensions.ObjectExtension.*


class DataGenerator {
// Generate declarations	
	// variables
	def static dispatch generateObjectDeclaration(Variable v) '''extern «v.CppPrimitiveTypeAsString» «v.name»;'''

	// constants
	def static dispatch generateObjectDeclaration(Constant c) '''extern const «c.CppPrimitiveTypeAsString» «c.name»;'''

	// Arrays objects
	def static dispatch generateObjectDeclaration(Array a) '''extern std::array<«a.CppPrimitiveTypeAsString», «a.sizeLocal»> «a.name»;'''

// Generate definitions	
	// variables
	def static dispatch generateObjectDefinition(Variable v) '''«v.CppPrimitiveTypeAsString» «v.name» = «v.ValueAsString»;'''

	// constants
	def static dispatch generateObjectDefinition(Constant c) '''const «c.CppPrimitiveTypeAsString» «c.name» = «c.ValueAsString»;'''

	// Arrays objects
	def static dispatch generateObjectDefinition(Array a) '''std::array<«a.CppPrimitiveTypeAsString», «a.sizeLocal»> «a.name»{};'''



	// Helper
	def static sizeLocal(Array a) {
		switch a.distributionMode {
			case DIST: a.size / Config.processes
			case COPY: a.size
			default: a.size
		}
	}
}
