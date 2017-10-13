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
	def static dispatch generateObjectDeclaration(
		Array a) '''extern std::array<«a.CppPrimitiveTypeAsString», «a.sizeLocal»> «a.name»;'''

// Generate definitions	
	// variables
	def static dispatch generateObjectDefinition(
		Variable v) '''«v.CppPrimitiveTypeAsString» «v.name» = «v.ValueAsString»;'''

	// constants
	def static dispatch generateObjectDefinition(
		Constant c) '''const «c.CppPrimitiveTypeAsString» «c.name» = «c.ValueAsString»;'''

	// Arrays objects
	def static dispatch generateObjectDefinition(
		Array a) '''std::array<«a.CppPrimitiveTypeAsString», «a.sizeLocal»> «a.name»{};'''

// Generate initialization
	def static generateArrayInitialization(Array a){
		var result = ""
		val sizeLocal = a.sizeLocal
		for (var p = 0; p < Config.processes; p++) {
			result += (generateArrayInitializationForProcess(p, a, a.ValuesAsString.drop(sizeLocal * p).take(sizeLocal)))
		}
		return result
	}

	def static generateArrayInitializationForProcess(int p, Array a, Iterable<String> values) '''
		if(«Config.var_pid» == «p»){
			«var value_id = 0»
			«FOR v : values»
				«a.name»[«value_id++»] = «v»;
			«ENDFOR»
		}«IF p != Config.processes - 1» else«ENDIF» '''

	// Helper
	def static sizeLocal(Array a) {
		switch a.distributionMode {
			case DIST: a.size / Config.processes
			case COPY: a.size
			default: a.size
		}
	}
}
