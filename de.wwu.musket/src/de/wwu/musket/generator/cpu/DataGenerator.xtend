package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.Array
import de.wwu.musket.musket.Constant
import de.wwu.musket.musket.Variable

import static extension de.wwu.musket.generator.extensions.ObjectExtension.*
import de.wwu.musket.musket.Matrix
import de.wwu.musket.musket.Struct

class DataGenerator {
// Generate declarations	
	// variables
	def static dispatch generateObjectDeclaration(Variable v) '''extern «v.CppPrimitiveTypeAsString» «v.name»;'''

	// constants
	def static dispatch generateObjectDeclaration(Constant c) '''extern const «c.CppPrimitiveTypeAsString» «c.name»;'''

	// Arrays objects
	def static dispatch generateObjectDeclaration(
		Array a) '''extern std::array<«a.CppPrimitiveTypeAsString», «a.sizeLocal»> «a.name»;'''
		
	// Matrix objects
	def static dispatch generateObjectDeclaration(
		Matrix m) '''extern std::array<«m.CppPrimitiveTypeAsString», «m.sizeLocal»> «m.name»;'''
		
	def static dispatch generateObjectDeclaration(
		Struct s) '''//TODO struct unimplemented''' //TODO struct unimplemented

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
		
	// Arrays objects
	def static dispatch generateObjectDefinition(
		Matrix m) '''std::array<«m.CppPrimitiveTypeAsString», «m.sizeLocal»> «m.name»{};'''
		
	def static dispatch generateObjectDefinition(
		Struct s) '''//TODO struct unimplemented''' //TODO struct unimplemented

// Generate initialization
	def static generateArrayInitializationForProcess(Array a, int p, Iterable<String> values) '''		
		«var value_id = 0»
		«FOR v : values»
			«a.name»[«value_id++»] = «v»;
		«ENDFOR»
	'''
	
	def static generateArrayInitializationWithSingleValue(Array a) '''
		#pragma omp parallel for
		for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter»  < «a.sizeLocal»; «Config.var_loop_counter» ++){
			«a.name»[«Config.var_loop_counter»] = «IF a.ValuesAsString.size == 0»0«ELSE»«a.ValuesAsString.head»«ENDIF»;
		}
	'''
}
