package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.ArrayType
import de.wwu.musket.musket.Constant
import de.wwu.musket.musket.Variable

import static extension de.wwu.musket.generator.extensions.ObjectExtension.*
import de.wwu.musket.musket.MatrixType
import de.wwu.musket.musket.Struct
import de.wwu.musket.musket.CollectionObject
import static extension de.wwu.musket.util.TypeHelper.*

/**
 * Generates declaration, definition and initialization of data object.
 * <p>
 * The Generator handles arrays and matrices, global variables and constants, and structs.
 * Methods for declarations and struct definitions are called by the header generator.
 * Other definitions and initialization methods are called by the source file generator.
 */
class DataGenerator {
// Generate declarations	
	// variables
	def static dispatch generateObjectDeclaration(Variable v) '''extern «v.calculateCollectionType.cppType» «v.name»;'''

	// constants
	def static dispatch generateObjectDeclaration(
		Constant c) '''extern const «c.calculateCollectionType.cppType» «c.name»;'''

	def static dispatch generateObjectDeclaration(CollectionObject c) {
		switch (c.type) {
			ArrayType: '''extern std::vector<«c.calculateCollectionType.cppType»> «c.name»;'''
			MatrixType: '''extern std::vector<«c.calculateCollectionType.cppType»> «c.name»;'''
		}
	}

	def static dispatch generateObjectDeclaration(Struct s) '''
		struct «s.name.toFirstUpper»{
			«FOR m : s.attributes»
				«m.calculateType.cppType» «m.name.toFirstLower»;
			«ENDFOR»
		};
	''' // this is done in StructGenerator.xtend

// Generate definitions	
	// variables
	def static dispatch generateObjectDefinition(
		Variable v) '''«v.calculateCollectionType.cppType» «v.name» = «v.ValueAsString»;'''

	// constants
	def static dispatch generateObjectDefinition(
		Constant c) '''const «c.calculateType.cppType» «c.name» = «c.ValueAsString»;'''

	// Arrays objects
	def static dispatch generateObjectDefinition(CollectionObject c) {
		switch (c.type) {
			ArrayType: '''std::vector<«c.calculateCollectionType.cppType»> «c.name»(«c.type.sizeLocal»);'''
			MatrixType: '''std::vector<«c.calculateCollectionType.cppType»> «c.name»(«c.type.sizeLocal»);'''
		}
	}

	def static dispatch generateObjectDefinition(Struct s) '''''' // this is done in StructGenerator.xtend

// Generate initialization
	def static generateArrayInitializationForProcess(CollectionObject a, Iterable<String> values) '''
		«var value_id = 0»
		«FOR v : values»
			«a.name»[«value_id++»] = «v»;
		«ENDFOR»
	'''

	def static generateInitializationWithSingleValue(CollectionObject a) '''
		«IF Config.cores > 1»
			#pragma omp parallel for simd
		«ENDIF»
		for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter»  < «a.type.sizeLocal»; ++«Config.var_loop_counter»){
			«a.name»[«Config.var_loop_counter»] = «IF a.ValuesAsString.size == 0»0«ELSE»«a.ValuesAsString.head»«ENDIF»;
		}
	'''
}
