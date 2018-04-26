package de.wwu.musket.generator.cpu.mpmd

import de.wwu.musket.musket.Struct
import static extension de.wwu.musket.util.MusketHelper.*
import static extension de.wwu.musket.util.TypeHelper.*
import static extension de.wwu.musket.generator.extensions.StringExtension.*
import static extension de.wwu.musket.generator.cpu.mpmd.util.DataHelper.*
import de.wwu.musket.musket.CollectionType
import de.wwu.musket.musket.CollectionObject

/**
 * Generates the declaration of structs.
 * 
 * TODO: possibly merge with data generator.
 */
class StructGenerator {

	/**
	 * Generates the declaration of structs.
	 * 
	 * @param s the struct
	 * @return the generated declaration
	 */
	def static generateStructDeclaration(Struct s) '''
		struct «s.name.toFirstUpper»{
			«FOR m : s.attributes»
				«IF m instanceof CollectionObject»
					«m.calculateType.cppType» «m.name.toFirstLower»;
				«ELSE»
					«m.calculateType.cppType» «m.name.toFirstLower»;
				«ENDIF»
			«ENDFOR»
			
			«s.name.toFirstUpper»();
		};
	'''
	
	/**
	 * Generates the default constructor for a struct.
	 * 
	 * @param s the struct
	 * @return the generated declaration
	 */
	def static generateStructDefaultConstructor(Struct s) '''
		«s.name.toFirstUpper»::«s.name.toFirstUpper»()«FOR m : s.attributes BEFORE " : " SEPARATOR ", "»«m.name.toFirstLower»«IF m.calculateType.collection»«m.calculateType.collectionType.CXXDefaultConstructorValue»«ELSE»«m.calculateType.primitiveType.CXXDefaultConstructorValue»«ENDIF»«ENDFOR» {}
	'''
}
