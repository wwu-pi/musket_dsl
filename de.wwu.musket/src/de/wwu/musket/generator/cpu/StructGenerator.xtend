package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.Struct
import static extension de.wwu.musket.util.MusketHelper.*
import static extension de.wwu.musket.util.TypeHelper.*

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
				«m.calculateType.cppType» «m.name.toFirstLower»;
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
