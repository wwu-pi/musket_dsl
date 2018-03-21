package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.Struct
import static extension de.wwu.musket.generator.extensions.ObjectExtension.*
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
				«IF m.calculateType.collection»std::array<«m.calculateType.calculateCollectionType», «m.calculateType.collectionType.size»>«ELSE»«m.calculateType.cppType»«ENDIF» «m.name.toFirstLower»;
			«ENDFOR»
		};
	'''
}
