package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.Struct
import static extension de.wwu.musket.generator.extensions.ObjectExtension.*

class StructGenerator {
	def static generateStructDeclaration(Struct s)'''
		struct «s.name.toFirstUpper»{
			«FOR m : s.attributes»
				«m.CppPrimitiveTypeAsString» «m.name.toFirstLower»;
			«ENDFOR»
		};
	'''
}