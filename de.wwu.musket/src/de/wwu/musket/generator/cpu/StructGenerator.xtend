package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.Struct
import static extension de.wwu.musket.util.TypeHelper.*


class StructGenerator {
	def static generateStructDeclaration(Struct s)'''
		struct «s.name.toFirstUpper»{
			«FOR m : s.attributes»
				«m.calculateType.cppType» «m.name.toFirstLower»;
			«ENDFOR»
		};
	'''
}