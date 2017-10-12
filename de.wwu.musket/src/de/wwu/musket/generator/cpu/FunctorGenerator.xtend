package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.CollectionParameter
import de.wwu.musket.musket.Function
import de.wwu.musket.musket.IndividualParameter

import static extension de.wwu.musket.generator.extensions.ObjectExtension.*

class FunctorGenerator {
	def static generateFunctorDeclaration(Function function) '''
		class «function.name.toFirstUpper»{
			public:
				«function.returnType» operator()«FOR p : function.params BEFORE '(' SEPARATOR ',' AFTER ')'»«p.generateParameter»«ENDFOR» const;
		}
	'''

	def static dispatch generateParameter(IndividualParameter p)
	 '''«p.CppPrimitiveTypeAsSting» «p.name»'''

	def static dispatch generateParameter(CollectionParameter p) '''
	
	'''

}
