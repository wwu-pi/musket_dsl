package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.CollectionParameter
import de.wwu.musket.musket.Function
import de.wwu.musket.musket.IndividualParameter

import static extension de.wwu.musket.generator.extensions.ObjectExtension.*

class FunctorGenerator {
	// declaration
	def static generateFunctorDeclaration(Function function) '''
		class «function.name.toFirstUpper»{
			public:
				«function.returnType» operator()«FOR p : function.params BEFORE '(' SEPARATOR ',' AFTER ')'»«p.generateParameter»«ENDFOR» const;
		};
	'''

	def static dispatch generateParameter(IndividualParameter p) '''«p.CppPrimitiveTypeAsString» «p.name»'''

	def static dispatch generateParameter(CollectionParameter p) '''
	
	'''

// definition
	def static generateFunctorDefinition(Function function) '''
		«function.returnType» «function.name.toFirstUpper»::operator() «FOR p : function.params BEFORE '(' SEPARATOR ',' AFTER ')'»«p.generateParameter»«ENDFOR» const{
			return i + i;
		}

«««		class «function.name.toFirstUpper»{
«««			public:
«««				«function.returnType» operator()«FOR p : function.params BEFORE '(' SEPARATOR ',' AFTER ')'»«p.generateParameter»«ENDFOR» const{
««««««					«function.statement»
«««					return i + i;
«««				}
«««		};
	'''

// instantiation
	def static generateFunctorInstantiation(Function function) '''
		«function.name.toFirstUpper» «function.name.toFirstLower»{};
	'''
}
