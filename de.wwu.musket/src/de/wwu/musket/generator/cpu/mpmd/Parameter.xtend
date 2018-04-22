package de.wwu.musket.generator.cpu.mpmd

import de.wwu.musket.musket.BoolVal
import de.wwu.musket.musket.IntVal
import de.wwu.musket.musket.DoubleVal
import de.wwu.musket.musket.StringVal
import de.wwu.musket.musket.CollectionObject
import de.wwu.musket.musket.ObjectRef
import de.wwu.musket.musket.FunctionCall

/**
 * @deprecated
 */
class Parameter {
	def static dispatch generateParameterInput(BoolVal pv)'''«pv.value»'''

	def static dispatch generateParameterInput(IntVal pv)'''«pv.value»'''
	
	def static dispatch generateParameterInput(DoubleVal pv)'''«pv.value»'''
	
	def static dispatch generateParameterInput(StringVal pv)'''"«pv.value.replaceAll("\n", "\\\\n").replaceAll("\t", "\\\\t")»"'''

	def static dispatch generateParameterInput(CollectionObject co)'''«co.name»'''
	
	def static dispatch generateParameterInput(ObjectRef or)'''«or.value.name»'''
	
	//functioncall
	def static dispatch generateParameterInput(FunctionCall or){
		throw new UnsupportedOperationException("Parameter.generateParameterInput: Function call as ParameterInput.")
	}
}