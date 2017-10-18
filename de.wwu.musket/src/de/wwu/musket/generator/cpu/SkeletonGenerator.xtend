package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.Array
import de.wwu.musket.musket.BoolVal
import de.wwu.musket.musket.DoubleArray
import de.wwu.musket.musket.DoubleVal
import de.wwu.musket.musket.FunctionCall
import de.wwu.musket.musket.IntVal
import de.wwu.musket.musket.InternalFunctionCall
import de.wwu.musket.musket.ObjectRef
import de.wwu.musket.musket.Parameter
import de.wwu.musket.musket.ParameterInput
import de.wwu.musket.musket.SkeletonStatement
import java.util.HashMap
import java.util.Map
import java.util.Map.Entry

import static extension de.wwu.musket.generator.cpu.FunctionGenerator.*
import static extension de.wwu.musket.generator.extensions.ObjectExtension.*

class SkeletonGenerator {
	def static generateSkeletonStatement(SkeletonStatement s) {
		switch s.function {
			case MAP_IN_PLACE: generateMapInPlaceSkeleton(s)
			default: ''''''
		}
	}

	def static generateMapInPlaceSkeleton(SkeletonStatement s) {
		switch s.obj {
			Array: generateArrayMapInPlaceSkeleton(s, s.obj as Array)
		}
	}

	def static generateArrayMapInPlaceSkeleton(SkeletonStatement s, Array a) '''
«««	create lookup table for parameters
		«val param_map = createParameterLookupTable(a, (s.param as InternalFunctionCall).value.params, (s.param as InternalFunctionCall).params)»
		«FOR p : param_map.entrySet»
			
		«ENDFOR»
		#pragma omp parallel for
		for(int «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «a.sizeLocal»; ++«Config.var_loop_counter»){
			«(s.param as InternalFunctionCall).generateInternalFunctionCallForSkeleton(a, param_map)»
		}
	'''
	
	def static  generateParameter(Entry<String, ParameterInput> set){
		switch set.value{
			DoubleArray: '''double «set.key» = «(set.value as DoubleArray).name»[«»];'''
			default: ''''''
		}
	}
	
	
	def static Map<String, String> createParameterLookupTable(Array a, Iterable<Parameter> parameters, Iterable<ParameterInput> inputs){
		val param_map = new HashMap<String, String>
		
		param_map.put(parameters.drop(inputs.size).head.name, '''«a.name»[«Config.var_loop_counter»]''')
		
		for(var i = 0; i < inputs.size; i++){
			param_map.put(parameters.get(i).name, inputs.get(i).asString)
		}
		return param_map
	}
	
	def static String asString(ParameterInput pi){				
		switch pi{
			FunctionCall:'''ERROR FUNCTION CALL'''
			ObjectRef:'''«pi.value»'''
			IntVal:'''«pi.value»'''
			DoubleVal:'''«pi.value»'''
			BoolVal:'''«pi.value»'''
			default: ''''''
		}
	}

}
