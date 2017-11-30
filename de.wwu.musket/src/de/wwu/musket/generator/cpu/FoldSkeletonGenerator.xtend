package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.FoldSkeleton
import de.wwu.musket.musket.InternalFunctionCall
import de.wwu.musket.musket.RegularFunction
import de.wwu.musket.musket.Array

import static extension de.wwu.musket.generator.extensions.StringExtension.*
import static extension de.wwu.musket.generator.extensions.ObjectExtension.*
import static extension de.wwu.musket.generator.cpu.FunctionGenerator.*
import java.util.Map
import de.wwu.musket.musket.ParameterInput
import java.util.HashMap
import de.wwu.musket.musket.Parameter
import de.wwu.musket.musket.FunctionCall
import de.wwu.musket.musket.ObjectRef
import de.wwu.musket.musket.DoubleVal
import de.wwu.musket.musket.IntVal
import de.wwu.musket.musket.BoolVal

class FoldSkeletonGenerator {
	def static generateMPIFoldFunction(FoldSkeleton foldSkeleton, Array a) '''
		void «((foldSkeleton.param as InternalFunctionCall).value as RegularFunction).name»(void *in, void *inout, int *len, MPI_Datatype *dptr){
			«val type = a.CppPrimitiveTypeAsString»
			«type»* inv = static_cast<«type»*>(in);
			«type»* inoutv = static_cast<«type»*>(inout);
			«val param_map = createParameterLookupTable(a, (foldSkeleton.param as InternalFunctionCall).value.params, (foldSkeleton.param as InternalFunctionCall).params)»
			«(foldSkeleton.param as InternalFunctionCall).generateInternalFunctionCallForSkeleton(foldSkeleton, a, param_map)»
		} 
	'''

	def static Map<String, String> createParameterLookupTable(Array a, Iterable<Parameter> parameters,
		Iterable<ParameterInput> inputs) {
		val param_map = new HashMap<String, String>

		param_map.put(parameters.drop(inputs.size).head.name, '''inoutv''')
		param_map.put(parameters.drop(inputs.size + 1).head.name, '''inv''')

		for (var i = 0; i < inputs.size; i++) {
			param_map.put(parameters.get(i).name, inputs.get(i).asString)
		}
		return param_map
	}

	def static String asString(ParameterInput pi) {
		switch pi {
			FunctionCall: '''ERROR FUNCTION CALL'''
			ObjectRef: '''«pi.value»'''
			IntVal: '''«pi.value»'''
			DoubleVal: '''«pi.value»'''
			BoolVal: '''«pi.value»'''
			default: ''''''
		}
	}

//	void myProd( void *in, void *inout, int *len, MPI_Datatype *dptr ) 
//{ 
//	int* inv = static_cast<int*>(in);
//	int* inoutv = static_cast<int*>(inout);
//	*inoutv = *inoutv + *inv;
//} 
}
