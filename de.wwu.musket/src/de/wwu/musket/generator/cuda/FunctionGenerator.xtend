package de.wwu.musket.generator.cuda

import de.wwu.musket.musket.ArrayType
import de.wwu.musket.musket.CollectionObject
import de.wwu.musket.musket.Expression
import de.wwu.musket.musket.MapIndexSkeleton
import de.wwu.musket.musket.MapLocalIndexSkeleton
import de.wwu.musket.musket.MapSkeleton
import de.wwu.musket.musket.MatrixType
import de.wwu.musket.musket.ZipSkeleton
import org.eclipse.emf.common.util.EList

import static extension de.wwu.musket.generator.cuda.ExpressionGenerator.*
import static extension de.wwu.musket.util.MusketHelper.*
import de.wwu.musket.musket.ZipInPlaceSkeleton
import de.wwu.musket.musket.ZipIndexSkeleton
import de.wwu.musket.musket.ZipLocalIndexSkeleton
import de.wwu.musket.musket.ZipIndexInPlaceSkeleton
import de.wwu.musket.musket.ZipLocalIndexInPlaceSkeleton

/**
 * Generates a function call.
 * <p>
 * For example, if a map skeleton is called with function f as argument, then this generator generates f as main
 * body of the map skeleton.
 * <p>
 * The entry point is the method generateInternalFunctionCallForSkeleton(InternalFunctionCall ifc, Skeleton skeleton, CollectionObject a,
 * 		Map<String, String> param_map).
 * The methods are mostly called from the skeleton generator.
 */
class FunctionGenerator {

// map variants
	def static dispatch generateFunctionCall(MapSkeleton s, CollectionObject input, int processId) '''«s.param.functionName.toFirstLower»_functor(«s.param.functionArguments.generateArguments(processId)»«input.name»[«Config.var_loop_counter»]);'''
	
	def static dispatch generateFunctionCall(MapIndexSkeleton s, ArrayType input, int processId) '''«s.param.functionName.toFirstLower»_functor(«s.param.functionArguments.generateArguments(processId)»«Config.var_loop_counter» + «Config.var_elem_offset», «(input.eContainer as CollectionObject).name»[«Config.var_loop_counter»]);'''
	def static dispatch generateFunctionCall(MapIndexSkeleton s, MatrixType input, int processId) '''«s.param.functionName.toFirstLower»_functor(«s.param.functionArguments.generateArguments(processId)»«Config.var_loop_counter_rows» + «Config.var_row_offset», «Config.var_loop_counter_cols» + «Config.var_col_offset», «(input.eContainer as CollectionObject).name»[«Config.var_loop_counter»]);'''
		
	def static dispatch generateFunctionCall(MapLocalIndexSkeleton s, ArrayType input, int processId) '''«s.param.functionName.toFirstLower»_functor(«s.param.functionArguments.generateArguments(processId)»«Config.var_loop_counter», «(input.eContainer as CollectionObject).name»[«Config.var_loop_counter»]);'''
	def static dispatch generateFunctionCall(MapLocalIndexSkeleton s, MatrixType input, int processId) '''«s.param.functionName.toFirstLower»_functor(«s.param.functionArguments.generateArguments(processId)»«Config.var_loop_counter_rows», «Config.var_loop_counter_cols», «(input.eContainer as CollectionObject).name»[«Config.var_loop_counter»]);'''
	
// zip variants
	def static dispatch generateFunctionCall(ZipSkeleton s, CollectionObject input, CollectionObject zip_input, int processId) '''«s.param.functionName.toFirstLower»_functor(«s.param.functionArguments.generateArguments(processId)»«zip_input.name»[«Config.var_loop_counter»], «input.name»[«Config.var_loop_counter»]);'''
	
	def static dispatch generateFunctionCall(ZipInPlaceSkeleton s, CollectionObject input, CollectionObject zip_input, int processId) '''«s.param.functionName.toFirstLower»_functor(«s.param.functionArguments.generateArguments(processId)»«zip_input.name»[«Config.var_loop_counter»], «input.name»[«Config.var_loop_counter»]);'''
	
	def static dispatch generateFunctionCall(ZipIndexSkeleton s, ArrayType input, ArrayType zip_input, int processId) '''«s.param.functionName.toFirstLower»_functor(«s.param.functionArguments.generateArguments(processId)»«Config.var_loop_counter» + «Config.var_elem_offset», «(zip_input.eContainer as CollectionObject).name»[«Config.var_loop_counter»], «(input.eContainer as CollectionObject).name»[«Config.var_loop_counter»]);'''
	def static dispatch generateFunctionCall(ZipIndexSkeleton s, MatrixType input, MatrixType zip_input, int processId) '''«s.param.functionName.toFirstLower»_functor(«s.param.functionArguments.generateArguments(processId)»«Config.var_loop_counter_rows» + «Config.var_row_offset», «Config.var_loop_counter_cols» + «Config.var_col_offset», «(zip_input.eContainer as CollectionObject).name»[«Config.var_loop_counter»], «(input.eContainer as CollectionObject).name»[«Config.var_loop_counter»]);'''
	
	def static dispatch generateFunctionCall(ZipLocalIndexSkeleton s, ArrayType input, ArrayType zip_input, int processId) '''«s.param.functionName.toFirstLower»_functor(«s.param.functionArguments.generateArguments(processId)»«Config.var_loop_counter», «(zip_input.eContainer as CollectionObject).name»[«Config.var_loop_counter»], «(input.eContainer as CollectionObject).name»[«Config.var_loop_counter»]);'''
	def static dispatch generateFunctionCall(ZipLocalIndexSkeleton s, MatrixType input, MatrixType zip_input, int processId) '''«s.param.functionName.toFirstLower»_functor(«s.param.functionArguments.generateArguments(processId)»«Config.var_loop_counter_rows», «Config.var_loop_counter_cols», «(zip_input.eContainer as CollectionObject).name»[«Config.var_loop_counter»], «(input.eContainer as CollectionObject).name»[«Config.var_loop_counter»]);'''
	
	def static dispatch generateFunctionCall(ZipIndexInPlaceSkeleton s, ArrayType input, ArrayType zip_input, int processId) '''«s.param.functionName.toFirstLower»_functor(«s.param.functionArguments.generateArguments(processId)»«Config.var_loop_counter» + «Config.var_elem_offset», «(zip_input.eContainer as CollectionObject).name»[«Config.var_loop_counter»], «(input.eContainer as CollectionObject).name»[«Config.var_loop_counter»]);'''
	def static dispatch generateFunctionCall(ZipIndexInPlaceSkeleton s, MatrixType input, MatrixType zip_input, int processId) '''«s.param.functionName.toFirstLower»_functor(«s.param.functionArguments.generateArguments(processId)»«Config.var_loop_counter_rows» + «Config.var_row_offset», «Config.var_loop_counter_cols» + «Config.var_col_offset», «(zip_input.eContainer as CollectionObject).name»[«Config.var_loop_counter»], «(input.eContainer as CollectionObject).name»[«Config.var_loop_counter»]);'''
	
	def static dispatch generateFunctionCall(ZipLocalIndexInPlaceSkeleton s, ArrayType input, ArrayType zip_input, int processId) '''«s.param.functionName.toFirstLower»_functor(«s.param.functionArguments.generateArguments(processId)»«Config.var_loop_counter», «(zip_input.eContainer as CollectionObject).name»[«Config.var_loop_counter»], «(input.eContainer as CollectionObject).name»[«Config.var_loop_counter»]);'''
	def static dispatch generateFunctionCall(ZipLocalIndexInPlaceSkeleton s, MatrixType input, MatrixType zip_input, int processId) '''«s.param.functionName.toFirstLower»_functor(«s.param.functionArguments.generateArguments(processId)»«Config.var_loop_counter_rows», «Config.var_loop_counter_cols», «(zip_input.eContainer as CollectionObject).name»[«Config.var_loop_counter»], «(input.eContainer as CollectionObject).name»[«Config.var_loop_counter»]);'''
	
	
	def static generateArguments(EList<Expression> args, int processId) '''«FOR arg : args SEPARATOR ", " AFTER ", "»«arg.generateExpression(null, processId)»«ENDFOR»'''	
}

	

