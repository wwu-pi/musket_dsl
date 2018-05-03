package de.wwu.musket.generator.cpu.mpmd

import de.wwu.musket.musket.Assignment
import de.wwu.musket.musket.CollectionObject
import de.wwu.musket.musket.ConditionalForLoop
import de.wwu.musket.musket.ControlStructure
import de.wwu.musket.musket.FoldSkeleton
import de.wwu.musket.musket.Function
import de.wwu.musket.musket.FunctionCall
import de.wwu.musket.musket.FunctionStatement
import de.wwu.musket.musket.GatherSkeleton
import de.wwu.musket.musket.IfClause
import de.wwu.musket.musket.InternalFunctionCall
import de.wwu.musket.musket.IteratorForLoop
import de.wwu.musket.musket.MapInPlaceSkeleton
import de.wwu.musket.musket.MapIndexInPlaceSkeleton
import de.wwu.musket.musket.MapLocalIndexInPlaceSkeleton
import de.wwu.musket.musket.MapSkeleton
import de.wwu.musket.musket.ReturnStatement
import de.wwu.musket.musket.ShiftPartitionsHorizontallySkeleton
import de.wwu.musket.musket.ShiftPartitionsVerticallySkeleton
import de.wwu.musket.musket.Skeleton
import de.wwu.musket.musket.Statement
import de.wwu.musket.musket.Variable
import java.util.Map

import static extension de.wwu.musket.generator.cpu.mpmd.ExpressionGenerator.*
import static extension de.wwu.musket.generator.cpu.mpmd.util.DataHelper.*
import static extension de.wwu.musket.util.TypeHelper.*
import de.wwu.musket.musket.SkeletonParameterInput
import de.wwu.musket.musket.LambdaFunction
import de.wwu.musket.musket.MapFoldSkeleton
import static extension de.wwu.musket.util.TypeHelper.*
import static extension de.wwu.musket.util.MusketHelper.*
import org.eclipse.emf.common.util.EList
import de.wwu.musket.musket.Expression

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

	def static generateFunctionCall(MapSkeleton s, CollectionObject input, int processId) '''«s.param.functionName»_functor(«s.param.functionArguments.generateArguments(processId)»«input.name»[«Config.var_loop_counter»]);'''
	
	def static generateArguments(EList<Expression> args, int processId) '''«FOR arg : args SEPARATOR ", " AFTER ", "»«arg.generateExpression(null, processId)»«ENDFOR»'''	
}

	

