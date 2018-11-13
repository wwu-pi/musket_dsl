package de.wwu.musket.util

import de.wwu.musket.musket.IntRef
import de.wwu.musket.musket.Function
import de.wwu.musket.musket.RegularFunction
import java.util.Map
import de.wwu.musket.musket.LambdaFunction
import java.util.LinkedHashMap
import de.wwu.musket.musket.InternalFunctionCall
import de.wwu.musket.musket.SkeletonParameterInput
import de.wwu.musket.musket.Expression
import org.eclipse.emf.common.util.BasicEList
import de.wwu.musket.musket.Type
import de.wwu.musket.musket.IntArrayType
import de.wwu.musket.musket.IntMatrixType
import de.wwu.musket.musket.PrimitiveTypeLiteral
import de.wwu.musket.musket.BoolMatrixType
import de.wwu.musket.musket.BoolArrayType
import de.wwu.musket.musket.FloatArrayType
import de.wwu.musket.musket.DoubleArrayType
import de.wwu.musket.musket.DoubleMatrixType
import de.wwu.musket.musket.FloatMatrixType
import de.wwu.musket.musket.PrimitiveType
import static extension de.wwu.musket.generator.extensions.ObjectExtension.*
import de.wwu.musket.musket.CollectionType

class MusketHelper {
	/**
	 * Resolve concrete values from references
	 */
	static def getConcreteValue(IntRef ref) {
		if (ref.ref !== null) {
			return ref.ref.value
		}
		return ref.value
	}

	/**
	 * Map the Musket type to a C++ primitive type
	 */
	static def getCXXPrimitiveDefaultValue(Type t) {
		switch (t) {
			IntArrayType,
			IntMatrixType,
			PrimitiveType case t.type == PrimitiveTypeLiteral.INT: '''0'''
			DoubleArrayType,
			DoubleMatrixType,
			PrimitiveType case t.type == PrimitiveTypeLiteral.DOUBLE: '''0.0'''
			FloatArrayType,
			FloatMatrixType,
			PrimitiveType case t.type == PrimitiveTypeLiteral.FLOAT: '''0.0f'''
			BoolArrayType,
			BoolMatrixType,
			PrimitiveType case t.type == PrimitiveTypeLiteral.BOOL: '''false'''
			PrimitiveType case t.type == PrimitiveTypeLiteral.STRING: ''''''
			default:
				null
		}
	}

	/**
	 * Map Musket type to a C++ constructor call
	 */
	static def getCXXDefaultConstructorValue(Type t) {
		switch (t) {
			CollectionType: '''(«t.sizeLocal», «t.CXXPrimitiveDefaultValue»)'''
			PrimitiveType: '''(«t.CXXPrimitiveDefaultValue»)'''
			default: '''()'''
		}
	}
	
	/**
	 * Map Musket type to a corresponding MPI primitive type
	 */
	static def toMPIPrimitiveType(Type t){
		switch(t){
			PrimitiveType case t.type == PrimitiveTypeLiteral.BOOL: '''MPI_BOOL'''
			PrimitiveType case t.type == PrimitiveTypeLiteral.INT: '''MPI_INT'''
			PrimitiveType case t.type == PrimitiveTypeLiteral.FLOAT: '''MPI_FLOAT'''
			PrimitiveType case t.type == PrimitiveTypeLiteral.DOUBLE: '''MPI_DOUBLE'''
			PrimitiveType case t.type == PrimitiveTypeLiteral.STRING: '''MPI_CHAR'''
			default: ''''''
		}
	}
	
	/**
	 * Shortcut to reach function from skeleton parameter
	 */
	static def toFunction(SkeletonParameterInput spi) {
		switch spi {
			InternalFunctionCall:
				spi.value
			LambdaFunction:
				spi
		}
	}

	/**
	 * Prepare list of custom arguments to a user function 
	 */
	static def getFunctionArguments(SkeletonParameterInput spi) {
		switch spi {
			InternalFunctionCall:
				spi.params
			LambdaFunction:
				new BasicEList<Expression>()
		}
	}

	/**
	 * Prepare list of parameters of a user function 
	 */
	static def getFunctionParameters(SkeletonParameterInput spi) {
		switch spi {
			InternalFunctionCall:
				spi.value.params
			LambdaFunction:
				spi.params
		}
	}

	/**
	 * Shortcut to reach function name from skeleton parameter
	 */
	static def getFunctionName(SkeletonParameterInput spi) {
		switch spi {
			InternalFunctionCall:
				spi.value.name
			LambdaFunction:
				spi.name
		}
	}

	/**
	 * Shortcut to reach function name from abstract function object
	 */
	static def getName(Function func) {
		switch func {
			RegularFunction:
				func.name
			LambdaFunction: {
				if (State.getLambdaNames.containsKey(func))
					State.getLambdaNames.get(func)
				else {
					val name = "lambda" + State.nextLambdaCounter
					State.addLambdaName(func, name)
					name
				}
			}
		}
	}

	/**
	 * Internal data structure to capture lambda abstractions within the model.
	 */
	static class State {
		private static int lambdaCounter = 0;

		private static Map<LambdaFunction, String> lambdaNames = new LinkedHashMap<LambdaFunction, String>();

		def static getNextLambdaCounter() {
			lambdaCounter++
		}

		def static getLambdaNames() {
			return lambdaNames
		}

		def static addLambdaName(LambdaFunction lf, String name) {
			lambdaNames.put(lf, name)
		}
	}
}
