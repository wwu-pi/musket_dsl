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

class MusketHelper {
	// Resolve concrete values from references
	static def getConcreteValue(IntRef ref) {
		if (ref.ref !== null) {
			return ref.ref.value
		}
		return ref.value
	}

	static def getFunctionArguments(SkeletonParameterInput spi) {
		switch spi {
			InternalFunctionCall:
				spi.params
			LambdaFunction:
				new BasicEList<Expression>()
		}
	}

	static def getFunctionParameters(SkeletonParameterInput spi) {
		switch spi {
			InternalFunctionCall:
				spi.value.params
			LambdaFunction:
				spi.params
		}
	}

	static def getFunctionName(SkeletonParameterInput spi) {
		switch spi {
			InternalFunctionCall:
				spi.value.name
			LambdaFunction:
				spi.name
		}
	}

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
