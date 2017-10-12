package de.wwu.musket.generator.extensions

import de.wwu.musket.musket.BoolArray
import de.wwu.musket.musket.BoolArrayParameter
import de.wwu.musket.musket.BoolConstant
import de.wwu.musket.musket.BoolParameter
import de.wwu.musket.musket.BoolVariable
import de.wwu.musket.musket.DoubleArray
import de.wwu.musket.musket.DoubleArrayParameter
import de.wwu.musket.musket.DoubleConstant
import de.wwu.musket.musket.DoubleParameter
import de.wwu.musket.musket.DoubleVariable
import de.wwu.musket.musket.IntArray
import de.wwu.musket.musket.IntArrayParameter
import de.wwu.musket.musket.IntConstant
import de.wwu.musket.musket.IntParameter
import de.wwu.musket.musket.IntVariable

import static extension de.wwu.musket.generator.extensions.ExpressionGenerator.*

class ObjectExtension {
	// get primitive cpp type as string for musket object element
	def static dispatch CppPrimitiveTypeAsString(IntArray o) {
		'int'
	}

	def static dispatch CppPrimitiveTypeAsString(DoubleArray o) {
		'double'
	}

	def static dispatch CppPrimitiveTypeAsString(BoolArray o) {
		'bool'
	}

	def static dispatch CppPrimitiveTypeAsString(IntVariable o) {
		'int'
	}

	def static dispatch CppPrimitiveTypeAsString(DoubleVariable o) {
		'double'
	}

	def static dispatch CppPrimitiveTypeAsString(BoolVariable o) {
		'bool'
	}

	def static dispatch CppPrimitiveTypeAsString(IntConstant o) {
		'int'
	}

	def static dispatch CppPrimitiveTypeAsString(DoubleConstant o) {
		'double'
	}

	def static dispatch CppPrimitiveTypeAsString(BoolConstant o) {
		'bool'
	}

	def static dispatch CppPrimitiveTypeAsString(IntArrayParameter o) {
		'int'
	}

	def static dispatch CppPrimitiveTypeAsString(DoubleArrayParameter o) {
		'double'
	}

	def static dispatch CppPrimitiveTypeAsString(BoolArrayParameter o) {
		'bool'
	}

	def static dispatch CppPrimitiveTypeAsString(IntParameter o) {
		'int'
	}

	def static dispatch CppPrimitiveTypeAsString(DoubleParameter o) {
		'double'
	}

	def static dispatch CppPrimitiveTypeAsString(BoolParameter o) {
		'bool'
	}

	// Value
	// Variable
	def static dispatch ValueAsString(IntVariable o) {
		if (o.initExpression !== null) {
			o.initExpression.generateString
		} else {
			o.initValue.toString
		}
	}

	def static dispatch ValueAsString(DoubleVariable o) {
		if (o.initExpression !== null) {
			o.initExpression.generateString
		} else {
			o.initValue.toString
		}
	}

	def static dispatch ValueAsString(BoolVariable o) {
		if (o.initExpression !== null) {
			o.initExpression.generateString
		} else {
			o.initValue.toString
		}
	}

	// Constants
	def static dispatch ValueAsString(IntConstant o) {
		o.value.toString
	}

	def static dispatch ValueAsString(DoubleConstant o) {
		o.value.toString
	}

	def static dispatch ValueAsString(BoolConstant o) {
		o.value.toString
	}
}
