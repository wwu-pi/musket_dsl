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

class ObjectExtension {
	// get primitive cpp type as string for musket object element
	def static dispatch CppPrimitiveTypeAsSting(IntArray o) {
		'int'
	}
	
	def static dispatch CppPrimitiveTypeAsSting(DoubleArray o) {
		'double'
	}
	
	def static dispatch CppPrimitiveTypeAsSting(BoolArray o) {
		'bool'
	}
	
	def static dispatch CppPrimitiveTypeAsSting(IntVariable o) {
		'int'
	}
	
	def static dispatch CppPrimitiveTypeAsSting(DoubleVariable o) {
		'double'
	}
	
	def static dispatch CppPrimitiveTypeAsSting(BoolVariable o) {
		'bool'
	}
	
	def static dispatch CppPrimitiveTypeAsSting(IntConstant o) {
		'int'
	}
	
	def static dispatch CppPrimitiveTypeAsSting(DoubleConstant o) {
		'double'
	}
	
	def static dispatch CppPrimitiveTypeAsSting(BoolConstant o) {
		'bool'
	}
	
	def static dispatch CppPrimitiveTypeAsSting(IntArrayParameter o) {
		'int'
	}
	
	def static dispatch CppPrimitiveTypeAsSting(DoubleArrayParameter o) {
		'double'
	}
	
	def static dispatch CppPrimitiveTypeAsSting(BoolArrayParameter o) {
		'bool'
	}
	
	def static dispatch CppPrimitiveTypeAsSting(IntParameter o) {
		'int'
	}
	
	def static dispatch CppPrimitiveTypeAsSting(DoubleParameter o) {
		'double'
	}
	
	def static dispatch CppPrimitiveTypeAsSting(BoolParameter o) {
		'bool'
	}
}