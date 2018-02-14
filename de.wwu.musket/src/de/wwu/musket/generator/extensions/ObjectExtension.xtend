package de.wwu.musket.generator.extensions

import de.wwu.musket.generator.cpu.Config
import de.wwu.musket.musket.Array
import de.wwu.musket.musket.BoolArray
import de.wwu.musket.musket.BoolArrayParameter
import de.wwu.musket.musket.BoolConstant
import de.wwu.musket.musket.BoolMatrix
import de.wwu.musket.musket.BoolParameter
import de.wwu.musket.musket.BoolVal
import de.wwu.musket.musket.BoolVariable
import de.wwu.musket.musket.DoubleArray
import de.wwu.musket.musket.DoubleArrayParameter
import de.wwu.musket.musket.DoubleConstant
import de.wwu.musket.musket.DoubleMatrix
import de.wwu.musket.musket.DoubleParameter
import de.wwu.musket.musket.DoubleVal
import de.wwu.musket.musket.DoubleVariable
import de.wwu.musket.musket.IntArray
import de.wwu.musket.musket.IntArrayParameter
import de.wwu.musket.musket.IntConstant
import de.wwu.musket.musket.IntMatrix
import de.wwu.musket.musket.IntParameter
import de.wwu.musket.musket.IntVal
import de.wwu.musket.musket.IntVariable
import de.wwu.musket.musket.Matrix

import static extension de.wwu.musket.generator.cpu.ExpressionGenerator.generateExpression
import de.wwu.musket.musket.StructArray
import de.wwu.musket.musket.StructMatrix
import org.eclipse.emf.common.util.EList
import org.eclipse.emf.common.util.BasicEList
import java.util.List

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
	
	def static dispatch CppPrimitiveTypeAsString(StructArray o) {
		o.name.toFirstUpper
	}
	
	def static dispatch CppPrimitiveTypeAsString(IntMatrix o) {
		'int'
	}

	def static dispatch CppPrimitiveTypeAsString(DoubleMatrix o) {
		'double'
	}

	def static dispatch CppPrimitiveTypeAsString(BoolMatrix o) {
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
	
	def static dispatch CppPrimitiveTypeAsString(IntVal o) {
		'int'
	}

	def static dispatch CppPrimitiveTypeAsString(DoubleVal o) {
		'double'
	}

	def static dispatch CppPrimitiveTypeAsString(BoolVal o) {
		'bool'
	}

	// Value
	// Array
	def static dispatch List<String> ValuesAsString(IntArray a){
		a.values.map[v|v.toString]
	}
	
	def static dispatch List<String> ValuesAsString(DoubleArray a){
		a.values.map[v|v.toString]
	}
	
	def static dispatch List<String> ValuesAsString(BoolArray a){
		a.values.map[v|v.toString]
	}
	
	def static dispatch List<String> ValuesAsString(StructArray a){
		newArrayList('')
	}
	
	// Matrix
	def static dispatch List<String> ValuesAsString(IntMatrix a){
		a.values.map[v|v.toString]
	}
	
	def static dispatch List<String> ValuesAsString(DoubleMatrix a){
		a.values.map[v|v.toString]
	}
	
	def static dispatch List<String> ValuesAsString(BoolMatrix a){
		a.values.map[v|v.toString]
	}
	
	def static dispatch List<String> ValuesAsString(StructMatrix a){
		newArrayList('')
	}
	
	// Variable
	def static dispatch ValueAsString(IntVariable o) {
		o.initExpression.generateExpression(null)
	}

	def static dispatch ValueAsString(DoubleVariable o) {
		o.initExpression.generateExpression(null)
	}

	def static dispatch ValueAsString(BoolVariable o) {
		o.initExpression.generateExpression(null)
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

	//Primitive Values
	def static dispatch ValueAsString(BoolVal o) {
		o.value.toString
	}

	def static dispatch ValueAsString(IntVal o) {
		o.value.toString
	}

	def static dispatch ValueAsString(DoubleVal o) {
		o.value.toString
	}
	
	
	// for arrays
	// for arrays
	def static dispatch size(Array a) {
		a.size
	}
	
	def static dispatch sizeLocal(Array a) {
		switch a.distributionMode {
			case DIST: a.size / Config.processes
			case COPY: a.size
			default: a.size
		}
	}
	
	// for matrices
	def static dispatch size(Matrix m) {
		m.cols * m.rows
	}
	
	def static dispatch sizeLocal(Matrix m) {
		switch m.distributionMode {
			case DIST: m.cols * m.rows / Config.processes
			case COPY: m.cols * m.rows
			case ROW_DIST: throw new UnsupportedOperationException("ObjectExetension.sizeLocal: case ROW_DIST")
			case COLUMN_DIST: throw new UnsupportedOperationException("ObjectExetension.sizeLocal: case COLUMN_DIST")
			default: 0
		}
	}
	
	def static rowsLocal(Matrix m) {
		switch m.distributionMode {
			case DIST: m.rows / m.blocksInRow
			case COPY: m.rows
			case ROW_DIST: throw new UnsupportedOperationException("ObjectExetension.rowsLocal: case ROW_DIST")
			case COLUMN_DIST: throw new UnsupportedOperationException("ObjectExetension.rowsLocal: case COLUMN_DIST")
			default: 0
		}
	}
	
	def static colsLocal(Matrix m) {
		switch m.distributionMode {
			case DIST: m.cols / m.blocksInColumn
			case COPY: m.cols
			case ROW_DIST: throw new UnsupportedOperationException("ObjectExetension.colsLocal: case ROW_DIST")
			case COLUMN_DIST: throw new UnsupportedOperationException("ObjectExetension.colsLocal: case COLUMN_DIST")
			default: 0
		}
	}
	
	def static blocksInRow(Matrix m) {
		switch m.distributionMode {
			case DIST: Math.sqrt(Config.processes).intValue
			case COPY: 1
			case ROW_DIST: throw new UnsupportedOperationException("ObjectExetension.colsLocal: case ROW_DIST")
			case COLUMN_DIST: throw new UnsupportedOperationException("ObjectExetension.colsLocal: case COLUMN_DIST")
			default: 0
		}
	}
	
	def static blocksInColumn(Matrix m) {
		switch m.distributionMode {
			case DIST: Math.sqrt(Config.processes).intValue
			case COPY: 1
			case ROW_DIST: throw new UnsupportedOperationException("ObjectExetension.colsLocal: case ROW_DIST")
			case COLUMN_DIST: throw new UnsupportedOperationException("ObjectExetension.colsLocal: case COLUMN_DIST")
			default: 0
		}
	}
	
	def static partitionPosition(Matrix m, int pid) {
		switch m.distributionMode {
			case DIST: (pid / m.blocksInColumn) -> (pid % m.blocksInColumn)
			case COPY: 0 -> 0
			case ROW_DIST: pid -> 0
			case COLUMN_DIST: 0 -> pid
			default: -1 -> -1
		}
	}
}
