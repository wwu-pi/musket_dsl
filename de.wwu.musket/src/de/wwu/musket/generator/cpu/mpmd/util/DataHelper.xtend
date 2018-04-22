package de.wwu.musket.generator.cpu.mpmd.util

import de.wwu.musket.generator.cpu.mpmd.Config
import de.wwu.musket.musket.ArrayType
import de.wwu.musket.musket.CollectionObject
import de.wwu.musket.musket.CollectionType
import de.wwu.musket.musket.Expression
import de.wwu.musket.musket.MatrixType
import java.util.Map
import org.eclipse.emf.common.util.EList

import static extension de.wwu.musket.generator.cpu.ExpressionGenerator.generateExpression
import static extension de.wwu.musket.util.MusketHelper.*
import java.util.List
import de.wwu.musket.musket.StructArrayType
import de.wwu.musket.musket.StructMatrixType
import de.wwu.musket.musket.IntVariable
import de.wwu.musket.musket.DoubleVariable
import de.wwu.musket.musket.FloatVariable
import de.wwu.musket.musket.BoolVariable
import de.wwu.musket.musket.IntConstant
import de.wwu.musket.musket.DoubleConstant
import de.wwu.musket.musket.FloatConstant
import de.wwu.musket.musket.BoolConstant
import de.wwu.musket.musket.BoolVal
import de.wwu.musket.musket.IntVal
import de.wwu.musket.musket.DoubleVal
import de.wwu.musket.musket.FloatVal
import de.wwu.musket.musket.CompareExpression
import de.wwu.musket.musket.ReferableObject
import de.wwu.musket.musket.CollectionParameter
import de.wwu.musket.musket.CollectionObjectOrParam
import de.wwu.musket.musket.TailObjectRef

class DataHelper {
// Value
	def static List<String> ValuesAsString(CollectionObject o) {
		if(o.type instanceof StructArrayType || o.type instanceof StructMatrixType) return newArrayList('')
		return o.values.map[v|v.ValueAsString]
	}

	// Variable
	def static dispatch ValueAsString(IntVariable o) {
		o.initExpression.generateExpression(null)
	}

	def static dispatch ValueAsString(DoubleVariable o) {
		o.initExpression.generateExpression(null)
	}

	def static dispatch ValueAsString(FloatVariable o) {
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

	def static dispatch ValueAsString(FloatConstant o) {
		o.value.toString + "f"
	}

	def static dispatch ValueAsString(BoolConstant o) {
		o.value.toString
	}

	// Primitive Values
	def static dispatch ValueAsString(BoolVal o) {
		o.value.toString
	}

	def static dispatch ValueAsString(IntVal o) {
		o.value.toString
	}

	def static dispatch ValueAsString(DoubleVal o) {
		o.value.toString
	}

	def static dispatch ValueAsString(FloatVal o) {
		o.value.toString + "f"
	}

	def static dispatch ValueAsString(CompareExpression co) {
		co.generateExpression(null)
	}

	// object references
	def static dispatch getCollectionType(ReferableObject ro) {
		switch ro {
			CollectionObject: ro.type
			CollectionParameter: ro.type
		}
	}

	def static dispatch getCollectionType(CollectionObjectOrParam coop) {
		switch coop {
			CollectionObject: coop.type
			CollectionParameter: coop.type
		}
	}

	def static getCollectionName(CollectionObjectOrParam coop) {
		switch coop {
			CollectionObject: coop.name
			CollectionParameter: coop.name
		}
	}

	// structs
	def static generateTail(TailObjectRef or) {
		var result = ''
		var tor = or
		while (tor !== null) {
			result += '.' + tor.value.name
			if(!tor.localCollectionIndex.nullOrEmpty){
				result += '['
				result += convertLocalCollectionIndex(tor.value.collectionType, tor.localCollectionIndex, null)
				result += ']'
			}else if(!tor.globalCollectionIndex.nullOrEmpty){
				result += '['
				result += convertGlobalCollectionIndex(tor.value.collectionType, tor.globalCollectionIndex, null)
				result += ']'
			}
			tor = tor.tail
		}
		return result
	}

	// calculate indices
	static def convertLocalCollectionIndex(CollectionType ct, EList<Expression> indices,
		Map<String, String> param_map) {
		switch ct {
			ArrayType:
				indices.head.generateExpression(param_map)
			MatrixType:
				indices.get(0).generateExpression(param_map) + "*" + ct.colsLocal + "+" +
					indices.get(1).generateExpression(param_map)
		}
	}

// TODO: todo, just copy paste, should not work
	static def convertGlobalCollectionIndex(CollectionType ct, EList<Expression> indices,
		Map<String, String> param_map) {
		switch ct {
			ArrayType:
				indices.head.generateExpression(param_map)
			MatrixType:
				indices.get(0).generateExpression(param_map) + "*" + ct.colsLocal + "+" +
					indices.get(1).generateExpression(param_map)
		}
	}

	// for arrays
	def static distributionMode(CollectionObject o) {
		switch o {
			ArrayType: o.distributionMode
			MatrixType: o.distributionMode
		}
	}

	// for arrays
	def static dispatch size(ArrayType a) {
		a.size.concreteValue
	}

	def static dispatch sizeLocal(ArrayType a, int processId) {
		switch a.distributionMode {
			case DIST: a.size.concreteValue / Config.processes//if(a.size.concreteValue % Config.processes > processId){ (a.size.concreteValue / Config.processes) + 1} else { a.size.concreteValue / Config.processes }
			case COPY: a.size.concreteValue
			default: a.size.concreteValue
		}
	}
	
	def static globalOffset(ArrayType a, int processId) {
		switch a.distributionMode {
			case DIST: processId * a.sizeLocal(processId)// {var o = 0l; for(i : 0 ..< processId){ o += a.sizeLocal(i)}} 
			case COPY: 0
			default: 0
		}
	}

	// for matrices
	def static dispatch size(MatrixType m) {
		m.cols.concreteValue * m.rows.concreteValue
	}

	def static dispatch sizeLocal(MatrixType m, int processId) {
		switch m.distributionMode {
			case DIST: m.cols.concreteValue * m.rows.concreteValue / Config.processes
			case COPY: m.cols.concreteValue * m.rows.concreteValue
			case ROW_DIST: throw new UnsupportedOperationException("ObjectExtension.sizeLocal: case ROW_DIST")
			case COLUMN_DIST: throw new UnsupportedOperationException("ObjectExtension.sizeLocal: case COLUMN_DIST")
			default: 0
		}
	}

	def static rowsLocal(MatrixType m) {
		switch m.distributionMode {
			case DIST: m.rows.concreteValue / m.blocksInRow
			case COPY: m.rows.concreteValue
			case ROW_DIST: throw new UnsupportedOperationException("ObjectExtension.rowsLocal: case ROW_DIST")
			case COLUMN_DIST: throw new UnsupportedOperationException("ObjectExtension.rowsLocal: case COLUMN_DIST")
			default: 0
		}
	}

	def static colsLocal(MatrixType m) {
		switch m.distributionMode {
			case DIST: m.cols.concreteValue / m.blocksInColumn
			case COPY: m.cols.concreteValue
			case ROW_DIST: throw new UnsupportedOperationException("ObjectExtension.colsLocal: case ROW_DIST")
			case COLUMN_DIST: throw new UnsupportedOperationException("ObjectExtension.colsLocal: case COLUMN_DIST")
			default: 0
		}
	}

	def static blocksInRow(MatrixType m) {
		switch m.distributionMode {
			case DIST: Math.sqrt(Config.processes).intValue
			case COPY: 1
			case ROW_DIST: throw new UnsupportedOperationException("ObjectExtension.colsLocal: case ROW_DIST")
			case COLUMN_DIST: throw new UnsupportedOperationException("ObjectExtension.colsLocal: case COLUMN_DIST")
			default: 0
		}
	}

	def static blocksInColumn(MatrixType m) {
		switch m.distributionMode {
			case DIST: Math.sqrt(Config.processes).intValue
			case COPY: 1
			case ROW_DIST: throw new UnsupportedOperationException("ObjectExtension.colsLocal: case ROW_DIST")
			case COLUMN_DIST: throw new UnsupportedOperationException("ObjectExtension.colsLocal: case COLUMN_DIST")
			default: 0
		}
	}

	/**
	 * Returns a tuple that holds the position of the partition for a given matrix and the process id.
	 * Key: row position
	 * Value: col position
	 */
	def static partitionPosition(MatrixType m, int pid) {
		switch m.distributionMode {
			case DIST: (pid / m.blocksInColumn) -> (pid % m.blocksInColumn)
			case COPY: 0 -> 0
			case ROW_DIST: pid -> 0
			case COLUMN_DIST: 0 -> pid
			default: -1 -> -1
		}
	}

}