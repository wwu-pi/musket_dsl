package de.wwu.musket.util

import de.wwu.musket.musket.PrimitiveTypeLiteral
import de.wwu.musket.musket.Function
import de.wwu.musket.musket.Struct
import java.util.Objects
import de.wwu.musket.musket.Type
import de.wwu.musket.musket.PrimitiveType
import de.wwu.musket.musket.DoubleArrayType
import de.wwu.musket.musket.IntArrayType
import de.wwu.musket.musket.BoolArrayType
import de.wwu.musket.musket.StructArrayType
import de.wwu.musket.musket.IntMatrixType
import de.wwu.musket.musket.BoolMatrixType
import de.wwu.musket.musket.StructMatrixType
import de.wwu.musket.musket.StructType
import de.wwu.musket.musket.DoubleMatrixType

class MusketType {
	
	public static final MusketType AUTO = new MusketType(PrimitiveTypeLiteral.AUTO)
	public static final MusketType INT = new MusketType(PrimitiveTypeLiteral.INT)
	public static final MusketType DOUBLE = new MusketType(PrimitiveTypeLiteral.DOUBLE)
	public static final MusketType BOOL = new MusketType(PrimitiveTypeLiteral.BOOL)
	public static final MusketType STRING = new MusketType(PrimitiveTypeLiteral.STRING)
	public static final MusketType INT_ARRAY = new MusketType(PrimitiveTypeLiteral.INT).toArray
	public static final MusketType DOUBLE_ARRAY = new MusketType(PrimitiveTypeLiteral.DOUBLE).toArray
	public static final MusketType BOOL_ARRAY = new MusketType(PrimitiveTypeLiteral.BOOL).toArray
	public static final MusketType INT_MATRIX = new MusketType(PrimitiveTypeLiteral.INT).toMatrix
	public static final MusketType DOUBLE_MATRIX = new MusketType(PrimitiveTypeLiteral.DOUBLE).toMatrix
	public static final MusketType BOOL_MATRIX = new MusketType(PrimitiveTypeLiteral.BOOL).toMatrix
	
	protected PrimitiveTypeLiteral type = null
	protected String structName = null
	protected boolean isArray = false
	protected boolean isMatrix = false
	
	new(PrimitiveTypeLiteral t){
		type = t
	}
	
	new(Type t){
		switch(t){
			IntArrayType: { type = PrimitiveTypeLiteral.INT; isArray = true }
			DoubleArrayType: { type = PrimitiveTypeLiteral.DOUBLE; isArray = true }
			BoolArrayType: { type = PrimitiveTypeLiteral.BOOL; isArray = true }
			StructArrayType: { structName = t.type.name; isArray = true }
			IntMatrixType: { type = PrimitiveTypeLiteral.INT; isMatrix = true }
			DoubleMatrixType: { type = PrimitiveTypeLiteral.DOUBLE; isMatrix = true }
			BoolMatrixType: { type = PrimitiveTypeLiteral.BOOL; isMatrix = true }
			StructMatrixType: { structName = t.type.name; isMatrix = true }
			PrimitiveType: type = t.type
			StructType: structName = t.type.name
		}
	}
	
	new(Struct s){
		structName = s.name
	}
	
	new(Function f){
		new MusketType(f.returnType)
	}
	
	def isArray() {
		return isArray
	}
	
	def isMatrix() {
		return isMatrix
	}
	
	def toArray(){
		isArray = true
		isMatrix = false
		return this
	}
	
	def toMatrix(){
		isMatrix = true
		isArray = false
		return this
	}
	
	def toSingleValued(){
		isMatrix = false
		isArray = false
		return this
	}
	
	def isNumeric(){
		return !isArray && !isMatrix && (type === PrimitiveTypeLiteral.AUTO || type === PrimitiveTypeLiteral.INT || type === PrimitiveTypeLiteral.DOUBLE)
	}
	
	def isCollection() {
		return isArray || isMatrix || type === PrimitiveTypeLiteral.AUTO
	}
	
	override hashCode() {
		Objects.hash(this.type, this.structName, this.isArray, this.isMatrix)
	}
	
	override def equals(Object obj){
		if(!(obj instanceof MusketType)) return false
		
		// Non-inferrable auto types are accepted
		if(this.type === PrimitiveTypeLiteral.AUTO || (obj as MusketType).type === PrimitiveTypeLiteral.AUTO) return true;
		
		return this.type === (obj as MusketType).type && this.structName == (obj as MusketType).structName
			&& this.isArray === (obj as MusketType).isArray && this.isMatrix === (obj as MusketType).isMatrix
	}
	
	override def String toString(){
		val name = if(structName !== null) structName else type.toString
		
		if(isArray) {
			return 'array<' + name + '>'
		} else if(isMatrix) {
			return 'matrix<' + name + '>'
		}
		return name;
	}
	
	static def toFullType(Type t){
		return new MusketType(t)
	}
	
	def getCppType(){
		switch (type) {
			case BOOL: return 'bool'
			case DOUBLE: return 'double'
			case INT: return 'int'
			default: return 'auto'
		}
	}
	
}