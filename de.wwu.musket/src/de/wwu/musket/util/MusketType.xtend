package de.wwu.musket.util

import de.wwu.musket.musket.BoolArrayType
import de.wwu.musket.musket.BoolMatrixType
import de.wwu.musket.musket.DistributionMode
import de.wwu.musket.musket.DoubleArrayType
import de.wwu.musket.musket.DoubleMatrixType
import de.wwu.musket.musket.Function
import de.wwu.musket.musket.IntArrayType
import de.wwu.musket.musket.IntMatrixType
import de.wwu.musket.musket.PrimitiveType
import de.wwu.musket.musket.PrimitiveTypeLiteral
import de.wwu.musket.musket.Struct
import de.wwu.musket.musket.StructArrayType
import de.wwu.musket.musket.StructMatrixType
import de.wwu.musket.musket.StructType
import de.wwu.musket.musket.Type
import java.util.Objects
import de.wwu.musket.musket.FloatArrayType
import de.wwu.musket.musket.FloatMatrixType

class MusketType {
	
	public static final MusketType AUTO = new MusketType(PrimitiveTypeLiteral.AUTO)
	public static final MusketType INT = new MusketType(PrimitiveTypeLiteral.INT)
	public static final MusketType DOUBLE = new MusketType(PrimitiveTypeLiteral.DOUBLE)
	public static final MusketType FLOAT = new MusketType(PrimitiveTypeLiteral.FLOAT)
	public static final MusketType BOOL = new MusketType(PrimitiveTypeLiteral.BOOL)
	public static final MusketType STRING = new MusketType(PrimitiveTypeLiteral.STRING)
	public static final MusketType INT_ARRAY = new MusketType(PrimitiveTypeLiteral.INT).toArray
	public static final MusketType DOUBLE_ARRAY = new MusketType(PrimitiveTypeLiteral.DOUBLE).toArray
	public static final MusketType FLOAT_ARRAY = new MusketType(PrimitiveTypeLiteral.FLOAT).toArray
	public static final MusketType BOOL_ARRAY = new MusketType(PrimitiveTypeLiteral.BOOL).toArray
	public static final MusketType INT_MATRIX = new MusketType(PrimitiveTypeLiteral.INT).toMatrix
	public static final MusketType DOUBLE_MATRIX = new MusketType(PrimitiveTypeLiteral.DOUBLE).toMatrix
	public static final MusketType FLOAT_MATRIX = new MusketType(PrimitiveTypeLiteral.FLOAT).toMatrix
	public static final MusketType BOOL_MATRIX = new MusketType(PrimitiveTypeLiteral.BOOL).toMatrix
	
	protected PrimitiveTypeLiteral type = null
	protected DistributionMode distributionMode = DistributionMode.COPY
	protected String structName = null
	protected boolean isArray = false
	protected boolean isMatrix = false
	
	new(PrimitiveTypeLiteral t){
		type = t
	}
	
	new(Type t){
		switch(t){
			IntArrayType: { type = PrimitiveTypeLiteral.INT; toArray; distributionMode = t.distributionMode }
			DoubleArrayType: { type = PrimitiveTypeLiteral.DOUBLE; toArray; distributionMode = t.distributionMode }
			FloatArrayType: { type = PrimitiveTypeLiteral.FLOAT; toArray; distributionMode = t.distributionMode }
			BoolArrayType: { type = PrimitiveTypeLiteral.BOOL; toArray; distributionMode = t.distributionMode }
			StructArrayType: { structName = t.type.name; toArray; distributionMode = t.distributionMode }
			IntMatrixType: { type = PrimitiveTypeLiteral.INT; toMatrix; distributionMode = t.distributionMode }
			DoubleMatrixType: { type = PrimitiveTypeLiteral.DOUBLE; toMatrix; distributionMode = t.distributionMode }
			FloatMatrixType: { type = PrimitiveTypeLiteral.FLOAT; toMatrix; distributionMode = t.distributionMode }
			BoolMatrixType: { type = PrimitiveTypeLiteral.BOOL; toMatrix; distributionMode = t.distributionMode }
			StructMatrixType: { structName = t.type.name; toMatrix; distributionMode = t.distributionMode }
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
	
	def toLocalCollection(){
		distributionMode = DistributionMode.LOC
		return this
	}
	
	def isNumeric(){
		return !isArray && !isMatrix && (type === PrimitiveTypeLiteral.AUTO || type === PrimitiveTypeLiteral.INT || type === PrimitiveTypeLiteral.DOUBLE || type === PrimitiveTypeLiteral.FLOAT)
	}
	
	def isCollection() {
		return isArray || isMatrix || type === PrimitiveTypeLiteral.AUTO
	}
	
	override hashCode() {
		Objects.hash(this.type, this.structName, this.isArray, this.isMatrix, this.distributionMode)
	}
	
	private def equals(Object obj, boolean ignoreDistribution){
		if(!(obj instanceof MusketType)) return false
		
		// Non-inferrable auto types are accepted
		if(this.type === PrimitiveTypeLiteral.AUTO || (obj as MusketType).type === PrimitiveTypeLiteral.AUTO) return true;
		
		val isDistributionOK = ignoreDistribution || !(this.isArray || this.isMatrix) || this.distributionMode == (obj as MusketType).distributionMode
	
		return this.type === (obj as MusketType).type && this.structName == (obj as MusketType).structName
			&& this.isArray === (obj as MusketType).isArray && this.isMatrix === (obj as MusketType).isMatrix
			&& isDistributionOK
	}
	
	override def equals(Object obj){
		return this.equals(obj, false)
	}
	
	def equalsIgnoreDistribution(Object obj){
		return this.equals(obj, true)
	}
	
	override def String toString(){
		val name = if(structName !== null) structName else type.toString
		
		if(isArray) {
			return 'array<' + name + ',' + distributionMode + '>'
		} else if(isMatrix) {
			return 'matrix<' + name + ',' + distributionMode + '>'
		}
		return name;
	}
	
	static def toFullType(Type t){
		return new MusketType(t)
	}
	
	def getCppType(){
		// struct
		if(structName !== null){
			return structName
		}
		
		// primitive type
		switch (type) {
			case BOOL: return 'bool'
			case DOUBLE: return 'double'
			case FLOAT: return 'float'
			case INT: return 'int'
			default: return 'auto'
		}
	}
	
}