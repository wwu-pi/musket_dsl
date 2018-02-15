package de.wwu.musket.util

import de.wwu.musket.musket.Type
import de.wwu.musket.musket.Function
import de.wwu.musket.musket.Struct
import java.util.Objects

class MusketType {
	
	public static final MusketType INT = new MusketType(Type.INT)
	public static final MusketType DOUBLE = new MusketType(Type.DOUBLE)
	public static final MusketType BOOL = new MusketType(Type.BOOL)
	public static final MusketType STRING = new MusketType(Type.STRING)
	public static final MusketType INT_ARRAY = new MusketType(Type.INT).toArray
	public static final MusketType DOUBLE_ARRAY = new MusketType(Type.DOUBLE).toArray
	public static final MusketType BOOL_ARRAY = new MusketType(Type.BOOL).toArray
	public static final MusketType INT_MATRIX = new MusketType(Type.INT).toMatrix
	public static final MusketType DOUBLE_MATRIX = new MusketType(Type.DOUBLE).toMatrix
	public static final MusketType BOOL_MATRIX = new MusketType(Type.BOOL).toMatrix
	
	protected Type type = null
	protected String structName = null
	protected boolean isArray = false
	protected boolean isMatrix = false
	
	new(Type t){
		type = t
	}
	
	new(Struct s){
		structName = s.name
	}
	
	new(Function f){
		if(f.returnType !== null){
			structName = f.returnType.name
		} else {
			type = f.returnTypePrimitive
		}
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
		return !isArray && !isMatrix && (type === Type.INT || type === Type.DOUBLE)
	}
	
	def isCollection() {
		return isArray || isMatrix
	}
	
	override hashCode() {
		Objects.hash(this.type, this.structName, this.isArray, this.isMatrix)
	}
	
	override def equals(Object obj){
		if(!(obj instanceof MusketType)) return false
		return this.type === (obj as MusketType).type && this.structName == (obj as MusketType).structName
			&& this.isArray === (obj as MusketType).isArray && this.isMatrix === (obj as MusketType).isMatrix
	}
	
	override def String toString(){
		if(structName !== null) return structName
		return type.toString
	}
	
	static def toFullType(Type t){
		return new MusketType(t)
	}
	
}