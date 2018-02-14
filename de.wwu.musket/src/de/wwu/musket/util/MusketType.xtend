package de.wwu.musket.util

import de.wwu.musket.musket.Type
import de.wwu.musket.musket.Function
import de.wwu.musket.musket.Struct

class MusketType {
	
	public static MusketType INT = new MusketType(Type.INT)
	public static MusketType DOUBLE = new MusketType(Type.DOUBLE)
	public static MusketType BOOL = new MusketType(Type.BOOL)
	public static MusketType STRING = new MusketType(Type.STRING)
	public static MusketType INT_ARRAY = new MusketType(Type.INT).toArray
	public static MusketType DOUBLE_ARRAY = new MusketType(Type.DOUBLE).toArray
	public static MusketType BOOL_ARRAY = new MusketType(Type.BOOL).toArray
	public static MusketType INT_MATRIX = new MusketType(Type.INT).toMatrix
	public static MusketType DOUBLE_MATRIX = new MusketType(Type.DOUBLE).toMatrix
	public static MusketType BOOL_MATRIX = new MusketType(Type.BOOL).toMatrix
	
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
	
	static def isNumeric(MusketType t){
		return !t.isArray && !t.isMatrix && (t.type === Type.INT || t.type === Type.DOUBLE)
	}
	
	def isCollection() {
		return isArray || isMatrix
	}
	
	def equals(MusketType t){
		return this.type === t.type && this.structName == t.structName && this.isArray === t.isArray && this.isMatrix === t.isMatrix
	}
	
	override def String toString(){
		if(structName !== null) return structName
		return type.toString
	}
	
	static def toFullType(Type t){
		return new MusketType(t)
	}
	
}