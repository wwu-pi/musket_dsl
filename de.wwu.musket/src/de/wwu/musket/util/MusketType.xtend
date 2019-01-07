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
import org.eclipse.xtend.lib.annotations.Accessors
import de.wwu.musket.musket.CollectionType
import de.wwu.musket.musket.ArrayType
import de.wwu.musket.musket.MatrixType
import static extension de.wwu.musket.util.MusketHelper.*

class MusketType {

	public static final MusketType VOID = new MusketType()
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
	protected PrimitiveType primitiveType = null
	protected CollectionType collectionType = null
	@Accessors
	protected DistributionMode distributionMode = DistributionMode.COPY
	@Accessors
	protected long size = 0
	protected String structName = null
	protected boolean isArray = false
	protected boolean isMatrix = false

	new(){
		// Void type constructor
	}
	
	new(PrimitiveTypeLiteral t) {
		type = t
	}

	new(Type t) {
		switch (t) {
			IntArrayType: {
				type = PrimitiveTypeLiteral.INT;
				toArray;
				distributionMode = t.distributionMode
				collectionType = t
				size = (t as ArrayType).size.concreteValue
			}
			DoubleArrayType: {
				type = PrimitiveTypeLiteral.DOUBLE;
				toArray;
				distributionMode = t.distributionMode
				collectionType = t
				size = (t as ArrayType).size.concreteValue
			}
			FloatArrayType: {
				type = PrimitiveTypeLiteral.FLOAT;
				toArray;
				distributionMode = t.distributionMode
				collectionType = t
				size = (t as ArrayType).size.concreteValue
			}
			BoolArrayType: {
				type = PrimitiveTypeLiteral.BOOL;
				toArray;
				distributionMode = t.distributionMode
				collectionType = t
				size = (t as ArrayType).size.concreteValue
			}
			StructArrayType: {
				structName = t.type.name;
				toArray;
				distributionMode = t.distributionMode
				collectionType = t
				size = (t as ArrayType).size.concreteValue
			}
			IntMatrixType: {
				type = PrimitiveTypeLiteral.INT;
				toMatrix;
				distributionMode = t.distributionMode
				collectionType = t
				size = (t as MatrixType).rows.concreteValue * (t as MatrixType).cols.concreteValue
			}
			DoubleMatrixType: {
				type = PrimitiveTypeLiteral.DOUBLE;
				toMatrix;
				distributionMode = t.distributionMode
				collectionType = t
				size = (t as MatrixType).rows.concreteValue * (t as MatrixType).cols.concreteValue
			}
			FloatMatrixType: {
				type = PrimitiveTypeLiteral.FLOAT;
				toMatrix;
				distributionMode = t.distributionMode
				collectionType = t
				size = (t as MatrixType).rows.concreteValue * (t as MatrixType).cols.concreteValue
			}
			BoolMatrixType: {
				type = PrimitiveTypeLiteral.BOOL;
				toMatrix;
				distributionMode = t.distributionMode
				collectionType = t
				size = (t as MatrixType).rows.concreteValue * (t as MatrixType).cols.concreteValue
			}
			StructMatrixType: {
				structName = t.type.name;
				toMatrix;
				distributionMode = t.distributionMode
				collectionType = t
				size = (t as MatrixType).rows.concreteValue * (t as MatrixType).cols.concreteValue
			}
			PrimitiveType: {
				type = t.type
				primitiveType = t
			}
			StructType:
				structName = t.type.name
		}
	}

	new(Struct s) {
		structName = s.name
	}

	new(Function f) {
		this(f.returnType)
	}

	def isArray() {
		return isArray
	}

	def isMatrix() {
		return isMatrix
	}

	def toArray() {
		isArray = true
		isMatrix = false
		return this
	}

	def toMatrix() {
		isMatrix = true
		isArray = false
		return this
	}

	def toSingleValued() {
		isMatrix = false
		isArray = false
		return this
	}

	def toLocalCollection() {
		distributionMode = DistributionMode.LOC
		return this
	}
	
	def toCopyCollection() {
		distributionMode = DistributionMode.COPY
		return this
	}
	
	def toDistributedCollection() {
		distributionMode = DistributionMode.DIST
		return this
	}

	def isNumeric() {
		return !isArray && !isMatrix &&
			(type === PrimitiveTypeLiteral.AUTO || type === PrimitiveTypeLiteral.INT ||
				type === PrimitiveTypeLiteral.DOUBLE || type === PrimitiveTypeLiteral.FLOAT)
	}

	def isCollection() {
		return isArray || isMatrix || type === PrimitiveTypeLiteral.AUTO
	}
	
	def isStruct(){
		return structName !== null
	}

	override hashCode() {
		Objects.hash(this.type, this.structName, this.isArray, this.isMatrix, this.distributionMode)
	}

	private def equals(Object obj, boolean ignoreDistribution) {
		if(!(obj instanceof MusketType)) return false

		// Non-inferrable auto types are accepted
		if(this.type === PrimitiveTypeLiteral.AUTO ||
			(obj as MusketType).type === PrimitiveTypeLiteral.AUTO) return true;

		val isDistributionOK = ignoreDistribution || !(this.isArray || this.isMatrix) ||
			this.distributionMode == (obj as MusketType).distributionMode

		return this.type === (obj as MusketType).type && this.structName == (obj as MusketType).structName &&
			this.isArray === (obj as MusketType).isArray && this.isMatrix === (obj as MusketType).isMatrix &&
			isDistributionOK
	}

	override def equals(Object obj) {
		return this.equals(obj, false)
	}

	def equalsIgnoreDistribution(Object obj) {
		return this.equals(obj, true)
	}

	override def String toString() {
		if (type === null) return 'void'
		
		val name = if(structName !== null) structName else type.toString

		if (isArray) {
			return 'array<' + name + ',' + distributionMode + '>'
		} else if (isMatrix) {
			return 'matrix<' + name + ',' + distributionMode + '>'
		}
		return name;
	}

	static def toFullType(Type t) {
		return new MusketType(t)
	}

	def getType() {
		return this.type
	}

	def getCollectionType() {
		return this.collectionType
	}

	def getPrimitiveType() {
		return this.primitiveType
	}

	/** 
	 * Map Musket type to C++ type
	 */
	def getCppType() {
		var primtype = ''
		// struct
		if (structName !== null) {
			primtype = structName
		} else {

			// primitive type
			switch (type) {
				case BOOL: primtype = 'bool'
				case DOUBLE: primtype = 'double'
				case FLOAT: primtype = 'float'
				case INT: primtype = 'int'
				default: primtype = 'auto'
			}
		}

		if (isArray || isMatrix) {
			if(distributionMode == DistributionMode.LOC)
				return 'std::array<' + primtype + ',' + size + '>'
			else{
				if(isArray){
					return 'mkt::DArray<' + primtype + '>'
				}else if(isMatrix){
					return 'mkt::DMatrix<' + primtype + '>'
				}					
				return ''
			}			
		} else {
			return primtype
		}
	}
	
	def getCXXDefaultValue() {
		if(primitiveType !== null){
			switch (this.primitiveType) {
				PrimitiveType case type == PrimitiveTypeLiteral.INT: '''0'''
				PrimitiveType case type == PrimitiveTypeLiteral.DOUBLE: '''0.0'''
				PrimitiveType case type == PrimitiveTypeLiteral.FLOAT: '''0.0f'''
				PrimitiveType case type == PrimitiveTypeLiteral.BOOL: '''false'''
				PrimitiveType case type == PrimitiveTypeLiteral.STRING: '''""'''
				default:
					'''/*getCXXDefaultValue: primitiveType*/'''
			}
		}else if (collectionType !== null){
			switch (this.collectionType) {
				IntArrayType,
				IntMatrixType: '''0'''
				DoubleArrayType,
				DoubleMatrixType: '''0.0'''
				FloatArrayType,
				FloatMatrixType: '''0.0f'''
				BoolArrayType,
				BoolMatrixType: '''false'''
				StructArrayType,
				StructMatrixType: '''«this.collectionType.toFullType.structName»{}'''
				default:
					'''/*getCXXDefaultValue: Collection*/'''
			}
		} else if(structName !== null) {
			'''«structName»{}'''			
		}else{
			'''/*getCXXDefaultValue: else*/'''
		}
	}

	/** 
	 * Map Musket type to MPI type
	 */
	def getMPIType() {
		var mpi_type = ''
		// struct
		if (structName !== null) {
			mpi_type = structName + "_mpi_type"
		} else {
			// primitive type
			switch (type) {
				case BOOL: mpi_type = 'MPI_BOOL'
				case DOUBLE: mpi_type = 'MPI_DOUBLE'
				case FLOAT: mpi_type = 'MPI_FLOAT'
				case INT: mpi_type = 'MPI_INT'
				case STRING: mpi_type = 'MPI_CHAR'
				default: mpi_type = 'MPI_BYTE'
			}
		}
	}

}
