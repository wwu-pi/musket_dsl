package de.wwu.musket.generator.gpu

import de.wwu.musket.musket.ArrayType
import de.wwu.musket.musket.Constant
import de.wwu.musket.musket.Variable

import de.wwu.musket.musket.MatrixType
import de.wwu.musket.musket.Struct
import de.wwu.musket.musket.CollectionObject
import static extension de.wwu.musket.util.TypeHelper.*
import static extension de.wwu.musket.util.MusketHelper.*
import static extension de.wwu.musket.generator.gpu.util.DataHelper.*
import de.wwu.musket.musket.StructArrayType
import de.wwu.musket.musket.StructMatrixType

/**
 * Generates declaration, definition and initialization of data object.
 * <p>
 * The Generator handles arrays and matrices, global variables and constants, and structs.
 * Methods for declarations and struct definitions are called by the header generator.
 * Other definitions and initialization methods are called by the source file generator.
 */
class DataGenerator {
// Generate declarations	
	// variables
	def static dispatch generateObjectDeclaration(Variable v) '''extern «v.calculateCollectionType.cppType» «v.name»;'''

	// constants
	def static dispatch generateObjectDeclaration(
		Constant c) '''extern const «c.calculateCollectionType.cppType» «c.name»;'''

	def static dispatch generateObjectDeclaration(CollectionObject c) {
		switch (c.type) {
			StructArrayType: generateObjectDeclarationStruct("Array", c.name, (c.type as StructArrayType).type)
			StructMatrixType: generateObjectDeclarationStruct("Matrix", c.name, (c.type as StructMatrixType).type)
			ArrayType: '''extern mkt::DArray<«c.calculateCollectionType.cppType»> «c.name»;'''			
			MatrixType: '''extern mkt::DMatrix<«c.calculateCollectionType.cppType»> «c.name»;'''
		}
	}
	
	def static generateObjectDeclarationStruct(String collectionType, String collectionName, Struct struct) '''
		«FOR m : struct.attributes»
			extern mkt::D«collectionType»<«m.calculateType.cppType»> «collectionName»_«m.name»;
		«ENDFOR»
	'''

	def static dispatch generateObjectDeclaration(Struct s) '''
		struct «s.name.toFirstUpper»{
			«FOR m : s.attributes»
				«m.calculateType.cppType» «m.name.toFirstLower»;
			«ENDFOR»
		};
	'''

// Generate definitions	
	// variables
	def static dispatch generateObjectDefinition(
		Variable v, int processId) '''«v.calculateCollectionType.cppType» «v.name» = «v.ValueAsString»;'''

	// constants
	def static dispatch generateObjectDefinition(
		Constant c, int processId) '''const «c.calculateType.cppType» «c.name» = «c.ValueAsString»;'''

	// Arrays objects
	def static dispatch generateObjectDefinition(CollectionObject c, int processId) {
		switch (c.type) {
			StructArrayType: generateObjectDefinitionStructArray(c, c.type as StructArrayType, processId)
			StructMatrixType: generateObjectDefinitionStructMatrix(c, c.type as StructMatrixType, processId)
			ArrayType: '''mkt::DArray<«c.calculateCollectionType.cppType»> «c.name»(«processId», «(c.type as ArrayType).size()», «c.type.sizeLocal(processId)», «IF c.values.size == 1»«c.values.head.ValueAsString»«ELSE»«c.calculateCollectionType.getCXXDefaultValue()»«ENDIF», «(c.type as ArrayType).blocks», «processId», «(c.type as ArrayType).globalOffset(processId)», mkt::«(c.type as ArrayType).distributionMode.toString.toUpperCase», mkt::«(c.type as ArrayType).gpuDistributionMode.toString.toUpperCase»);'''
			MatrixType: '''mkt::DMatrix<«c.calculateCollectionType.cppType»> «c.name»(«processId», «(c.type as MatrixType).rows.concreteValue», «(c.type as MatrixType).cols.concreteValue», «(c.type as MatrixType).rowsLocal()», «(c.type as MatrixType).colsLocal()», «(c.type as MatrixType).size()», «c.type.sizeLocal(processId)», «IF c.values.size == 1»«c.values.head.ValueAsString»«ELSE»«c.calculateCollectionType.getCXXDefaultValue()»«ENDIF», «(c.type as MatrixType).blocksInRow», «(c.type as MatrixType).blocksInColumn», «(c.type as MatrixType).partitionPosition(processId).key», «(c.type as MatrixType).partitionPosition(processId).value», «(c.type as MatrixType).globalRowOffset(processId)», «(c.type as MatrixType).globalColOffset(processId)», mkt::«(c.type as MatrixType).distributionMode.toString.toUpperCase», mkt::«(c.type as MatrixType).gpuDistributionMode.toString.toUpperCase»);'''
		}
	}

	def static generateObjectDefinitionStructArray(CollectionObject co, StructArrayType sat, int processId) '''
		«val struct = sat.type»
		«FOR m : struct.attributes»
			«IF m.calculateType.isCollection»
				mkt::DArray<«m.calculateCollectionType.cppType»> «co.name»_«m.name»(«processId», «sat.size() * m.calculateCollectionType.size», «sat.sizeLocal(processId) * m.calculateCollectionType.size», «m.calculateCollectionType.getCXXDefaultValue()», «sat.blocks», «processId», «sat.globalOffset(processId)», mkt::«sat.distributionMode.toString.toUpperCase», mkt::«sat.gpuDistributionMode.toString.toUpperCase»);
			«ELSE»
				mkt::DArray<«m.calculateType.cppType»> «co.name»_«m.name»(«processId», «sat.size()», «sat.sizeLocal(processId)», «m.calculateType.getCXXDefaultValue()», «sat.blocks», «processId», «sat.globalOffset(processId)», mkt::«sat.distributionMode.toString.toUpperCase», mkt::«sat.gpuDistributionMode.toString.toUpperCase»);
			«ENDIF»
		«ENDFOR»
	'''
	
	def static generateObjectDefinitionStructMatrix(CollectionObject co, StructMatrixType smt, int processId) '''
		«val struct = smt.type»
		«FOR m : struct.attributes»
			«IF m.calculateType.isCollection»
«««	TODO
				// TODO
			«ELSE»
				mkt::DMatrix<«m.calculateType.cppType»> «co.name»_«m.name»(«processId», «smt.rows.concreteValue», «smt.cols.concreteValue», «smt.rowsLocal()», «smt.colsLocal()», «smt.size()», «smt.sizeLocal(processId)», «m.calculateType.getCXXDefaultValue()», «smt.blocksInRow», «smt.blocksInColumn», «smt.partitionPosition(processId).key», «smt.partitionPosition(processId).value», «smt.globalRowOffset(processId)», «smt.globalColOffset(processId)», mkt::«smt.distributionMode.toString.toUpperCase», mkt::«smt.gpuDistributionMode.toString.toUpperCase»);
			«ENDIF»
		«ENDFOR»
	'''
	
	def static generateObjectDefinitionStructMatrix(String collectionType, String collectionName, Struct struct) '''
		«FOR m : struct.attributes»
			extern mkt::D«collectionType»<«m.calculateType.cppType»> «collectionName»_«m.name»;
		«ENDFOR»
	'''

	def static dispatch generateObjectDefinition(Struct s, int processId) '''''' // this is done in StructGenerator.xtend

// Generate initialization
	def static generateArrayInitializationForProcess(CollectionObject a, Iterable<String> values) '''
		«var value_id = 0»
		«FOR v : values»
			«a.name».set_local(«value_id++», «v»);
		«ENDFOR»
	'''
}
