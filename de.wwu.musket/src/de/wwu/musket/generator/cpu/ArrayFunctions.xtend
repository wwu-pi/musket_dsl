package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.ArrayType
import de.wwu.musket.musket.CollectionFunctionCall
import de.wwu.musket.musket.CollectionObject
import de.wwu.musket.musket.DistributionMode
import de.wwu.musket.musket.MatrixType
import de.wwu.musket.musket.Struct
import de.wwu.musket.musket.StructArrayType
import de.wwu.musket.musket.StructMatrixType

import static de.wwu.musket.generator.cpu.MPIRoutines.*
import static extension de.wwu.musket.util.MusketHelper.*
import static extension de.wwu.musket.generator.extensions.ObjectExtension.*

class ArrayFunctions {

	def static generateCollectionFunctionCall(CollectionFunctionCall afc) {
		switch afc.function {
			case SIZE:
				generateSize(afc)
			case SIZE_LOCAL:
				generateSizeLocal(afc)
			case SHOW: {
				switch (afc.^var.type){
					ArrayType: generateShowArray(afc.^var)
					MatrixType: generateShowMatrix(afc.^var)
				}
			}
			case COLUMNS:
				(afc.^var.type as MatrixType).cols
			case COLUMNS_LOCAL:
				(afc.^var.type as MatrixType).colsLocal
			case ROWS:
				(afc.^var.type as MatrixType).rows
			case ROWS_LOCAL:
				(afc.^var.type as MatrixType).rowsLocal
			case BLOCKS_IN_ROW:
				(afc.^var.type as MatrixType).blocksInRow
			case BLOCKS_IN_COLUMN:
				(afc.^var.type as MatrixType).blocksInColumn
		}
	}

	def static generateSize(CollectionFunctionCall afc) '''«afc.^var.type.size»'''

	def static generateSizeLocal(CollectionFunctionCall afc) '''«afc.^var.type.sizeLocal»'''

	def static generateShowArray(CollectionObject a) '''
		«IF a.type.distributionMode == DistributionMode.COPY»
			«State.setArrayName(a.name)»
		«ELSE»
			«State.setArrayName("temp" + State.counter)»			
			std::array<«a.CppPrimitiveTypeAsString», «a.type.size»> «State.arrayName»{};
			
			«generateMPIGather(a.name + '.data()', a.type.sizeLocal, a.CppPrimitiveTypeAsString, State.arrayName + '.data()')»
		«ENDIF»
				
		if («Config.var_pid» == 0) {
			«val streamName = 's' + State.counter»
			«State.incCounter»
			std::ostringstream «streamName»;
			«streamName» << "«a.name»: " << std::endl << "[";
			for (int i = 0; i < «(a.type as ArrayType).size.concreteValue - 1»; i++) {
				«streamName» << «generateShowElements(a, State.arrayName, "i")»;				
				«streamName» << "; ";
			}
			«streamName» << «generateShowElements(a, State.arrayName, ((a.type as ArrayType).size.concreteValue - 1).toString)» << "]" << std::endl;
			«streamName» << std::endl;
			printf("%s", «streamName».str().c_str());
		}
	'''

	def static generateShowMatrix(CollectionObject m) '''
		«val type = m.type as MatrixType»
		«IF m.type.distributionMode == DistributionMode.COPY»
			«State.setArrayName(m.name)»
		«ELSE»
			«State.setArrayName("temp" + State.counter)»			
			std::array<«m.CppPrimitiveTypeAsString», «m.type.size»> «State.arrayName»{};
			
			«generateMPIGather(m.name + '.data()', m.type.sizeLocal, m.CppPrimitiveTypeAsString, State.arrayName + '.data()')»
		«ENDIF»

		«val streamName = 's' + State.counter»
		«State.incCounter»
		if («Config.var_pid» == 0) {
			std::ostringstream «streamName»;
			«streamName» << "«m.name»: " << std::endl;
			«FOR i : 0..<type.rows.concreteValue BEFORE streamName + '<< "[";' SEPARATOR streamName + '<< std::endl;' AFTER streamName + '<< "]" << std::endl;'»
				«FOR j : 0..<type.cols.concreteValue BEFORE streamName + '<< "[";' SEPARATOR streamName + '<< "; ";'  AFTER streamName + '<< "]";'»
					«IF type.blocksInRow == 1»
						«streamName» << «State.arrayName»[«(i % type.rowsLocal) * type.colsLocal + (i / type.rowsLocal) * type.sizeLocal + (j / type.colsLocal) * type.sizeLocal + j % type.colsLocal»];									
					«ELSE»
						«streamName» << «State.arrayName»[«(i % type.rowsLocal) * type.colsLocal + (i / type.rowsLocal) * type.sizeLocal * type.blocksInColumn + (j / type.colsLocal) * type.sizeLocal + j % type.colsLocal»];
					«ENDIF»
				«ENDFOR»
			«ENDFOR»
				
			«streamName» << std::endl;							
			printf("%s", «streamName».str().c_str());
		}
	'''

	// helper
	def static String generateShowElements(CollectionObject co, String arrayName, String index) {
		switch co.type {
			StructArrayType: (co.type as StructArrayType).type.generateShowStruct(arrayName, index)
			StructMatrixType: (co.type as StructMatrixType).type.generateShowStruct(arrayName, index)
			default: '''«arrayName»[«index»]'''
		}
	}

	def static String generateShowStruct(Struct s, String arrayName, String index) {
		var result = '"["'

		for (i : 0 ..< s.attributes.size) {
			val ca = s.attributes.get(i)
			var separator = ' << ", "'

			if (i == s.attributes.size - 1) {
				separator = ''
			}

			switch ca {
				Struct:
					result += ca.generateShowStruct(arrayName, index) + separator
				default:
					result +=
						' << "' + s.attributes.get(i).name + ' = " << ' + arrayName + '[' + index + '].' +
							s.attributes.get(i).name + separator
			}
		}

		return result + ' << "]"'
	}

	static class State {
		static String arrayName;
		static int counter = 0;

		def static void setArrayName(String arrayName) {
			State.arrayName = arrayName;
		}

		def static void incCounter() {
			counter = counter + 1;
		}
	}
}
