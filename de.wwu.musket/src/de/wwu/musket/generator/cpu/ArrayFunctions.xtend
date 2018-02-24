package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.Array
import de.wwu.musket.musket.CollectionFunctionCall
import de.wwu.musket.musket.CollectionObject
import de.wwu.musket.musket.DistributionMode
import de.wwu.musket.musket.Matrix
import de.wwu.musket.musket.Struct
import de.wwu.musket.musket.StructArray
import de.wwu.musket.musket.StructMatrix

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
			case SHOW:
				generateShow(afc.^var)
			case COLUMNS:
				(afc.^var as Matrix).cols
			case COLUMNS_LOCAL:
				(afc.^var as Matrix).colsLocal
			case ROWS:
				(afc.^var as Matrix).rows
			case ROWS_LOCAL:
				(afc.^var as Matrix).rowsLocal
			case BLOCKS_IN_ROW:
				(afc.^var as Matrix).blocksInRow
			case BLOCKS_IN_COLUMN:
				(afc.^var as Matrix).blocksInColumn
		}
	}

	def static generateSize(CollectionFunctionCall afc) '''«afc.^var.size»'''

	def static generateSizeLocal(CollectionFunctionCall afc) '''«afc.^var.sizeLocal»'''

	def static dispatch generateShow(Array a) '''
		«IF a.distributionMode == DistributionMode.COPY»
			«State.setArrayName(a.name)»
		«ELSE»
			«State.setArrayName("temp" + State.counter)»			
			std::array<«a.CppPrimitiveTypeAsString», «a.size»> «State.arrayName»{};
			
			«generateMPIGather(a.name + '.data()', a.sizeLocal, a.CppPrimitiveTypeAsString, State.arrayName + '.data()')»
		«ENDIF»
				
		if («Config.var_pid» == 0) {
			«val streamName = 's' + State.counter»
			«State.incCounter»
			std::ostringstream «streamName»;
			«streamName» << "«a.name»: " << std::endl << "[";
			for (int i = 0; i < «a.size.concreteValue - 1»; i++) {
				«streamName» << «generateShowElements(a, State.arrayName, "i")»;				
				«streamName» << "; ";
			}
			«streamName» << «generateShowElements(a, State.arrayName, (a.size.concreteValue - 1).toString)» << "]" << std::endl;
			«streamName» << std::endl;
			printf("%s", «streamName».str().c_str());
		}
	'''

	def static dispatch generateShow(Matrix m) '''
		«IF m.distributionMode == DistributionMode.COPY»
			«State.setArrayName(m.name)»
		«ELSE»
			«State.setArrayName("temp" + State.counter)»			
			std::array<«m.CppPrimitiveTypeAsString», «m.size»> «State.arrayName»{};
			
			«generateMPIGather(m.name + '.data()', m.sizeLocal, m.CppPrimitiveTypeAsString, State.arrayName + '.data()')»
		«ENDIF»

		«val streamName = 's' + State.counter»
		«State.incCounter»
		if («Config.var_pid» == 0) {
			std::ostringstream «streamName»;
			«streamName» << "«m.name»: " << std::endl;
			«FOR i : 0..<m.rows.concreteValue BEFORE streamName + '<< "[";' SEPARATOR streamName + '<< std::endl;' AFTER streamName + '<< "]" << std::endl;'»
				«FOR j : 0..<m.cols.concreteValue BEFORE streamName + '<< "[";' SEPARATOR streamName + '<< "; ";'  AFTER streamName + '<< "]";'»
					«IF m.blocksInRow == 1»
						«streamName» << «State.arrayName»[«(i % m.rowsLocal) * m.colsLocal + (i / m.rowsLocal) * m.sizeLocal + (j / m.colsLocal) * m.sizeLocal + j % m.colsLocal»];									
					«ELSE»
						«streamName» << «State.arrayName»[«(i % m.rowsLocal) * m.colsLocal + (i / m.rowsLocal) * m.sizeLocal * m.blocksInColumn + (j / m.colsLocal) * m.sizeLocal + j % m.colsLocal»];
					«ENDIF»
				«ENDFOR»
			«ENDFOR»
				
			«streamName» << std::endl;							
			printf("%s", «streamName».str().c_str());
		}
	'''

	// helper
	def static String generateShowElements(CollectionObject co, String arrayName, String index) {
		switch co {
			StructArray: co.type.generateShowStruct(arrayName, index)
			StructMatrix: co.type.generateShowStruct(arrayName, index)
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
