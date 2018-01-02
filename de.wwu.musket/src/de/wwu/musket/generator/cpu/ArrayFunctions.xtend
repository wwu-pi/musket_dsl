package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.CollectionFunctionCall

import static de.wwu.musket.generator.cpu.MPIRoutines.*

import static extension de.wwu.musket.generator.extensions.ObjectExtension.*
import de.wwu.musket.musket.Array
import de.wwu.musket.musket.Matrix

class ArrayFunctions {

	def static generateCollectionFunctionCall(CollectionFunctionCall afc) {
		switch afc.function {
			case SIZE:
				generateSize(afc)
			case SIZE_LOCAL:
				generateSizeLocal(afc)
			case SHOW:
				generateShow(afc)
			case COLUMNS: { // TODO unimplemented
			}
			case COLUMS_LOCAL: { // TODO unimplemented
			}
			case ROWS: { // TODO unimplemented
			}
			case ROWS_LOCAL: { // TODO unimplemented
			}
		}
	}

	def static generateSize(CollectionFunctionCall afc) '''«afc.^var.size»'''

	def static generateSizeLocal(CollectionFunctionCall afc) '''«afc.^var.sizeLocal»'''

	def static generateShow(CollectionFunctionCall afc) {
		switch afc.^var.distributionMode {
			case COPY: generateShowCopy(afc.^var)
			case DIST: generateShowDist(afc.^var)
			default: ''''''
		}
	}

	def static dispatch generateShowCopy(Array a) '''
		«val streamName = 's' + Status.temp_count++»
		if («Config.var_pid» == 0) {
			std::ostringstream «streamName»;
			«streamName» << "[";
			for (int i = 0; i < «a.size» - 1; i++) {
				«streamName» << «a.name»[i];
				«streamName» << "; ";
			}
			«streamName» << «a.name»[«a.size» - 1] << "]" << std::endl;
			«streamName» << std::endl;
			printf("%s", «streamName».str().c_str());
		}
	'''

	def static dispatch generateShowCopy(Matrix m) '''
		«val streamName = 's' + Status.temp_count++»
				if («Config.var_pid» == 0) {
					std::ostringstream «streamName»;
					«streamName» << "[";
					for (size_t i = 0; i < «m.rows»; ++i) {
						«streamName» << "[";
						for(size_t j = 0; j < «m.cols - 1»; ++j){
							«streamName» << «m.name»[«m.cols» * i + j];						
							«streamName» << "; ";
						}
						«streamName» << «m.name»[«m.cols» * i + «m.cols - 1»] << "]" << std::endl;
						if(i == «m.rows - 1»){
							«streamName» << "]";
						}
						«streamName» << std::endl;
					}
					
					printf("%s", «streamName».str().c_str());
				}
	'''

	def static dispatch generateShowDist(Array a) '''
		«val array_name = 'temp' + Status.temp_count++»
		std::array<«a.CppPrimitiveTypeAsString», «a.size»> «array_name»{};
		
		«generateMPIGather(a.name + '.data()', a.sizeLocal, a.CppPrimitiveTypeAsString, array_name + '.data()')»
		
		if («Config.var_pid» == 0) {
			«val streamName = 's' + Status.temp_count++»
			std::ostringstream «streamName»;
			«streamName» << "[";
			for (int i = 0; i < «a.size» - 1; i++) {
				«streamName» << «array_name»[i];
				«streamName» << "; ";
			}
			«streamName» << «array_name»[«a.size» - 1] << "]" << std::endl;
			«streamName» << std::endl;
			printf("%s", «streamName».str().c_str());
		}
	'''

	def static dispatch generateShowDist(Matrix m) '''
			«val array_name = 'temp' + Status.temp_count++»
			std::array<«m.CppPrimitiveTypeAsString», «m.size»> «array_name»{};
			
			«generateMPIGather(m.name + '.data()', m.sizeLocal, m.CppPrimitiveTypeAsString, array_name + '.data()')»
		
			«val streamName = 's' + Status.temp_count++»
			if («Config.var_pid» == 0) {
				std::ostringstream «streamName»;
				«FOR i : 0..<m.rows BEFORE streamName + '<< "[";' SEPARATOR streamName + '<< std::endl;' AFTER streamName + '<< "]" << std::endl;'»
					«FOR j : 0..<m.cols BEFORE streamName + '<< "[";' SEPARATOR streamName + '<< "; ";'  AFTER streamName + '<< "]";'»
						«IF m.blocksInRow == 1»
							«streamName» << «array_name»[«(i % m.rowsLocal) * m.colsLocal + (i / m.rowsLocal) * m.sizeLocal + (j / m.colsLocal) * m.sizeLocal + j % m.colsLocal»];									
						«ELSE»
							«streamName» << «array_name»[«(i % m.rowsLocal) * m.colsLocal + (i / m.rowsLocal) * m.sizeLocal * m.blocksInColumn + (j / m.colsLocal) * m.sizeLocal + j % m.colsLocal»];
						«ENDIF»
					«ENDFOR»
				«ENDFOR»
				
				«streamName» << "]" << std::endl;							
				printf("%s", «streamName».str().c_str());
			}
	'''
}
