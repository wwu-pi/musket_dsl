package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.CollectionFunctionCall

import static de.wwu.musket.generator.cpu.MPIRoutines.*

import static extension de.wwu.musket.generator.extensions.ObjectExtension.*
import de.wwu.musket.musket.Array

class ArrayFunctions {

	def static generateCollectionFunctionCall(CollectionFunctionCall afc) {
		switch afc.function {
			case SIZE: generateSize(afc)
			case SIZE_LOCAL: generateSizeLocal(afc)
			case SHOW: generateShow(afc)
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

	def static generateSize(CollectionFunctionCall afc) '''
	'''

	def static generateSizeLocal(CollectionFunctionCall afc) '''
	'''

	def static generateShow(CollectionFunctionCall afc) {
		switch afc.^var.distributionMode {
			case COPY: generateShowCopy(afc)
			case DIST: generateShowDist(afc)
			default: ''''''
		}
	}

	def static generateShowCopy(CollectionFunctionCall afc) {
		if(afc.^var instanceof Array){
			return generateArrayShowCopy(afc)
		} else {
			return generateMatrixShowCopy(afc)
		}
	} 
	
	def static generateArrayShowCopy(CollectionFunctionCall afc) '''
		«val streamName = 's' + Status.temp_count++»
		if («Config.var_pid» == 0) {
			std::ostringstream «streamName»;
			«streamName» << "[";
			for (int i = 0; i < «(afc.^var as Array).size» - 1; i++) {
				«streamName» << «afc.^var.name»[i];
				«streamName» << " ";
			}
			«streamName» << «afc.^var.name»[«(afc.^var as Array).size» - 1] << "]" << std::endl;
			«streamName» << std::endl;
			printf("%s", «streamName».str().c_str());
		}
	'''
	
	// TODO unimplemented
	def static generateMatrixShowCopy(CollectionFunctionCall afc) '''
		// TODO unimplemented
	'''

	def static generateShowDist(CollectionFunctionCall afc) {
		if(afc.^var instanceof Array){
			return generateArrayShowDist(afc)
		} else {
			return generateMatrixShowDist(afc)
		}
	} 
	
	def static generateArrayShowDist(CollectionFunctionCall afc) '''
		«val array_name = 'temp' + Status.temp_count++»
		std::array<«afc.^var.CppPrimitiveTypeAsString», «(afc.^var as Array).size»> «array_name»{};
		
		«generateMPIGather(afc.^var.name + '.data()', (afc.^var as Array).sizeLocal, afc.^var.CppPrimitiveTypeAsString, array_name + '.data()')»
		
		if («Config.var_pid» == 0) {
			«val streamName = 's' + Status.temp_count++»
			std::ostringstream «streamName»;
			«streamName» << "[";
			for (int i = 0; i < «(afc.^var as Array).size» - 1; i++) {
				«streamName» << «array_name»[i];
				«streamName» << " ";
			}
			«streamName» << «array_name»[«(afc.^var as Array).size» - 1] << "]" << std::endl;
			«streamName» << std::endl;
			printf("%s", «streamName».str().c_str());
		}
	'''
	
	// TODO unimplemented
	def static generateMatrixShowDist(CollectionFunctionCall afc) '''
		// TODO unimplemented
	'''
}
