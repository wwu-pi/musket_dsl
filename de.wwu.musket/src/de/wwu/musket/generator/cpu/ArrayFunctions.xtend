package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.ArrayFunctionCall
import de.wwu.musket.musket.ArrayFunctionName
import static extension de.wwu.musket.generator.extensions.ObjectExtension.*
import static de.wwu.musket.generator.cpu.MPIRoutines.*

class ArrayFunctions {

	def static generateArrayFunctionCall(ArrayFunctionCall afc) {
		switch afc.function {
			case SIZE: generateSize(afc)
			case SIZE_LOCAL: generateSizeLocal(afc)
			case SHOW: generateShow(afc)
		}
	}

	def static generateSize(ArrayFunctionCall afc) '''
	'''

	def static generateSizeLocal(ArrayFunctionCall afc) '''
	'''

	def static generateShow(ArrayFunctionCall afc) {
		switch afc.^var.distributionMode {
			case COPY: generateShowCopy(afc)
			case DIST: generateShowDist(afc)
			default: ''''''
		}
	}

	def static generateShowCopy(ArrayFunctionCall afc) '''
		«val streamName = 's' + Status.temp_count++»
		if («Config.var_pid» == 0) {
			std::ostringstream «streamName»;
			«streamName» << "[";
			for (int i = 0; i < «afc.^var.size» - 1; i++) {
				«streamName» << «afc.^var.name»[i];
				«streamName» << " ";
			}
			«streamName» << «afc.^var.name»[«afc.^var.size» - 1] << "]" << std::endl;
			«streamName» << std::endl;
			printf("%s", «streamName».str().c_str());
		}
	'''

	def static generateShowDist(ArrayFunctionCall afc) '''
		«val array_name = 'temp' + Status.temp_count++»
		std::array<«afc.^var.CppPrimitiveTypeAsString», «afc.^var.size»> «array_name»{};
		
		«generateMPIGather(afc.^var.name + '.data()', afc.^var.sizeLocal, afc.^var.CppPrimitiveTypeAsString, array_name + '.data()')»
		
		if («Config.var_pid» == 0) {
			«val streamName = 's' + Status.temp_count++»
			std::ostringstream «streamName»;
			«streamName» << "[";
			for (int i = 0; i < «afc.^var.size» - 1; i++) {
				«streamName» << «array_name»[i];
				«streamName» << " ";
			}
			«streamName» << «array_name»[«afc.^var.size» - 1] << "]" << std::endl;
			«streamName» << std::endl;
			printf("%s", «streamName».str().c_str());
		}
	'''

	def static generateArrayFunctionCalls(ArrayFunctionCall afc) '''
		T* b = new T[n];
		std::ostringstream s;
		if (descr.size() > 0)
		s << descr << ": " << std::endl;
		
		gather(b);
		
		if (msl::isRootProcess()) {
		s << "[";
		for (int i = 0; i < n - 1; i++) {
		s << b[i];
		s << " ";
		}
		s << b[n - 1] << "]" << std::endl;
		s << std::endl;
		}
		
		delete b;
		
		if (msl::isRootProcess()) {
		printf("%s", s.str().c_str());
		}
	'''
}
