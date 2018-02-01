package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.Mode

class RngGenerator {
	def static generateRandomEnginesArray(int cores, Mode mode)'''				
		«IF mode == Mode.RELEASE»
			std::random_device rd;
			std::array<std::mt19937, «cores»> «Config.var_rng_array» = { std::mt19937(rd()) };
		«ELSE»
			std::array<std::mt19937, «cores»> «Config.var_rng_array» = «FOR i : 0 ..< cores BEFORE '{' SEPARATOR ',' AFTER '};'»std::mt19937(«i»)«ENDFOR» 
		«ENDIF»
	'''
}