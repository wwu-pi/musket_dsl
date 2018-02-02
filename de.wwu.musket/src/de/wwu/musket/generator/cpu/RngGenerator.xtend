package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.Mode
import de.wwu.musket.musket.MusketFunctionCall
import static extension de.wwu.musket.generator.extensions.ObjectExtension.*
import de.wwu.musket.musket.IntVal
import de.wwu.musket.musket.DoubleVal

class RngGenerator {
	def static generateRandomEnginesArray(int cores, Mode mode) '''				
		«IF mode == Mode.RELEASE»
			std::random_device rd;
			std::array<std::mt19937, «cores»> «Config.var_rng_array» = { std::mt19937(rd()) };
		«ELSE»
			std::array<std::mt19937, «cores»> «Config.var_rng_array» = «FOR i : 0 ..< cores BEFORE '{' SEPARATOR ',' AFTER '};'»std::mt19937(«i»)«ENDFOR» 
		«ENDIF»
	'''

	def static generateDistributionArrays(Iterable<MusketFunctionCall> calls, int cores) {
		var result = ''
//		TODO: remove duplicates
		val uniqueCalls = calls.filter[c1 | true]
		for (rc : uniqueCalls) {
			// assumes that rand takes two params
			val lower = rc.params.head
			val higher = rc.params.get(1)

			switch lower {
				IntVal: result += '''std::array<std::uniform_int_distribution<int>, «cores»> rand_dist_int_«lower.ValueAsString»_«higher.ValueAsString» = {std::uniform_int_distribution<int>(«lower.ValueAsString», «higher.ValueAsString»)};'''
				DoubleVal: result += '''std::array<std::uniform_real_distribution<double>, «cores»> rand_dist_double_«lower.ValueAsString»_«higher.ValueAsString» = {std::uniform_real_distribution<double>(«lower.ValueAsString», «higher.ValueAsString»)};'''
				default: throw new UnsupportedOperationException('Random number generation only for ints and doubles!')
			}
		}
		result
	}
}
