package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.Mode
import de.wwu.musket.musket.MusketFunctionCall
import static extension de.wwu.musket.generator.extensions.ObjectExtension.*
import de.wwu.musket.musket.IntVal
import de.wwu.musket.musket.DoubleVal

class RngGenerator {
	def static generateRandomEnginesArray(int cores, Mode mode) '''			
		
		«IF mode == Mode.RELEASE»
			std::vector<std::mt19937> «Config.var_rng_array»;
			«Config.var_rng_array».reserve(«cores»);
			std::random_device rd;
			for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «cores»; ++«Config.var_loop_counter»){
				«Config.var_rng_array».push_back(std::mt19937(rd()));
			}
		«ELSE»
			std::array<std::mt19937, «cores»> «Config.var_rng_array» = «FOR i : 0 ..< cores BEFORE '{' SEPARATOR ',' AFTER '};'»std::mt19937(«Config.var_pid» * «cores» + «i»)«ENDFOR» 
		«ENDIF»
	'''

	def static generateDistributionArrays(Iterable<MusketFunctionCall> calls, int cores) {
		var result = ''

		val uniqueCalls = newArrayList

		calls.forEach[e1| if(!uniqueCalls.exists[e2 | e1.params.get(0).CppPrimitiveTypeAsString ==
					(e2 as MusketFunctionCall).params.get(0).CppPrimitiveTypeAsString &&
					e1.params.get(0).ValueAsString == (e2 as MusketFunctionCall).params.get(0).ValueAsString &&
					e1.params.get(0).ValueAsString == (e2 as MusketFunctionCall).params.get(0).ValueAsString]) uniqueCalls.add(e1)]

		for (rc : uniqueCalls) {
			// assumes that rand takes two params
			val lower = rc.params.head
			val higher = rc.params.get(1)

			switch lower {
				IntVal:
					result +=
						'''std::vector<std::uniform_int_distribution<int>> rand_dist_int_«lower.ValueAsString»_«higher.ValueAsString»;
						rand_dist_int_«lower.ValueAsString»_«higher.ValueAsString».reserve(«cores»);
						for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «cores»; ++«Config.var_loop_counter»){
							rand_dist_int_«lower.ValueAsString»_«higher.ValueAsString».push_back(std::uniform_int_distribution<int>(«lower.ValueAsString», «higher.ValueAsString»));
						}'''
				DoubleVal:
					result +=
						'''std::vector<std::uniform_real_distribution<double>> rand_dist_double_«lower.ValueAsString.replace('.', '_')»_«higher.ValueAsString.replace('.', '_')»;
						rand_dist_double_«lower.ValueAsString.replace('.', '_')»_«higher.ValueAsString.replace('.', '_')».reserve(«cores»);
						for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «cores»; ++«Config.var_loop_counter»){
							rand_dist_double_«lower.ValueAsString.replace('.', '_')»_«higher.ValueAsString.replace('.', '_')».push_back(std::uniform_real_distribution<double>(«lower.ValueAsString», «higher.ValueAsString»));
						}'''
				default:
					throw new UnsupportedOperationException('Random number generation only for ints and doubles!')
			}
		}
		result
	}
}
