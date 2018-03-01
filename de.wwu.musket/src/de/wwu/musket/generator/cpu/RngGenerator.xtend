package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.Mode
import de.wwu.musket.musket.MusketFunctionCall
import static extension de.wwu.musket.generator.extensions.ObjectExtension.*
import static extension de.wwu.musket.util.TypeHelper.*
import de.wwu.musket.util.MusketType

/**
 * Generates everything that is required to get random numbers.
 * <p>
 * Random engines are not thread-safe, therefore there are as many engines as threads, and each thread uses its own engine.
 */
class RngGenerator {
	
	/**
	 * Generates the array with the random engines. In release mode, a random device is used for initialization. 
	 * In debug mode, engines are initialized with a consecutive number process id * cores + thread id
	 * 
	 * @param cores the number of cores
	 * @param mode the mode release or debug
	 */
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

	/**
	 * Generates the array with distributions. For ints it is uniform_int_dist<int> and for double uniform_real_dist<double>.
	 * There are multiple arrays based on the borders of each musket function call mkt::rand(lower, higher).
	 * 
	 * @param calls all musket function calls
	 * @param cores number of cores
	 * @return generated code
	 */
	def static generateDistributionArrays(Iterable<MusketFunctionCall> calls, int cores) {
		var result = ''

		val uniqueCalls = newArrayList

		calls.forEach[e1| if(!uniqueCalls.exists[e2 | e1.params.get(0).calculateType.cppType ==
					(e2 as MusketFunctionCall).params.get(0).calculateType.cppType &&
					e1.params.get(0).ValueAsString == (e2 as MusketFunctionCall).params.get(0).ValueAsString &&
					e1.params.get(0).ValueAsString == (e2 as MusketFunctionCall).params.get(0).ValueAsString]) uniqueCalls.add(e1)]

		for (rc : uniqueCalls) {
			// assumes that rand takes two params
			val lower = rc.params.head
			val higher = rc.params.get(1)

			switch lower.calculateType {
				case MusketType.INT:
					result +=
						'''std::vector<std::uniform_int_distribution<int>> rand_dist_int_«lower.ValueAsString»_«higher.ValueAsString»;
						rand_dist_int_«lower.ValueAsString»_«higher.ValueAsString».reserve(«cores»);
						for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «cores»; ++«Config.var_loop_counter»){
							rand_dist_int_«lower.ValueAsString»_«higher.ValueAsString».push_back(std::uniform_int_distribution<int>(«lower.ValueAsString», «higher.ValueAsString»));
						}'''
				case MusketType.DOUBLE:
					result +=
						'''std::vector<std::uniform_real_distribution<double>> rand_dist_double_«lower.ValueAsString.replace('.', '_')»_«higher.ValueAsString.replace('.', '_')»;
						rand_dist_double_«lower.ValueAsString.replace('.', '_')»_«higher.ValueAsString.replace('.', '_')».reserve(«cores»);
						for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «cores»; ++«Config.var_loop_counter»){
							rand_dist_double_«lower.ValueAsString.replace('.', '_')»_«higher.ValueAsString.replace('.', '_')».push_back(std::uniform_real_distribution<double>(«lower.ValueAsString», «higher.ValueAsString»));
						}'''
				case MusketType.FLOAT:
					result +=
						'''std::vector<std::uniform_real_distribution<float>> rand_dist_float_«lower.ValueAsString.replace('.', '_')»_«higher.ValueAsString.replace('.', '_')»;
						rand_dist_float_«lower.ValueAsString.replace('.', '_')»_«higher.ValueAsString.replace('.', '_')».reserve(«cores»);
						for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «cores»; ++«Config.var_loop_counter»){
							rand_dist_float_«lower.ValueAsString.replace('.', '_')»_«higher.ValueAsString.replace('.', '_')».push_back(std::uniform_real_distribution<float>(«lower.ValueAsString», «higher.ValueAsString»));
						}'''
				default:
					throw new UnsupportedOperationException('Random number generation only for ints, floats, and doubles!')
			}
		}
		result
	}
}
