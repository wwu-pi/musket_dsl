package de.wwu.musket.generator.gpu

import de.wwu.musket.musket.Mode
import de.wwu.musket.musket.MusketFunctionCall
import static extension de.wwu.musket.generator.gpu.util.DataHelper.*
import static extension de.wwu.musket.generator.extensions.StringExtension.*
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
			std::vector<std::mt19937> «Config.var_rng_array»;
	'''
	
	def static generateRandomDeviceVariables() '''
			std::array<float, «Config.number_of_random_numbers»> «Config.var_rns_array»;			
			size_t «Config.var_rns_index» = 0;
	'''
	
	def static generateRandomDeviceVariablesInit(int cores, Mode mode, int processId) '''
		std::mt19937 d_rng_gen(rd());
		std::uniform_real_distribution<float> d_rng_dis(0.0f, 1.0f);
		for(int random_number = 0; random_number < «Config.number_of_random_numbers»; random_number++){
			«Config.var_rns_array»[random_number] = d_rng_dis(d_rng_gen);
		}
		
		#pragma omp parallel for
		for(int gpu = 0; gpu < «Config.gpus»; ++gpu){
			acc_set_device_num(gpu, acc_device_not_host);
			#pragma acc enter data copyin(«Config.var_rns_array»)
			#pragma acc enter data copyin(«Config.var_rns_index»)
		}
	'''
	
	def static generateRandomDeviceVariablesFree() '''
		#pragma omp parallel for
		for(int gpu = 0; gpu < «Config.gpus»; ++gpu){
			acc_set_device_num(gpu, acc_device_not_host);
			#pragma acc exit data delete(«Config.var_rns_array»)
			#pragma acc exit data delete(«Config.var_rns_index»)
		}
	'''
	
	def static generateRandomEnginesArrayInit(int cores, Mode mode, int processId) '''			
		«Config.var_rng_array».reserve(«cores»);
		«IF mode == Mode.RELEASE»			
			std::random_device rd;
			for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «cores»; ++«Config.var_loop_counter»){
				«Config.var_rng_array».push_back(std::mt19937(rd()));
			}
		«ELSE»
			«FOR i : 0 ..< cores»
				«Config.var_rng_array».push_back(std::mt19937(«IF Config.processes > 1»«processId * cores + i»«ELSE»«i»«ENDIF»));
			«ENDFOR»
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
					result +='''std::vector<std::uniform_int_distribution<int>> rand_dist_int_«lower.ValueAsString.toCXXIdentifier»_«higher.ValueAsString.toCXXIdentifier»;'''
				case MusketType.DOUBLE:
					result +=
						'''std::vector<std::uniform_real_distribution<double>> rand_dist_double_«lower.ValueAsString.toCXXIdentifier»_«higher.ValueAsString.toCXXIdentifier»;'''
				case MusketType.FLOAT:
					result +='''std::vector<std::uniform_real_distribution<float>> rand_dist_float_«lower.ValueAsString.toCXXIdentifier»_«higher.ValueAsString.toCXXIdentifier»;'''
				default:
					throw new UnsupportedOperationException('Random number generation only for ints, floats, and doubles!')
			}
		}
		result
	}
	
	def static generateDistributionArraysInit(Iterable<MusketFunctionCall> calls, int cores) {
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
					result +='''
						rand_dist_int_«lower.ValueAsString.toCXXIdentifier»_«higher.ValueAsString.toCXXIdentifier».reserve(«cores»);
						for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «cores»; ++«Config.var_loop_counter»){
							rand_dist_int_«lower.ValueAsString.toCXXIdentifier»_«higher.ValueAsString.toCXXIdentifier».push_back(std::uniform_int_distribution<int>(«lower.ValueAsString», «higher.ValueAsString»));
						}'''
				case MusketType.DOUBLE:
					result +=
						'''
						rand_dist_double_«lower.ValueAsString.toCXXIdentifier»_«higher.ValueAsString.toCXXIdentifier».reserve(«cores»);
						for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «cores»; ++«Config.var_loop_counter»){
							rand_dist_double_«lower.ValueAsString.toCXXIdentifier»_«higher.ValueAsString.toCXXIdentifier».push_back(std::uniform_real_distribution<double>(«lower.ValueAsString», «higher.ValueAsString»));
						}'''
				case MusketType.FLOAT:
					result +=
						'''
						rand_dist_float_«lower.ValueAsString.toCXXIdentifier»_«higher.ValueAsString.toCXXIdentifier».reserve(«cores»);
						for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «cores»; ++«Config.var_loop_counter»){
							rand_dist_float_«lower.ValueAsString.toCXXIdentifier»_«higher.ValueAsString.toCXXIdentifier».push_back(std::uniform_real_distribution<float>(«lower.ValueAsString», «higher.ValueAsString»));
						}'''
				default:
					throw new UnsupportedOperationException('Random number generation only for ints, floats, and doubles!')
			}
		}
		result
	}
	
	
	def static generateGetRandomDeviceFunctions() '''
		#pragma acc routine seq nohost // present(«Config.var_rns_index», «Config.var_rns_array»)
		int get_random_int(int lower, int higher){
			size_t t_rng_index = 0;
			#pragma acc atomic capture
			t_rng_index = «Config.var_rns_index»++;
			
			t_rng_index = t_rng_index % «Config.number_of_random_numbers»;
			
			return static_cast<int>(«Config.var_rns_array»[t_rng_index] * (higher - lower + 0.999999) + lower);
		}
		
		#pragma acc routine seq nohost
		float get_random_float(float lower, float higher){
			size_t t_rng_index = 0;
			#pragma acc atomic capture
			t_rng_index = «Config.var_rns_index»++;
			
			t_rng_index = t_rng_index % «Config.number_of_random_numbers»;
			
			return «Config.var_rns_array»[t_rng_index] * (higher - lower + 0.999999) + lower;
		}
		
		#pragma acc routine seq nohost
		double get_random_double(double lower, double higher){
			size_t t_rng_index = 0;
			#pragma acc atomic capture
			t_rng_index = «Config.var_rns_index»++;
			
			t_rng_index = t_rng_index % «Config.number_of_random_numbers»;
			
			return static_cast<double>(«Config.var_rns_array»[t_rng_index] * (higher - lower + 0.999999) + lower);
		}
	'''

}
