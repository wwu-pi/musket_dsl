	
	#include <omp.h>
	#include <openacc.h>
	#include <stdlib.h>
	#include <math.h>
	#include <array>
	#include <vector>
	#include <sstream>
	#include <chrono>
	#include <random>
	#include <limits>
	#include <memory>
	#include <cstddef>
	#include <type_traits>
	//#include <cuda.h>
	//#include <openacc_curand.h>
	
	#include "../include/musket.hpp"
	#include "../include/fss_0.hpp"
	
	
	std::vector<std::mt19937> random_engines;
	std::array<float*, 1> rns_pointers;
	std::array<float, 100000> rns;	
	std::vector<std::uniform_real_distribution<double>> rand_dist_double_INIT_LOWER_BOUND_INIT_UPPER_BOUND;std::vector<std::uniform_real_distribution<double>> rand_dist_double_minus1_0_1_0;std::vector<std::uniform_real_distribution<double>> rand_dist_double_0_0_1_0;
	
			
	const double PI = 3.141592653589793;
	const double EULER = 2.718281828459045;
	const double UPPER_BOUND = 5.12;
	const double LOWER_BOUND = -5.12;
	const double PROBLEM_RANGE = 10.24;
	const double INIT_UPPER_BOUND = 5.12;
	const double INIT_LOWER_BOUND = -5.12;
	const double WEIGHT_UPPER_BOUND = 5000.0;
	const double WEIGHT_LOWER_BOUND = 1.0;
	const double STEP_SIZE_INITIAL = 0.1;
	const double STEP_SIZE_FINAL = 1.0E-5;
	const double STEP_SIZE_VOLITIVE_INITIAL = 0.2;
	const double STEP_SIZE_VOLITIVE_FINAL = 2.0E-5;
	const int NUMBER_OF_FISH = 1024;
	const int ITERATIONS = 50;
	const int DIMENSIONS = 512;
	mkt::DArray<Fish> population(0, 1024, 1024, Fish{}, 1, 0, 0, mkt::DIST, mkt::COPY);
	mkt::DArray<double> instinctive_movement_vector_copy(0, 512, 512, 0.0, 1, 0, 0, mkt::COPY, mkt::COPY);
	mkt::DArray<Fish> weighted_fishes(0, 1024, 1024, Fish{}, 1, 0, 0, mkt::DIST, mkt::COPY);
	mkt::DArray<double> barycenter_copy(0, 512, 512, 0.0, 1, 0, 0, mkt::COPY, mkt::COPY);
	
	//Fish::Fish() : position(0, 0.0), fitness(), candidate_position(0, 0.0), candidate_fitness(), displacement(0, 0.0), fitness_variation(), weight(), best_position(0, 0.0), best_fitness() {}
	

	
	struct InitFish_map_in_place_array_functor{
		
		InitFish_map_in_place_array_functor(std::array<float*, 1> rns_pointers){
			for(int gpu = 0; gpu < 1; gpu++){
			 	_rns_pointers[gpu] = rns_pointers[gpu];
			}
			_rns_index = 0;
		}
		
		~InitFish_map_in_place_array_functor() {}
		
		auto operator()(Fish& fi){
			size_t local_rns_index  = _gang + _worker + _vector + _rns_index; // this can probably be improved
			local_rns_index  = (local_rns_index + 0x7ed55d16) + (local_rns_index << 12);
			local_rns_index = (local_rns_index ^ 0xc761c23c) ^ (local_rns_index >> 19);
			local_rns_index = (local_rns_index + 0x165667b1) + (local_rns_index << 5);
			local_rns_index = (local_rns_index + 0xd3a2646c) ^ (local_rns_index << 9);
			local_rns_index = (local_rns_index + 0xfd7046c5) + (local_rns_index << 3);
			local_rns_index = (local_rns_index ^ 0xb55a4f09) ^ (local_rns_index >> 16);
			local_rns_index = local_rns_index % 100000;
			_rns_index++;
			fi.fitness = std::numeric_limits<double>::lowest();
			fi.candidate_fitness = std::numeric_limits<double>::lowest();
			fi.weight = (WEIGHT_LOWER_BOUND);
			fi.fitness_variation = 0.0;
			fi.best_fitness = std::numeric_limits<double>::lowest();
			for(int i = 0; ((i) < (DIMENSIONS)); ++i){
				fi.position[(i)] = static_cast<double>(_rns[local_rns_index++] * ((INIT_UPPER_BOUND) - (INIT_LOWER_BOUND) + 0.999999) + (INIT_LOWER_BOUND));
				fi.candidate_position[(i)] = 0.0;
				fi.displacement[(i)] = 0.0;
				fi.best_position[(i)] = 0.0;
			}
		}
	
		void init(int gpu){
			_rns = _rns_pointers[gpu];
			std::random_device rd{};
			std::mt19937 d_rng_gen(rd());
			std::uniform_int_distribution<> d_rng_dis(0, 100000);
			_rns_index = d_rng_dis(d_rng_gen);
		}
		
		void set_id(int gang, int worker, int vector){
			_gang = gang;
			_worker = worker;
			_vector = vector;
		}
		
		
		
		float* _rns;
		std::array<float*, 1> _rns_pointers;
		size_t _rns_index;
		
		int _gang;
		int _worker;
		int _vector;
	};
	struct EvaluateFitness_map_in_place_array_functor{
		
		EvaluateFitness_map_in_place_array_functor(){
		}
		
		~EvaluateFitness_map_in_place_array_functor() {}
		
		auto operator()(Fish& fi){
			double sum = 0.0;
			for(int j = 0; ((j) < (DIMENSIONS)); ++j){
				double value = (fi).position[(j)];
				sum += (std::pow((value), 2) - (10 * std::cos(((2 * (PI)) * (value)))));
			}
			fi.fitness = -(((10 * (DIMENSIONS)) + (sum)));
			
			if(((fi).fitness > (fi).best_fitness)){
			fi.best_fitness = (fi).fitness;
			for(int k = 0; ((k) < (DIMENSIONS)); ++k){
				fi.best_position[(k)] = (fi).position[(k)];
			}
			}
		}
	
		void init(int gpu){
		}
		
		void set_id(int gang, int worker, int vector){
			_gang = gang;
			_worker = worker;
			_vector = vector;
		}
		
		
		
		
		int _gang;
		int _worker;
		int _vector;
	};
	struct IndividualMovement_map_in_place_array_functor{
		
		IndividualMovement_map_in_place_array_functor(std::array<float*, 1> rns_pointers){
			for(int gpu = 0; gpu < 1; gpu++){
			 	_rns_pointers[gpu] = rns_pointers[gpu];
			}
			_rns_index = 0;
		}
		
		~IndividualMovement_map_in_place_array_functor() {}
		
		auto operator()(Fish& fi){
			size_t local_rns_index  = _gang + _worker + _vector + _rns_index; // this can probably be improved
			local_rns_index  = (local_rns_index + 0x7ed55d16) + (local_rns_index << 12);
			local_rns_index = (local_rns_index ^ 0xc761c23c) ^ (local_rns_index >> 19);
			local_rns_index = (local_rns_index + 0x165667b1) + (local_rns_index << 5);
			local_rns_index = (local_rns_index + 0xd3a2646c) ^ (local_rns_index << 9);
			local_rns_index = (local_rns_index + 0xfd7046c5) + (local_rns_index << 3);
			local_rns_index = (local_rns_index ^ 0xb55a4f09) ^ (local_rns_index >> 16);
			local_rns_index = local_rns_index % 100000;
			_rns_index++;
			for(int i = 0; ((i) < (DIMENSIONS)); ++i){
				double rand_factor = static_cast<double>(_rns[local_rns_index++] * (1.0 - -(1.0) + 0.999999) + -(1.0));
				double direction = (((rand_factor) * (step_size)) * ((UPPER_BOUND) - (LOWER_BOUND)));
				double new_value = ((fi).position[(i)] + (direction));
				
				if(((new_value) < (LOWER_BOUND))){
				new_value = (LOWER_BOUND);
				} else 
				if(((new_value) > (UPPER_BOUND))){
				new_value = (UPPER_BOUND);
				}
				fi.candidate_position[(i)] = (new_value);
			}
			double sum = 0.0;
			for(int j = 0; ((j) < (DIMENSIONS)); ++j){
				double value = (fi).candidate_position[(j)];
				sum += (std::pow((value), 2) - (10 * std::cos(((2 * (PI)) * (value)))));
			}
			fi.candidate_fitness = -(((10 * (DIMENSIONS)) + (sum)));
			
			if(((fi).candidate_fitness > (fi).fitness)){
			fi.fitness_variation = ((fi).candidate_fitness - (fi).fitness);
			fi.fitness = (fi).candidate_fitness;
			for(int k = 0; ((k) < (DIMENSIONS)); ++k){
				fi.displacement[(k)] = ((fi).candidate_position[(k)] - (fi).position[(k)]);
				fi.position[(k)] = (fi).candidate_position[(k)];
			}
			
			if(((fi).fitness > (fi).best_fitness)){
			fi.best_fitness = (fi).fitness;
			for(int k = 0; ((k) < (DIMENSIONS)); ++k){
				fi.best_position[(k)] = (fi).position[(k)];
			}
			}
			}
			 else {
					fi.fitness_variation = 0.0;
					for(int k = 0; ((k) < (DIMENSIONS)); ++k){
						fi.displacement[(k)] = 0.0;
					}
				}
		}
	
		void init(int gpu){
			_rns = _rns_pointers[gpu];
			std::random_device rd{};
			std::mt19937 d_rng_gen(rd());
			std::uniform_int_distribution<> d_rng_dis(0, 100000);
			_rns_index = d_rng_dis(d_rng_gen);
		}
		
		void set_id(int gang, int worker, int vector){
			_gang = gang;
			_worker = worker;
			_vector = vector;
		}
		
		double step_size;
		
		
		float* _rns;
		std::array<float*, 1> _rns_pointers;
		size_t _rns_index;
		
		int _gang;
		int _worker;
		int _vector;
	};
	struct Feeding_map_in_place_array_functor{
		
		Feeding_map_in_place_array_functor(){
		}
		
		~Feeding_map_in_place_array_functor() {}
		
		auto operator()(Fish& fi){
			
			if(((max_fitness_variation) != 0.0)){
			double result = ((fi).weight + ((fi).fitness_variation / (max_fitness_variation)));
			
			if(((result) > (WEIGHT_UPPER_BOUND))){
			result = (WEIGHT_UPPER_BOUND);
			} else 
			if(((result) < (WEIGHT_LOWER_BOUND))){
			result = (WEIGHT_LOWER_BOUND);
			}
			fi.weight = (result);
			}
		}
	
		void init(int gpu){
		}
		
		void set_id(int gang, int worker, int vector){
			_gang = gang;
			_worker = worker;
			_vector = vector;
		}
		
		double max_fitness_variation;
		
		
		
		int _gang;
		int _worker;
		int _vector;
	};
	struct CalcDisplacementMap_map_in_place_array_functor{
		
		CalcDisplacementMap_map_in_place_array_functor(){
		}
		
		~CalcDisplacementMap_map_in_place_array_functor() {}
		
		auto operator()(Fish& fi){
			for(int i = 0; ((i) < (DIMENSIONS)); ++i){
				fi.displacement[(i)] *= (fi).fitness_variation;
			}
		}
	
		void init(int gpu){
		}
		
		void set_id(int gang, int worker, int vector){
			_gang = gang;
			_worker = worker;
			_vector = vector;
		}
		
		
		
		
		int _gang;
		int _worker;
		int _vector;
	};
	struct CalcInstinctiveMovementVector_map_in_place_array_functor{
		
		CalcInstinctiveMovementVector_map_in_place_array_functor(){
		}
		
		~CalcInstinctiveMovementVector_map_in_place_array_functor() {}
		
		auto operator()(double& x){
			double result = (x);
			
			if(((sum_fitness_variation) != 0.0)){
			result = ((x) / (sum_fitness_variation));
			}
			x = (result);
		}
	
		void init(int gpu){
		}
		
		void set_id(int gang, int worker, int vector){
			_gang = gang;
			_worker = worker;
			_vector = vector;
		}
		
		double sum_fitness_variation;
		
		
		
		int _gang;
		int _worker;
		int _vector;
	};
	struct InstinctiveMovement_map_in_place_array_functor{
		
		InstinctiveMovement_map_in_place_array_functor(const mkt::DArray<double>& _instinctive_movement_vector_copy) : instinctive_movement_vector_copy(_instinctive_movement_vector_copy){
		}
		
		~InstinctiveMovement_map_in_place_array_functor() {}
		
		auto operator()(Fish& fi){
			for(int i = 0; ((i) < (DIMENSIONS)); ++i){
				double new_position = ((fi).position[(i)] + instinctive_movement_vector_copy.get_data_local((i)));
				
				if(((new_position) < (LOWER_BOUND))){
				new_position = (LOWER_BOUND);
				} else 
				if(((new_position) > (UPPER_BOUND))){
				new_position = (UPPER_BOUND);
				}
				fi.position[(i)] = (new_position);
			}
		}
	
		void init(int gpu){
			instinctive_movement_vector_copy.init(gpu);
		}
		
		void set_id(int gang, int worker, int vector){
			_gang = gang;
			_worker = worker;
			_vector = vector;
		}
		
		
		mkt::DeviceArray<double> instinctive_movement_vector_copy;
		
		
		int _gang;
		int _worker;
		int _vector;
	};
	struct CalcWeightedFish_map_array_functor{
		
		CalcWeightedFish_map_array_functor(){
		}
		
		~CalcWeightedFish_map_array_functor() {}
		
		auto operator()(const Fish& fi){
			Fish _fi{fi};
			for(int i = 0; ((i) < (DIMENSIONS)); ++i){
				_fi.position[(i)] *= (_fi).weight;
			}
			return (_fi);
		}
	
		void init(int gpu){
		}
		
		void set_id(int gang, int worker, int vector){
			_gang = gang;
			_worker = worker;
			_vector = vector;
		}
		
		
		
		
		int _gang;
		int _worker;
		int _vector;
	};
	struct CalcBarycenterMap_map_in_place_array_functor{
		
		CalcBarycenterMap_map_in_place_array_functor(){
		}
		
		~CalcBarycenterMap_map_in_place_array_functor() {}
		
		auto operator()(double& x){
			double result = (x);
			
			if(((sum_weight) != 0)){
			result = ((x) / (sum_weight));
			}
			x = (result);
		}
	
		void init(int gpu){
		}
		
		void set_id(int gang, int worker, int vector){
			_gang = gang;
			_worker = worker;
			_vector = vector;
		}
		
		double sum_weight;
		
		
		
		int _gang;
		int _worker;
		int _vector;
	};
	struct VolitiveMovement_map_in_place_array_functor{
		
		VolitiveMovement_map_in_place_array_functor(const mkt::DArray<double>& _barycenter_copy, std::array<float*, 1> rns_pointers) : barycenter_copy(_barycenter_copy){
			for(int gpu = 0; gpu < 1; gpu++){
			 	_rns_pointers[gpu] = rns_pointers[gpu];
			}
			_rns_index = 0;
		}
		
		~VolitiveMovement_map_in_place_array_functor() {}
		
		auto operator()(Fish& fi){
			size_t local_rns_index  = _gang + _worker + _vector + _rns_index; // this can probably be improved
			local_rns_index  = (local_rns_index + 0x7ed55d16) + (local_rns_index << 12);
			local_rns_index = (local_rns_index ^ 0xc761c23c) ^ (local_rns_index >> 19);
			local_rns_index = (local_rns_index + 0x165667b1) + (local_rns_index << 5);
			local_rns_index = (local_rns_index + 0xd3a2646c) ^ (local_rns_index << 9);
			local_rns_index = (local_rns_index + 0xfd7046c5) + (local_rns_index << 3);
			local_rns_index = (local_rns_index ^ 0xb55a4f09) ^ (local_rns_index >> 16);
			local_rns_index = local_rns_index % 100000;
			_rns_index++;
			double distance = 0.0;
			for(int i = 0; ((i) < (DIMENSIONS)); ++i){
				distance += (((fi).position[(i)] - barycenter_copy.get_data_local((i))) * ((fi).position[(i)] - barycenter_copy.get_data_local((i))));
			}
			distance = sqrt((distance));
			
			if(((distance) != 0.0)){
			double rand_factor = static_cast<double>(_rns[local_rns_index++] * (1.0 - 0.0 + 0.999999) + 0.0);
			for(int i = 0; ((i) < (DIMENSIONS)); ++i){
				double direction = ((((rand_factor) * (step_size)) * ((UPPER_BOUND) - (LOWER_BOUND))) * (((fi).position[(i)] - barycenter_copy.get_data_local((i))) / (distance)));
				double new_position = (fi).position[(i)];
				
				if(((sum_weight) > (sum_weight_last_iteration))){
				new_position -= (direction);
				}
				 else {
						new_position += (direction);
					}
				
				if(((new_position) < (LOWER_BOUND))){
				new_position = (LOWER_BOUND);
				} else 
				if(((new_position) > (UPPER_BOUND))){
				new_position = (UPPER_BOUND);
				}
				fi.position[(i)] = (new_position);
			}
			}
		}
	
		void init(int gpu){
			barycenter_copy.init(gpu);
			_rns = _rns_pointers[gpu];
			std::random_device rd{};
			std::mt19937 d_rng_gen(rd());
			std::uniform_int_distribution<> d_rng_dis(0, 100000);
			_rns_index = d_rng_dis(d_rng_gen);
		}
		
		void set_id(int gang, int worker, int vector){
			_gang = gang;
			_worker = worker;
			_vector = vector;
		}
		
		double step_size;
		double sum_weight;
		double sum_weight_last_iteration;
		
		mkt::DeviceArray<double> barycenter_copy;
		
		float* _rns;
		std::array<float*, 1> _rns_pointers;
		size_t _rns_index;
		
		int _gang;
		int _worker;
		int _vector;
	};
	struct Lambda63_map_reduce_array_functor{
		
		Lambda63_map_reduce_array_functor(){
		}
		
		~Lambda63_map_reduce_array_functor() {}
		
		auto operator()(Fish fi){
			return (fi).weight;
		}
	
		void init(int gpu){
		}
		
		void set_id(int gang, int worker, int vector){
			_gang = gang;
			_worker = worker;
			_vector = vector;
		}
		
		
		
		
		int _gang;
		int _worker;
		int _vector;
	};
	struct Lambda64_map_reduce_array_functor{
		
		Lambda64_map_reduce_array_functor(){
		}
		
		~Lambda64_map_reduce_array_functor() {}
		
		auto operator()(Fish fi){
			return (fi).fitness_variation;
		}
	
		void init(int gpu){
		}
		
		void set_id(int gang, int worker, int vector){
			_gang = gang;
			_worker = worker;
			_vector = vector;
		}
		
		
		
		
		int _gang;
		int _worker;
		int _vector;
	};
	struct Lambda65_map_reduce_array_functor{
		
		Lambda65_map_reduce_array_functor(){
		}
		
		~Lambda65_map_reduce_array_functor() {}
		
		auto operator()(Fish fi){
			return (fi).fitness_variation;
		}
	
		void init(int gpu){
		}
		
		void set_id(int gang, int worker, int vector){
			_gang = gang;
			_worker = worker;
			_vector = vector;
		}
		
		
		
		
		int _gang;
		int _worker;
		int _vector;
	};
	struct Lambda66_map_reduce_array_functor{
		
		Lambda66_map_reduce_array_functor(){
		}
		
		~Lambda66_map_reduce_array_functor() {}
		
		auto operator()(Fish fi){
			return (fi).displacement;
		}
	
		void init(int gpu){
		}
		
		void set_id(int gang, int worker, int vector){
			_gang = gang;
			_worker = worker;
			_vector = vector;
		}
		
		
		
		
		int _gang;
		int _worker;
		int _vector;
	};
	struct Lambda67_map_reduce_array_functor{
		
		Lambda67_map_reduce_array_functor(){
		}
		
		~Lambda67_map_reduce_array_functor() {}
		
		auto operator()(Fish fi){
			return (fi).weight;
		}
	
		void init(int gpu){
		}
		
		void set_id(int gang, int worker, int vector){
			_gang = gang;
			_worker = worker;
			_vector = vector;
		}
		
		
		
		
		int _gang;
		int _worker;
		int _vector;
	};
	struct Lambda68_map_reduce_array_functor{
		
		Lambda68_map_reduce_array_functor(){
		}
		
		~Lambda68_map_reduce_array_functor() {}
		
		auto operator()(Fish fi){
			return (fi).position;
		}
	
		void init(int gpu){
		}
		
		void set_id(int gang, int worker, int vector){
			_gang = gang;
			_worker = worker;
			_vector = vector;
		}
		
		
		
		
		int _gang;
		int _worker;
		int _vector;
	};
	struct Lambda69_map_reduce_array_functor{
		
		Lambda69_map_reduce_array_functor(){
		}
		
		~Lambda69_map_reduce_array_functor() {}
		
		auto operator()(Fish fi){
			return (fi).best_fitness;
		}
	
		void init(int gpu){
		}
		
		void set_id(int gang, int worker, int vector){
			_gang = gang;
			_worker = worker;
			_vector = vector;
		}
		
		
		
		
		int _gang;
		int _worker;
		int _vector;
	};
	
	
	
	
	template<>
	double mkt::map_reduce_plus<Fish, double, Lambda63_map_reduce_array_functor>(mkt::DArray<Fish>& a, Lambda63_map_reduce_array_functor f){
		double local_result = 0.0;
		
		acc_set_device_num(0, acc_device_not_host);
		Fish* devptr = a.get_device_pointer(0);
		f.init(0);
		const int gpu_elements = a.get_size_gpu();
		
		#pragma acc parallel loop deviceptr(devptr) present_or_copy(local_result) reduction(+:local_result) async(0)
		for(unsigned int counter = 0; counter < gpu_elements; ++counter) {
			#pragma acc cache(local_result, devptr[0:gpu_elements])
			double map_result = f(devptr[counter]);
			local_result = local_result + map_result;
		}
		acc_wait(0);
		
		return local_result;
	}
	template<>
	double mkt::map_reduce_max<Fish, double, Lambda64_map_reduce_array_functor>(mkt::DArray<Fish>& a, Lambda64_map_reduce_array_functor f){
		double local_result = std::numeric_limits<double>::lowest();
		
		acc_set_device_num(0, acc_device_not_host);
		Fish* devptr = a.get_device_pointer(0);
		f.init(0);
		const int gpu_elements = a.get_size_gpu();
		
		#pragma acc parallel loop deviceptr(devptr) present_or_copy(local_result) reduction(max:local_result) async(0)
		for(unsigned int counter = 0; counter < gpu_elements; ++counter) {
			#pragma acc cache(local_result, devptr[0:gpu_elements])
			double map_result = f(devptr[counter]);
			local_result = local_result > map_result ? local_result : map_result;
		}
		acc_wait(0);
		
		return local_result;
	}
	template<>
	double mkt::map_reduce_plus<Fish, double, Lambda65_map_reduce_array_functor>(mkt::DArray<Fish>& a, Lambda65_map_reduce_array_functor f){
		double local_result = 0.0;
		
		acc_set_device_num(0, acc_device_not_host);
		Fish* devptr = a.get_device_pointer(0);
		f.init(0);
		const int gpu_elements = a.get_size_gpu();
		
		#pragma acc parallel loop deviceptr(devptr) present_or_copy(local_result) reduction(+:local_result) async(0)
		for(unsigned int counter = 0; counter < gpu_elements; ++counter) {
			#pragma acc cache(local_result, devptr[0:gpu_elements])
			double map_result = f(devptr[counter]);
			local_result = local_result + map_result;
		}
		acc_wait(0);
		
		return local_result;
	}
	template<>
	std::array<double,512> mkt::map_reduce_plus<Fish, std::array<double,512>, Lambda66_map_reduce_array_functor>(mkt::DArray<Fish>& a, Lambda66_map_reduce_array_functor f){
		std::array<double,512> local_result;
		local_result.fill(0.0);
		
		acc_set_device_num(0, acc_device_not_host);
		f.init(0);
		Fish* devptr = a.get_device_pointer(0);
		const int gpu_elements = a.get_size_gpu();
		
		#pragma acc parallel loop deviceptr(devptr) present_or_copy(local_result) async(0)
		for(unsigned int counter = 0; counter < 512; ++counter) {
			double element_result = 0.0;
			#pragma acc loop reduction(+:element_result)
			for(unsigned int inner_counter = 0; inner_counter < gpu_elements; ++inner_counter) {
				double map_result = (f(devptr[inner_counter]))[counter];
				element_result = element_result + map_result;
			}
			local_result[counter] = local_result[counter] + element_result;
		}
		acc_wait(0);
		
		return local_result;
	}
	template<>
	double mkt::map_reduce_plus<Fish, double, Lambda67_map_reduce_array_functor>(mkt::DArray<Fish>& a, Lambda67_map_reduce_array_functor f){
		double local_result = 0.0;
		
		acc_set_device_num(0, acc_device_not_host);
		Fish* devptr = a.get_device_pointer(0);
		f.init(0);
		const int gpu_elements = a.get_size_gpu();
		
		#pragma acc parallel loop deviceptr(devptr) present_or_copy(local_result) reduction(+:local_result) async(0)
		for(unsigned int counter = 0; counter < gpu_elements; ++counter) {
			#pragma acc cache(local_result, devptr[0:gpu_elements])
			double map_result = f(devptr[counter]);
			local_result = local_result + map_result;
		}
		acc_wait(0);
		
		return local_result;
	}
	template<>
	std::array<double,512> mkt::map_reduce_plus<Fish, std::array<double,512>, Lambda68_map_reduce_array_functor>(mkt::DArray<Fish>& a, Lambda68_map_reduce_array_functor f){
		std::array<double,512> local_result;
		local_result.fill(0.0);
		
		acc_set_device_num(0, acc_device_not_host);
		f.init(0);
		Fish* devptr = a.get_device_pointer(0);
		const int gpu_elements = a.get_size_gpu();
		
		#pragma acc parallel loop deviceptr(devptr) present_or_copy(local_result) async(0)
		for(unsigned int counter = 0; counter < 512; ++counter) {
			double element_result = 0.0;
			#pragma acc loop reduction(+:element_result)
			for(unsigned int inner_counter = 0; inner_counter < gpu_elements; ++inner_counter) {
				double map_result = (f(devptr[inner_counter]))[counter];
				element_result = element_result + map_result;
			}
			local_result[counter] = local_result[counter] + element_result;
		}
		acc_wait(0);
		
		return local_result;
	}
	template<>
	double mkt::map_reduce_max<Fish, double, Lambda69_map_reduce_array_functor>(mkt::DArray<Fish>& a, Lambda69_map_reduce_array_functor f){
		double local_result = std::numeric_limits<double>::lowest();
		
		acc_set_device_num(0, acc_device_not_host);
		Fish* devptr = a.get_device_pointer(0);
		f.init(0);
		const int gpu_elements = a.get_size_gpu();
		
		#pragma acc parallel loop deviceptr(devptr) present_or_copy(local_result) reduction(max:local_result) async(0)
		for(unsigned int counter = 0; counter < gpu_elements; ++counter) {
			#pragma acc cache(local_result, devptr[0:gpu_elements])
			double map_result = f(devptr[counter]);
			local_result = local_result > map_result ? local_result : map_result;
		}
		acc_wait(0);
		
		return local_result;
	}
	
	
	
	int main(int argc, char** argv) {
		
		
		random_engines.reserve(4);
		std::random_device rd;
		for(size_t counter = 0; counter < 4; ++counter){
			random_engines.push_back(std::mt19937(rd()));
		}
		std::mt19937 d_rng_gen(rd());
		std::uniform_real_distribution<float> d_rng_dis(0.0f, 1.0f);
		for(int random_number = 0; random_number < 100000; random_number++){
			rns[random_number] = d_rng_dis(d_rng_gen);
		}
		
		#pragma omp parallel for
		for(int gpu = 0; gpu < 1; ++gpu){
			acc_set_device_num(gpu, acc_device_not_host);
			float* devptr = static_cast<float*>(acc_malloc(100000 * sizeof(float)));
			rns_pointers[gpu] = devptr;
			acc_memcpy_to_device(devptr, rns.data(), 100000 * sizeof(float));
		}
		
		InitFish_map_in_place_array_functor initFish_map_in_place_array_functor{rns_pointers};
		EvaluateFitness_map_in_place_array_functor evaluateFitness_map_in_place_array_functor{};
		IndividualMovement_map_in_place_array_functor individualMovement_map_in_place_array_functor{rns_pointers};
		Feeding_map_in_place_array_functor feeding_map_in_place_array_functor{};
		CalcDisplacementMap_map_in_place_array_functor calcDisplacementMap_map_in_place_array_functor{};
		CalcInstinctiveMovementVector_map_in_place_array_functor calcInstinctiveMovementVector_map_in_place_array_functor{};
		InstinctiveMovement_map_in_place_array_functor instinctiveMovement_map_in_place_array_functor{instinctive_movement_vector_copy};
		CalcWeightedFish_map_array_functor calcWeightedFish_map_array_functor{};
		CalcBarycenterMap_map_in_place_array_functor calcBarycenterMap_map_in_place_array_functor{};
		VolitiveMovement_map_in_place_array_functor volitiveMovement_map_in_place_array_functor{barycenter_copy, rns_pointers};
		Lambda63_map_reduce_array_functor lambda63_map_reduce_array_functor{};
		Lambda64_map_reduce_array_functor lambda64_map_reduce_array_functor{};
		Lambda65_map_reduce_array_functor lambda65_map_reduce_array_functor{};
		Lambda66_map_reduce_array_functor lambda66_map_reduce_array_functor{};
		Lambda67_map_reduce_array_functor lambda67_map_reduce_array_functor{};
		Lambda68_map_reduce_array_functor lambda68_map_reduce_array_functor{};
		Lambda69_map_reduce_array_functor lambda69_map_reduce_array_functor{};
		
		rand_dist_double_INIT_LOWER_BOUND_INIT_UPPER_BOUND.reserve(4);
		for(size_t counter = 0; counter < 4; ++counter){
			rand_dist_double_INIT_LOWER_BOUND_INIT_UPPER_BOUND.push_back(std::uniform_real_distribution<double>((INIT_LOWER_BOUND), (INIT_UPPER_BOUND)));
		}rand_dist_double_minus1_0_1_0.reserve(4);
		for(size_t counter = 0; counter < 4; ++counter){
			rand_dist_double_minus1_0_1_0.push_back(std::uniform_real_distribution<double>(-(1.0), 1.0));
		}rand_dist_double_0_0_1_0.reserve(4);
		for(size_t counter = 0; counter < 4; ++counter){
			rand_dist_double_0_0_1_0.push_back(std::uniform_real_distribution<double>(0.0, 1.0));
		}
		
				
		
		for(int gpu = 0; gpu < 1; ++gpu){
			acc_set_device_num(gpu, acc_device_not_host);
			acc_wait_all();
		}
		std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
		mkt::map_in_place<Fish, InitFish_map_in_place_array_functor>(population, initFish_map_in_place_array_functor);
		double step_size = (STEP_SIZE_INITIAL);
		double step_size_vol = (STEP_SIZE_VOLITIVE_INITIAL);
		double sum_weight_last_iteration = 0.0;
		sum_weight_last_iteration = mkt::map_reduce_plus<Fish, double, Lambda63_map_reduce_array_functor>(population, lambda63_map_reduce_array_functor);
		for(int iteration = 0; ((iteration) < (ITERATIONS)); ++iteration){
			mkt::map_in_place<Fish, EvaluateFitness_map_in_place_array_functor>(population, evaluateFitness_map_in_place_array_functor);
			if(((iteration) > 0)){
				step_size = ((step_size) - (((STEP_SIZE_INITIAL) - (STEP_SIZE_FINAL)) / static_cast<double>(((ITERATIONS) - 1))));
				step_size_vol = ((step_size_vol) - (((STEP_SIZE_VOLITIVE_INITIAL) - (STEP_SIZE_VOLITIVE_FINAL)) / static_cast<double>(((ITERATIONS) - 1))));
			}
			individualMovement_map_in_place_array_functor.step_size = (step_size);
			mkt::map_in_place<Fish, IndividualMovement_map_in_place_array_functor>(population, individualMovement_map_in_place_array_functor);
			double max_fitness_variation = 0.0;
			max_fitness_variation = mkt::map_reduce_max<Fish, double, Lambda64_map_reduce_array_functor>(population, lambda64_map_reduce_array_functor);
			feeding_map_in_place_array_functor.max_fitness_variation = (max_fitness_variation);
			mkt::map_in_place<Fish, Feeding_map_in_place_array_functor>(population, feeding_map_in_place_array_functor);
			double sum_fitness_variation = 0.0;
			sum_fitness_variation = mkt::map_reduce_plus<Fish, double, Lambda65_map_reduce_array_functor>(population, lambda65_map_reduce_array_functor);
			mkt::map_in_place<Fish, CalcDisplacementMap_map_in_place_array_functor>(population, calcDisplacementMap_map_in_place_array_functor);
			instinctive_movement_vector_copy = mkt::map_reduce_plus<Fish, std::array<double,512>, Lambda66_map_reduce_array_functor>(population, lambda66_map_reduce_array_functor);
			calcInstinctiveMovementVector_map_in_place_array_functor.sum_fitness_variation = (sum_fitness_variation);
			mkt::map_in_place<double, CalcInstinctiveMovementVector_map_in_place_array_functor>(instinctive_movement_vector_copy, calcInstinctiveMovementVector_map_in_place_array_functor);
			mkt::map_in_place<Fish, InstinctiveMovement_map_in_place_array_functor>(population, instinctiveMovement_map_in_place_array_functor);
			double sum_weight = 0.0;
			sum_weight = mkt::map_reduce_plus<Fish, double, Lambda67_map_reduce_array_functor>(population, lambda67_map_reduce_array_functor);
			mkt::map<Fish, Fish, CalcWeightedFish_map_array_functor>(population, weighted_fishes, calcWeightedFish_map_array_functor);
			barycenter_copy = mkt::map_reduce_plus<Fish, std::array<double,512>, Lambda68_map_reduce_array_functor>(weighted_fishes, lambda68_map_reduce_array_functor);
			calcBarycenterMap_map_in_place_array_functor.sum_weight = (sum_weight);
			mkt::map_in_place<double, CalcBarycenterMap_map_in_place_array_functor>(barycenter_copy, calcBarycenterMap_map_in_place_array_functor);
			volitiveMovement_map_in_place_array_functor.step_size = (step_size_vol);volitiveMovement_map_in_place_array_functor.sum_weight = (sum_weight);volitiveMovement_map_in_place_array_functor.sum_weight_last_iteration = (sum_weight_last_iteration);
			mkt::map_in_place<Fish, VolitiveMovement_map_in_place_array_functor>(population, volitiveMovement_map_in_place_array_functor);
			sum_weight_last_iteration = (sum_weight);
		}
		double global_best_fitness = 0.0;
		global_best_fitness = mkt::map_reduce_max<Fish, double, Lambda69_map_reduce_array_functor>(population, lambda69_map_reduce_array_functor);
		for(int gpu = 0; gpu < 1; ++gpu){
			acc_set_device_num(gpu, acc_device_not_host);
			acc_wait_all();
		}
		std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
		double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
		printf("Best solution: %.5f\n",(global_best_fitness));
		
		printf("Execution time: %.5fs\n", seconds);
		printf("Threads: %i\n", omp_get_max_threads());
		printf("Processes: %i\n", 1);
		
		#pragma omp parallel for
		for(int gpu = 0; gpu < 1; ++gpu){
			acc_set_device_num(gpu, acc_device_not_host);
			acc_free(rns_pointers[gpu]);
		}
		return EXIT_SUCCESS;
		}
