
#include <array>
#include <sstream>
#include <chrono>
#include <random>
#include <limits>
#include <memory>
#include "../include/fss-n-1-c-1.hpp"

size_t tmp_size_t = 0;


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
const int NUMBER_OF_FISH = 4096;
const int ITERATIONS = 5000;
const int DIMENSIONS = 512;
std::vector<Fish> population(4096);
std::vector<double> instinctive_movement_vector_copy(512);
std::vector<Fish> weighted_fishes(4096);
std::vector<double> barycenter_copy(512);

Fish::Fish() : position(512, 0.0), fitness(), candidate_position(512, 0.0), candidate_fitness(), displacement(512, 0.0), fitness_variation(), weight(), best_position(512, 0.0), best_fitness() {}

int main(int argc, char** argv) {
	
	printf("Run Fss-n-1-c-1\n\n");			
	
	
	std::vector<std::mt19937> random_engines;
	random_engines.reserve(1);
	std::random_device rd;
	for(size_t counter = 0; counter < 1; ++counter){
		random_engines.push_back(std::mt19937(rd()));
	}
	
	std::vector<std::uniform_real_distribution<double>> rand_dist_double_INIT_LOWER_BOUND_INIT_UPPER_BOUND;
							rand_dist_double_INIT_LOWER_BOUND_INIT_UPPER_BOUND.reserve(1);
							for(size_t counter = 0; counter < 1; ++counter){
								rand_dist_double_INIT_LOWER_BOUND_INIT_UPPER_BOUND.push_back(std::uniform_real_distribution<double>((INIT_LOWER_BOUND), (INIT_UPPER_BOUND)));
							}std::vector<std::uniform_real_distribution<double>> rand_dist_double_minus1_0_1_0;
							rand_dist_double_minus1_0_1_0.reserve(1);
							for(size_t counter = 0; counter < 1; ++counter){
								rand_dist_double_minus1_0_1_0.push_back(std::uniform_real_distribution<double>(-(1.0), 1.0));
							}std::vector<std::uniform_real_distribution<double>> rand_dist_double_0_0_1_0;
							rand_dist_double_0_0_1_0.reserve(1);
							for(size_t counter = 0; counter < 1; ++counter){
								rand_dist_double_0_0_1_0.push_back(std::uniform_real_distribution<double>(0.0, 1.0));
							}
	
	
	for(size_t counter = 0; counter  < 512; ++counter){
		instinctive_movement_vector_copy[counter] = 0;
	}
	
	for(size_t counter = 0; counter  < 512; ++counter){
		barycenter_copy[counter] = 0;
	}
	
	
	std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
	#pragma omp simd
	for(size_t counter = 0; counter < 4096; ++counter){
		
		population[counter].fitness = std::numeric_limits<double>::lowest();
		population[counter].candidate_fitness = std::numeric_limits<double>::lowest();
		population[counter].weight = (WEIGHT_LOWER_BOUND);
		population[counter].fitness_variation = 0.0;
		population[counter].best_fitness = std::numeric_limits<double>::lowest();
		for(int i = 0; ((i) < (DIMENSIONS)); ++i){
			population[counter].position[(i)] = rand_dist_double_INIT_LOWER_BOUND_INIT_UPPER_BOUND[0](random_engines[0]);
			population[counter].candidate_position[(i)] = 0.0;
			population[counter].displacement[(i)] = 0.0;
			population[counter].best_position[(i)] = 0.0;
		}
	}
	double step_size  = (STEP_SIZE_INITIAL);
	double step_size_vol  = (STEP_SIZE_VOLITIVE_INITIAL);
	double sum_weight_last_iteration = 0.0;
	sum_weight_last_iteration = 0.0;
	
	
	
	#pragma omp simd reduction(sumWeight:sum_weight_last_iteration)
	for(size_t counter = 0; counter < 4096; ++counter){
		double map_fold_tmp;
		
		map_fold_tmp = (population[counter]).weight;
	
		
		sum_weight_last_iteration = ((sum_weight_last_iteration) + (map_fold_tmp));
	}		
	
	for(int iteration = 0; ((iteration) < (ITERATIONS)); ++iteration){
		#pragma omp simd
		for(size_t counter = 0; counter < 4096; ++counter){
			
			double sum = 0.0;
			for(int j = 0; ((j) < (DIMENSIONS)); ++j){
				double value = (population[counter]).position[(j)];
				sum += (std::pow((value), 2) - (10 * std::cos(((2 * (PI)) * (value)))));
			}
			population[counter].fitness = -(((10 * (DIMENSIONS)) + (sum)));
			
			if(((population[counter]).fitness > (population[counter]).best_fitness)){
			population[counter].best_fitness = (population[counter]).fitness;
			for(int k = 0; ((k) < (DIMENSIONS)); ++k){
				population[counter].best_position[(k)] = (population[counter]).position[(k)];
			}
			}
		}
		if(((iteration) > 0)){
			step_size = ((step_size) - (((STEP_SIZE_INITIAL) - (STEP_SIZE_FINAL)) / static_cast<double>(((ITERATIONS) - 1))));
			step_size_vol = ((step_size_vol) - (((STEP_SIZE_VOLITIVE_INITIAL) - (STEP_SIZE_VOLITIVE_FINAL)) / static_cast<double>(((ITERATIONS) - 1))));
		}
		#pragma omp simd
		for(size_t counter = 0; counter < 4096; ++counter){
			
			for(int i = 0; ((i) < (DIMENSIONS)); ++i){
				double rand_factor = rand_dist_double_minus1_0_1_0[0](random_engines[0]);
				double direction = (((rand_factor) * ((step_size))) * ((UPPER_BOUND) - (LOWER_BOUND)));
				double new_value = ((population[counter]).position[(i)] + (direction));
				
				if(((new_value) < (LOWER_BOUND))){
				new_value = (LOWER_BOUND);
				} else 
				if(((new_value) > (UPPER_BOUND))){
				new_value = (UPPER_BOUND);
				}
				population[counter].candidate_position[(i)] = (new_value);
			}
			double sum = 0.0;
			for(int j = 0; ((j) < (DIMENSIONS)); ++j){
				double value = (population[counter]).candidate_position[(j)];
				sum += (std::pow((value), 2) - (10 * std::cos(((2 * (PI)) * (value)))));
			}
			population[counter].candidate_fitness = -(((10 * (DIMENSIONS)) + (sum)));
			
			if(((population[counter]).candidate_fitness > (population[counter]).fitness)){
			population[counter].fitness_variation = ((population[counter]).candidate_fitness - (population[counter]).fitness);
			population[counter].fitness = (population[counter]).candidate_fitness;
			for(int k = 0; ((k) < (DIMENSIONS)); ++k){
				population[counter].displacement[(k)] = ((population[counter]).candidate_position[(k)] - (population[counter]).position[(k)]);
				population[counter].position[(k)] = (population[counter]).candidate_position[(k)];
			}
			
			if(((population[counter]).fitness > (population[counter]).best_fitness)){
			population[counter].best_fitness = (population[counter]).fitness;
			for(int k = 0; ((k) < (DIMENSIONS)); ++k){
				population[counter].best_position[(k)] = (population[counter]).position[(k)];
			}
			}
			}
			 else {
				population[counter].fitness_variation = 0.0;
				for(int k = 0; ((k) < (DIMENSIONS)); ++k){
					population[counter].displacement[(k)] = 0.0;
				}
			}
		}
		double max_fitness_variation = 0.0;
		max_fitness_variation = 0.0;
		
		
		
		#pragma omp simd reduction(maxFitnessVariation:max_fitness_variation)
		for(size_t counter = 0; counter < 4096; ++counter){
			double map_fold_tmp;
			
			map_fold_tmp = (population[counter]).fitness_variation;
		
			
			double result;
			
			if(((max_fitness_variation) > (map_fold_tmp))){
			result = (max_fitness_variation);
			}
			 else {
				result = (map_fold_tmp);
			}
			max_fitness_variation = (result);
		}		
		
		#pragma omp simd
		for(size_t counter = 0; counter < 4096; ++counter){
			
			
			if((((max_fitness_variation)) != 0.0)){
			double result = ((population[counter]).weight + ((population[counter]).fitness_variation / ((max_fitness_variation))));
			
			if(((result) > (WEIGHT_UPPER_BOUND))){
			result = (WEIGHT_UPPER_BOUND);
			} else 
			if(((result) < (WEIGHT_LOWER_BOUND))){
			result = (WEIGHT_LOWER_BOUND);
			}
			population[counter].weight = (result);
			}
		}
		double sum_fitness_variation = 0.0;
		sum_fitness_variation = 0.0;
		
		
		
		#pragma omp simd reduction(sumFitnessVariation:sum_fitness_variation)
		for(size_t counter = 0; counter < 4096; ++counter){
			double map_fold_tmp;
			
			map_fold_tmp = (population[counter]).fitness_variation;
		
			
			sum_fitness_variation = ((sum_fitness_variation) + (map_fold_tmp));
		}		
		
		#pragma omp simd
		for(size_t counter = 0; counter < 4096; ++counter){
			
			for(int i = 0; ((i) < (DIMENSIONS)); ++i){
				population[counter].displacement[(i)] *= (population[counter]).fitness_variation;
			}
		}
		instinctive_movement_vector_copy.assign(512, 0.0);
		
		
		
		#pragma omp simd reduction(calcDisplacementFold:instinctive_movement_vector_copy)
		for(size_t counter = 0; counter < 4096; ++counter){
			std::array<double,0> map_fold_tmp;
			
			map_fold_tmp = (population[counter]).displacement;
		
			
			for(int i = 0; ((i) < (DIMENSIONS)); ++i){
				instinctive_movement_vector_copy[(i)] += (map_fold_tmp)[(i)];
			}
		}		
		
		#pragma omp simd
		for(size_t counter = 0; counter < 512; ++counter){
			
			double result = (instinctive_movement_vector_copy[counter]);
			
			if((((sum_fitness_variation)) != 0.0)){
			result = ((instinctive_movement_vector_copy[counter]) / ((sum_fitness_variation)));
			}
			instinctive_movement_vector_copy[counter] = (result);
		}
		#pragma omp simd
		for(size_t counter = 0; counter < 4096; ++counter){
			
			for(int i = 0; ((i) < (DIMENSIONS)); ++i){
				double new_position = ((population[counter]).position[(i)] + (instinctive_movement_vector_copy)[(i)]);
				
				if(((new_position) < (LOWER_BOUND))){
				new_position = (LOWER_BOUND);
				} else 
				if(((new_position) > (UPPER_BOUND))){
				new_position = (UPPER_BOUND);
				}
				population[counter].position[(i)] = (new_position);
			}
		}
		double sum_weight = 0.0;
		sum_weight = 0.0;
		
		
		
		#pragma omp simd reduction(sumWeight:sum_weight)
		for(size_t counter = 0; counter < 4096; ++counter){
			double map_fold_tmp;
			
			map_fold_tmp = (population[counter]).weight;
		
			
			sum_weight = ((sum_weight) + (map_fold_tmp));
		}		
		
				#pragma omp simd
				for(size_t counter = 0; counter < 4096; ++counter){
					Fish map_input = population[counter];
					
					for(int i = 0; ((i) < (DIMENSIONS)); ++i){
						map_input.position[(i)] *= (map_input).weight;
					}
					weighted_fishes[counter] = (map_input);
				}
		barycenter_copy.assign(512, 0.0);
		
		
		
		#pragma omp simd reduction(calcBarycenterFold:barycenter_copy)
		for(size_t counter = 0; counter < 4096; ++counter){
			std::array<double,0> map_fold_tmp;
			
			map_fold_tmp = (weighted_fishes[counter]).position;
		
			
			for(int i = 0; ((i) < (DIMENSIONS)); ++i){
				barycenter_copy[(i)] += (map_fold_tmp)[(i)];
			}
		}		
		
		#pragma omp simd
		for(size_t counter = 0; counter < 512; ++counter){
			
			double result = (barycenter_copy[counter]);
			
			if((((sum_weight)) != 0)){
			result = ((barycenter_copy[counter]) / ((sum_weight)));
			}
			barycenter_copy[counter] = (result);
		}
		#pragma omp simd
		for(size_t counter = 0; counter < 4096; ++counter){
			
			double distance = 0.0;
			for(int i = 0; ((i) < (DIMENSIONS)); ++i){
				distance += (((population[counter]).position[(i)] - (barycenter_copy)[(i)]) * ((population[counter]).position[(i)] - (barycenter_copy)[(i)]));
			}
			distance = std::sqrt((distance));
			
			if(((distance) != 0.0)){
			double rand_factor = rand_dist_double_0_0_1_0[0](random_engines[0]);
			for(int i = 0; ((i) < (DIMENSIONS)); ++i){
				double direction = ((((rand_factor) * ((step_size_vol))) * ((UPPER_BOUND) - (LOWER_BOUND))) * (((population[counter]).position[(i)] - (barycenter_copy)[(i)]) / (distance)));
				double new_position = (population[counter]).position[(i)];
				
				if((((sum_weight)) > ((sum_weight_last_iteration)))){
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
				population[counter].position[(i)] = (new_position);
			}
			}
		}
		sum_weight_last_iteration = (sum_weight);
	}
	double global_best_fitness = 0.0;
	global_best_fitness = std::numeric_limits<double>::lowest();
	
	
	
	#pragma omp simd reduction(getBestSolution:global_best_fitness)
	for(size_t counter = 0; counter < 4096; ++counter){
		double map_fold_tmp;
		
		map_fold_tmp = (population[counter]).best_fitness;
	
		
		double result = (global_best_fitness);
		
		if(((map_fold_tmp) > (global_best_fitness))){
		result = (map_fold_tmp);
		}
		global_best_fitness = (result);
	}		
	
	std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
	double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
		printf("Best solution: %.5f\n",(global_best_fitness));
	
	printf("Execution time: %.5fs\n", seconds);
	printf("Threads: %i\n", 1);
	printf("Processes: %i\n", 1);
	
	return EXIT_SUCCESS;
}
