#include <mpi.h>

#include <omp.h>
#include <array>
#include <sstream>
#include <chrono>
#include <random>
#include <limits>
#include <memory>
#include "../include/fss.hpp"

const size_t number_of_processes = 4;
int process_id = -1;
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
const int NUMBER_OF_FISH = 128;
const int ITERATIONS = 20;
const int DIMENSIONS = 128;
std::vector<Fish> population(32);
std::vector<double> instinctive_movement_vector_copy(128);
std::vector<Fish> weighted_fishes(32);
std::vector<double> barycenter_copy(128);

Fish::Fish() : position(128, 0.0), fitness(), candidate_position(128, 0.0), candidate_fitness(), displacement(128, 0.0), fitness_variation(), weight(), best_position(128, 0.0), best_fitness() {}

void sumWeight(void *in, void *inout, int *len, MPI_Datatype *dptr){
	double* inv = static_cast<double*>(in);
	double* inoutv = static_cast<double*>(inout);
	
	*inoutv = ((*inoutv) + (*inv));
} 
void maxFitnessVariation(void *in, void *inout, int *len, MPI_Datatype *dptr){
	double* inv = static_cast<double*>(in);
	double* inoutv = static_cast<double*>(inout);
	
	double result;
	
	if(((*inoutv) > (*inv))){
	result = (*inoutv);
	}
	 else {
		result = (*inv);
	}
	*inoutv = (result);
} 
void sumFitnessVariation(void *in, void *inout, int *len, MPI_Datatype *dptr){
	double* inv = static_cast<double*>(in);
	double* inoutv = static_cast<double*>(inout);
	
	*inoutv = ((*inoutv) + (*inv));
} 
void calcDisplacementFold(void *in, void *inout, int *len, MPI_Datatype *dptr){
	double* inv = static_cast<double*>(in);
	double* inoutv = static_cast<double*>(inout);
	
	for(int i = 0; ((i) < (DIMENSIONS)); ++i){
		inoutv[(i)] += (inv)[(i)];
	}
} 
void calcBarycenterFold(void *in, void *inout, int *len, MPI_Datatype *dptr){
	double* inv = static_cast<double*>(in);
	double* inoutv = static_cast<double*>(inout);
	
	for(int i = 0; ((i) < (DIMENSIONS)); ++i){
		inoutv[(i)] += (inv)[(i)];
	}
} 
void getBestSolution(void *in, void *inout, int *len, MPI_Datatype *dptr){
	double* inv = static_cast<double*>(in);
	double* inoutv = static_cast<double*>(inout);
	
	double result = (*inoutv);
	
	if(((*inv) > (*inoutv))){
	result = (*inv);
	}
	*inoutv = (result);
} 
int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	
	int mpi_world_size = 0;
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
	
	if(mpi_world_size != number_of_processes){
		MPI_Finalize();
		return EXIT_FAILURE;
	}
	
	MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
	
	if(process_id == 0){
	printf("Run Fss\n\n");			
	}
	
	
	std::vector<std::mt19937> random_engines;
	random_engines.reserve(24);
	std::random_device rd;
	for(size_t counter = 0; counter < 24; ++counter){
		random_engines.push_back(std::mt19937(rd()));
	}
	
	std::vector<std::uniform_real_distribution<double>> rand_dist_double_INIT_LOWER_BOUND_INIT_UPPER_BOUND;
							rand_dist_double_INIT_LOWER_BOUND_INIT_UPPER_BOUND.reserve(24);
							for(size_t counter = 0; counter < 24; ++counter){
								rand_dist_double_INIT_LOWER_BOUND_INIT_UPPER_BOUND.push_back(std::uniform_real_distribution<double>((INIT_LOWER_BOUND), (INIT_UPPER_BOUND)));
							}std::vector<std::uniform_real_distribution<double>> rand_dist_double_minus1_0_1_0;
							rand_dist_double_minus1_0_1_0.reserve(24);
							for(size_t counter = 0; counter < 24; ++counter){
								rand_dist_double_minus1_0_1_0.push_back(std::uniform_real_distribution<double>(-(1.0), 1.0));
							}std::vector<std::uniform_real_distribution<double>> rand_dist_double_0_0_1_0;
							rand_dist_double_0_0_1_0.reserve(24);
							for(size_t counter = 0; counter < 24; ++counter){
								rand_dist_double_0_0_1_0.push_back(std::uniform_real_distribution<double>(0.0, 1.0));
							}
	
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter  < 128; ++counter){
		instinctive_movement_vector_copy[counter] = 0;
	}
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter  < 128; ++counter){
		barycenter_copy[counter] = 0;
	}
	#pragma omp declare reduction(sumWeight : double : omp_out = [&](){return ((omp_out) + (omp_in));}()) initializer(omp_priv = omp_orig)
	#pragma omp declare reduction(maxFitnessVariation : double : omp_out = [&](){double result;if(((omp_out) > (omp_in))){result = (omp_out);} else {	result = (omp_in);}return (result);}()) initializer(omp_priv = omp_orig)
	#pragma omp declare reduction(sumFitnessVariation : double : omp_out = [&](){return ((omp_out) + (omp_in));}()) initializer(omp_priv = omp_orig)
	#pragma omp declare reduction(calcDisplacementFold : std::array<double,0> : omp_out = [&](){for(int i = 0; ((i) < (DIMENSIONS)); ++i){	omp_out[(i)] += (omp_in)[(i)];}return (omp_out);}()) initializer(omp_priv = omp_orig)
	#pragma omp declare reduction(calcBarycenterFold : std::array<double,0> : omp_out = [&](){for(int i = 0; ((i) < (DIMENSIONS)); ++i){	omp_out[(i)] += (omp_in)[(i)];}return (omp_out);}()) initializer(omp_priv = omp_orig)
	#pragma omp declare reduction(getBestSolution : double : omp_out = [&](){double result = (omp_out);if(((omp_in) > (omp_out))){result = (omp_in);}return (result);}()) initializer(omp_priv = omp_orig)
	
	MPI_Op sumWeight_mpi_op;
	MPI_Op_create( sumWeight, 0, &sumWeight_mpi_op );
	MPI_Op maxFitnessVariation_mpi_op;
	MPI_Op_create( maxFitnessVariation, 0, &maxFitnessVariation_mpi_op );
	MPI_Op sumFitnessVariation_mpi_op;
	MPI_Op_create( sumFitnessVariation, 0, &sumFitnessVariation_mpi_op );
	MPI_Op calcDisplacementFold_mpi_op;
	MPI_Op_create( calcDisplacementFold, 0, &calcDisplacementFold_mpi_op );
	MPI_Op calcBarycenterFold_mpi_op;
	MPI_Op_create( calcBarycenterFold, 0, &calcBarycenterFold_mpi_op );
	MPI_Op getBestSolution_mpi_op;
	MPI_Op_create( getBestSolution, 0, &getBestSolution_mpi_op );
	double fold_result_double;std::array<double,0> fold_result_std_array_double_0_;
	
	std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 32; ++counter){
		
		population[counter].fitness = std::numeric_limits<double>::lowest();
		population[counter].candidate_fitness = std::numeric_limits<double>::lowest();
		population[counter].weight = (WEIGHT_LOWER_BOUND);
		population[counter].fitness_variation = 0.0;
		population[counter].best_fitness = std::numeric_limits<double>::lowest();
		for(int i = 0; ((i) < (DIMENSIONS)); ++i){
			population[counter].position[(i)] = rand_dist_double_INIT_LOWER_BOUND_INIT_UPPER_BOUND[omp_get_thread_num()](random_engines[omp_get_thread_num()]);
			population[counter].candidate_position[(i)] = 0.0;
			population[counter].displacement[(i)] = 0.0;
			population[counter].best_position[(i)] = 0.0;
		}
	}
	double step_size  = (STEP_SIZE_INITIAL);
	double step_size_vol  = (STEP_SIZE_VOLITIVE_INITIAL);
	double sum_weight_last_iteration = 0.0;
	fold_result_double = 0.0;
	
	
	
	#pragma omp parallel for simd reduction(sumWeight:fold_result_double)
	for(size_t counter = 0; counter < 32; ++counter){
		double map_fold_tmp;
		
		map_fold_tmp = (population[counter]).weight;
	
		
		fold_result_double = ((fold_result_double) + (map_fold_tmp));
	}		
	
	MPI_Allreduce(&fold_result_double, &sum_weight_last_iteration, sizeof(double), MPI_BYTE, sumWeight_mpi_op, MPI_COMM_WORLD); 
	for(int iteration = 0; ((iteration) < (ITERATIONS)); ++iteration){
		#pragma omp parallel for simd
		for(size_t counter = 0; counter < 32; ++counter){
			
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
		#pragma omp parallel for simd
		for(size_t counter = 0; counter < 32; ++counter){
			
			for(int i = 0; ((i) < (DIMENSIONS)); ++i){
				double rand_factor = rand_dist_double_minus1_0_1_0[omp_get_thread_num()](random_engines[omp_get_thread_num()]);
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
		fold_result_double = 0.0;
		
		
		
		#pragma omp parallel for simd reduction(maxFitnessVariation:fold_result_double)
		for(size_t counter = 0; counter < 32; ++counter){
			double map_fold_tmp;
			
			map_fold_tmp = (population[counter]).fitness_variation;
		
			
			double result;
			
			if(((fold_result_double) > (map_fold_tmp))){
			result = (fold_result_double);
			}
			 else {
				result = (map_fold_tmp);
			}
			fold_result_double = (result);
		}		
		
		MPI_Allreduce(&fold_result_double, &max_fitness_variation, sizeof(double), MPI_BYTE, maxFitnessVariation_mpi_op, MPI_COMM_WORLD); 
		#pragma omp parallel for simd
		for(size_t counter = 0; counter < 32; ++counter){
			
			
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
		fold_result_double = 0.0;
		
		
		
		#pragma omp parallel for simd reduction(sumFitnessVariation:fold_result_double)
		for(size_t counter = 0; counter < 32; ++counter){
			double map_fold_tmp;
			
			map_fold_tmp = (population[counter]).fitness_variation;
		
			
			fold_result_double = ((fold_result_double) + (map_fold_tmp));
		}		
		
		MPI_Allreduce(&fold_result_double, &sum_fitness_variation, sizeof(double), MPI_BYTE, sumFitnessVariation_mpi_op, MPI_COMM_WORLD); 
		#pragma omp parallel for simd
		for(size_t counter = 0; counter < 32; ++counter){
			
			for(int i = 0; ((i) < (DIMENSIONS)); ++i){
				population[counter].displacement[(i)] *= (population[counter]).fitness_variation;
			}
		}
		fold_result_std_array_double_0_.assign(128, 0.0);
		
		
		
		#pragma omp parallel for simd reduction(calcDisplacementFold:fold_result_std_array_double_0_)
		for(size_t counter = 0; counter < 32; ++counter){
			std::array<double,0> map_fold_tmp;
			
			map_fold_tmp = (population[counter]).displacement;
		
			
			for(int i = 0; ((i) < (DIMENSIONS)); ++i){
				fold_result_std_array_double_0_[(i)] += (map_fold_tmp)[(i)];
			}
		}		
		
		MPI_Allreduce(fold_result_std_array_double_0_.data(), instinctive_movement_vector_copy.data(), 128 * sizeof(double), MPI_BYTE, calcDisplacementFold_mpi_op, MPI_COMM_WORLD); 
		#pragma omp parallel for simd
		for(size_t counter = 0; counter < 128; ++counter){
			
			double result = (instinctive_movement_vector_copy[counter]);
			
			if((((sum_fitness_variation)) != 0.0)){
			result = ((instinctive_movement_vector_copy[counter]) / ((sum_fitness_variation)));
			}
			instinctive_movement_vector_copy[counter] = (result);
		}
		#pragma omp parallel for simd
		for(size_t counter = 0; counter < 32; ++counter){
			
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
		fold_result_double = 0.0;
		
		
		
		#pragma omp parallel for simd reduction(sumWeight:fold_result_double)
		for(size_t counter = 0; counter < 32; ++counter){
			double map_fold_tmp;
			
			map_fold_tmp = (population[counter]).weight;
		
			
			fold_result_double = ((fold_result_double) + (map_fold_tmp));
		}		
		
		MPI_Allreduce(&fold_result_double, &sum_weight, sizeof(double), MPI_BYTE, sumWeight_mpi_op, MPI_COMM_WORLD); 
				#pragma omp parallel for simd
				for(size_t counter = 0; counter < 32; ++counter){
					Fish map_input = population[counter];
					
					for(int i = 0; ((i) < (DIMENSIONS)); ++i){
						map_input.position[(i)] *= (map_input).weight;
					}
					weighted_fishes[counter] = (map_input);
				}
		fold_result_std_array_double_0_.assign(128, 0.0);
		
		
		
		#pragma omp parallel for simd reduction(calcBarycenterFold:fold_result_std_array_double_0_)
		for(size_t counter = 0; counter < 32; ++counter){
			std::array<double,0> map_fold_tmp;
			
			map_fold_tmp = (weighted_fishes[counter]).position;
		
			
			for(int i = 0; ((i) < (DIMENSIONS)); ++i){
				fold_result_std_array_double_0_[(i)] += (map_fold_tmp)[(i)];
			}
		}		
		
		MPI_Allreduce(fold_result_std_array_double_0_.data(), barycenter_copy.data(), 128 * sizeof(double), MPI_BYTE, calcBarycenterFold_mpi_op, MPI_COMM_WORLD); 
		#pragma omp parallel for simd
		for(size_t counter = 0; counter < 128; ++counter){
			
			double result = (barycenter_copy[counter]);
			
			if((((sum_weight)) != 0)){
			result = ((barycenter_copy[counter]) / ((sum_weight)));
			}
			barycenter_copy[counter] = (result);
		}
		#pragma omp parallel for simd
		for(size_t counter = 0; counter < 32; ++counter){
			
			double distance = 0.0;
			for(int i = 0; ((i) < (DIMENSIONS)); ++i){
				distance += (((population[counter]).position[(i)] - (barycenter_copy)[(i)]) * ((population[counter]).position[(i)] - (barycenter_copy)[(i)]));
			}
			distance = std::sqrt((distance));
			
			if(((distance) != 0.0)){
			double rand_factor = rand_dist_double_0_0_1_0[omp_get_thread_num()](random_engines[omp_get_thread_num()]);
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
	fold_result_double = std::numeric_limits<double>::lowest();
	
	
	
	#pragma omp parallel for simd reduction(getBestSolution:fold_result_double)
	for(size_t counter = 0; counter < 32; ++counter){
		double map_fold_tmp;
		
		map_fold_tmp = (population[counter]).best_fitness;
	
		
		double result = (fold_result_double);
		
		if(((map_fold_tmp) > (fold_result_double))){
		result = (map_fold_tmp);
		}
		fold_result_double = (result);
	}		
	
	MPI_Allreduce(&fold_result_double, &global_best_fitness, sizeof(double), MPI_BYTE, getBestSolution_mpi_op, MPI_COMM_WORLD); 
	std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
	double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
	if(process_id == 0){
		printf("Best solution: %.5f\n",(global_best_fitness));
	}
	
	if(process_id == 0){
	printf("Execution time: %.5fs\n", seconds);
	printf("Threads: %i\n", omp_get_max_threads());
	printf("Processes: %i\n", mpi_world_size);
	}
	
	MPI_Finalize();
	return EXIT_SUCCESS;
}
