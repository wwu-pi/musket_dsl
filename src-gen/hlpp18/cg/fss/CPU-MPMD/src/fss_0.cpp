#include <mpi.h>

#include <omp.h>
#include <array>
#include <vector>
#include <sstream>
#include <chrono>
#include <random>
#include <limits>
#include <memory>
#include <cstddef>
#include "../include/fss_0.hpp"

const size_t number_of_processes = 4;
const size_t process_id = 0;
int mpi_rank = -1;
int mpi_world_size = 0;

std::vector<std::mt19937> random_engines;
std::vector<std::uniform_real_distribution<double>> rand_dist_double_INIT_LOWER_BOUND_INIT_UPPER_BOUND;std::vector<std::uniform_real_distribution<double>> rand_dist_double_minus1_0_1_0;std::vector<std::uniform_real_distribution<double>> rand_dist_double_0_0_1_0;

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
const int ITERATIONS = 5;
const int DIMENSIONS = 512;
std::vector<Fish> population(32);
std::vector<double> instinctive_movement_vector_copy(512);
std::vector<Fish> weighted_fishes(32);
std::vector<double> barycenter_copy(512);

//Fish::Fish() : position(512, 0.0), fitness(), candidate_position(512, 0.0), candidate_fitness(), displacement(512, 0.0), fitness_variation(), weight(), best_position(512, 0.0), best_fitness() {}


// generate Function
inline auto sumWeight_function(double sum_weight, double fishWeight){
	return ((sum_weight) + (fishWeight));
}
// generate Function
inline auto maxFitnessVariation_function(double max_fitness_variation, double fitness_variation){
	double result;
	
	if(((max_fitness_variation) > (fitness_variation))){
	result = (max_fitness_variation);
	}
	 else {
			result = (fitness_variation);
		}
	return (result);
}
// generate Function
inline auto sumFitnessVariation_function(double sum_fitness_variation, double fitness_variation){
	return ((sum_fitness_variation) + (fitness_variation));
}
// generate Function
inline auto calcDisplacementFold_function(std::array<double,512> arr, std::array<double,512> displacement){
	for(int i = 0; ((i) < (DIMENSIONS)); ++i){
		arr[(i)] += (displacement)[(i)];
	}
	return (arr);
}
// generate Function
inline auto calcBarycenterFold_function(std::array<double,512> arr, std::array<double,512> position){
	for(int i = 0; ((i) < (DIMENSIONS)); ++i){
		arr[(i)] += (position)[(i)];
	}
	return (arr);
}
// generate Function
inline auto getBestSolution_function(double best_solution, double best_fitness){
	double result = (best_solution);
	
	if(((best_fitness) > (best_solution))){
	result = (best_fitness);
	}
	return (result);
}

struct InitFish_functor{
	auto operator()(Fish fi) const{
		fi.fitness = std::numeric_limits<double>::lowest();
		fi.candidate_fitness = std::numeric_limits<double>::lowest();
		fi.weight = (WEIGHT_LOWER_BOUND);
		fi.fitness_variation = 0.0;
		fi.best_fitness = std::numeric_limits<double>::lowest();
		for(int i = 0; ((i) < (DIMENSIONS)); ++i){
			fi.position[(i)] = rand_dist_double_INIT_LOWER_BOUND_INIT_UPPER_BOUND[omp_get_thread_num()](random_engines[omp_get_thread_num()]);
			fi.candidate_position[(i)] = 0.0;
			fi.displacement[(i)] = 0.0;
			fi.best_position[(i)] = 0.0;
		}
		return (fi);
	}
};
struct EvaluateFitness_functor{
	auto operator()(Fish fi) const{
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
		return (fi);
	}
};
struct IndividualMovement_functor{
	auto operator()(double step_size, Fish fi) const{
		for(int i = 0; ((i) < (DIMENSIONS)); ++i){
			double rand_factor = rand_dist_double_minus1_0_1_0[omp_get_thread_num()](random_engines[omp_get_thread_num()]);
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
		return (fi);
	}
};
struct Feeding_functor{
	auto operator()(double max_fitness_variation, Fish fi) const{
		
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
		return (fi);
	}
};
struct MaxFitnessVariation_functor{
	auto operator()(double max_fitness_variation, double fitness_variation) const{
		double result;
		
		if(((max_fitness_variation) > (fitness_variation))){
		result = (max_fitness_variation);
		}
		 else {
				result = (fitness_variation);
			}
		return (result);
	}
};
struct SumFitnessVariation_functor{
	auto operator()(double sum_fitness_variation, double fitness_variation) const{
		return ((sum_fitness_variation) + (fitness_variation));
	}
};
struct CalcDisplacementMap_functor{
	auto operator()(Fish fi) const{
		for(int i = 0; ((i) < (DIMENSIONS)); ++i){
			fi.displacement[(i)] *= (fi).fitness_variation;
		}
		return (fi);
	}
};
struct CalcDisplacementFold_functor{
	auto operator()(std::array<double,512> arr, std::array<double,512> displacement) const{
		for(int i = 0; ((i) < (DIMENSIONS)); ++i){
			arr[(i)] += (displacement)[(i)];
		}
		return (arr);
	}
};
struct CalcInstinctiveMovementVector_functor{
	auto operator()(double sum_fitness_variation, double x) const{
		double result = (x);
		
		if(((sum_fitness_variation) != 0.0)){
		result = ((x) / (sum_fitness_variation));
		}
		return (result);
	}
};
struct InstinctiveMovement_functor{
	auto operator()(Fish fi) const{
		for(int i = 0; ((i) < (DIMENSIONS)); ++i){
			double new_position = ((fi).position[(i)] + (instinctive_movement_vector_copy)[(i)]);
			
			if(((new_position) < (LOWER_BOUND))){
			new_position = (LOWER_BOUND);
			} else 
			if(((new_position) > (UPPER_BOUND))){
			new_position = (UPPER_BOUND);
			}
			fi.position[(i)] = (new_position);
		}
		return (fi);
	}
};
struct SumWeight_functor{
	auto operator()(double sum_weight, double fishWeight) const{
		return ((sum_weight) + (fishWeight));
	}
};
struct CalcWeightedFish_functor{
	auto operator()(Fish fi) const{
		for(int i = 0; ((i) < (DIMENSIONS)); ++i){
			fi.position[(i)] *= (fi).weight;
		}
		return (fi);
	}
};
struct CalcBarycenterFold_functor{
	auto operator()(std::array<double,512> arr, std::array<double,512> position) const{
		for(int i = 0; ((i) < (DIMENSIONS)); ++i){
			arr[(i)] += (position)[(i)];
		}
		return (arr);
	}
};
struct CalcBarycenterMap_functor{
	auto operator()(double sum_weight, double x) const{
		double result = (x);
		
		if(((sum_weight) != 0)){
		result = ((x) / (sum_weight));
		}
		return (result);
	}
};
struct VolitiveMovement_functor{
	auto operator()(double step_size, double sum_weight, double sum_weight_last_iteration, Fish fi) const{
		double distance = 0.0;
		for(int i = 0; ((i) < (DIMENSIONS)); ++i){
			distance += (((fi).position[(i)] - (barycenter_copy)[(i)]) * ((fi).position[(i)] - (barycenter_copy)[(i)]));
		}
		distance = std::sqrt((distance));
		
		if(((distance) != 0.0)){
		double rand_factor = rand_dist_double_0_0_1_0[omp_get_thread_num()](random_engines[omp_get_thread_num()]);
		for(int i = 0; ((i) < (DIMENSIONS)); ++i){
			double direction = ((((rand_factor) * (step_size)) * ((UPPER_BOUND) - (LOWER_BOUND))) * (((fi).position[(i)] - (barycenter_copy)[(i)]) / (distance)));
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
		return (fi);
	}
};
struct GetBestSolution_functor{
	auto operator()(double best_solution, double best_fitness) const{
		double result = (best_solution);
		
		if(((best_fitness) > (best_solution))){
		result = (best_fitness);
		}
		return (result);
	}
};
struct Lambda0_functor{
	auto operator()(Fish fi) const{
		return (fi).weight;
	}
};
struct Lambda1_functor{
	auto operator()(Fish fi) const{
		return (fi).fitness_variation;
	}
};
struct Lambda2_functor{
	auto operator()(Fish fi) const{
		return (fi).fitness_variation;
	}
};
struct Lambda3_functor{
	auto operator()(Fish fi) const{
		return (fi).displacement;
	}
};
struct Lambda4_functor{
	auto operator()(Fish fi) const{
		return (fi).weight;
	}
};
struct Lambda5_functor{
	auto operator()(Fish fi) const{
		return (fi).position;
	}
};
struct Lambda6_functor{
	auto operator()(Fish fi) const{
		return (fi).best_fitness;
	}
};

void sumWeight(void* in, void* inout, int *len, MPI_Datatype *dptr){
	double* inv = static_cast<double*>(in);
	double* inoutv = static_cast<double*>(inout);
	*inoutv = sumWeight_function(*inv, *inoutv);
} 
void maxFitnessVariation(void* in, void* inout, int *len, MPI_Datatype *dptr){
	double* inv = static_cast<double*>(in);
	double* inoutv = static_cast<double*>(inout);
	*inoutv = maxFitnessVariation_function(*inv, *inoutv);
} 
void sumFitnessVariation(void* in, void* inout, int *len, MPI_Datatype *dptr){
	double* inv = static_cast<double*>(in);
	double* inoutv = static_cast<double*>(inout);
	*inoutv = sumFitnessVariation_function(*inv, *inoutv);
} 
void calcDisplacementFold(void* in, void* inout, int *len, MPI_Datatype *dptr){
	std::array<double,512>* inv = static_cast<std::array<double,512>*>(in);
	std::array<double,512>* inoutv = static_cast<std::array<double,512>*>(inout);
	*inoutv = calcDisplacementFold_function(*inv, *inoutv);
} 
void calcBarycenterFold(void* in, void* inout, int *len, MPI_Datatype *dptr){
	std::array<double,512>* inv = static_cast<std::array<double,512>*>(in);
	std::array<double,512>* inoutv = static_cast<std::array<double,512>*>(inout);
	*inoutv = calcBarycenterFold_function(*inv, *inoutv);
} 
void getBestSolution(void* in, void* inout, int *len, MPI_Datatype *dptr){
	double* inv = static_cast<double*>(in);
	double* inoutv = static_cast<double*>(inout);
	*inoutv = getBestSolution_function(*inv, *inoutv);
} 

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	
	if(mpi_world_size != number_of_processes || mpi_rank != process_id){
		MPI_Finalize();
		return EXIT_FAILURE;
	}			
	
	
	printf("Run Fss\n\n");			
	
	InitFish_functor initFish_functor{};
	EvaluateFitness_functor evaluateFitness_functor{};
	IndividualMovement_functor individualMovement_functor{};
	Feeding_functor feeding_functor{};
	MaxFitnessVariation_functor maxFitnessVariation_functor{};
	SumFitnessVariation_functor sumFitnessVariation_functor{};
	CalcDisplacementMap_functor calcDisplacementMap_functor{};
	CalcDisplacementFold_functor calcDisplacementFold_functor{};
	CalcInstinctiveMovementVector_functor calcInstinctiveMovementVector_functor{};
	InstinctiveMovement_functor instinctiveMovement_functor{};
	SumWeight_functor sumWeight_functor{};
	CalcWeightedFish_functor calcWeightedFish_functor{};
	CalcBarycenterFold_functor calcBarycenterFold_functor{};
	CalcBarycenterMap_functor calcBarycenterMap_functor{};
	VolitiveMovement_functor volitiveMovement_functor{};
	GetBestSolution_functor getBestSolution_functor{};
	Lambda0_functor lambda0_functor{};
	Lambda1_functor lambda1_functor{};
	Lambda2_functor lambda2_functor{};
	Lambda3_functor lambda3_functor{};
	Lambda4_functor lambda4_functor{};
	Lambda5_functor lambda5_functor{};
	Lambda6_functor lambda6_functor{};
	
	random_engines.reserve(24);
	std::random_device rd;
	for(size_t counter = 0; counter < 24; ++counter){
		random_engines.push_back(std::mt19937(rd()));
	}
	
	rand_dist_double_INIT_LOWER_BOUND_INIT_UPPER_BOUND.reserve(24);
	for(size_t counter = 0; counter < 24; ++counter){
		rand_dist_double_INIT_LOWER_BOUND_INIT_UPPER_BOUND.push_back(std::uniform_real_distribution<double>((INIT_LOWER_BOUND), (INIT_UPPER_BOUND)));
	}rand_dist_double_minus1_0_1_0.reserve(24);
	for(size_t counter = 0; counter < 24; ++counter){
		rand_dist_double_minus1_0_1_0.push_back(std::uniform_real_distribution<double>(-(1.0), 1.0));
	}rand_dist_double_0_0_1_0.reserve(24);
	for(size_t counter = 0; counter < 24; ++counter){
		rand_dist_double_0_0_1_0.push_back(std::uniform_real_distribution<double>(0.0, 1.0));
	}
	
	#pragma omp declare reduction(sumWeight_reduction : double : omp_out = sumWeight_function(omp_in, omp_out)) initializer(omp_priv = omp_orig)
	#pragma omp declare reduction(maxFitnessVariation_reduction : double : omp_out = maxFitnessVariation_function(omp_in, omp_out)) initializer(omp_priv = omp_orig)
	#pragma omp declare reduction(sumFitnessVariation_reduction : double : omp_out = sumFitnessVariation_function(omp_in, omp_out)) initializer(omp_priv = omp_orig)
	#pragma omp declare reduction(calcDisplacementFold_reduction : std::array<double,512> : omp_out = calcDisplacementFold_function(omp_in, omp_out)) initializer(omp_priv = omp_orig)
	#pragma omp declare reduction(calcBarycenterFold_reduction : std::array<double,512> : omp_out = calcBarycenterFold_function(omp_in, omp_out)) initializer(omp_priv = omp_orig)
	#pragma omp declare reduction(getBestSolution_reduction : double : omp_out = getBestSolution_function(omp_in, omp_out)) initializer(omp_priv = omp_orig)
	
	MPI_Datatype Fish_mpi_type_temp, Fish_mpi_type;
	MPI_Type_create_struct(9, (std::array<int,9>{1, 1, 1, 1, 1, 1, 1, 1, 1}).data(), (std::array<MPI_Aint,9>{static_cast<MPI_Aint>(offsetof(struct Fish, position)), static_cast<MPI_Aint>(offsetof(struct Fish, fitness)), static_cast<MPI_Aint>(offsetof(struct Fish, candidate_position)), static_cast<MPI_Aint>(offsetof(struct Fish, candidate_fitness)), static_cast<MPI_Aint>(offsetof(struct Fish, displacement)), static_cast<MPI_Aint>(offsetof(struct Fish, fitness_variation)), static_cast<MPI_Aint>(offsetof(struct Fish, weight)), static_cast<MPI_Aint>(offsetof(struct Fish, best_position)), static_cast<MPI_Aint>(offsetof(struct Fish, best_fitness))}).data(), (std::array<MPI_Datatype,9>{MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE}).data(), &Fish_mpi_type_temp);
	MPI_Type_create_resized(Fish_mpi_type_temp, 0, sizeof(Fish), &Fish_mpi_type);
	MPI_Type_free(&Fish_mpi_type_temp);
	MPI_Type_commit(&Fish_mpi_type);
	
	

	MPI_Op sumWeight_reduction_mpi_op;
	MPI_Op_create( sumWeight, 0, &sumWeight_reduction_mpi_op );
	MPI_Op maxFitnessVariation_reduction_mpi_op;
	MPI_Op_create( maxFitnessVariation, 0, &maxFitnessVariation_reduction_mpi_op );
	MPI_Op sumFitnessVariation_reduction_mpi_op;
	MPI_Op_create( sumFitnessVariation, 0, &sumFitnessVariation_reduction_mpi_op );
	MPI_Op calcDisplacementFold_reduction_mpi_op;
	MPI_Op_create( calcDisplacementFold, 0, &calcDisplacementFold_reduction_mpi_op );
	MPI_Op calcBarycenterFold_reduction_mpi_op;
	MPI_Op_create( calcBarycenterFold, 0, &calcBarycenterFold_reduction_mpi_op );
	MPI_Op getBestSolution_reduction_mpi_op;
	MPI_Op_create( getBestSolution, 0, &getBestSolution_reduction_mpi_op );
	double fold_result_double;std::array<double,512> fold_result_std_array_double_512_;
	
	
	
	std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 32; ++counter){
		population[counter] = initFish_functor(population[counter]);
	}
	double step_size  = (STEP_SIZE_INITIAL);
	double step_size_vol  = (STEP_SIZE_VOLITIVE_INITIAL);
	double sum_weight_last_iteration = 0.0;
	fold_result_double = 0.0;
	
	
	
	#pragma omp parallel for simd reduction(sumWeight_reduction:fold_result_double)
	for(size_t counter = 0; counter < 32; ++counter){
		double map_fold_tmp = lambda0_functor(population[counter]);
	
		fold_result_double = sumWeight_functor(fold_result_double, map_fold_tmp);
	}		
	
	MPI_Allreduce(&fold_result_double, &sum_weight_last_iteration, 4, MPI_DOUBLE, sumWeight_reduction_mpi_op, MPI_COMM_WORLD); 
	for(int iteration = 0; ((iteration) < (ITERATIONS)); ++iteration){
		#pragma omp parallel for simd
		for(size_t counter = 0; counter < 32; ++counter){
			population[counter] = evaluateFitness_functor(population[counter]);
		}
		if(((iteration) > 0)){
			step_size = ((step_size) - (((STEP_SIZE_INITIAL) - (STEP_SIZE_FINAL)) / static_cast<double>(((ITERATIONS) - 1))));
			step_size_vol = ((step_size_vol) - (((STEP_SIZE_VOLITIVE_INITIAL) - (STEP_SIZE_VOLITIVE_FINAL)) / static_cast<double>(((ITERATIONS) - 1))));
		}
		#pragma omp parallel for simd
		for(size_t counter = 0; counter < 32; ++counter){
			population[counter] = individualMovement_functor((step_size), population[counter]);
		}
		double max_fitness_variation = 0.0;
		fold_result_double = 0.0;
		
		
		
		#pragma omp parallel for simd reduction(maxFitnessVariation_reduction:fold_result_double)
		for(size_t counter = 0; counter < 32; ++counter){
			double map_fold_tmp = lambda1_functor(population[counter]);
		
			fold_result_double = maxFitnessVariation_functor(fold_result_double, map_fold_tmp);
		}		
		
		MPI_Allreduce(&fold_result_double, &max_fitness_variation, 4, MPI_DOUBLE, maxFitnessVariation_reduction_mpi_op, MPI_COMM_WORLD); 
		#pragma omp parallel for simd
		for(size_t counter = 0; counter < 32; ++counter){
			population[counter] = feeding_functor((max_fitness_variation), population[counter]);
		}
		double sum_fitness_variation = 0.0;
		fold_result_double = 0.0;
		
		
		
		#pragma omp parallel for simd reduction(sumFitnessVariation_reduction:fold_result_double)
		for(size_t counter = 0; counter < 32; ++counter){
			double map_fold_tmp = lambda2_functor(population[counter]);
		
			fold_result_double = sumFitnessVariation_functor(fold_result_double, map_fold_tmp);
		}		
		
		MPI_Allreduce(&fold_result_double, &sum_fitness_variation, 4, MPI_DOUBLE, sumFitnessVariation_reduction_mpi_op, MPI_COMM_WORLD); 
		#pragma omp parallel for simd
		for(size_t counter = 0; counter < 32; ++counter){
			population[counter] = calcDisplacementMap_functor(population[counter]);
		}
		fold_result_std_array_double_512_.fill(0.0);
		
		
		
		#pragma omp parallel for simd reduction(calcDisplacementFold_reduction:fold_result_std_array_double_512_)
		for(size_t counter = 0; counter < 32; ++counter){
			std::array<double,512> map_fold_tmp = lambda3_functor(population[counter]);
		
			fold_result_std_array_double_512_ = calcDisplacementFold_functor(fold_result_std_array_double_512_, map_fold_tmp);
		}		
		
		MPI_Allreduce(fold_result_std_array_double_512_.data(), instinctive_movement_vector_copy.data(), 512, MPI_DOUBLE, calcDisplacementFold_reduction_mpi_op, MPI_COMM_WORLD); 
		#pragma omp parallel for simd
		for(size_t counter = 0; counter < 512; ++counter){
			instinctive_movement_vector_copy[counter] = calcInstinctiveMovementVector_functor((sum_fitness_variation), instinctive_movement_vector_copy[counter]);
		}
		#pragma omp parallel for simd
		for(size_t counter = 0; counter < 32; ++counter){
			population[counter] = instinctiveMovement_functor(population[counter]);
		}
		double sum_weight = 0.0;
		fold_result_double = 0.0;
		
		
		
		#pragma omp parallel for simd reduction(sumWeight_reduction:fold_result_double)
		for(size_t counter = 0; counter < 32; ++counter){
			double map_fold_tmp = lambda4_functor(population[counter]);
		
			fold_result_double = sumWeight_functor(fold_result_double, map_fold_tmp);
		}		
		
		MPI_Allreduce(&fold_result_double, &sum_weight, 4, MPI_DOUBLE, sumWeight_reduction_mpi_op, MPI_COMM_WORLD); 
		#pragma omp parallel for simd
		for(size_t counter = 0; counter < 32; ++counter){
			weighted_fishes[counter] = calcWeightedFish_functor(population[counter]);
		}
		fold_result_std_array_double_512_.fill(0.0);
		
		
		
		#pragma omp parallel for simd reduction(calcBarycenterFold_reduction:fold_result_std_array_double_512_)
		for(size_t counter = 0; counter < 32; ++counter){
			std::array<double,512> map_fold_tmp = lambda5_functor(weighted_fishes[counter]);
		
			fold_result_std_array_double_512_ = calcBarycenterFold_functor(fold_result_std_array_double_512_, map_fold_tmp);
		}		
		
		MPI_Allreduce(fold_result_std_array_double_512_.data(), barycenter_copy.data(), 512, MPI_DOUBLE, calcBarycenterFold_reduction_mpi_op, MPI_COMM_WORLD); 
		#pragma omp parallel for simd
		for(size_t counter = 0; counter < 512; ++counter){
			barycenter_copy[counter] = calcBarycenterMap_functor((sum_weight), barycenter_copy[counter]);
		}
		#pragma omp parallel for simd
		for(size_t counter = 0; counter < 32; ++counter){
			population[counter] = volitiveMovement_functor((step_size_vol), (sum_weight), (sum_weight_last_iteration), population[counter]);
		}
		sum_weight_last_iteration = (sum_weight);
	}
	double global_best_fitness = 0.0;
	fold_result_double = std::numeric_limits<double>::lowest();
	
	
	
	#pragma omp parallel for simd reduction(getBestSolution_reduction:fold_result_double)
	for(size_t counter = 0; counter < 32; ++counter){
		double map_fold_tmp = lambda6_functor(population[counter]);
	
		fold_result_double = getBestSolution_functor(fold_result_double, map_fold_tmp);
	}		
	
	MPI_Allreduce(&fold_result_double, &global_best_fitness, 4, MPI_DOUBLE, getBestSolution_reduction_mpi_op, MPI_COMM_WORLD); 
	std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
	double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
	printf("Best solution: %.5f\n",(global_best_fitness));
	
	printf("Execution time: %.5fs\n", seconds);
	printf("Threads: %i\n", omp_get_max_threads());
	printf("Processes: %i\n", mpi_world_size);
	
	MPI_Finalize();
	return EXIT_SUCCESS;
}
