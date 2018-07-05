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
#include "../include/fss.hpp"

int mpi_rank = -1;
int mpi_world_size = -1;

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
const int NUMBER_OF_FISH = 8192;
const int ITERATIONS = 5000;
const int DIMENSIONS = 1024;
const int POP = 2048;
std::vector<Fish> population(POP);


// generate Function
inline auto getBestSolution_function(double best_solution, double best_fitness){
    double result = (best_solution);
    
    if(((best_fitness) > (best_solution))){
    result = (best_fitness);
    }
    return (result);
}

struct InitFish_functor{
    void operator()(Fish& fi) const{
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
    }
};
struct EvaluateFitness_functor{
    void operator()(Fish& fi) const{
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
};
struct IndividualMovement_functor{
    void operator()(double step_size, Fish& fi) const{
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

struct Lambda6_functor{
    auto operator()(const Fish& fi) const{
        return (fi).best_fitness;
    }
};

void getBestSolution(void* in, void* inout, int *len, MPI_Datatype *dptr){
    double* inv = static_cast<double*>(in);
    double* inoutv = static_cast<double*>(inout);
    *inoutv = getBestSolution_function(*inv, *inoutv);
} 

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    
    if(mpi_rank == 0){
        printf("Run Fss Call by reference\n\n");
    }    			
    
    InitFish_functor initFish_functor{};
    EvaluateFitness_functor evaluateFitness_functor{};
    IndividualMovement_functor individualMovement_functor{};
    GetBestSolution_functor getBestSolution_functor{};
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
    
    #pragma omp declare reduction(getBestSolution_reduction : double : omp_out = getBestSolution_function(omp_in, omp_out)) initializer(omp_priv = omp_orig)
    
    MPI_Datatype Fish_mpi_type_temp, Fish_mpi_type;
    MPI_Type_create_struct(9, (std::array<int,9>{1, 1, 1, 1, 1, 1, 1, 1, 1}).data(), (std::array<MPI_Aint,9>{static_cast<MPI_Aint>(offsetof(struct Fish, position)), static_cast<MPI_Aint>(offsetof(struct Fish, fitness)), static_cast<MPI_Aint>(offsetof(struct Fish, candidate_position)), static_cast<MPI_Aint>(offsetof(struct Fish, candidate_fitness)), static_cast<MPI_Aint>(offsetof(struct Fish, displacement)), static_cast<MPI_Aint>(offsetof(struct Fish, fitness_variation)), static_cast<MPI_Aint>(offsetof(struct Fish, weight)), static_cast<MPI_Aint>(offsetof(struct Fish, best_position)), static_cast<MPI_Aint>(offsetof(struct Fish, best_fitness))}).data(), (std::array<MPI_Datatype,9>{MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE}).data(), &Fish_mpi_type_temp);
    MPI_Type_create_resized(Fish_mpi_type_temp, 0, sizeof(Fish), &Fish_mpi_type);
    MPI_Type_free(&Fish_mpi_type_temp);
    MPI_Type_commit(&Fish_mpi_type);
    
    MPI_Op getBestSolution_reduction_mpi_op;
    MPI_Op_create( getBestSolution, 0, &getBestSolution_reduction_mpi_op );
        
    std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for simd
    for(size_t counter = 0; counter < POP; ++counter){
        initFish_functor(population[counter]);
    }
    double step_size  = (STEP_SIZE_INITIAL);
    double step_size_vol  = (STEP_SIZE_VOLITIVE_INITIAL);
    double fold_result_double = 0.0;

    for(int iteration = 0; ((iteration) < (ITERATIONS)); ++iteration){
        #pragma omp parallel for simd
        for(size_t counter = 0; counter < POP; ++counter){
            evaluateFitness_functor(population[counter]);
        }
        if(((iteration) > 0)){
            step_size = ((step_size) - (((STEP_SIZE_INITIAL) - (STEP_SIZE_FINAL)) / static_cast<double>(((ITERATIONS) - 1))));
            step_size_vol = ((step_size_vol) - (((STEP_SIZE_VOLITIVE_INITIAL) - (STEP_SIZE_VOLITIVE_FINAL)) / static_cast<double>(((ITERATIONS) - 1))));
        }
        #pragma omp parallel for simd
        for(size_t counter = 0; counter < POP; ++counter){
            individualMovement_functor((step_size), population[counter]);
        }
        
    }
    double global_best_fitness = 0.0;
    fold_result_double = std::numeric_limits<double>::lowest();
        
    #pragma omp parallel for simd reduction(getBestSolution_reduction:fold_result_double)
    for(size_t counter = 0; counter < POP; ++counter){
        double map_fold_tmp = lambda6_functor(population[counter]);
    
        fold_result_double = getBestSolution_functor(fold_result_double, map_fold_tmp);
    }		
    
    MPI_Allreduce(&fold_result_double, &global_best_fitness, mpi_world_size, MPI_DOUBLE, getBestSolution_reduction_mpi_op, MPI_COMM_WORLD); 
    std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
    
    if(mpi_rank == 0){
        printf("Best solution: %.5f\n",(global_best_fitness));
        
        printf("Execution time: %.5fs\n", seconds);
        printf("Threads: %i\n", omp_get_max_threads());
        printf("Processes: %i\n", mpi_world_size);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
