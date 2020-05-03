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
const int NUMBER_OF_FISH = 2048;
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

int main(int argc, char** argv) {
    
    printf("Run Fss Call by reference on one node\n\n");

    InitFish_functor initFish_functor{};
    EvaluateFitness_functor evaluateFitness_functor{};
    IndividualMovement_functor individualMovement_functor{};
    GetBestSolution_functor getBestSolution_functor{};
    Lambda6_functor lambda6_functor{};
    
    int cores = omp_get_max_threads();

    random_engines.reserve(cores);
    std::random_device rd;
    for(size_t counter = 0; counter < cores; ++counter){
        random_engines.push_back(std::mt19937(rd()));
    }
    
    rand_dist_double_INIT_LOWER_BOUND_INIT_UPPER_BOUND.reserve(cores);
    for(size_t counter = 0; counter < cores; ++counter){
        rand_dist_double_INIT_LOWER_BOUND_INIT_UPPER_BOUND.push_back(std::uniform_real_distribution<double>((INIT_LOWER_BOUND), (INIT_UPPER_BOUND)));
    }rand_dist_double_minus1_0_1_0.reserve(cores);
    for(size_t counter = 0; counter < cores; ++counter){
        rand_dist_double_minus1_0_1_0.push_back(std::uniform_real_distribution<double>(-(1.0), 1.0));
    }rand_dist_double_0_0_1_0.reserve(cores);
    for(size_t counter = 0; counter < cores; ++counter){
        rand_dist_double_0_0_1_0.push_back(std::uniform_real_distribution<double>(0.0, 1.0));
    }
    
    #pragma omp declare reduction(getBestSolution_reduction : double : omp_out = getBestSolution_function(omp_in, omp_out)) initializer(omp_priv = omp_orig)
        
    std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for simd
    for(size_t counter = 0; counter < POP; ++counter){
        initFish_functor(population[counter]);
    }
    double step_size  = (STEP_SIZE_INITIAL);
    double step_size_vol  = (STEP_SIZE_VOLITIVE_INITIAL);

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
    double global_best_fitness = std::numeric_limits<double>::lowest();
    
        
    #pragma omp parallel for simd reduction(getBestSolution_reduction:global_best_fitness)
    for(size_t counter = 0; counter < POP; ++counter){
        double map_fold_tmp = lambda6_functor(population[counter]);
    
        global_best_fitness = getBestSolution_functor(global_best_fitness, map_fold_tmp);
    }		
    
    std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
    

        printf("Best solution: %.5f\n",(global_best_fitness));
        
        printf("Execution time: %.5fs\n", seconds);
        printf("Threads: %i\n", omp_get_max_threads());
        printf("Processes: %i\n", 1);

    return EXIT_SUCCESS;
}
