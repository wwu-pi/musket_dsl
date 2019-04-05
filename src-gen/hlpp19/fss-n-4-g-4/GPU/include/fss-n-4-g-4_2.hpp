#pragma once

extern mkt::DArray<std::array<double,512>> population_position;
extern mkt::DArray<double> population_fitness;
extern mkt::DArray<std::array<double,512>> population_candidate_position;
extern mkt::DArray<double> population_candidate_fitness;
extern mkt::DArray<std::array<double,512>> population_displacement;
extern mkt::DArray<double> population_fitness_variation;
extern mkt::DArray<double> population_weight;
extern mkt::DArray<std::array<double,512>> population_best_position;
extern mkt::DArray<double> population_best_fitness;
extern mkt::DArray<double> instinctive_movement_vector_copy;
extern mkt::DArray<std::array<double,512>> weighted_fishes_position;
extern mkt::DArray<double> weighted_fishes_fitness;
extern mkt::DArray<std::array<double,512>> weighted_fishes_candidate_position;
extern mkt::DArray<double> weighted_fishes_candidate_fitness;
extern mkt::DArray<std::array<double,512>> weighted_fishes_displacement;
extern mkt::DArray<double> weighted_fishes_fitness_variation;
extern mkt::DArray<double> weighted_fishes_weight;
extern mkt::DArray<std::array<double,512>> weighted_fishes_best_position;
extern mkt::DArray<double> weighted_fishes_best_fitness;
extern mkt::DArray<double> barycenter_copy;
