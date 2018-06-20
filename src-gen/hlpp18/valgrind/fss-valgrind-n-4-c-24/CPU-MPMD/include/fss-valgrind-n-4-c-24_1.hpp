#pragma once

struct Fish{
	std::array<double,32> position;
	double fitness;
	std::array<double,32> candidate_position;
	double candidate_fitness;
	std::array<double,32> displacement;
	double fitness_variation;
	double weight;
	std::array<double,32> best_position;
	double best_fitness;
	
	Fish();
};

extern std::vector<Fish> population;
extern std::vector<double> instinctive_movement_vector_copy;
extern std::vector<Fish> weighted_fishes;
extern std::vector<double> barycenter_copy;