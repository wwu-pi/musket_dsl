#pragma once

struct Fish{
	std::vector<double> position;
	double fitness;
	std::vector<double> candidate_position;
	double candidate_fitness;
	std::vector<double> displacement;
	double fitness_variation;
	double weight;
	std::vector<double> best_position;
	double best_fitness;
	
	Fish();
};

extern std::vector<Fish> population;
extern std::vector<double> instinctive_movement_vector_copy;
extern std::vector<Fish> weighted_fishes;
extern std::vector<double> barycenter_copy;
