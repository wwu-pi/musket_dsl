#pragma once

struct Fish{
	std::array<double,1024> position;
	double fitness;
	std::array<double,1024> candidate_position;
	double candidate_fitness;
	std::array<double,1024> displacement;
	double fitness_variation;
	double weight;
	std::array<double,1024> best_position;
	double best_fitness;

};

extern std::vector<Fish> population;

