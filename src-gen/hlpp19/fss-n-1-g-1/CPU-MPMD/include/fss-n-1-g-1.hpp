#pragma once

struct Fish{
	std::array<double,512> position;
	double fitness;
	std::array<double,512> candidate_position;
	double candidate_fitness;
	std::array<double,512> displacement;
	double fitness_variation;
	double weight;
	std::array<double,512> best_position;
	double best_fitness;
	
	//Fish();
};
		
