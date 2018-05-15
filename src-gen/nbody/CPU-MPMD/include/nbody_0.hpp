#pragma once

struct Particle{
	double x;
	double y;
	double z;
	double vx;
	double vy;
	double vz;
	double mass;
	double charge;
	
	//Particle();
};

extern std::vector<Particle> P;
extern std::vector<Particle> oldP;
