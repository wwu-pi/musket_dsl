#pragma once

struct Particle{
	float x;
	float y;
	float z;
	float vx;
	float vy;
	float vz;
	float mass;
	float charge;
	
	Particle();
};

extern std::vector<Particle> P;
extern std::vector<Particle> oldP;
