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
	
	//Particle();
};
		
MPI_Datatype Particle_mpi_type;

