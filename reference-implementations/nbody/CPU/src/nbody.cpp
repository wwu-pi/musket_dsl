	// reference implementation for nbody simulation with 2 processes and 4 cores per process
	
	#include <mpi.h>
	#include <omp.h>
	#include <array>
	#include <sstream>
	#include <chrono>
	#include <random>
	#include "../include/nbody.hpp"
	
	
	// constants and global vars
	const size_t number_of_processes = 2;
	const int steps = 1;
	int process_id = -1;
	size_t tmp_size_t = 0;
	
	const double EPSILON = 0.0000000001;
	const double DT = 0.01;
	
	// structs
	struct Particle{
		double x;
		double y;
		double z;
		double vx;
		double vy;
		double vz;
		double mass;
		double charge;
	};	

	// data structures
	std::array<Particle, 8> P;
	std::array<Particle, 16> oldP;

	int main(int argc, char** argv) {
	
		// mpi setup
		MPI_Init(&argc, &argv);
		
		int mpi_world_size = 0;
		MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
		
		if(mpi_world_size != number_of_processes){
			MPI_Finalize();
			return EXIT_FAILURE;
		}
		
		MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
		
		if(process_id == 0){
			printf("Run Nbody\n\n");			
		}
		
		// prepare random engines and distributions
		
		std::vector<std::mt19937> random_engines;
		random_engines.reserve(4);

		for(size_t counter = 0; counter < 4; ++counter){
			random_engines.push_back(std::mt19937(counter));
		}
		
		std::vector<std::uniform_real_distribution<double>> rand_dist_double_0_1;
		rand_dist_double_0_1.reserve(4);
		
		for(size_t counter = 0; counter < 4; ++counter){
			rand_dist_double_0_1.push_back(std::uniform_real_distribution<double>(0, 1));
		}
		
		// init particles		
		int offset = 0;		
		if(process_id == 1){
			offset = 8;
		}
		
		#pragma omp parallel for simd
		for(size_t counter = 0; counter < 8; ++counter){
			int tid = omp_get_thread_num();
		  P[counter].x = rand_dist_double_0_1[tid](random_engines[tid]);
		  P[counter].y = rand_dist_double_0_1[tid](random_engines[tid]);
		  P[counter].z = rand_dist_double_0_1[tid](random_engines[tid]);
		  P[counter].vx = 0.0;
		  P[counter].vy = 0.0;
		  P[counter].vz = 0.0;
		  P[counter].mass = 1.0;
		  P[counter].charge = 1.0 - 2.0 * static_cast<double>((counter + offset) % 2);
		}
			
		// start main program
		std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
		
		
		// gather P into oldP show initial particle system		
		tmp_size_t = 8 * sizeof(Particle);
		MPI_Allgather(P.data(), tmp_size_t, MPI_BYTE, oldP.data(), tmp_size_t,MPI_BYTE, MPI_COMM_WORLD);
		
		if (process_id == 0) {
			std::ostringstream s1;
			s1 << "[";
			for (int i = 0; i < 16 - 1; i++) {
				s1 << oldP[i].x;
				s1 << "; ";
			}
			s1 << oldP[16 - 1].x << "]" << std::endl;
			s1 << std::endl;
			printf("%s", s1.str().c_str());
		}
		
		
		// start main loop
		for(int step = 0; step < steps; ++step){
		
			// calc force		
			#pragma omp parallel for simd
			for(size_t counter = 0; counter < 8; ++counter){
				
				int curIndex = counter + offset;
				Particle curParticle = P[counter];
			
				double ax = 0.0;
				double ay = 0.0;
				double az = 0.0;

				// calculate forces for the current particle
				for (int j = 0; j < 16; j++) {

				  // do not evaluate interaction with yourself.
				  if (j != curIndex) {

				    // Evaluate forces that j-particles exert on the i-particle.
				    double dx, dy, dz, r2, r, qj_by_r3;

				    // Here we absorb the minus sign by changing the order of i and j.
				    dx = curParticle.x - oldP[j].x;
				    dy = curParticle.y - oldP[j].y;
				    dz = curParticle.z - oldP[j].z;

				    r2 = dx * dx + dy * dy + dz * dz;
				    r = sqrtf(r2);

				    // Quench the force if the particles are too close.
				    if (r < EPSILON)
				      qj_by_r3 = 0.0;
				    else
				      qj_by_r3 = oldP[j].charge / (r2 * r);

				    // accumulate the contribution from particle j.
				    ax += qj_by_r3 * dx;
				    ay += qj_by_r3 * dy;
				    az += qj_by_r3 * dz;
				  }
				}

				// advance current particle
				double vx0 = curParticle.vx;
				double vy0 = curParticle.vy;
				double vz0 = curParticle.vz;

				double qidt_by_m = curParticle.charge * DT / curParticle.mass;
				curParticle.vx += ax * qidt_by_m;
				curParticle.vy += ay * qidt_by_m;
				curParticle.vz += az * qidt_by_m;

				// Use average velocity in the interval to advance the particles' positions
				curParticle.x += (vx0 + curParticle.vx) * DT * 0.5f;
				curParticle.y += (vy0 + curParticle.vy) * DT * 0.5f;
				curParticle.z += (vz0 + curParticle.vz) * DT * 0.5f;

				P[counter] = curParticle;
			} // end calc force
		
			// gather particles
			MPI_Allgather(P.data(), tmp_size_t, MPI_BYTE, oldP.data(), tmp_size_t,MPI_BYTE, MPI_COMM_WORLD);

		} // end main loop
		
		// show final particle system
		
		if (process_id == 0) {
			std::ostringstream s3;
			s3 << "[";
			for (int i = 0; i < 16 - 1; i++) {
				s3 << oldP[i].x;
				s3 << "; ";
			}
			s3 << oldP[16 - 1].x << "]" << std::endl;
			s3 << std::endl;
			printf("%s", s3.str().c_str());
		}
		
		std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
		double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
	
		// final output
		if(process_id == 0){
			printf("Execution time: %.5fs\n", seconds);
			printf("Threads: %i\n", omp_get_max_threads());
			printf("Processes: %i\n", mpi_world_size);
		}
		
		MPI_Finalize();
		return EXIT_SUCCESS;
	}
