	// sequential reference implementation for nbody simulation
	
	#include <array>
	#include <sstream>
	#include <chrono>
	#include <random>
	#include "../include/nbody.hpp"
	
	
	// constants and global vars
	const int steps = 5;
	
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
	std::array<Particle, 16> P;
	std::array<Particle, 16> oldP;

	int main(int argc, char** argv) {
	
		printf("Run Nbody\n\n");			

			

		for(size_t counter = 0; counter < 16; ++counter){
			std::mt19937 engine(counter);
			std::uniform_real_distribution<double> unif(0.0, 1.0);
		  P[counter].x = unif(engine);
		  P[counter].y = unif(engine);
		  P[counter].z = unif(engine);
		  P[counter].vx = 0.0;
		  P[counter].vy = 0.0;
		  P[counter].vz = 0.0;
		  P[counter].mass = 1.0;
		  P[counter].charge = 1.0 - 2.0 * static_cast<double>(counter % 2);
		}
			
		// start main program
		std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
		
		
		// gather P into oldP show initial particle system		
		std::copy(P.begin(), P.end(), oldP.begin());
		
		std::ostringstream s1;
			s1 << "[";
			for (int i = 0; i < 16 - 1; i++) {
				s1 << oldP[i].x << ", ";
				s1 << oldP[i].y << ", ";
				s1 << oldP[i].z;
				s1 << "; ";
			}
			s1 << oldP[16 - 1].x << ", " << oldP[16 - 1].y << ", " << oldP[16 - 1].z << "]" << std::endl;
			s1 << std::endl;
			printf("%s", s1.str().c_str());
		
		
		// start main loop
		for(int step = 0; step < steps; ++step){
		
			// calc force		
			for(size_t counter = 0; counter < 16; ++counter){
				
				int curIndex = counter;
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
				curParticle.x += (vx0 + curParticle.vx) * DT * 0.5;
				curParticle.y += (vy0 + curParticle.vy) * DT * 0.5;
				curParticle.z += (vz0 + curParticle.vz) * DT * 0.5;

				P[counter] = curParticle;
			} // end calc force
		
			// gather particles
			std::copy(P.begin(), P.end(), oldP.begin());

		} // end main loop
		
		// show final particle system
		
		std::ostringstream s2;
			s2 << "[";
			for (int i = 0; i < 16 - 1; i++) {
				s2 << oldP[i].x << ", ";
				s2 << oldP[i].y << ", ";
				s2 << oldP[i].z;
				s2 << "; ";
			}
			s2 << oldP[16 - 1].x << ", " << oldP[16 - 1].y << ", " << oldP[16 - 1].z << "]" << std::endl;
			s2 << std::endl;
			printf("%s", s2.str().c_str());
		
		std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
		double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
	
		// final output
			printf("Execution time: %.5fs\n", seconds);
			printf("Threads: %i\n", 1);
			printf("Processes: %i\n", 1);
		
		return EXIT_SUCCESS;
	}
