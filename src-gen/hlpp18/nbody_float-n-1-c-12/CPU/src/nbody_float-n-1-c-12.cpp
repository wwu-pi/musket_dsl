
#include <omp.h>
#include <array>
#include <sstream>
#include <chrono>
#include <random>
#include <limits>
#include <memory>
#include "../include/nbody_float-n-1-c-12.hpp"

size_t tmp_size_t = 0;


const int steps = 5;
const float EPSILON = 1.0E-10f;
const float DT = 0.01f;
std::vector<Particle> P(500000);
std::vector<Particle> oldP(500000);

Particle::Particle() : x(), y(), z(), vx(), vy(), vz(), mass(), charge() {}

int main(int argc, char** argv) {
	
	printf("Run Nbody_float-n-1-c-12\n\n");			
	
	
	std::vector<std::mt19937> random_engines;
	random_engines.reserve(12);
	std::random_device rd;
	for(size_t counter = 0; counter < 12; ++counter){
		random_engines.push_back(std::mt19937(rd()));
	}
	
	std::vector<std::uniform_real_distribution<float>> rand_dist_float_0_0f_1_0f;
							rand_dist_float_0_0f_1_0f.reserve(12);
							for(size_t counter = 0; counter < 12; ++counter){
								rand_dist_float_0_0f_1_0f.push_back(std::uniform_real_distribution<float>(0.0f, 1.0f));
							}
	
	
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 500000; ++counter){
		
		P[counter].x = rand_dist_float_0_0f_1_0f[omp_get_thread_num()](random_engines[omp_get_thread_num()]);
		P[counter].y = rand_dist_float_0_0f_1_0f[omp_get_thread_num()](random_engines[omp_get_thread_num()]);
		P[counter].z = rand_dist_float_0_0f_1_0f[omp_get_thread_num()](random_engines[omp_get_thread_num()]);
		P[counter].vx = 0.0f;
		P[counter].vy = 0.0f;
		P[counter].vz = 0.0f;
		P[counter].mass = 1.0f;
		P[counter].charge = (1.0f - (2.0f * static_cast<float>(((counter) % 2))));
	}
	std::copy(P.begin(), P.end(), oldP.begin());
	std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
	for(int i = 0; ((i) < (steps)); ++i){
		#pragma omp parallel for simd
		for(size_t counter = 0; counter < 500000; ++counter){
			
			float ax = 0.0f;
			float ay = 0.0f;
			float az = 0.0f;
			for(int j = 0; ((j) < 500000); j++){
				
				if(((j) != (counter))){
				float dx;
				float dy;
				float dz;
				float r2;
				float r;
				float qj_by_r3;
				dx = ((P[counter]).x - (oldP)[(j)].x);
				dy = ((P[counter]).y - (oldP)[(j)].y);
				dz = ((P[counter]).z - (oldP)[(j)].z);
				r2 = ((((dx) * (dx)) + ((dy) * (dy))) + ((dz) * (dz)));
				r = std::sqrt((r2));
				
				if(((r) < (EPSILON))){
				qj_by_r3 = 0.0f;
				}
				 else {
					qj_by_r3 = ((oldP)[(j)].charge / ((r2) * (r)));
				}
				ax += ((qj_by_r3) * (dx));
				ay += ((qj_by_r3) * (dy));
				az += ((qj_by_r3) * (dz));
				}
			}
			float vx0 = (P[counter]).vx;
			float vy0 = (P[counter]).vy;
			float vz0 = (P[counter]).vz;
			float qidt_by_m = (((P[counter]).charge * (DT)) / (P[counter]).mass);
			P[counter].vx += ((ax) * (qidt_by_m));
			P[counter].vy += ((ay) * (qidt_by_m));
			P[counter].vz += ((az) * (qidt_by_m));
			P[counter].x += ((((vx0) + (P[counter]).vx) * (DT)) * 0.5f);
			P[counter].y += ((((vy0) + (P[counter]).vy) * (DT)) * 0.5f);
			P[counter].z += ((((vz0) + (P[counter]).vz) * (DT)) * 0.5f);
		}
		std::copy(P.begin(), P.end(), oldP.begin());
	}
	std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
	double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
	
	printf("Execution time: %.5fs\n", seconds);
	printf("Threads: %i\n", omp_get_max_threads());
	printf("Processes: %i\n", 1);
	
	return EXIT_SUCCESS;
}
