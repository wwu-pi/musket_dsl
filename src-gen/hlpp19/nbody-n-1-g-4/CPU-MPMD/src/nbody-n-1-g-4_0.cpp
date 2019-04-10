	
	#include <omp.h>
	#include <array>
	#include <vector>
	#include <sstream>
	#include <chrono>
	#include <random>
	#include <limits>
	#include <memory>
	#include <cstddef>
	#include <type_traits>
	
	#include "../include/musket.hpp"
	#include "../include/nbody-n-1-g-4_0.hpp"
	
	
	std::vector<std::mt19937> random_engines;
	std::vector<std::uniform_real_distribution<float>> rand_dist_float_0_0f_1_0f;
	

	
	const int dim = 500000;
	const int steps = 5;
	const float EPSILON = 1.0E-10f;
	const float DT = 0.01f;
	mkt::DArray<Particle> P(0, 500000, 500000, Particle{}, 1, 0, 0, mkt::DIST);
	mkt::DArray<Particle> oldP(0, 500000, 500000, Particle{}, 1, 0, 0, mkt::COPY);
	
	//Particle::Particle() : x(), y(), z(), vx(), vy(), vz(), mass(), charge() {}
	

	
	struct Init_particles_map_index_in_place_array_functor{
		auto operator()(int i, Particle p) const{
			p.x = rand_dist_float_0_0f_1_0f[omp_get_thread_num()](random_engines[omp_get_thread_num()]);
			p.y = rand_dist_float_0_0f_1_0f[omp_get_thread_num()](random_engines[omp_get_thread_num()]);
			p.z = rand_dist_float_0_0f_1_0f[omp_get_thread_num()](random_engines[omp_get_thread_num()]);
			p.vx = 0.0f;
			p.vy = 0.0f;
			p.vz = 0.0f;
			p.mass = 1.0f;
			p.charge = (1.0f - (2.0f * static_cast<float>(((i) % 2))));
			return (p);
		}
		
	};
	struct Calc_force_map_index_in_place_array_functor{
		auto operator()(int curIndex, Particle curParticle) const{
			float ax = 0.0f;
			float ay = 0.0f;
			float az = 0.0f;
			for(int j = 0; ((j) < 500000); j++){
				
				if(((j) != (curIndex))){
				float dx;
				float dy;
				float dz;
				float r2;
				float r;
				float qj_by_r3;
				dx = ((curParticle).x - oldP[(j)].x);
				dy = ((curParticle).y - oldP[(j)].y);
				dz = ((curParticle).z - oldP[(j)].z);
				r2 = ((((dx) * (dx)) + ((dy) * (dy))) + ((dz) * (dz)));
				r = std::sqrt((r2));
				
				if(((r) < (EPSILON))){
				qj_by_r3 = 0.0f;
				}
				 else {
						qj_by_r3 = (oldP[(j)].charge / ((r2) * (r)));
					}
				ax += ((qj_by_r3) * (dx));
				ay += ((qj_by_r3) * (dy));
				az += ((qj_by_r3) * (dz));
				}
			}
			float vx0 = (curParticle).vx;
			float vy0 = (curParticle).vy;
			float vz0 = (curParticle).vz;
			float qidt_by_m = (((curParticle).charge * (DT)) / (curParticle).mass);
			curParticle.vx += ((ax) * (qidt_by_m));
			curParticle.vy += ((ay) * (qidt_by_m));
			curParticle.vz += ((az) * (qidt_by_m));
			curParticle.x += ((((vx0) + (curParticle).vx) * (DT)) * 0.5f);
			curParticle.y += ((((vy0) + (curParticle).vy) * (DT)) * 0.5f);
			curParticle.z += ((((vz0) + (curParticle).vz) * (DT)) * 0.5f);
			return (curParticle);
		}
		
	};
	
	
	
	
	
	
	int main(int argc, char** argv) {
		
		
				Init_particles_map_index_in_place_array_functor init_particles_map_index_in_place_array_functor{};
				Calc_force_map_index_in_place_array_functor calc_force_map_index_in_place_array_functor{};
		
		random_engines.reserve(24);
		std::random_device rd;
		for(size_t counter = 0; counter < 24; ++counter){
			random_engines.push_back(std::mt19937(rd()));
		}
		
		rand_dist_float_0_0f_1_0f.reserve(24);
		for(size_t counter = 0; counter < 24; ++counter){
			rand_dist_float_0_0f_1_0f.push_back(std::uniform_real_distribution<float>(0.0f, 1.0f));
		}
		
				
		
		mkt::map_index_in_place<Particle, Init_particles_map_index_in_place_array_functor>(P, init_particles_map_index_in_place_array_functor);
		mkt::gather<void>(P, oldP);
		std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
		for(int i = 0; ((i) < (steps)); ++i){
			mkt::map_index_in_place<Particle, Calc_force_map_index_in_place_array_functor>(P, calc_force_map_index_in_place_array_functor);
			mkt::gather<void>(P, oldP);
		}
		std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
		double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
		
		printf("Execution time: %.5fs\n", seconds);
		printf("Threads: %i\n", omp_get_max_threads());
		printf("Processes: %i\n", 1);
		
		return EXIT_SUCCESS;
		}
