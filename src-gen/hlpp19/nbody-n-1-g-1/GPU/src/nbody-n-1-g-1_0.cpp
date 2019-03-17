	
	#include <omp.h>
	#include <openacc.h>
	#include <stdlib.h>
	#include <math.h>
	#include <array>
	#include <vector>
	#include <sstream>
	#include <chrono>
	#include <random>
	#include <limits>
	#include <memory>
	#include <cstddef>
	#include <type_traits>
	//#include <cuda.h>
	//#include <openacc_curand.h>
	
	#include "../include/musket.hpp"
	#include "../include/nbody-n-1-g-1_0.hpp"
	
	
	std::vector<std::mt19937> random_engines;
	std::vector<std::uniform_real_distribution<float>> rand_dist_float_0_0f_1_0f;
	
	std::array<float*, 1> rns_pointers;
	std::array<float, 3> rns;			
// extern size_t rns_index;
	// #pragma acc routine seq
	// float get_random_float(float lower, float higher, float * rns){
	// 	size_t t_rng_index = 0;
	// 	//#pragma acc atomic capture
	// 	t_rng_index = rns_index++;
		
	// 	t_rng_index = t_rng_index % 100000;
		
	// 	return rns[t_rng_index] * (higher - lower + 0.999999) + lower;
	// }
	
	const int dim = 500000;
	const int steps = 5;
	const float EPSILON = 1.0E-10f;
	const float DT = 0.01f;
	mkt::DArray<Particle> P(0, 500000, 500000, Particle{}, 1, 0, 0, mkt::DIST, mkt::DIST);
	mkt::DArray<Particle> oldP(0, 500000, 500000, Particle{}, 1, 0, 0, mkt::COPY, mkt::COPY);
	
	//Particle::Particle() : x(), y(), z(), vx(), vy(), vz(), mass(), charge() {}
	

	
	struct Init_particles_map_index_in_place_array_functor{
		
		Init_particles_map_index_in_place_array_functor(std::array<float*, 1> rns_pointers)  : _rns_index(0) {
			// std::random_device rd;
			// std::mt19937 d_rng_gen(rd());
			// std::uniform_real_distribution<float> d_rng_dis(0.0f, 1.0f);
			//rns = new float[3];
			// for(int random_number = 0; random_number < 3; random_number++){
			// 	rns[random_number] = d_rng_dis(d_rng_gen);
			// }

			for(int gpu = 0; gpu < 1; gpu++){
			 	_rns_pointers[gpu] = rns_pointers[gpu];
			}
		}

		~Init_particles_map_index_in_place_array_functor() {
			//delete [] rns;
		}
		
		
		#pragma acc routine seq
		auto operator()(int i, Particle& p) const{
	p.x = 0.3f; // rns[rns_index];
	p.y = 0.3f; // rns[rns_index];
	p.z = 0.3f; // rns[rns_index];
	p.vx = 0.0f;
	p.vy = 0.0f;
	p.vz = 0.0f;
	p.mass = 1.0f;
	p.charge = (1.0f - (2.0f * static_cast<float>(((i) % 2))));
		}
	
		void init(int gpu){
			_rns = _rns_pointers[gpu];
			// _rns_index = random number from host;
		}
		
		float* _rns;
		
		std::array<float*, 1> _rns_pointers;	
	 	size_t _rns_index;
		
	};
	struct Calc_force_map_index_in_place_array_functor{
		
		Calc_force_map_index_in_place_array_functor(const mkt::DArray<Particle>& _oldP) : oldP(_oldP) {}
		#pragma acc routine seq
		auto operator()(int curIndex, Particle& curParticle) const{
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
		dx = ((curParticle).x - oldP.get_data_local((j)).x);
		dy = ((curParticle).y - oldP.get_data_local((j)).y);
		dz = ((curParticle).z - oldP.get_data_local((j)).z);
		r2 = ((((dx) * (dx)) + ((dy) * (dy))) + ((dz) * (dz)));
		r = sqrtf((r2));
		
		if(((r) < (EPSILON))){
		qj_by_r3 = 0.0f;
		}
		 else {
				qj_by_r3 = (oldP.get_data_local((j)).charge / ((r2) * (r)));
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
		}
	
		void init(int gpu){
			oldP.init(gpu);
		}
		
		
		mkt::DeviceArray<Particle> oldP;
	};
	
	
	
	
	
	
	int main(int argc, char** argv) {

	
		
		random_engines.reserve(24);
		std::random_device rd;
		for(size_t counter = 0; counter < 24; ++counter){
			random_engines.push_back(std::mt19937(rd()));
		}

		std::mt19937 d_rng_gen(rd());
		std::uniform_real_distribution<float> d_rng_dis(0.0f, 1.0f);
		//rns = new float[3];
		for(int random_number = 0; random_number < 3; random_number++){
			rns[random_number] = d_rng_dis(d_rng_gen);
		}
		
		#pragma omp parallel for
		for(int gpu = 0; gpu < 1; ++gpu){
			acc_set_device_num(gpu, acc_device_not_host);
			float* devptr = static_cast<float*>(acc_malloc(3 * sizeof(float)));
			rns_pointers[gpu] = devptr;
			acc_memcpy_to_device(devptr, rns.data(), 3 * sizeof(float));
		}
		
		rand_dist_float_0_0f_1_0f.reserve(24);
		for(size_t counter = 0; counter < 24; ++counter){
			rand_dist_float_0_0f_1_0f.push_back(std::uniform_real_distribution<float>(0.0f, 1.0f));
		}
		
		Init_particles_map_index_in_place_array_functor init_particles_map_index_in_place_array_functor{rns_pointers};
		Calc_force_map_index_in_place_array_functor calc_force_map_index_in_place_array_functor{oldP};
		
		mkt::map_index_in_place<Particle, Init_particles_map_index_in_place_array_functor>(P, init_particles_map_index_in_place_array_functor);
		mkt::gather<Particle>(P, oldP);
		std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
		for(int i = 0; ((i) < (steps)); ++i){
			mkt::map_index_in_place<Particle, Calc_force_map_index_in_place_array_functor>(P, calc_force_map_index_in_place_array_functor);
			mkt::gather<Particle>(P, oldP);
		}
		std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
		double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
		
		printf("Execution time: %.5fs\n", seconds);
		printf("Threads: %i\n", omp_get_max_threads());
		printf("Processes: %i\n", 1);
		


		#pragma omp parallel for
		for(int gpu = 0; gpu < 1; ++gpu){
			acc_set_device_num(gpu, acc_device_not_host);
			// #pragma acc exit data delete(rns)
			// #pragma acc exit data delete(rns_index)
			acc_free(rns_pointers[gpu]);
		}
		return EXIT_SUCCESS;
		}
