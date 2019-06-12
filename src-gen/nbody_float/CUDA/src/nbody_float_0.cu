	#include <cuda.h>
	#include <omp.h>
	#include <stdlib.h>
	#include <math.h>
	#include <array>
	#include <vector>
	#include <sstream>
	#include <chrono>
	#include <curand_kernel.h>
	#include <limits>
	#include <memory>
	#include <cstddef>
	#include <type_traits>
	
	
	#include "../include/musket.cuh"
	#include "../include/nbody_float_0.cuh"
	
	
			
	//Particle::Particle() : x(), y(), z(), vx(), vy(), vz(), mass(), charge() {}
	

	
	struct Init_particles_map_index_in_place_array_functor{
		
		Init_particles_map_index_in_place_array_functor(){}
		
		~Init_particles_map_index_in_place_array_functor() {}
		
		__device__
		auto operator()(int i, Particle p){
			curandState_t curand_state; // performance could be improved by creating states before
			size_t id = blockIdx.x * blockDim.x + threadIdx.x;
			curand_init(clock64(), id, 0, &curand_state);
			p.x = (curand_uniform(&curand_state) * (1.0f - 0.0f) + 0.0f);
			p.y = (curand_uniform(&curand_state) * (1.0f - 0.0f) + 0.0f);
			p.z = (curand_uniform(&curand_state) * (1.0f - 0.0f) + 0.0f);
			p.vx = 0.0f;
			p.vy = 0.0f;
			p.vz = 0.0f;
			p.mass = 1.0f;
			p.charge = (1.0f - (2.0f * static_cast<float>(((i) % 2))));
			return (p);
		}
	
		void init(int device){
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		
	};
	struct Calc_force_map_index_in_place_array_functor{
		
		Calc_force_map_index_in_place_array_functor(const mkt::DArray<Particle>& _oldP) : oldP(_oldP){}
		
		~Calc_force_map_index_in_place_array_functor() {}
		
		__device__
		auto operator()(int curIndex, Particle curParticle){
			float ax = 0.0f;
			float ay = 0.0f;
			float az = 0.0f;
			for(int j = 0; ((j) < 32); j++){
				
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
			return (curParticle);
		}
	
		void init(int device){
			oldP.init(device);
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		
		mkt::DeviceArray<Particle> oldP;
	};
	
	
	
	
	
	
	
	int main(int argc, char** argv) {
		mkt::init();
		
		
		const int dim = 32;
		const int steps = 5;
		const float EPSILON = 1.0E-10f;
		const float DT = 0.01f;
		mkt::DArray<Particle> P(0, 32, 32, Particle{}, 1, 0, 0, mkt::DIST, mkt::DIST);
		mkt::DArray<Particle> oldP(0, 32, 32, Particle{}, 1, 0, 0, mkt::COPY, mkt::COPY);
		
		Init_particles_map_index_in_place_array_functor init_particles_map_index_in_place_array_functor{};
		Calc_force_map_index_in_place_array_functor calc_force_map_index_in_place_array_functor{oldP};
		
		
				
		
		mkt::map_index_in_place<Particle, Init_particles_map_index_in_place_array_functor>(P, init_particles_map_index_in_place_array_functor);
		mkt::gather<Particle>(P, oldP);
		oldP.update_self();
		mkt::print("oldP", oldP);
		mkt::sync_streams();
		std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
		for(int i = 0; ((i) < (steps)); ++i){
			mkt::map_index_in_place<Particle, Calc_force_map_index_in_place_array_functor>(P, calc_force_map_index_in_place_array_functor);
			mkt::gather<Particle>(P, oldP);
		}
		mkt::sync_streams();
		std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
		double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
		oldP.update_self();
		mkt::print("oldP", oldP);
		
		printf("Execution time: %.5fs\n", seconds);
		printf("Threads: %i\n", omp_get_max_threads());
		printf("Processes: %i\n", 1);
		
		return EXIT_SUCCESS;
		}
