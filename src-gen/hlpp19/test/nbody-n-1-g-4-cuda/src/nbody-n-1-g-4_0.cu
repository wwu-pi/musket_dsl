	
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
	#include <curand_kernel.h>
	
	#include "../include/musket.hpp"
	#include "../include/nbody-n-1-g-4_0.hpp"
		
			
	const int dim = 500000;
	const int steps = 5;
	const float EPSILON = 1.0E-10f;
	const float DT = 0.01f;
	mkt::DArray<Particle> P(0, 500000, 500000, Particle{}, 1, 0, 0, mkt::DIST, mkt::DIST);
	mkt::DArray<Particle> oldP(0, 500000, 500000, Particle{}, 1, 0, 0, mkt::COPY, mkt::COPY);
	
	//Particle::Particle() : x(), y(), z(), vx(), vy(), vz(), mass(), charge() {}
	
	// __global__ void setup_kernel(curandState *state)
	// {
	// 	int id = threadIdx.x + blockIdx.x * 1024;
	// 	/* Each thread gets same seed, a different sequence 
	// 	   number, no offset */
	// 	curand_init(1234, id, 0, &state[id]);
	// }
	
	struct Init_particles_map_index_in_place_array_functor{
		
		Init_particles_map_index_in_place_array_functor(){
			printf("init functor constructor\n");
		}
		
		~Init_particles_map_index_in_place_array_functor() {}
		
		__device__
		auto operator()(int i, Particle& p){
			curandState state;
			curand_init(clock64(), 0, 0, &state);
			
			p.x = static_cast<float>(curand_uniform(&state) * (1.0f - 0.0f + 0.999999) + 0.0f);
			p.y = static_cast<float>(curand_uniform(&state) * (1.0f - 0.0f + 0.999999) + 0.0f);
			p.z = static_cast<float>(curand_uniform(&state) * (1.0f - 0.0f + 0.999999) + 0.0f);
			p.vx = 0.0f;
			p.vy = 0.0f;
			p.vz = 0.0f;
			p.mass = 1.0f;
			p.charge = (1.0f - (2.0f * static_cast<float>(((i) % 2))));

		}
	
		void init(int gpu){

		}
	};
	struct Calc_force_map_index_in_place_array_functor{
		
		Calc_force_map_index_in_place_array_functor(const mkt::DArray<Particle>& _oldP) : oldP(_oldP){
			printf("functor constructor \n");
		}
		
		~Calc_force_map_index_in_place_array_functor() {}
		
		__device__
		auto operator()(int curIndex, Particle& curParticle){
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
			printf("functor init %i\n", gpu)
			oldP.init(gpu);
		}
			
		
		mkt::DeviceArray<Particle> oldP;

	};
	
	
	
	
	int main(int argc, char** argv) {
		
		//curandState* devStates[4];
		
		// #pragma omp parallel for
		// for(int gpu = 0; gpu < 4; ++gpu){
		// 	cudaSetDevice(gpu);
		// 	cudaMalloc((void **)&devStates[gpu], 64 * 1024 * sizeof(curandState)));
		// 	setup_kernel<<<64, 1024>>>(devStates[gpu]);
		// }
		
		Init_particles_map_index_in_place_array_functor init_particles_map_index_in_place_array_functor{};
		Calc_force_map_index_in_place_array_functor calc_force_map_index_in_place_array_functor{oldP};
		
				
		printf("map init\n");
		mkt::map_index_in_place<Particle, Init_particles_map_index_in_place_array_functor>(P, init_particles_map_index_in_place_array_functor);
		printf("gather init\n");
		mkt::gather<Particle>(P, oldP);
		mkt::sync_streams();

		double gather_time = 0.0;
		double map_time = 0.0;

		std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
		for(int i = 0; ((i) < (steps)); ++i){
			mkt::sync_streams();
			std::chrono::high_resolution_clock::time_point map_timer_start = std::chrono::high_resolution_clock::now();
			printf("map iteration %i\n", i);
			mkt::map_index_in_place<Particle, Calc_force_map_index_in_place_array_functor>(P, calc_force_map_index_in_place_array_functor);

			mkt::sync_streams();
			printf("map end iteration %i\n", i);
			std::chrono::high_resolution_clock::time_point map_timer_end = std::chrono::high_resolution_clock::now();

			map_time += std::chrono::duration<double>(map_timer_end - map_timer_start).count();

			std::chrono::high_resolution_clock::time_point gather_timer_start = std::chrono::high_resolution_clock::now();

			printf("gather iteration %i\n", i);
			mkt::gather<Particle>(P, oldP);
			mkt::sync_streams();
			printf("gather iteration %i\n", i);

			std::chrono::high_resolution_clock::time_point gather_timer_end = std::chrono::high_resolution_clock::now();

			gather_time += std::chrono::duration<double>(gather_timer_end - gather_timer_start).count();
		}
		mkt::sync_streams();
		std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
		double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
		
		printf("Execution time: %.5fs\n", seconds);

		printf("map time: %.5fs\n", map_time);
		printf("gather time: %.5fs\n", gather_time);

		printf("Threads: %i\n", omp_get_max_threads());
		printf("Processes: %i\n", 1);
		
		// #pragma omp parallel for
		// for(int gpu = 0; gpu < 4; ++gpu){
		// 	cudaSetDevice(gpu);
		// 	cudaFree(devStates[gpu]);
		// }
		return EXIT_SUCCESS;
		}
