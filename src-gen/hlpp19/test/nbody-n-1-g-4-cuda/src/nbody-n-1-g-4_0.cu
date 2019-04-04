	
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
		
			
	const int dim = 50000;
	const int steps = 5;
	const float EPSILON = 1.0E-10f;
	const float DT = 0.01f;
	
	
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
			//printf("init functor constructor\n");
		}
		
		~Init_particles_map_index_in_place_array_functor() {}
		
		__device__
		void operator()(int i, Particle& p){
			int deviceId = -1;
			cudaGetDevice(&deviceId);
			//curandState_t curand_state;
			size_t id = blockIdx.x * blockDim.x + threadIdx.x;
			//curand_init(1234, id, 0, &curand_state);
			float random_number = 1.1f; //curand_uniform(&curand_state);
			if(id == 0 || id == 12499){
				printf("init functor() id %i on Device %i, random number: %.5f\n", id, deviceId, random_number);
			}
			//printf("init functor()\n");
			p.x = 1.2f; //curand_uniform(&curand_state);
			p.y = 1.3f;//curand_uniform(&curand_state);
			p.z = 1.4f;//curand_uniform(&curand_state);
			p.vx = 0.0f;
			p.vy = 0.0f;
			p.vz = 0.0f;
			p.mass = 1.0f;
			p.charge = (1.0f - (2.0f * static_cast<float>(((i) % 2))));

			if(id == 0 || id == 12499){
				printf("init functor() on device %i for thread %i: particle after changes: p.x = %.5f; p.y = %.5f; p.z = %.5f; p.vx = %.5f; p.vy = %.5f; p.vz = %.5f; p.mass = %.5f; p.charge = %.5f\n", deviceId, id, p.x, p.y, p.z, p.vx, p.vy, p.vz, p.mass, p.charge);
			}
		}
	
		void init(int gpu){

		}
	};
	struct Calc_force_map_index_in_place_array_functor{
		
		Calc_force_map_index_in_place_array_functor(const mkt::DArray<Particle>& _oldP) : oldP(_oldP){
			//printf("functor constructor \n");
		}
		
		~Calc_force_map_index_in_place_array_functor() {}
		
		__device__
		void operator()(int curIndex, Particle& curParticle){
			//printf("calc force %i\n", id);

			float ax = 0.0f;
			float ay = 0.0f;
			float az = 0.0f;
			for(int j = 0; ((j) < 50000); j++){
				
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
			//printf("functor init %i\n", gpu);
			oldP.init(gpu);
		}
			
		
		mkt::DeviceArray<Particle> oldP;

	};
	
	
	void showParticle(Particle p) {
		printf("x: %.5f, y: %.5f, z: %.5f, vx: %.5f, vy: %.5f, vz: %.5f, mass: %.5f, charge: %.5f \n", p.x, p.y, p.z, p.vx, p.vy, p.vz, p.mass, p.charge);
	}
	
	int main(int argc, char** argv) {
		mkt::init_mkt();
		int number_of_devices = 0;
		cudaGetDeviceCount(&number_of_devices);
		printf("number of devices: %i\n", number_of_devices);

		mkt::DArray<Particle> P(0, 50000, 50000, Particle{}, 1, 0, 0, mkt::DIST, mkt::DIST);
		mkt::DArray<Particle> oldP(0, 50000, 50000, Particle{}, 1, 0, 0, mkt::COPY, mkt::COPY);
		//curandState* devStates[4];
		
		// #pragma omp parallel for
		// for(int gpu = 0; gpu < 4; ++gpu){
		// 	cudaSetDevice(gpu);
		// 	cudaMalloc((void **)&devStates[gpu], 64 * 1024 * sizeof(curandState)));
		// 	setup_kernel<<<64, 1024>>>(devStates[gpu]);
		// }
		
		

		Init_particles_map_index_in_place_array_functor init_particles_map_index_in_place_array_functor{};
		Calc_force_map_index_in_place_array_functor calc_force_map_index_in_place_array_functor{oldP};
		
		
		
		//printf("map init\n");
		mkt::map_index_in_place<Particle, Init_particles_map_index_in_place_array_functor>(P, init_particles_map_index_in_place_array_functor);
		//printf("gather init\n");
		mkt::gather<Particle>(P, oldP);
		mkt::sync_streams();

		Particle in_particle = oldP.get_global(0);
		showParticle(in_particle);
		double gather_time = 0.0;
		double map_time = 0.0;

		std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
		for(int i = 0; ((i) < (0)); ++i){
			mkt::sync_streams();
			std::chrono::high_resolution_clock::time_point map_timer_start = std::chrono::high_resolution_clock::now();
			//printf("map iteration %i\n", i);
			mkt::map_index_in_place<Particle, Calc_force_map_index_in_place_array_functor>(P, calc_force_map_index_in_place_array_functor);

			mkt::sync_streams();
			//printf("map end iteration %i\n", i);
			std::chrono::high_resolution_clock::time_point map_timer_end = std::chrono::high_resolution_clock::now();

			map_time += std::chrono::duration<double>(map_timer_end - map_timer_start).count();

			std::chrono::high_resolution_clock::time_point gather_timer_start = std::chrono::high_resolution_clock::now();

			//printf("gather iteration %i\n", i);
			mkt::gather<Particle>(P, oldP);
			mkt::sync_streams();
			//printf("gather iteration %i\n", i);

			std::chrono::high_resolution_clock::time_point gather_timer_end = std::chrono::high_resolution_clock::now();

			gather_time += std::chrono::duration<double>(gather_timer_end - gather_timer_start).count();
		}
		mkt::sync_streams();
		std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
		double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
		
		Particle out_particle = oldP.get_global(0);
		showParticle(out_particle);

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
