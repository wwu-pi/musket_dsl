	
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
	#include "../include/nbody-n-1-g-4_0.hpp"
	
	
	std::vector<std::mt19937> random_engines;
	std::array<float*, 4> rns_pointers;
	std::array<float, 100000> rns;	
	std::vector<std::uniform_real_distribution<float>> rand_dist_float_0_0f_1_0f;
	
			
	const int dim = 500000;
	const int steps = 5;
	const float EPSILON = 1.0E-10f;
	const float DT = 0.01f;


// 	struct Particle{
// 	float x;
// 	float y;
// 	float z;
// 	float vx;
// 	float vy;
// 	float vz;
// 	float mass;
// 	float charge;
	
// 	//Particle();
// };

mkt::DArray<float> P_x(0, 500000, 500000, 0, 1, 0, 0, mkt::DIST, mkt::DIST);
mkt::DArray<float> P_y(0, 500000, 500000, 0, 1, 0, 0, mkt::DIST, mkt::DIST);
mkt::DArray<float> P_z(0, 500000, 500000, 0, 1, 0, 0, mkt::DIST, mkt::DIST);
mkt::DArray<float> P_vx(0, 500000, 500000, 0, 1, 0, 0, mkt::DIST, mkt::DIST);
mkt::DArray<float> P_vy(0, 500000, 500000, 0, 1, 0, 0, mkt::DIST, mkt::DIST);
mkt::DArray<float> P_vz(0, 500000, 500000, 0, 1, 0, 0, mkt::DIST, mkt::DIST);
mkt::DArray<float> P_charge(0, 500000, 500000, 0, 1, 0, 0, mkt::DIST, mkt::DIST);
mkt::DArray<float> P_mass(0, 500000, 500000, 0, 1, 0, 0, mkt::DIST, mkt::DIST);

mkt::DArray<float> oldP_x(0, 500000, 500000, 0, 1, 0, 0, mkt::COPY, mkt::COPY);
mkt::DArray<float> oldP_y(0, 500000, 500000, 0, 1, 0, 0, mkt::COPY, mkt::COPY);
mkt::DArray<float> oldP_z(0, 500000, 500000, 0, 1, 0, 0, mkt::COPY, mkt::COPY);
mkt::DArray<float> oldP_vx(0, 500000, 500000, 0, 1, 0, 0, mkt::COPY, mkt::COPY);
mkt::DArray<float> oldP_vy(0, 500000, 500000, 0, 1, 0, 0, mkt::COPY, mkt::COPY);
mkt::DArray<float> oldP_vz(0, 500000, 500000, 0, 1, 0, 0, mkt::COPY, mkt::COPY);
mkt::DArray<float> oldP_mass(0, 500000, 500000, 0, 1, 0, 0, mkt::COPY, mkt::COPY);
mkt::DArray<float> oldP_charge(0, 500000, 500000, 0, 1, 0, 0, mkt::COPY, mkt::COPY);

	mkt::DArray<Particle> P(0, 500000, 500000, Particle{}, 1, 0, 0, mkt::DIST, mkt::DIST);
	mkt::DArray<Particle> oldP(0, 500000, 500000, Particle{}, 1, 0, 0, mkt::COPY, mkt::COPY);
	
	//Particle::Particle() : x(), y(), z(), vx(), vy(), vz(), mass(), charge() {}
	

	
	struct Init_particles_map_index_in_place_array_functor{
		
		Init_particles_map_index_in_place_array_functor(std::array<float*, 4> rns_pointers){
			for(int gpu = 0; gpu < 4; gpu++){
			 	_rns_pointers[gpu] = rns_pointers[gpu];
			}
			_rns_index = 0;
		}
		
		~Init_particles_map_index_in_place_array_functor() {}
		
		auto operator()(int i, Particle& p){
			size_t local_rns_index  = _gang + _worker + _vector + _rns_index; // this can probably be improved
			local_rns_index  = (local_rns_index + 0x7ed55d16) + (local_rns_index << 12);
			local_rns_index = (local_rns_index ^ 0xc761c23c) ^ (local_rns_index >> 19);
			local_rns_index = (local_rns_index + 0x165667b1) + (local_rns_index << 5);
			local_rns_index = (local_rns_index + 0xd3a2646c) ^ (local_rns_index << 9);
			local_rns_index = (local_rns_index + 0xfd7046c5) + (local_rns_index << 3);
			local_rns_index = (local_rns_index ^ 0xb55a4f09) ^ (local_rns_index >> 16);
			local_rns_index = local_rns_index % 100000;
			_rns_index++;
			p.x = static_cast<float>(_rns[local_rns_index++] * (1.0f - 0.0f + 0.999999) + 0.0f);
			p.y = static_cast<float>(_rns[local_rns_index++] * (1.0f - 0.0f + 0.999999) + 0.0f);
			p.z = static_cast<float>(_rns[local_rns_index++] * (1.0f - 0.0f + 0.999999) + 0.0f);
			p.vx = 0.0f;
			p.vy = 0.0f;
			p.vz = 0.0f;
			p.mass = 1.0f;
			p.charge = (1.0f - (2.0f * static_cast<float>(((i) % 2))));
		}
	
		void init(int gpu){
			_rns = _rns_pointers[gpu];
			std::random_device rd{};
			std::mt19937 d_rng_gen(rd());
			std::uniform_int_distribution<> d_rng_dis(0, 100000);
			_rns_index = d_rng_dis(d_rng_gen);
		}
		
		void set_id(int gang, int worker, int vector){
			_gang = gang;
			_worker = worker;
			_vector = vector;
		}
		
		
		
		float* _rns;
		std::array<float*, 4> _rns_pointers;
		size_t _rns_index;
		
		int _gang;
		int _worker;
		int _vector;
	};



	struct Calc_force_map_index_in_place_array_functor{
		
		// Calc_force_map_index_in_place_array_functor(const mkt::DArray<Particle>& _oldP_x, const mkt::DArray<Particle>& _oldP_y, const mkt::DArray<Particle>& _oldP_z, const mkt::DArray<Particle>& _oldP_charge) : oldP_x(_oldP_x), oldP_y(_oldP_y), oldP_z(_oldP_z), oldP_charge(_oldP_charge){
		// }
		Calc_force_map_index_in_place_array_functor(){
		}
		
		~Calc_force_map_index_in_place_array_functor() {}
		
		auto operator()(int curIndex, Particle& curParticle, float* p_oldP_x, float* p_oldP_y, float* p_oldP_z, float* p_oldP_charge){
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
				dx = (curParticle.x - p_oldP_x[j]);
				dy = (curParticle.y - p_oldP_y[j]);
				dz = (curParticle.z - p_oldP_z[j]);
				r2 = ((((dx) * (dx)) + ((dy) * (dy))) + ((dz) * (dz)));
				r = sqrtf((r2));
				
				if(((r) < (EPSILON))){
				qj_by_r3 = 0.0f;
				}
				 else {
						qj_by_r3 = (p_oldP_charge[j] / ((r2) * (r)));
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
	
		// void init(int gpu){
		// 	oldP_y.init(gpu);
		// 	oldP_x.init(gpu);
		// 	oldP_z.init(gpu);
		// 	oldP_charge.init(gpu);
		// }
		
		// void set_id(int gang, int worker, int vector){
		// 	_gang = gang;
		// 	_worker = worker;
		// 	_vector = vector;
		// }
		
		
		// mkt::DeviceArray<Particle> oldP_x;
		// mkt::DeviceArray<Particle> oldP_y;
		// mkt::DeviceArray<Particle> oldP_z;
		// mkt::DeviceArray<Particle> oldP_charge;
		
		
		// int _gang;
		// int _worker;
		// int _vector;
	};
	
	
	
	void wait_all_gpus(){
		#pragma omp parallel for
		for(int gpu = 0; gpu < 4; ++gpu){
			acc_set_device_num(gpu, acc_device_not_host);
			acc_wait_all();
		}
	}
	
	template<typename T>
	void mkt::map_index_in_place(mkt::DArray<T>& a){
	unsigned int offset = a.get_offset();
	unsigned int gpu_elements = a.get_size_gpu();
			  
  	//#pragma omp parallel for
  	for(int gpu = 0; gpu < 4; ++gpu){
		acc_set_device_num(gpu, acc_device_not_host);
		T* devptr = a.get_device_pointer(gpu);
		
		unsigned int gpu_offset = offset;
		if(a.get_device_distribution() == mkt::Distribution::DIST){
			gpu_offset += gpu * gpu_elements;
		}

		float* d_oldP_x = oldP_x.get_device_pointer(gpu);
		unsigned int size_oldP_x = oldP_x.get_size_gpu();

		float* d_oldP_y = oldP_y.get_device_pointer(gpu);
		unsigned int size_oldP_y = oldP_x.get_size_gpu();

		float* d_oldP_z = oldP_z.get_device_pointer(gpu);
		unsigned int size_oldP_z = oldP_x.get_size_gpu();

		float* d_oldP_charge = oldP_charge.get_device_pointer(gpu);
		unsigned int size_oldP_charge = oldP_x.get_size_gpu();

		#pragma acc parallel loop deviceptr(devptr, d_oldP_x, d_oldP_y, d_oldP_z, d_oldP_charge) copyin(gpu_offset) async(0)
	  	for(unsigned int i = 0; i < gpu_elements; ++i) {
			#pragma acc cache(d_oldP_x[0:500000], d_oldP_y[0:500000], d_oldP_z[0:500000], d_oldP_charge[0:500000])
			// #pragma acc cache(devptr[:gpu_elements], oldpdevice[0:500000] )
	    	// f(i + gpu_offset, devptr[i]);
			//Particle* curParticle = devptr[i];
			// const int curIndex = i + gpu_offset;

			Calc_force_map_index_in_place_array_functor f{};
			f(i + gpu_offset, devptr[i], d_oldP_x, d_oldP_y,d_oldP_z,d_oldP_charge);
	  	}
  	}
}
	
	
	int main(int argc, char** argv) {
		
		
		random_engines.reserve(24);
		std::random_device rd;
		for(size_t counter = 0; counter < 24; ++counter){
			random_engines.push_back(std::mt19937(rd()));
		}
		std::mt19937 d_rng_gen(rd());
		std::uniform_real_distribution<float> d_rng_dis(0.0f, 1.0f);
		for(int random_number = 0; random_number < 100000; random_number++){
			rns[random_number] = d_rng_dis(d_rng_gen);
		}
		
		#pragma omp parallel for
		for(int gpu = 0; gpu < 4; ++gpu){
			acc_set_device_num(gpu, acc_device_not_host);
			float* devptr = static_cast<float*>(acc_malloc(100000 * sizeof(float)));
			rns_pointers[gpu] = devptr;
			acc_memcpy_to_device(devptr, rns.data(), 100000 * sizeof(float));
		}
		
		Init_particles_map_index_in_place_array_functor init_particles_map_index_in_place_array_functor{rns_pointers};
		// Calc_force_map_index_in_place_array_functor calc_force_map_index_in_place_array_functor{oldP};
		
		rand_dist_float_0_0f_1_0f.reserve(24);
		for(size_t counter = 0; counter < 24; ++counter){
			rand_dist_float_0_0f_1_0f.push_back(std::uniform_real_distribution<float>(0.0f, 1.0f));
		}
		
				
		
		mkt::map_local_index_in_place<Particle, Init_particles_map_index_in_place_array_functor>(P, init_particles_map_index_in_place_array_functor);
		mkt::gather<Particle>(P, oldP);
		for(int gpu = 0; gpu < 4; ++gpu){
			acc_set_device_num(gpu, acc_device_not_host);
			acc_wait_all();
		}

		double gather_time = 0.0;
		double map_time = 0.0;

		std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
		for(int i = 0; ((i) < (steps)); ++i){
			wait_all_gpus();
			std::chrono::high_resolution_clock::time_point map_timer_start = std::chrono::high_resolution_clock::now();
			mkt::map_index_in_place<Particle>(P);

			wait_all_gpus();
			std::chrono::high_resolution_clock::time_point map_timer_end = std::chrono::high_resolution_clock::now();

			map_time += std::chrono::duration<double>(map_timer_end - map_timer_start).count();

			std::chrono::high_resolution_clock::time_point gather_timer_start = std::chrono::high_resolution_clock::now();

			mkt::gather<Particle>(P, oldP);
			wait_all_gpus();

			std::chrono::high_resolution_clock::time_point gather_timer_end = std::chrono::high_resolution_clock::now();

			gather_time += std::chrono::duration<double>(gather_timer_end - gather_timer_start).count();
		}
		for(int gpu = 0; gpu < 4; ++gpu){
			acc_set_device_num(gpu, acc_device_not_host);
			acc_wait_all();
		}
		std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
		double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
		
		printf("Execution time: %.5fs\n", seconds);

		printf("map time: %.5fs\n", map_time);
		printf("gather time: %.5fs\n", gather_time);

		printf("Threads: %i\n", omp_get_max_threads());
		printf("Processes: %i\n", 1);
		
		#pragma omp parallel for
		for(int gpu = 0; gpu < 4; ++gpu){
			acc_set_device_num(gpu, acc_device_not_host);
			acc_free(rns_pointers[gpu]);
		}
		return EXIT_SUCCESS;
		}
