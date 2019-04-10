	#include <mpi.h>
	
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
	#include "../include/nbody-n-16-g-2_13.hpp"
	
	const size_t number_of_processes = 16;
	const size_t process_id = 13;
	int mpi_rank = -1;
	int mpi_world_size = 0;
	
	std::vector<std::mt19937> random_engines;
	std::array<float*, 2> rns_pointers;
	std::array<float, 100000> rns;	
	std::vector<std::uniform_real_distribution<float>> rand_dist_float_0_0f_1_0f;
	
			
	const int dim = 500000;
	const int steps = 5;
	const float EPSILON = 1.0E-10f;
	const float DT = 0.01f;
	mkt::DArray<Particle> P(13, 500000, 31250, Particle{}, 4, 13, 406250, mkt::DIST, mkt::DIST);
	mkt::DArray<Particle> oldP(13, 500000, 500000, Particle{}, 1, 13, 0, mkt::COPY, mkt::COPY);
	
	//Particle::Particle() : x(), y(), z(), vx(), vy(), vz(), mass(), charge() {}
	

	
	struct Init_particles_map_index_in_place_array_functor{
		
		Init_particles_map_index_in_place_array_functor(std::array<float*, 2> rns_pointers){
			for(int gpu = 0; gpu < 2; gpu++){
			 	_rns_pointers[gpu] = rns_pointers[gpu];
			}
			_rns_index = 0;
		}
		
		~Init_particles_map_index_in_place_array_functor() {}
		
		auto operator()(int i, Particle p){
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
			return (p);
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
		std::array<float*, 2> _rns_pointers;
		size_t _rns_index;
		
		int _gang;
		int _worker;
		int _vector;
	};
	struct Calc_force_map_index_in_place_array_functor{
		
		Calc_force_map_index_in_place_array_functor(const mkt::DArray<Particle>& _oldP) : oldP(_oldP){
		}
		
		~Calc_force_map_index_in_place_array_functor() {}
		
		auto operator()(int curIndex, Particle curParticle){
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
			return (curParticle);
		}
	
		void init(int gpu){
			oldP.init(gpu);
		}
		
		void set_id(int gang, int worker, int vector){
			_gang = gang;
			_worker = worker;
			_vector = vector;
		}
		
		
		mkt::DeviceArray<Particle> oldP;
		
		
		int _gang;
		int _worker;
		int _vector;
	};
	
	
	
	
	
	
	
	int main(int argc, char** argv) {
		MPI_Init(&argc, &argv);
		
		MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
		MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
		
		if(mpi_world_size != number_of_processes || mpi_rank != process_id){
			MPI_Finalize();
			return EXIT_FAILURE;
		}			
		
		
		
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
		for(int gpu = 0; gpu < 2; ++gpu){
			acc_set_device_num(gpu, acc_device_not_host);
			float* devptr = static_cast<float*>(acc_malloc(100000 * sizeof(float)));
			rns_pointers[gpu] = devptr;
			acc_memcpy_to_device(devptr, rns.data(), 100000 * sizeof(float));
		}
		
		Init_particles_map_index_in_place_array_functor init_particles_map_index_in_place_array_functor{rns_pointers};
		Calc_force_map_index_in_place_array_functor calc_force_map_index_in_place_array_functor{oldP};
		
		rand_dist_float_0_0f_1_0f.reserve(24);
		for(size_t counter = 0; counter < 24; ++counter){
			rand_dist_float_0_0f_1_0f.push_back(std::uniform_real_distribution<float>(0.0f, 1.0f));
		}
		
				
			MPI_Datatype Particle_mpi_type_temp;
			MPI_Type_create_struct(8, (std::array<int,8>{1, 1, 1, 1, 1, 1, 1, 1}).data(), (std::array<MPI_Aint,8>{static_cast<MPI_Aint>(offsetof(struct Particle, x)), static_cast<MPI_Aint>(offsetof(struct Particle, y)), static_cast<MPI_Aint>(offsetof(struct Particle, z)), static_cast<MPI_Aint>(offsetof(struct Particle, vx)), static_cast<MPI_Aint>(offsetof(struct Particle, vy)), static_cast<MPI_Aint>(offsetof(struct Particle, vz)), static_cast<MPI_Aint>(offsetof(struct Particle, mass)), static_cast<MPI_Aint>(offsetof(struct Particle, charge))}).data(), (std::array<MPI_Datatype,8>{MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT}).data(), &Particle_mpi_type_temp);
			MPI_Type_create_resized(Particle_mpi_type_temp, 0, sizeof(Particle), &Particle_mpi_type);
			MPI_Type_free(&Particle_mpi_type_temp);
			MPI_Type_commit(&Particle_mpi_type);
			
			
		
			
		
		
		mkt::map_index_in_place<Particle, Init_particles_map_index_in_place_array_functor>(P, init_particles_map_index_in_place_array_functor);
		mkt::gather<Particle>(P, oldP);
		for(int gpu = 0; gpu < 2; ++gpu){
			acc_set_device_num(gpu, acc_device_not_host);
			acc_wait_all();
		}
		for(int i = 0; ((i) < (steps)); ++i){
			mkt::map_index_in_place<Particle, Calc_force_map_index_in_place_array_functor>(P, calc_force_map_index_in_place_array_functor);
			mkt::gather<Particle>(P, oldP);
		}
		for(int gpu = 0; gpu < 2; ++gpu){
			acc_set_device_num(gpu, acc_device_not_host);
			acc_wait_all();
		}
		
		
		#pragma omp parallel for
		for(int gpu = 0; gpu < 2; ++gpu){
			acc_set_device_num(gpu, acc_device_not_host);
			acc_free(rns_pointers[gpu]);
		}
		MPI_Finalize();
		return EXIT_SUCCESS;
		}
