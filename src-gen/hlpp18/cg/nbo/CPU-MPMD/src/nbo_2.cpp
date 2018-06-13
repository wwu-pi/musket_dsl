#include <mpi.h>

#include <omp.h>
#include <array>
#include <vector>
#include <sstream>
#include <chrono>
#include <random>
#include <limits>
#include <memory>
#include <cstddef>
#include "../include/nbo_2.hpp"

const size_t number_of_processes = 4;
const size_t process_id = 2;
int mpi_rank = -1;
int mpi_world_size = 0;

std::vector<std::mt19937> random_engines;
std::vector<std::uniform_real_distribution<float>> rand_dist_float_0_0f_1_0f;
size_t tmp_size_t = 0;


const int steps = 2;
const float EPSILON = 1.0E-10f;
const float DT = 0.01f;
std::vector<Particle> P(1250);
std::vector<Particle> oldP(5000);

Particle::Particle() : x(), y(), z(), vx(), vy(), vz(), mass(), charge() {}



struct Init_particles_functor{
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
struct Calc_force_functor{
	auto operator()(int curIndex, Particle curParticle) const{
		float ax = 0.0f;
		float ay = 0.0f;
		float az = 0.0f;
		for(int j = 0; ((j) < 5000); j++){
			
			if(((j) != (curIndex))){
			float dx;
			float dy;
			float dz;
			float r2;
			float r;
			float qj_by_r3;
			dx = ((curParticle).x - (oldP)[(j)].x);
			dy = ((curParticle).y - (oldP)[(j)].y);
			dz = ((curParticle).z - (oldP)[(j)].z);
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
	MPI_Init(&argc, &argv);
	
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	
	if(mpi_world_size != number_of_processes || mpi_rank != process_id){
		MPI_Finalize();
		return EXIT_FAILURE;
	}			
	
	
	
	Init_particles_functor init_particles_functor{};
	Calc_force_functor calc_force_functor{};
	
	random_engines.reserve(24);
	std::random_device rd;
	for(size_t counter = 0; counter < 24; ++counter){
		random_engines.push_back(std::mt19937(rd()));
	}
	
	rand_dist_float_0_0f_1_0f.reserve(24);
	for(size_t counter = 0; counter < 24; ++counter){
		rand_dist_float_0_0f_1_0f.push_back(std::uniform_real_distribution<float>(0.0f, 1.0f));
	}
	
	
	MPI_Datatype Particle_mpi_type_temp, Particle_mpi_type;
	MPI_Type_create_struct(8, (std::array<int,8>{1, 1, 1, 1, 1, 1, 1, 1}).data(), (std::array<MPI_Aint,8>{static_cast<MPI_Aint>(offsetof(struct Particle, x)), static_cast<MPI_Aint>(offsetof(struct Particle, y)), static_cast<MPI_Aint>(offsetof(struct Particle, z)), static_cast<MPI_Aint>(offsetof(struct Particle, vx)), static_cast<MPI_Aint>(offsetof(struct Particle, vy)), static_cast<MPI_Aint>(offsetof(struct Particle, vz)), static_cast<MPI_Aint>(offsetof(struct Particle, mass)), static_cast<MPI_Aint>(offsetof(struct Particle, charge))}).data(), (std::array<MPI_Datatype,8>{MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT}).data(), &Particle_mpi_type_temp);
	MPI_Type_create_resized(Particle_mpi_type_temp, 0, sizeof(Particle), &Particle_mpi_type);
	MPI_Type_free(&Particle_mpi_type_temp);
	MPI_Type_commit(&Particle_mpi_type);
	
	

	
	
	size_t elem_offset = 0;
	
	elem_offset = 2500;
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 1250; ++counter){
		P[counter] = init_particles_functor(elem_offset + counter, P[counter]);
	}
	MPI_Allgather(P.data(), 1250, Particle_mpi_type, oldP.data(), 1250, Particle_mpi_type, MPI_COMM_WORLD);
	for(int i = 0; ((i) < (steps)); ++i){
		elem_offset = 2500;
		#pragma omp parallel for simd
		for(size_t counter = 0; counter < 1250; ++counter){
			P[counter] = calc_force_functor(elem_offset + counter, P[counter]);
		}
		MPI_Allgather(P.data(), 1250, Particle_mpi_type, oldP.data(), 1250, Particle_mpi_type, MPI_COMM_WORLD);
	}
	
	
	MPI_Finalize();
	return EXIT_SUCCESS;
}
