#include <mpi.h>

#include <omp.h>
#include <array>
#include <sstream>
#include <chrono>
#include <random>
#include <limits>
#include <memory>
#include "../include/nbody_float-n-4-c-24.hpp"

const size_t number_of_processes = 4;
int process_id = -1;
size_t tmp_size_t = 0;


const int steps = 5;
const float EPSILON = 1.0E-10f;
const float DT = 0.01f;
std::vector<Particle> P(125000);
std::vector<Particle> oldP(500000);

Particle::Particle() : x(), y(), z(), vx(), vy(), vz(), mass(), charge() {}

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	
	int mpi_world_size = 0;
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
	
	if(mpi_world_size != number_of_processes){
		MPI_Finalize();
		return EXIT_FAILURE;
	}
	
	MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
	
	if(process_id == 0){
	printf("Run Nbody_float-n-4-c-24\n\n");			
	}
	
	
	std::vector<std::mt19937> random_engines;
	random_engines.reserve(24);
	std::random_device rd;
	for(size_t counter = 0; counter < 24; ++counter){
		random_engines.push_back(std::mt19937(rd()));
	}
	
	std::vector<std::uniform_real_distribution<float>> rand_dist_float_0_0f_1_0f;
							rand_dist_float_0_0f_1_0f.reserve(24);
							for(size_t counter = 0; counter < 24; ++counter){
								rand_dist_float_0_0f_1_0f.push_back(std::uniform_real_distribution<float>(0.0f, 1.0f));
							}
	
	
	size_t elem_offset = 0;
	
	switch(process_id){
	case 0: {
		elem_offset = 0;
		break;
	}
	case 1: {
		elem_offset = 125000;
		break;
	}
	case 2: {
		elem_offset = 250000;
		break;
	}
	case 3: {
		elem_offset = 375000;
		break;
	}
	}
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 125000; ++counter){
		
		P[counter].x = rand_dist_float_0_0f_1_0f[omp_get_thread_num()](random_engines[omp_get_thread_num()]);
		P[counter].y = rand_dist_float_0_0f_1_0f[omp_get_thread_num()](random_engines[omp_get_thread_num()]);
		P[counter].z = rand_dist_float_0_0f_1_0f[omp_get_thread_num()](random_engines[omp_get_thread_num()]);
		P[counter].vx = 0.0f;
		P[counter].vy = 0.0f;
		P[counter].vz = 0.0f;
		P[counter].mass = 1.0f;
		P[counter].charge = (1.0f - (2.0f * static_cast<float>((((elem_offset + counter)) % 2))));
	}
	tmp_size_t = 125000 * sizeof(Particle);
	MPI_Allgather(P.data(), tmp_size_t, MPI_BYTE, oldP.data(), tmp_size_t, MPI_BYTE, MPI_COMM_WORLD);
	std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
	for(int i = 0; ((i) < (steps)); ++i){
		switch(process_id){
		case 0: {
			elem_offset = 0;
			break;
		}
		case 1: {
			elem_offset = 125000;
			break;
		}
		case 2: {
			elem_offset = 250000;
			break;
		}
		case 3: {
			elem_offset = 375000;
			break;
		}
		}
		#pragma omp parallel for simd
		for(size_t counter = 0; counter < 125000; ++counter){
			
			float ax = 0.0f;
			float ay = 0.0f;
			float az = 0.0f;
			for(int j = 0; ((j) < 500000); j++){
				
				if(((j) != ((elem_offset + counter)))){
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
		tmp_size_t = 125000 * sizeof(Particle);
		MPI_Allgather(P.data(), tmp_size_t, MPI_BYTE, oldP.data(), tmp_size_t, MPI_BYTE, MPI_COMM_WORLD);
	}
	std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
	double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
	
	if(process_id == 0){
	printf("Execution time: %.5fs\n", seconds);
	printf("Threads: %i\n", omp_get_max_threads());
	printf("Processes: %i\n", mpi_world_size);
	}
	
	MPI_Finalize();
	return EXIT_SUCCESS;
}
