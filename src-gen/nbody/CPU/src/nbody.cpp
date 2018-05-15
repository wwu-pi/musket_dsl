#include <mpi.h>

#include <omp.h>
#include <array>
#include <sstream>
#include <chrono>
#include <random>
#include <limits>
#include <memory>
#include "../include/nbody.hpp"

const size_t number_of_processes = 4;
int process_id = -1;
size_t tmp_size_t = 0;


const int steps = 5;
const double EPSILON = 1.0E-10;
const double DT = 0.01;
std::vector<Particle> P(131072);
std::vector<Particle> oldP(524288);

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
	printf("Run Nbody\n\n");			
	}
	
	
	std::vector<std::mt19937> random_engines;
	random_engines.reserve(24);
	std::random_device rd;
	for(size_t counter = 0; counter < 24; ++counter){
		random_engines.push_back(std::mt19937(rd()));
	}
	
	std::vector<std::uniform_real_distribution<double>> rand_dist_double_0_0_1_0;
							rand_dist_double_0_0_1_0.reserve(24);
							for(size_t counter = 0; counter < 24; ++counter){
								rand_dist_double_0_0_1_0.push_back(std::uniform_real_distribution<double>(0.0, 1.0));
							}
	
	
	size_t elem_offset = 0;
	
	switch(process_id){
	case 0: {
		elem_offset = 0;
		break;
	}
	case 1: {
		elem_offset = 131072;
		break;
	}
	case 2: {
		elem_offset = 262144;
		break;
	}
	case 3: {
		elem_offset = 393216;
		break;
	}
	}
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 131072; ++counter){
		
		P[counter].x = rand_dist_double_0_0_1_0[omp_get_thread_num()](random_engines[omp_get_thread_num()]);
		P[counter].y = rand_dist_double_0_0_1_0[omp_get_thread_num()](random_engines[omp_get_thread_num()]);
		P[counter].z = rand_dist_double_0_0_1_0[omp_get_thread_num()](random_engines[omp_get_thread_num()]);
		P[counter].vx = 0.0;
		P[counter].vy = 0.0;
		P[counter].vz = 0.0;
		P[counter].mass = 1.0;
		P[counter].charge = (1.0 - (2.0 * static_cast<double>((((elem_offset + counter)) % 2))));
	}
	tmp_size_t = 131072 * sizeof(Particle);
	MPI_Allgather(P.data(), tmp_size_t, MPI_BYTE, oldP.data(), tmp_size_t, MPI_BYTE, MPI_COMM_WORLD);
	std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
	for(int i = 0; ((i) < (steps)); ++i){
		switch(process_id){
		case 0: {
			elem_offset = 0;
			break;
		}
		case 1: {
			elem_offset = 131072;
			break;
		}
		case 2: {
			elem_offset = 262144;
			break;
		}
		case 3: {
			elem_offset = 393216;
			break;
		}
		}
		#pragma omp parallel for simd
		for(size_t counter = 0; counter < 131072; ++counter){
			
			double ax = 0.0;
			double ay = 0.0;
			double az = 0.0;
			for(int j = 0; ((j) < 524288); j++){
				
				if(((j) != ((elem_offset + counter)))){
				double dx;
				double dy;
				double dz;
				double r2;
				double r;
				double qj_by_r3;
				dx = ((P[counter]).x - (oldP)[(j)].x);
				dy = ((P[counter]).y - (oldP)[(j)].y);
				dz = ((P[counter]).z - (oldP)[(j)].z);
				r2 = ((((dx) * (dx)) + ((dy) * (dy))) + ((dz) * (dz)));
				r = std::sqrt((r2));
				
				if(((r) < (EPSILON))){
				qj_by_r3 = 0.0;
				}
				 else {
					qj_by_r3 = ((oldP)[(j)].charge / ((r2) * (r)));
				}
				ax += ((qj_by_r3) * (dx));
				ay += ((qj_by_r3) * (dy));
				az += ((qj_by_r3) * (dz));
				}
			}
			double vx0 = (P[counter]).vx;
			double vy0 = (P[counter]).vy;
			double vz0 = (P[counter]).vz;
			double qidt_by_m = (((P[counter]).charge * (DT)) / (P[counter]).mass);
			P[counter].vx += ((ax) * (qidt_by_m));
			P[counter].vy += ((ay) * (qidt_by_m));
			P[counter].vz += ((az) * (qidt_by_m));
			P[counter].x += ((((vx0) + (P[counter]).vx) * (DT)) * 0.5);
			P[counter].y += ((((vy0) + (P[counter]).vy) * (DT)) * 0.5);
			P[counter].z += ((((vz0) + (P[counter]).vz) * (DT)) * 0.5);
		}
		tmp_size_t = 131072 * sizeof(Particle);
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
