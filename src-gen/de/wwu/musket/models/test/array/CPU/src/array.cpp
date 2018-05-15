#include <mpi.h>

#include <omp.h>
#include <array>
#include <sstream>
#include <chrono>
#include <random>
#include <limits>
#include <memory>
#include "../include/array.hpp"

const size_t number_of_processes = 4;
int process_id = -1;
size_t tmp_size_t = 0;


const int dim = 16;
std::vector<int> ads(4);
std::vector<int> bds(4);
std::vector<int> cds(4);
std::vector<int> acs(16);
std::vector<int> bcs(16);
std::vector<int> ccs(16);
std::vector<int> temp(4);
std::vector<int> temp_copy(16);


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
	printf("Run Array\n\n");			
	}
	
	
	
	switch(process_id){
	case 0: {
	cds[0] = 1;
	cds[1] = 2;
	cds[2] = 3;
	cds[3] = 4;
	break;
	}
	case 1: {
	cds[0] = 5;
	cds[1] = 6;
	cds[2] = 7;
	cds[3] = 8;
	break;
	}
	case 2: {
	cds[0] = 9;
	cds[1] = 10;
	cds[2] = 11;
	cds[3] = 12;
	break;
	}
	case 3: {
	cds[0] = 13;
	cds[1] = 14;
	cds[2] = 15;
	cds[3] = 16;
	break;
	}
	}
	
	ccs[0] = 1;
	ccs[1] = 2;
	ccs[2] = 3;
	ccs[3] = 4;
	ccs[4] = 5;
	ccs[5] = 6;
	ccs[6] = 7;
	ccs[7] = 8;
	ccs[8] = 9;
	ccs[9] = 10;
	ccs[10] = 11;
	ccs[11] = 12;
	ccs[12] = 13;
	ccs[13] = 14;
	ccs[14] = 15;
	ccs[15] = 16;
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter  < 4; ++counter){
		ads[counter] = 1;
	}
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter  < 4; ++counter){
		bds[counter] = 0;
	}
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter  < 16; ++counter){
		acs[counter] = 7;
	}
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter  < 16; ++counter){
		bcs[counter] = 0;
	}
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter  < 4; ++counter){
		temp[counter] = 0;
	}
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter  < 16; ++counter){
		temp_copy[counter] = 0;
	}
	
	
	std::array<int, 16> temp0{};
	
	tmp_size_t = 4 * sizeof(int);
	MPI_Gather(ads.data(), tmp_size_t, MPI_BYTE, temp0.data(), tmp_size_t, MPI_BYTE, 0, MPI_COMM_WORLD);
	
	if (process_id == 0) {
		std::ostringstream s0;
		s0 << "ads: " << std::endl << "[";
		for (int i = 0; i < 15; i++) {
		s0 << temp0[i];				
		s0 << "; ";
		}
		s0 << temp0[15] << "]" << std::endl;
		s0 << std::endl;
		printf("%s", s0.str().c_str());
	}
	std::array<int, 16> temp1{};
	
	tmp_size_t = 4 * sizeof(int);
	MPI_Gather(bds.data(), tmp_size_t, MPI_BYTE, temp1.data(), tmp_size_t, MPI_BYTE, 0, MPI_COMM_WORLD);
	
	if (process_id == 0) {
		std::ostringstream s1;
		s1 << "bds: " << std::endl << "[";
		for (int i = 0; i < 15; i++) {
		s1 << temp1[i];				
		s1 << "; ";
		}
		s1 << temp1[15] << "]" << std::endl;
		s1 << std::endl;
		printf("%s", s1.str().c_str());
	}
	std::array<int, 16> temp2{};
	
	tmp_size_t = 4 * sizeof(int);
	MPI_Gather(cds.data(), tmp_size_t, MPI_BYTE, temp2.data(), tmp_size_t, MPI_BYTE, 0, MPI_COMM_WORLD);
	
	if (process_id == 0) {
		std::ostringstream s2;
		s2 << "cds: " << std::endl << "[";
		for (int i = 0; i < 15; i++) {
		s2 << temp2[i];				
		s2 << "; ";
		}
		s2 << temp2[15] << "]" << std::endl;
		s2 << std::endl;
		printf("%s", s2.str().c_str());
	}
	
	if (process_id == 0) {
		std::ostringstream s3;
		s3 << "acs: " << std::endl << "[";
		for (int i = 0; i < 15; i++) {
		s3 << acs[i];				
		s3 << "; ";
		}
		s3 << acs[15] << "]" << std::endl;
		s3 << std::endl;
		printf("%s", s3.str().c_str());
	}
	
	if (process_id == 0) {
		std::ostringstream s4;
		s4 << "bcs: " << std::endl << "[";
		for (int i = 0; i < 15; i++) {
		s4 << bcs[i];				
		s4 << "; ";
		}
		s4 << bcs[15] << "]" << std::endl;
		s4 << std::endl;
		printf("%s", s4.str().c_str());
	}
	
	if (process_id == 0) {
		std::ostringstream s5;
		s5 << "ccs: " << std::endl << "[";
		for (int i = 0; i < 15; i++) {
		s5 << ccs[i];				
		s5 << "; ";
		}
		s5 << ccs[15] << "]" << std::endl;
		s5 << std::endl;
		printf("%s", s5.str().c_str());
	}
			#pragma omp parallel for simd
			for(size_t counter = 0; counter < 4; ++counter){
				int map_input = ads[counter];
				
				temp[counter] = ((map_input) + 1);
			}
	std::array<int, 16> temp6{};
	
	tmp_size_t = 4 * sizeof(int);
	MPI_Gather(temp.data(), tmp_size_t, MPI_BYTE, temp6.data(), tmp_size_t, MPI_BYTE, 0, MPI_COMM_WORLD);
	
	if (process_id == 0) {
		std::ostringstream s6;
		s6 << "temp: " << std::endl << "[";
		for (int i = 0; i < 15; i++) {
		s6 << temp6[i];				
		s6 << "; ";
		}
		s6 << temp6[15] << "]" << std::endl;
		s6 << std::endl;
		printf("%s", s6.str().c_str());
	}
			#pragma omp parallel for simd
			for(size_t counter = 0; counter < 16; ++counter){
				int map_input = bcs[counter];
				
				temp_copy[counter] = ((map_input) + 1);
			}
	
	if (process_id == 0) {
		std::ostringstream s7;
		s7 << "temp_copy: " << std::endl << "[";
		for (int i = 0; i < 15; i++) {
		s7 << temp_copy[i];				
		s7 << "; ";
		}
		s7 << temp_copy[15] << "]" << std::endl;
		s7 << std::endl;
		printf("%s", s7.str().c_str());
	}
	
	if(process_id == 0){
	printf("Threads: %i\n", omp_get_max_threads());
	printf("Processes: %i\n", mpi_world_size);
	}
	
	MPI_Finalize();
	return EXIT_SUCCESS;
}
