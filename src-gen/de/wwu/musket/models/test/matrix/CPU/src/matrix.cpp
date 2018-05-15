#include <mpi.h>

#include <omp.h>
#include <array>
#include <sstream>
#include <chrono>
#include <random>
#include <limits>
#include <memory>
#include "../include/matrix.hpp"

const size_t number_of_processes = 4;
int process_id = -1;
size_t tmp_size_t = 0;


std::vector<int> ads(4);
std::vector<int> bds(4);
std::vector<int> cds(4);
std::vector<int> acs(16);
std::vector<int> bcs(16);
std::vector<int> ccs(16);


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
	printf("Run Matrix\n\n");			
	}
	
	
	
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter  < 4; ++counter){
		ads[counter] = 7;
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
	
	size_t row_offset = 0;size_t col_offset = 0;
	
	std::array<int, 16> temp8{};
	
	tmp_size_t = 4 * sizeof(int);
	MPI_Gather(ads.data(), tmp_size_t, MPI_BYTE, temp8.data(), tmp_size_t, MPI_BYTE, 0, MPI_COMM_WORLD);
	
	if (process_id == 0) {
		std::ostringstream s8;
		s8 << "ads: " << std::endl << "[" << std::endl;
		
		for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
			s8 << "[";
			for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
				s8 << temp8[(counter_rows%2) * 2 + (counter_rows / 2) * 4 * 2 + (counter_cols/ 2) * 4 + counter_cols%2];
				if(counter_cols < 3){
					s8 << "; ";
				}else{
					s8 << "]" << std::endl;
				}
			}
		}
		
		s8 << "]" << std::endl;
		printf("%s", s8.str().c_str());
	}
	std::array<int, 16> temp9{};
	
	tmp_size_t = 4 * sizeof(int);
	MPI_Gather(bds.data(), tmp_size_t, MPI_BYTE, temp9.data(), tmp_size_t, MPI_BYTE, 0, MPI_COMM_WORLD);
	
	if (process_id == 0) {
		std::ostringstream s9;
		s9 << "bds: " << std::endl << "[" << std::endl;
		
		for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
			s9 << "[";
			for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
				s9 << temp9[(counter_rows%2) * 2 + (counter_rows / 2) * 4 * 2 + (counter_cols/ 2) * 4 + counter_cols%2];
				if(counter_cols < 3){
					s9 << "; ";
				}else{
					s9 << "]" << std::endl;
				}
			}
		}
		
		s9 << "]" << std::endl;
		printf("%s", s9.str().c_str());
	}
	
	if (process_id == 0) {
		std::ostringstream s10;
		s10 << "acs: " << std::endl << "[" << std::endl;
		
		for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
			s10 << "[";
			for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
				s10 << acs[counter_rows * 4 + counter_cols];
				if(counter_cols < 3){
					s10 << "; ";
				}else{
					s10 << "]" << std::endl;
				}
			}
		}
		
		s10 << "]" << std::endl;
		printf("%s", s10.str().c_str());
	}
	
	if (process_id == 0) {
		std::ostringstream s11;
		s11 << "bcs: " << std::endl << "[" << std::endl;
		
		for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
			s11 << "[";
			for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
				s11 << bcs[counter_rows * 4 + counter_cols];
				if(counter_cols < 3){
					s11 << "; ";
				}else{
					s11 << "]" << std::endl;
				}
			}
		}
		
		s11 << "]" << std::endl;
		printf("%s", s11.str().c_str());
	}
	switch(process_id){
	case 0: {
		row_offset = 0;
		col_offset = 0;
		break;
	}
	case 1: {
		row_offset = 0;
		col_offset = 2;
		break;
	}
	case 2: {
		row_offset = 2;
		col_offset = 0;
		break;
	}
	case 3: {
		row_offset = 2;
		col_offset = 2;
		break;
	}
	}		
	#pragma omp parallel for 
	for(size_t counter_rows = 0; counter_rows < 2; ++counter_rows){
		#pragma omp simd
		for(size_t counter_cols = 0; counter_cols < 2; ++counter_cols){
			
			cds[counter_rows * 2 + counter_cols] = ((((row_offset + counter_rows)) * 4) + ((col_offset + counter_cols)));
		}
	}
	#pragma omp parallel for 
	for(size_t counter_rows = 0; counter_rows < 4; ++counter_rows){
		#pragma omp simd
		for(size_t counter_cols = 0; counter_cols < 4; ++counter_cols){
			
			ccs[counter_rows * 4 + counter_cols] = (((counter_rows) * 4) + (counter_cols));
		}
	}
	std::array<int, 16> temp12{};
	
	tmp_size_t = 4 * sizeof(int);
	MPI_Gather(cds.data(), tmp_size_t, MPI_BYTE, temp12.data(), tmp_size_t, MPI_BYTE, 0, MPI_COMM_WORLD);
	
	if (process_id == 0) {
		std::ostringstream s12;
		s12 << "cds: " << std::endl << "[" << std::endl;
		
		for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
			s12 << "[";
			for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
				s12 << temp12[(counter_rows%2) * 2 + (counter_rows / 2) * 4 * 2 + (counter_cols/ 2) * 4 + counter_cols%2];
				if(counter_cols < 3){
					s12 << "; ";
				}else{
					s12 << "]" << std::endl;
				}
			}
		}
		
		s12 << "]" << std::endl;
		printf("%s", s12.str().c_str());
	}
	
	if (process_id == 0) {
		std::ostringstream s13;
		s13 << "ccs: " << std::endl << "[" << std::endl;
		
		for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
			s13 << "[";
			for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
				s13 << ccs[counter_rows * 4 + counter_cols];
				if(counter_cols < 3){
					s13 << "; ";
				}else{
					s13 << "]" << std::endl;
				}
			}
		}
		
		s13 << "]" << std::endl;
		printf("%s", s13.str().c_str());
	}
	
	if(process_id == 0){
	printf("Threads: %i\n", omp_get_max_threads());
	printf("Processes: %i\n", mpi_world_size);
	}
	
	MPI_Finalize();
	return EXIT_SUCCESS;
}
