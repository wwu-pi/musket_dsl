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
#include "../include/scatter_3.hpp"

const size_t number_of_processes = 4;
const size_t process_id = 3;
int mpi_rank = -1;
int mpi_world_size = 0;

size_t tmp_size_t = 0;


std::vector<int> ad(4, 0);
std::vector<int> ac(16);
std::vector<int> md(4, 0);
std::vector<int> mc(16, 0);




struct Lambda7_functor{
	auto operator()(int r, int c, int x) const{
		return (((r) * 4) + (c));
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
	
	
	
	Lambda7_functor lambda7_functor{};
	
	
	
	ac[0] = 1;
	ac[1] = 2;
	ac[2] = 3;
	ac[3] = 4;
	ac[4] = 5;
	ac[5] = 6;
	ac[6] = 7;
	ac[7] = 8;
	ac[8] = 9;
	ac[9] = 10;
	ac[10] = 11;
	ac[11] = 12;
	ac[12] = 13;
	ac[13] = 14;
	ac[14] = 15;
	ac[15] = 16;
	ac[0] = 1;
	ac[1] = 2;
	ac[2] = 3;
	ac[3] = 4;
	ac[4] = 5;
	ac[5] = 6;
	ac[6] = 7;
	ac[7] = 8;
	ac[8] = 9;
	ac[9] = 10;
	ac[10] = 11;
	ac[11] = 12;
	ac[12] = 13;
	ac[13] = 14;
	ac[14] = 15;
	ac[15] = 16;
	
	
	
	MPI_Datatype md_partition_type, md_partition_type_resized;
	MPI_Type_vector(2, 2, 4, MPI_INT, &md_partition_type);
	MPI_Type_create_resized(md_partition_type, 0, sizeof(int) * 2, &md_partition_type_resized);
	MPI_Type_free(&md_partition_type);
	MPI_Type_commit(&md_partition_type_resized);

	
	
	size_t row_offset = 0;size_t col_offset = 0;size_t elem_offset = 0;
	
	
	MPI_Gather(ad.data(), 4, MPI_INT, nullptr, 4, MPI_INT, 0, MPI_COMM_WORLD);
	
	std::ostringstream s100;
	s100 << "ac on 3: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s100 << ac[i];
	s100 << "; ";
	}
	s100 << ac[15] << "]" << std::endl;
	s100 << std::endl;
	printf("%s", s100.str().c_str());
	std::ostringstream s101;
	s101 << "ad on 3: " << std::endl << "[";
	for (int i = 0; i < 3; i++) {
	s101 << ad[i];
	s101 << "; ";
	}
	s101 << ad[3] << "]" << std::endl;
	s101 << std::endl;
	printf("%s", s101.str().c_str());
	elem_offset = 12;
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 4; ++counter){
		ad[counter] = ac[elem_offset + counter];
	}
	MPI_Gather(ad.data(), 4, MPI_INT, nullptr, 4, MPI_INT, 0, MPI_COMM_WORLD);
	
	std::ostringstream s102;
	s102 << "ad on 3: " << std::endl << "[";
	for (int i = 0; i < 3; i++) {
	s102 << ad[i];
	s102 << "; ";
	}
	s102 << ad[3] << "]" << std::endl;
	s102 << std::endl;
	printf("%s", s102.str().c_str());
	row_offset = 0;
	col_offset = 0;
	#pragma omp parallel for 
	for(size_t counter_rows = 0; counter_rows < 4; ++counter_rows){
		#pragma omp simd
		for(size_t counter_cols = 0; counter_cols < 4; ++counter_cols){
			size_t counter = counter_rows * 4 + counter_cols;
			mc[counter] = lambda7_functor(row_offset + counter_rows, col_offset + counter_cols, mc[counter]);
		}
	}
	
	MPI_Gather(md.data(), 4, MPI_INT, nullptr, 4, MPI_INT, 0, MPI_COMM_WORLD);
	
	std::ostringstream s105;
	s105 << "mc on 3: " << std::endl << "[";
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s105 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s105 << mc[counter_rows * 4 + counter_cols];
			if(counter_cols < 3){
				s105 << "; ";
			}else{
				s105 << "]" << std::endl;
			}
		}
	}
	printf("%s", s105.str().c_str());
	std::ostringstream s106;
	s106 << "md on 3: " << std::endl << "[";
	
	for(int counter_rows = 0; counter_rows < 2; ++counter_rows){
		s106 << "[";
		for(int counter_cols = 0; counter_cols < 2; ++counter_cols){
			s106 << md[counter_rows * 2 + counter_cols];
			if(counter_cols < 1){
				s106 << "; ";
			}else{
				s106 << "]" << std::endl;
			}
		}
	}
	printf("%s", s106.str().c_str());
	row_offset = 2;
	col_offset = 2;
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 4; ++counter){
		md[counter] = mc[(counter / 2) * 4 + row_offset * 4 + (counter % 2 + col_offset)];
	}
	MPI_Gather(md.data(), 4, MPI_INT, nullptr, 4, MPI_INT, 0, MPI_COMM_WORLD);
	
	std::ostringstream s108;
	s108 << "md on 3: " << std::endl << "[";
	
	for(int counter_rows = 0; counter_rows < 2; ++counter_rows){
		s108 << "[";
		for(int counter_cols = 0; counter_cols < 2; ++counter_cols){
			s108 << md[counter_rows * 2 + counter_cols];
			if(counter_cols < 1){
				s108 << "; ";
			}else{
				s108 << "]" << std::endl;
			}
		}
	}
	printf("%s", s108.str().c_str());
	
	
	MPI_Finalize();
	return EXIT_SUCCESS;
}
