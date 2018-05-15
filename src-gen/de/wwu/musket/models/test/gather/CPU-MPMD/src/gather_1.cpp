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
#include "../include/gather_1.hpp"

const size_t number_of_processes = 4;
const size_t process_id = 1;
int mpi_rank = -1;
int mpi_world_size = 0;

size_t tmp_size_t = 0;


std::vector<int> ad(4);
std::vector<int> ac(16, 0);
std::vector<int> md(4, 0);
std::vector<int> mc(16, 0);




struct Lambda8_functor{
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
	
	
	
	Lambda8_functor lambda8_functor{};
	
	
	
	ad[0] = 5;
	ad[1] = 6;
	ad[2] = 7;
	ad[3] = 8;
	
	
	
	MPI_Datatype md_partition_type, md_partition_type_resized;
	MPI_Type_vector(2, 2, 4, MPI_INT, &md_partition_type);
	MPI_Type_create_resized(md_partition_type, 0, sizeof(int) * 2, &md_partition_type_resized);
	MPI_Type_free(&md_partition_type);
	MPI_Type_commit(&md_partition_type_resized);

	
	
	size_t row_offset = 0;size_t col_offset = 0;
	
	
	MPI_Gather(ad.data(), 4, MPI_INT, nullptr, 4, MPI_INT, 0, MPI_COMM_WORLD);
	
	std::ostringstream s121;
	s121 << "ac on 1: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s121 << ac[i];
	s121 << "; ";
	}
	s121 << ac[15] << "]" << std::endl;
	s121 << std::endl;
	printf("%s", s121.str().c_str());
	std::ostringstream s122;
	s122 << "ad on 1: " << std::endl << "[";
	for (int i = 0; i < 3; i++) {
	s122 << ad[i];
	s122 << "; ";
	}
	s122 << ad[3] << "]" << std::endl;
	s122 << std::endl;
	printf("%s", s122.str().c_str());
	MPI_Allgather(ad.data(), 4, MPI_INT, ac.data(), 4, MPI_INT, MPI_COMM_WORLD);
	
	std::ostringstream s123;
	s123 << "ac on 1: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s123 << ac[i];
	s123 << "; ";
	}
	s123 << ac[15] << "]" << std::endl;
	s123 << std::endl;
	printf("%s", s123.str().c_str());
	row_offset = 0;
	col_offset = 2;
	#pragma omp parallel for 
	for(size_t counter_rows = 0; counter_rows < 2; ++counter_rows){
		#pragma omp simd
		for(size_t counter_cols = 0; counter_cols < 2; ++counter_cols){
			size_t counter = counter_rows * 2 + counter_cols;
			md[counter] = lambda8_functor(row_offset + counter_rows, col_offset + counter_cols, md[counter]);
		}
	}
	
	MPI_Gather(md.data(), 4, MPI_INT, nullptr, 4, MPI_INT, 0, MPI_COMM_WORLD);
	
	std::ostringstream s126;
	s126 << "mc on 1: " << std::endl << "[";
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s126 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s126 << mc[counter_rows * 4 + counter_cols];
			if(counter_cols < 3){
				s126 << "; ";
			}else{
				s126 << "]" << std::endl;
			}
		}
	}
	printf("%s", s126.str().c_str());
	std::ostringstream s127;
	s127 << "md on 1: " << std::endl << "[";
	
	for(int counter_rows = 0; counter_rows < 2; ++counter_rows){
		s127 << "[";
		for(int counter_cols = 0; counter_cols < 2; ++counter_cols){
			s127 << md[counter_rows * 2 + counter_cols];
			if(counter_cols < 1){
				s127 << "; ";
			}else{
				s127 << "]" << std::endl;
			}
		}
	}
	printf("%s", s127.str().c_str());
	MPI_Allgatherv(md.data(), 4, MPI_INT, mc.data(), (std::array<int, 4>{1, 1, 1, 1}).data(), (std::array<int, 4>{0, 1, 4, 5}).data(), md_partition_type_resized, MPI_COMM_WORLD);
	//MPI_Allgather(md.data(), 4, MPI_INT, mc.data(), 1, md_partition_type, MPI_COMM_WORLD);
	
	std::ostringstream s129;
	s129 << "mc on 1: " << std::endl << "[";
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s129 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s129 << mc[counter_rows * 4 + counter_cols];
			if(counter_cols < 3){
				s129 << "; ";
			}else{
				s129 << "]" << std::endl;
			}
		}
	}
	printf("%s", s129.str().c_str());
	
	
	MPI_Finalize();
	return EXIT_SUCCESS;
}
