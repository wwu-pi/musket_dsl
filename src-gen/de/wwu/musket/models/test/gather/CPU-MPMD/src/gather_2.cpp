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
#include "../include/gather_2.hpp"

const size_t number_of_processes = 4;
const size_t process_id = 2;
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
	
	
	
	ad[0] = 9;
	ad[1] = 10;
	ad[2] = 11;
	ad[3] = 12;
	
	
	
	MPI_Datatype md_partition_type, md_partition_type_resized;
	MPI_Type_vector(2, 2, 4, MPI_INT, &md_partition_type);
	MPI_Type_create_resized(md_partition_type, 0, sizeof(int) * 2, &md_partition_type_resized);
	MPI_Type_free(&md_partition_type);
	MPI_Type_commit(&md_partition_type_resized);

	
	
	size_t row_offset = 0;size_t col_offset = 0;
	
	
	MPI_Gather(ad.data(), 4, MPI_INT, nullptr, 4, MPI_INT, 0, MPI_COMM_WORLD);
	
	std::ostringstream s130;
	s130 << "ac on 2: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s130 << ac[i];
	s130 << "; ";
	}
	s130 << ac[15] << "]" << std::endl;
	s130 << std::endl;
	printf("%s", s130.str().c_str());
	std::ostringstream s131;
	s131 << "ad on 2: " << std::endl << "[";
	for (int i = 0; i < 3; i++) {
	s131 << ad[i];
	s131 << "; ";
	}
	s131 << ad[3] << "]" << std::endl;
	s131 << std::endl;
	printf("%s", s131.str().c_str());
	MPI_Allgather(ad.data(), 4, MPI_INT, ac.data(), 4, MPI_INT, MPI_COMM_WORLD);
	
	std::ostringstream s132;
	s132 << "ac on 2: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s132 << ac[i];
	s132 << "; ";
	}
	s132 << ac[15] << "]" << std::endl;
	s132 << std::endl;
	printf("%s", s132.str().c_str());
	row_offset = 2;
	col_offset = 0;
	#pragma omp parallel for 
	for(size_t counter_rows = 0; counter_rows < 2; ++counter_rows){
		#pragma omp simd
		for(size_t counter_cols = 0; counter_cols < 2; ++counter_cols){
			size_t counter = counter_rows * 2 + counter_cols;
			md[counter] = lambda8_functor(row_offset + counter_rows, col_offset + counter_cols, md[counter]);
		}
	}
	
	MPI_Gather(md.data(), 4, MPI_INT, nullptr, 4, MPI_INT, 0, MPI_COMM_WORLD);
	
	std::ostringstream s135;
	s135 << "mc on 2: " << std::endl << "[";
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s135 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s135 << mc[counter_rows * 4 + counter_cols];
			if(counter_cols < 3){
				s135 << "; ";
			}else{
				s135 << "]" << std::endl;
			}
		}
	}
	printf("%s", s135.str().c_str());
	std::ostringstream s136;
	s136 << "md on 2: " << std::endl << "[";
	
	for(int counter_rows = 0; counter_rows < 2; ++counter_rows){
		s136 << "[";
		for(int counter_cols = 0; counter_cols < 2; ++counter_cols){
			s136 << md[counter_rows * 2 + counter_cols];
			if(counter_cols < 1){
				s136 << "; ";
			}else{
				s136 << "]" << std::endl;
			}
		}
	}
	printf("%s", s136.str().c_str());
	MPI_Allgatherv(md.data(), 4, MPI_INT, mc.data(), (std::array<int, 4>{1, 1, 1, 1}).data(), (std::array<int, 4>{0, 1, 4, 5}).data(), md_partition_type_resized, MPI_COMM_WORLD);
	//MPI_Allgather(md.data(), 4, MPI_INT, mc.data(), 1, md_partition_type, MPI_COMM_WORLD);
	
	std::ostringstream s138;
	s138 << "mc on 2: " << std::endl << "[";
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s138 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s138 << mc[counter_rows * 4 + counter_cols];
			if(counter_cols < 3){
				s138 << "; ";
			}else{
				s138 << "]" << std::endl;
			}
		}
	}
	printf("%s", s138.str().c_str());
	
	
	MPI_Finalize();
	return EXIT_SUCCESS;
}
