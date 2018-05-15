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
#include "../include/gather_0.hpp"

const size_t number_of_processes = 4;
const size_t process_id = 0;
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
	
	
	printf("Run Gather\n\n");			
	
	Lambda8_functor lambda8_functor{};
	
	
	
	ad[0] = 1;
	ad[1] = 2;
	ad[2] = 3;
	ad[3] = 4;
	
	
	
	MPI_Datatype md_partition_type, md_partition_type_resized;
	MPI_Type_vector(2, 2, 4, MPI_INT, &md_partition_type);
	MPI_Type_create_resized(md_partition_type, 0, sizeof(int) * 2, &md_partition_type_resized);
	MPI_Type_free(&md_partition_type);
	MPI_Type_commit(&md_partition_type_resized);

	
	
	size_t row_offset = 0;size_t col_offset = 0;
	
	printf("Array:\n");
	
	std::ostringstream s109;
	s109 << "ac: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s109 << ac[i];
	s109 << "; ";
	}
	s109 << ac[15] << "]" << std::endl;
	s109 << std::endl;
	printf("%s", s109.str().c_str());
	std::array<int, 16> temp110{};
	MPI_Gather(ad.data(), 4, MPI_INT, temp110.data(), 4, MPI_INT, 0, MPI_COMM_WORLD);
	
	std::ostringstream s110;
	s110 << "ad: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s110 << temp110[i];
	s110 << "; ";
	}
	s110 << temp110[15] << "]" << std::endl;
	s110 << std::endl;
	printf("%s", s110.str().c_str());
	std::ostringstream s111;
	s111 << "ac on 0: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s111 << ac[i];
	s111 << "; ";
	}
	s111 << ac[15] << "]" << std::endl;
	s111 << std::endl;
	printf("%s", s111.str().c_str());
	std::ostringstream s112;
	s112 << "ad on 0: " << std::endl << "[";
	for (int i = 0; i < 3; i++) {
	s112 << ad[i];
	s112 << "; ";
	}
	s112 << ad[3] << "]" << std::endl;
	s112 << std::endl;
	printf("%s", s112.str().c_str());
	printf("call gather.\n");
	MPI_Allgather(ad.data(), 4, MPI_INT, ac.data(), 4, MPI_INT, MPI_COMM_WORLD);
	
	std::ostringstream s113;
	s113 << "ac: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s113 << ac[i];
	s113 << "; ";
	}
	s113 << ac[15] << "]" << std::endl;
	s113 << std::endl;
	printf("%s", s113.str().c_str());
	std::ostringstream s114;
	s114 << "ac on 0: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s114 << ac[i];
	s114 << "; ";
	}
	s114 << ac[15] << "]" << std::endl;
	s114 << std::endl;
	printf("%s", s114.str().c_str());
	printf("Matrix:\n");
	row_offset = 0;
	col_offset = 0;
	#pragma omp parallel for 
	for(size_t counter_rows = 0; counter_rows < 2; ++counter_rows){
		#pragma omp simd
		for(size_t counter_cols = 0; counter_cols < 2; ++counter_cols){
			size_t counter = counter_rows * 2 + counter_cols;
			md[counter] = lambda8_functor(row_offset + counter_rows, col_offset + counter_cols, md[counter]);
		}
	}
	
	std::ostringstream s115;
	s115 << "mc: " << std::endl << "[" << std::endl;
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s115 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s115 << mc[counter_rows * 4 + counter_cols];
			if(counter_cols < 3){
				s115 << "; ";
			}else{
				s115 << "]" << std::endl;
			}
		}
	}
	
	s115 << "]" << std::endl;
	printf("%s", s115.str().c_str());
	std::array<int, 16> temp116{};
	MPI_Gather(md.data(), 4, MPI_INT, temp116.data(), 4, MPI_INT, 0, MPI_COMM_WORLD);
	
	std::ostringstream s116;
	s116 << "md: " << std::endl << "[" << std::endl;
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s116 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s116 << temp116[(counter_rows%2) * 2 + (counter_rows / 2) * 4 * 2 + (counter_cols/ 2) * 4 + counter_cols%2];
			if(counter_cols < 3){
				s116 << "; ";
			}else{
				s116 << "]" << std::endl;
			}
		}
	}
	
	s116 << "]" << std::endl;
	printf("%s", s116.str().c_str());
	std::ostringstream s117;
	s117 << "mc on 0: " << std::endl << "[";
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s117 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s117 << mc[counter_rows * 4 + counter_cols];
			if(counter_cols < 3){
				s117 << "; ";
			}else{
				s117 << "]" << std::endl;
			}
		}
	}
	printf("%s", s117.str().c_str());
	std::ostringstream s118;
	s118 << "md on 0: " << std::endl << "[";
	
	for(int counter_rows = 0; counter_rows < 2; ++counter_rows){
		s118 << "[";
		for(int counter_cols = 0; counter_cols < 2; ++counter_cols){
			s118 << md[counter_rows * 2 + counter_cols];
			if(counter_cols < 1){
				s118 << "; ";
			}else{
				s118 << "]" << std::endl;
			}
		}
	}
	printf("%s", s118.str().c_str());
	printf("call gather.\n");
	MPI_Allgatherv(md.data(), 4, MPI_INT, mc.data(), (std::array<int, 4>{1, 1, 1, 1}).data(), (std::array<int, 4>{0, 1, 4, 5}).data(), md_partition_type_resized, MPI_COMM_WORLD);
	//MPI_Allgather(md.data(), 4, MPI_INT, mc.data(), 1, md_partition_type, MPI_COMM_WORLD);
	
	std::ostringstream s119;
	s119 << "mc: " << std::endl << "[" << std::endl;
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s119 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s119 << mc[counter_rows * 4 + counter_cols];
			if(counter_cols < 3){
				s119 << "; ";
			}else{
				s119 << "]" << std::endl;
			}
		}
	}
	
	s119 << "]" << std::endl;
	printf("%s", s119.str().c_str());
	std::ostringstream s120;
	s120 << "mc on 0: " << std::endl << "[";
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s120 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s120 << mc[counter_rows * 4 + counter_cols];
			if(counter_cols < 3){
				s120 << "; ";
			}else{
				s120 << "]" << std::endl;
			}
		}
	}
	printf("%s", s120.str().c_str());
	
	printf("Threads: %i\n", omp_get_max_threads());
	printf("Processes: %i\n", mpi_world_size);
	
	MPI_Finalize();
	return EXIT_SUCCESS;
}
