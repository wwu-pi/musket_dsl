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
#include "../include/scatter_0.hpp"

const size_t number_of_processes = 4;
const size_t process_id = 0;
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
	
	
	printf("Run Scatter\n\n");			
	
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
	
	printf("Array:\n");
	
	std::ostringstream s70;
	s70 << "ac: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s70 << ac[i];
	s70 << "; ";
	}
	s70 << ac[15] << "]" << std::endl;
	s70 << std::endl;
	printf("%s", s70.str().c_str());
	std::array<int, 16> temp71{};
	MPI_Gather(ad.data(), 4, MPI_INT, temp71.data(), 4, MPI_INT, 0, MPI_COMM_WORLD);
	
	std::ostringstream s71;
	s71 << "ad: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s71 << temp71[i];
	s71 << "; ";
	}
	s71 << temp71[15] << "]" << std::endl;
	s71 << std::endl;
	printf("%s", s71.str().c_str());
	std::ostringstream s72;
	s72 << "ac on 0: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s72 << ac[i];
	s72 << "; ";
	}
	s72 << ac[15] << "]" << std::endl;
	s72 << std::endl;
	printf("%s", s72.str().c_str());
	std::ostringstream s73;
	s73 << "ad on 0: " << std::endl << "[";
	for (int i = 0; i < 3; i++) {
	s73 << ad[i];
	s73 << "; ";
	}
	s73 << ad[3] << "]" << std::endl;
	s73 << std::endl;
	printf("%s", s73.str().c_str());
	printf("call scatter.\n");
	elem_offset = 0;
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 4; ++counter){
		ad[counter] = ac[elem_offset + counter];
	}
	std::array<int, 16> temp74{};
	MPI_Gather(ad.data(), 4, MPI_INT, temp74.data(), 4, MPI_INT, 0, MPI_COMM_WORLD);
	
	std::ostringstream s74;
	s74 << "ad: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s74 << temp74[i];
	s74 << "; ";
	}
	s74 << temp74[15] << "]" << std::endl;
	s74 << std::endl;
	printf("%s", s74.str().c_str());
	std::ostringstream s75;
	s75 << "ad on 0: " << std::endl << "[";
	for (int i = 0; i < 3; i++) {
	s75 << ad[i];
	s75 << "; ";
	}
	s75 << ad[3] << "]" << std::endl;
	s75 << std::endl;
	printf("%s", s75.str().c_str());
	printf("Matrix:\n");
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
	
	std::ostringstream s76;
	s76 << "mc: " << std::endl << "[" << std::endl;
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s76 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s76 << mc[counter_rows * 4 + counter_cols];
			if(counter_cols < 3){
				s76 << "; ";
			}else{
				s76 << "]" << std::endl;
			}
		}
	}
	
	s76 << "]" << std::endl;
	printf("%s", s76.str().c_str());
	std::array<int, 16> temp77{};
	MPI_Gather(md.data(), 4, MPI_INT, temp77.data(), 4, MPI_INT, 0, MPI_COMM_WORLD);
	
	std::ostringstream s77;
	s77 << "md: " << std::endl << "[" << std::endl;
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s77 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s77 << temp77[(counter_rows%2) * 2 + (counter_rows / 2) * 4 * 2 + (counter_cols/ 2) * 4 + counter_cols%2];
			if(counter_cols < 3){
				s77 << "; ";
			}else{
				s77 << "]" << std::endl;
			}
		}
	}
	
	s77 << "]" << std::endl;
	printf("%s", s77.str().c_str());
	std::ostringstream s78;
	s78 << "mc on 0: " << std::endl << "[";
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s78 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s78 << mc[counter_rows * 4 + counter_cols];
			if(counter_cols < 3){
				s78 << "; ";
			}else{
				s78 << "]" << std::endl;
			}
		}
	}
	printf("%s", s78.str().c_str());
	std::ostringstream s79;
	s79 << "md on 0: " << std::endl << "[";
	
	for(int counter_rows = 0; counter_rows < 2; ++counter_rows){
		s79 << "[";
		for(int counter_cols = 0; counter_cols < 2; ++counter_cols){
			s79 << md[counter_rows * 2 + counter_cols];
			if(counter_cols < 1){
				s79 << "; ";
			}else{
				s79 << "]" << std::endl;
			}
		}
	}
	printf("%s", s79.str().c_str());
	printf("call scatter.\n");
	row_offset = 0;
	col_offset = 0;
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 4; ++counter){
		md[counter] = mc[(counter / 2) * 4 + row_offset * 4 + (counter % 2 + col_offset)];
	}
	std::array<int, 16> temp80{};
	MPI_Gather(md.data(), 4, MPI_INT, temp80.data(), 4, MPI_INT, 0, MPI_COMM_WORLD);
	
	std::ostringstream s80;
	s80 << "md: " << std::endl << "[" << std::endl;
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s80 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s80 << temp80[(counter_rows%2) * 2 + (counter_rows / 2) * 4 * 2 + (counter_cols/ 2) * 4 + counter_cols%2];
			if(counter_cols < 3){
				s80 << "; ";
			}else{
				s80 << "]" << std::endl;
			}
		}
	}
	
	s80 << "]" << std::endl;
	printf("%s", s80.str().c_str());
	std::ostringstream s81;
	s81 << "md on 0: " << std::endl << "[";
	
	for(int counter_rows = 0; counter_rows < 2; ++counter_rows){
		s81 << "[";
		for(int counter_cols = 0; counter_cols < 2; ++counter_cols){
			s81 << md[counter_rows * 2 + counter_cols];
			if(counter_cols < 1){
				s81 << "; ";
			}else{
				s81 << "]" << std::endl;
			}
		}
	}
	printf("%s", s81.str().c_str());
	
	printf("Threads: %i\n", omp_get_max_threads());
	printf("Processes: %i\n", mpi_world_size);
	
	MPI_Finalize();
	return EXIT_SUCCESS;
}
