#include <mpi.h>

#include <omp.h>
#include <array>
#include <sstream>
#include <chrono>
#include <random>
#include <limits>
#include <memory>
#include "../include/lambda.hpp"

const size_t number_of_processes = 4;
int process_id = -1;
size_t tmp_size_t = 0;


const int dim = 16;
std::vector<int> cs(4);


void lambda9(void *in, void *inout, int *len, MPI_Datatype *dptr){
	int* inv = static_cast<int*>(in);
	int* inoutv = static_cast<int*>(inout);
	
	*inoutv = ((*inoutv) + (*inv));
} 
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
	printf("Run Lambda\n\n");			
	}
	
	
	
	switch(process_id){
	case 0: {
	cs[0] = 1;
	cs[1] = 2;
	cs[2] = 3;
	cs[3] = 4;
	break;
	}
	case 1: {
	cs[0] = 5;
	cs[1] = 6;
	cs[2] = 7;
	cs[3] = 8;
	break;
	}
	case 2: {
	cs[0] = 9;
	cs[1] = 10;
	cs[2] = 11;
	cs[3] = 12;
	break;
	}
	case 3: {
	cs[0] = 13;
	cs[1] = 14;
	cs[2] = 15;
	cs[3] = 16;
	break;
	}
	}
	
	#pragma omp declare reduction(lambda9 : int : omp_out = [&](){return ((omp_out) + (omp_in));}()) initializer(omp_priv = omp_orig)
	
	MPI_Op lambda9_mpi_op;
	MPI_Op_create( lambda9, 0, &lambda9_mpi_op );
	int fold_result_int;
	
	std::array<int, 16> temp14{};
	
	tmp_size_t = 4 * sizeof(int);
	MPI_Gather(cs.data(), tmp_size_t, MPI_BYTE, temp14.data(), tmp_size_t, MPI_BYTE, 0, MPI_COMM_WORLD);
	
	if (process_id == 0) {
		std::ostringstream s14;
		s14 << "cs: " << std::endl << "[";
		for (int i = 0; i < 15; i++) {
		s14 << temp14[i];				
		s14 << "; ";
		}
		s14 << temp14[15] << "]" << std::endl;
		s14 << std::endl;
		printf("%s", s14.str().c_str());
	}
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 4; ++counter){
		
		cs[counter] = ((cs[counter]) - 1);
	}
	std::array<int, 16> temp15{};
	
	tmp_size_t = 4 * sizeof(int);
	MPI_Gather(cs.data(), tmp_size_t, MPI_BYTE, temp15.data(), tmp_size_t, MPI_BYTE, 0, MPI_COMM_WORLD);
	
	if (process_id == 0) {
		std::ostringstream s15;
		s15 << "cs: " << std::endl << "[";
		for (int i = 0; i < 15; i++) {
		s15 << temp15[i];				
		s15 << "; ";
		}
		s15 << temp15[15] << "]" << std::endl;
		s15 << std::endl;
		printf("%s", s15.str().c_str());
	}
	int sum = 0;
	
		fold_result_int  = 0;
		
		#pragma omp parallel for simd reduction(lambda9:fold_result_int)
		for(size_t counter = 0; counter < 4; ++counter){
			
			fold_result_int = ((fold_result_int) + (cs[counter]));
		}		
		
		MPI_Allreduce(&fold_result_int, &sum, sizeof(int), MPI_BYTE, lambda9_mpi_op, MPI_COMM_WORLD); 
	if(process_id == 0){
		printf("Sum is: %i! \n",(sum));
	}
	
	if(process_id == 0){
	printf("Threads: %i\n", omp_get_max_threads());
	printf("Processes: %i\n", mpi_world_size);
	}
	
	MPI_Finalize();
	return EXIT_SUCCESS;
}
