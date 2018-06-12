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
#include "../include/frobenius-valgrind-n-4-c-24_3.hpp"

const size_t number_of_processes = 4;
const size_t process_id = 3;
int mpi_rank = -1;
int mpi_world_size = 0;

size_t tmp_size_t = 0;


std::vector<double> as(65536);



// generate Function
inline auto sum_function(double a, double b){
	return ((a) + (b));
}

struct Init_functor{
	auto operator()(int x, int y, double a) const{
		return ((static_cast<double>((x)) + (y)) + 1);
	}
};
struct Square_functor{
	auto operator()(double a) const{
		return ((a) * (a));
	}
};
struct Sum_functor{
	auto operator()(double a, double b) const{
		return ((a) + (b));
	}
};

void sum(void* in, void* inout, int *len, MPI_Datatype *dptr){
	double* inv = static_cast<double*>(in);
	double* inoutv = static_cast<double*>(inout);
	*inoutv = sum_function(*inv, *inoutv);
} 

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	
	if(mpi_world_size != number_of_processes || mpi_rank != process_id){
		MPI_Finalize();
		return EXIT_FAILURE;
	}			
	
	
	
	Init_functor init_functor{};
	Square_functor square_functor{};
	Sum_functor sum_functor{};
	
	
	
	#pragma omp declare reduction(sum_reduction : double : omp_out = sum_function(omp_in, omp_out)) initializer(omp_priv = omp_orig)
	
	
	
	MPI_Datatype as_partition_type, as_partition_type_resized;
	MPI_Type_vector(256, 256, 512, MPI_DOUBLE, &as_partition_type);
	MPI_Type_create_resized(as_partition_type, 0, sizeof(double) * 256, &as_partition_type_resized);
	MPI_Type_free(&as_partition_type);
	MPI_Type_commit(&as_partition_type_resized);

	MPI_Op sum_reduction_mpi_op;
	MPI_Op_create( sum, 0, &sum_reduction_mpi_op );
	double fold_result_double;
	
	
	size_t row_offset = 0;size_t col_offset = 0;
	
	row_offset = 256;
	col_offset = 256;
	#pragma omp parallel for 
	for(size_t counter_rows = 0; counter_rows < 256; ++counter_rows){
		#pragma omp simd
		for(size_t counter_cols = 0; counter_cols < 256; ++counter_cols){
			size_t counter = counter_rows * 256 + counter_cols;
			as[counter] = init_functor(row_offset + counter_rows, col_offset + counter_cols, as[counter]);
		}
	}
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 65536; ++counter){
		as[counter] = square_functor(as[counter]);
	}
	double fn = 0.0;
	
	fold_result_double = 0.0;
	
	#pragma omp parallel for simd reduction(sum_reduction:fold_result_double)
	for(size_t counter = 0; counter < 65536; ++counter){
		fold_result_double = sum_functor(fold_result_double, as[counter]);
	}
	
	MPI_Allreduce(&fold_result_double, &fn, 4, MPI_DOUBLE, sum_reduction_mpi_op, MPI_COMM_WORLD); 
	fn = std::sqrt((fn));
	
	
	MPI_Finalize();
	return EXIT_SUCCESS;
}
