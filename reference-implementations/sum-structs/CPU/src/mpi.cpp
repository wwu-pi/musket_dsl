// reference implementation for nbody simulation with 4 processes and 4 cores per process

#include <mpi.h>
#include <omp.h>
#include <vector>
#include <sstream>
#include <chrono>
#include <random>
#include "../include/mpi.hpp"

struct Complex{

	int number;
	//std::vector<int> numbers;
	std::array<int,4> numbers;
	int number2;

	Complex(): number{42}, numbers{1,2,3,4}, number2{17}{}

	void show() const {printf("Number: %i; Numbers: %i, %i, %i, %i; Number2: %i.\n", number, numbers[0], numbers[1], numbers[2], numbers[3], number2);}
};

// constants and global vars
const size_t number_of_processes = 4;
int process_id = -1;
size_t tmp_size_t = 0;

void vector_sum(void *in, void *inout, int *len, MPI_Datatype *dptr){
	int* inv = static_cast<int*>(in);
	int* inoutv = static_cast<int*>(inout);

	for(size_t i = 0; i < *len; ++i){
		inoutv[i] += inv[i];
	}
} 

void complex_sum(void *in, void *inout, int *len, MPI_Datatype *dptr){
	Complex* inv = static_cast<Complex*>(in);
	Complex* inoutv = static_cast<Complex*>(inout);

	inoutv->number += inv->number;

	for(int i = 0; i < inoutv->numbers.size(); ++i){
		inoutv->numbers[i] += inv->numbers[i];
	}

	inoutv->number2 += inv->number2;
}


int main(int argc, char** argv) {

	// mpi setup
	MPI_Init(&argc, &argv);
	
	int mpi_world_size = 0;
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
	
	if(mpi_world_size != number_of_processes){
		MPI_Finalize();
		return EXIT_FAILURE;
	}
	
	MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
	

	Complex c{};

	if(process_id == 0){
		printf("Run MPI reduce test\n\n");	
		c.show(); 
	}

	MPI_Op vector_sum_op;
	MPI_Op_create( vector_sum, 0, &vector_sum_op );
	
	MPI_Op complex_sum_op;
	MPI_Op_create( complex_sum, 0, &complex_sum_op );
	
	MPI_Datatype complex_mpi_type_temp, complex_mpi_type;
	MPI_Type_create_struct(3, (std::array<int,3>{1, 4, 1}).data(), (std::array<MPI_Aint,3>{static_cast<MPI_Aint>(offsetof(struct Complex, number)), static_cast<MPI_Aint>(offsetof(struct Complex, numbers)), static_cast<MPI_Aint>(offsetof(struct Complex, number2))}).data(), (std::array<MPI_Datatype,3>{MPI_INT, MPI_INT, MPI_INT}).data(), &complex_mpi_type_temp);
	MPI_Type_create_resized(complex_mpi_type_temp, 0, sizeof(Complex), &complex_mpi_type);
	MPI_Type_free(&complex_mpi_type_temp);
	MPI_Type_commit(&complex_mpi_type);	


	Complex r{};
	MPI_Allreduce(&c, &r, 1, complex_mpi_type, complex_sum_op, MPI_COMM_WORLD);

	if(process_id == 0){
		printf("\nResult:\n");	
		r.show(); 
	}
	MPI_Finalize();
	return EXIT_SUCCESS;
}
