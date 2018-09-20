#include <mpi.h>
#include <omp.h>
#include <array>
#include <chrono>
#include <stddef.h> 
#include <cstdio>
#include <algorithm>
#include "../include/mpi.hpp"

struct Data{
	std::array<float,N> numbers;
	void print() const;
};

void Data::print() const{
	printf("[");
	for(int i = 0; i < N-1; i++){
		printf("%.0f, ", numbers[i]);
	}
	printf("%.0f]\n", numbers[N-1]);
}

Data set_2(Data d){	
	for(int i = 0; i < N; ++i){
		d.numbers[i] = 2;
	}
	return d;
} 

Data square(Data d){	
	for(int i = 0; i < N; ++i){
		d.numbers[i] = d.numbers[i] * d.numbers[i];
	}
	return d;
}

Data sum_omp(Data a, Data b){

	for(int i = 0; i < a.numbers.size(); ++i){
		a.numbers[i] += b.numbers[i];
	}

	return a;
}


void sum_mpi(void *in, void *inout, int *len, MPI_Datatype *dptr){
	Data* inv = static_cast<Data*>(in);
	Data* inoutv = static_cast<Data*>(inout);

	for(int i = 0; i < inoutv->numbers.size(); ++i){
		inoutv->numbers[i] += inv->numbers[i];
	}
}

void print(const std::array<Data, M>& a){
	printf("[\n");
	for(int i = 0; i < a.size(); i++){
		a[i].print();
		printf("\n");
	}
	printf("]");
}

int main(int argc, char** argv) {

	// mpi setup
	MPI_Init(&argc, &argv);
	
	// constants and global vars
	int process_id = -1;
	int mpi_world_size = -1;

	MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
	
	MPI_Comm_rank(MPI_COMM_WORLD, &process_id);

	std::array<Data, M> ds;

	if(process_id == 0){
		printf("Run MPI reduce test\n\n");	
		printf("Initial: %.0f\n", ds[42].numbers[17]); 

		//print(ds);
	}	

	#pragma omp declare reduction(data_reduction : Data : omp_out = sum_omp(omp_out, omp_in)) initializer(omp_priv = omp_orig)

	Data r{};

	std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();

	#pragma omp parallel for simd reduction(data_reduction:r)
	for(int i = 0; i < M; ++i){
		// mapInPlace
		ds[i] = set_2(ds[i]);
		ds[i] = square(ds[i]);
		r = sum_omp(r, ds[i]);
	}

	MPI_Op sum_op;
	MPI_Op_create( sum_mpi, 0, &sum_op );
	
	MPI_Datatype data_mpi_type_temp, data_mpi_type;
	MPI_Type_create_struct(
		1, 
		(std::array<int,1>{N}).data(),
		(std::array<MPI_Aint,1>{static_cast<MPI_Aint>(offsetof(struct Data, numbers))}).data(), 
		(std::array<MPI_Datatype,1>{MPI_FLOAT}).data(), 
		&data_mpi_type_temp);
	MPI_Type_create_resized(data_mpi_type_temp, 0, sizeof(Data), &data_mpi_type);
	MPI_Type_free(&data_mpi_type_temp);
	MPI_Type_commit(&data_mpi_type);	


	Data rg{};

	//rg.print();

	MPI_Allreduce(&r, &rg, 1, data_mpi_type, sum_op, MPI_COMM_WORLD);

	//rg.print();

	std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
	double seconds = std::chrono::duration<double>(timer_end - timer_start).count();

	if(process_id == 0){
		printf("\nResult Map: %.0f", ds[42].numbers[17]);
		printf("\nResult local Fold: %.5f", r.numbers[42]);
		printf("\nResult Fold: %.0f", rg.numbers[32]);
		//r.print();
		std::string s;
		if(rg.numbers[42] == (2 * 2 * M * mpi_world_size)){
			s = "correct";
		}else{
			s = "incorrect";
		}
		printf("\nThe fold result seems to be %s.", s.c_str());

		printf("\nExecution time: %.5fs\n", seconds);
	}
	MPI_Type_free(&data_mpi_type);
	MPI_Finalize();
	return EXIT_SUCCESS;
}
