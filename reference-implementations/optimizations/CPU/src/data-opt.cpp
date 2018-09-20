#include <mpi.h>
#include <omp.h>
#include <array>
#include <chrono>
#include <cstdio>
#include <utility>
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

void set(int const value, Data& d){
	for(int i = 0; i < N; ++i){
		d.numbers[i] = value;
	}
} 

void square(Data& d){	
	for(int i = 0; i < N; ++i){
		d.numbers[i] = d.numbers[i] * d.numbers[i];
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

	std::array<Data, M> a;
	std::array<Data, M> b;

	#pragma omp parallel for simd
	for(int i = 0; i < M; ++i){
		set(2, a[i]);
		set(3, b[i]);
	}

	if(process_id == 0){
		printf("Run MPI reduce test\n\n");	
		printf("Initial: a = %.0f, b = %.0f\n", a[42].numbers[17], b[32].numbers[19]); 
	}	

	std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();

	std::copy(a.begin(), a.end(), b.begin());

	#pragma omp parallel for simd
	for(int i = 0; i < M; ++i){
		square(b[i]);
	}

	std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
	double seconds = std::chrono::duration<double>(timer_end - timer_start).count();

	if(process_id == 0){
		printf("\nResult Map: a = %.0f, b = %.0f", a[20].numbers[4], b[46].numbers[12]);
		std::string s;
		if(b[32].numbers[19] == 4){
			s = "correct";
		}else{
			s = "incorrect";
		}
		printf("\nThe fold result seems to be %s.", s.c_str());

		printf("\nExecution time: %.5fs\n", seconds);
	}

	MPI_Finalize();
	return EXIT_SUCCESS;
}
