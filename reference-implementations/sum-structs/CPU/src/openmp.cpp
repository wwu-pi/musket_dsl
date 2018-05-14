#include <omp.h>
#include <vector>
#include <array>
#include <sstream>
#include <chrono>
#include <random>
#include "../include/mpi.hpp"

struct Complex{

	int number;
	std::array<int,4> numbers;
	int number2;

	Complex(): number{42}, numbers{1,2,3,4}, number2{17}{}

	void show() const {printf("Number: %i; Numbers: %i, %i, %i, %i; Number2: %i.\n", number, numbers[0], numbers[1], numbers[2], numbers[3], number2);}
};

// constants and global vars
size_t tmp_size_t = 0;

Complex complex_sum(Complex a, Complex b){

	a.number += b.number;

	for(int i = 0; i < a.numbers.size(); ++i){
		a.numbers[i] += b.numbers[i];
	}

	a.number2 += b.number2;

	return a;
}


int main(int argc, char** argv) {

	const int complex_numbers = 10;

	Complex r{};
	r.number = 0;
	r.numbers[0] = 0;
	r.numbers[1] = 0;
	r.numbers[2] = 0;
	r.numbers[3] = 0;
	r.number2 = 0;

	std::vector<Complex> cs {};

	cs.reserve(complex_numbers);

	for(int i = 0; i< complex_numbers; ++i){
		cs.push_back(Complex{});
	}

	printf("Run OpenMP reduce test\n\n");	
	cs[0].show();

	#pragma omp declare reduction(complex_reduction : Complex : omp_out = complex_sum(omp_out, omp_in)) initializer(omp_priv = omp_orig)

	
	#pragma omp parallel for reduction(complex_reduction:r)
	for(int i = 0; i < complex_numbers; ++i){
		r = complex_sum(r, cs[i]);
	}
	
	printf("\nResult:\n");	
	r.show(); 

	return EXIT_SUCCESS;
}
