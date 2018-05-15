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
#include "../include/zip_0.hpp"

const size_t number_of_processes = 4;
const size_t process_id = 0;
int mpi_rank = -1;
int mpi_world_size = 0;

size_t tmp_size_t = 0;


const int dimA = 16;
const int dimM = 4;
std::vector<Complex> ad1(4);
std::vector<Complex> ad2(4);
std::vector<Complex> adr(4);
std::vector<Complex> ac1(16);
std::vector<Complex> ac2(16);
std::vector<Complex> acr(16);
std::vector<Complex> md1(4);
std::vector<Complex> md2(4);
std::vector<Complex> mdr(4);
std::vector<Complex> mc1(16);
std::vector<Complex> mc2(16);
std::vector<Complex> mcr(16);

//Complex::Complex() : real(), imaginary() {}



struct InitIndexA1_functor{
	auto operator()(int i, Complex x) const{
		x.real = static_cast<float>((i));
		x.imaginary = (static_cast<float>((i)) / 2.0f);
		return (x);
	}
};
struct InitIndexA2_functor{
	auto operator()(int i, Complex x) const{
		x.real = static_cast<float>((i));
		x.imaginary = (static_cast<float>((i)) / 2.0f);
		return (x);
	}
};
struct InitIndexM1_functor{
	auto operator()(int r, int c, Complex x) const{
		x.real = static_cast<float>((r));
		x.imaginary = static_cast<float>((c));
		return (x);
	}
};
struct InitIndexM2_functor{
	auto operator()(int r, int c, Complex x) const{
		x.real = static_cast<float>((r));
		x.imaginary = static_cast<float>((c));
		return (x);
	}
};
struct SumComplex_functor{
	auto operator()(Complex y, Complex x) const{
		Complex result;
		result.real = ((x).real + (y).real);
		result.imaginary = ((x).imaginary + (y).imaginary);
		return (result);
	}
};
struct SubComplex_functor{
	auto operator()(Complex y, Complex x) const{
		Complex result;
		result.real = ((x).real - (y).real);
		result.imaginary = ((x).imaginary - (y).imaginary);
		return (result);
	}
};
struct SumComplexPlusAIndex_functor{
	auto operator()(int in, Complex y, Complex x) const{
		Complex result;
		result.real = (((x).real + (y).real) + (in));
		result.imaginary = (((x).imaginary + (y).imaginary) + (in));
		return (result);
	}
};
struct SumComplexPlusMIndex_functor{
	auto operator()(int r, int c, Complex y, Complex x) const{
		Complex result;
		result.real = (((x).real + (y).real) + (r));
		result.imaginary = (((x).imaginary + (y).imaginary) + (c));
		return (result);
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
	
	
	printf("Run Zip\n\n");			
	
	InitIndexA1_functor initIndexA1_functor{};
	InitIndexA2_functor initIndexA2_functor{};
	InitIndexM1_functor initIndexM1_functor{};
	InitIndexM2_functor initIndexM2_functor{};
	SumComplex_functor sumComplex_functor{};
	SubComplex_functor subComplex_functor{};
	SumComplexPlusAIndex_functor sumComplexPlusAIndex_functor{};
	SumComplexPlusMIndex_functor sumComplexPlusMIndex_functor{};
	
	
	
	
	MPI_Datatype Complex_mpi_type_temp, Complex_mpi_type;
	MPI_Type_create_struct(2, (std::array<int,2>{1, 1}).data(), (std::array<MPI_Aint,2>{static_cast<MPI_Aint>(offsetof(struct Complex, real)), static_cast<MPI_Aint>(offsetof(struct Complex, imaginary))}).data(), (std::array<MPI_Datatype,2>{MPI_FLOAT, MPI_FLOAT}).data(), &Complex_mpi_type_temp);
	MPI_Type_create_resized(Complex_mpi_type_temp, 0, sizeof(Complex), &Complex_mpi_type);
	MPI_Type_free(&Complex_mpi_type_temp);
	MPI_Type_commit(&Complex_mpi_type);
	
	
	MPI_Datatype md1_partition_type, md1_partition_type_resized;
	MPI_Type_vector(2, 2, 4, Complex_mpi_type, &md1_partition_type);
	MPI_Type_create_resized(md1_partition_type, 0, sizeof(Complex) * 2, &md1_partition_type_resized);
	MPI_Type_free(&md1_partition_type);
	MPI_Type_commit(&md1_partition_type_resized);
	MPI_Datatype md2_partition_type, md2_partition_type_resized;
	MPI_Type_vector(2, 2, 4, Complex_mpi_type, &md2_partition_type);
	MPI_Type_create_resized(md2_partition_type, 0, sizeof(Complex) * 2, &md2_partition_type_resized);
	MPI_Type_free(&md2_partition_type);
	MPI_Type_commit(&md2_partition_type_resized);
	MPI_Datatype mdr_partition_type, mdr_partition_type_resized;
	MPI_Type_vector(2, 2, 4, Complex_mpi_type, &mdr_partition_type);
	MPI_Type_create_resized(mdr_partition_type, 0, sizeof(Complex) * 2, &mdr_partition_type_resized);
	MPI_Type_free(&mdr_partition_type);
	MPI_Type_commit(&mdr_partition_type_resized);

	
	
	size_t row_offset = 0;size_t col_offset = 0;size_t elem_offset = 0;
	
	printf("Inital:\n");
	elem_offset = 0;
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 4; ++counter){
		ad1[counter] = initIndexA1_functor(elem_offset + counter, ad1[counter]);
	}
	elem_offset = 0;
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 4; ++counter){
		ad2[counter] = initIndexA2_functor(elem_offset + counter, ad2[counter]);
	}
	elem_offset = 0;
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 16; ++counter){
		ac1[counter] = initIndexA1_functor(elem_offset + counter, ac1[counter]);
	}
	elem_offset = 0;
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 16; ++counter){
		ac2[counter] = initIndexA2_functor(elem_offset + counter, ac2[counter]);
	}
	row_offset = 0;
	col_offset = 0;
	#pragma omp parallel for 
	for(size_t counter_rows = 0; counter_rows < 2; ++counter_rows){
		#pragma omp simd
		for(size_t counter_cols = 0; counter_cols < 2; ++counter_cols){
			size_t counter = counter_rows * 2 + counter_cols;
			md1[counter] = initIndexM1_functor(row_offset + counter_rows, col_offset + counter_cols, md1[counter]);
		}
	}
	row_offset = 0;
	col_offset = 0;
	#pragma omp parallel for 
	for(size_t counter_rows = 0; counter_rows < 2; ++counter_rows){
		#pragma omp simd
		for(size_t counter_cols = 0; counter_cols < 2; ++counter_cols){
			size_t counter = counter_rows * 2 + counter_cols;
			md2[counter] = initIndexM2_functor(row_offset + counter_rows, col_offset + counter_cols, md2[counter]);
		}
	}
	row_offset = 0;
	col_offset = 0;
	#pragma omp parallel for 
	for(size_t counter_rows = 0; counter_rows < 4; ++counter_rows){
		#pragma omp simd
		for(size_t counter_cols = 0; counter_cols < 4; ++counter_cols){
			size_t counter = counter_rows * 4 + counter_cols;
			mc1[counter] = initIndexM1_functor(row_offset + counter_rows, col_offset + counter_cols, mc1[counter]);
		}
	}
	row_offset = 0;
	col_offset = 0;
	#pragma omp parallel for 
	for(size_t counter_rows = 0; counter_rows < 4; ++counter_rows){
		#pragma omp simd
		for(size_t counter_cols = 0; counter_cols < 4; ++counter_cols){
			size_t counter = counter_rows * 4 + counter_cols;
			mc2[counter] = initIndexM2_functor(row_offset + counter_rows, col_offset + counter_cols, mc2[counter]);
		}
	}
	std::array<Complex, 16> temp148{};
	MPI_Gather(ad1.data(), 4, Complex_mpi_type, temp148.data(), 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	std::ostringstream s148;
	s148 << "ad1: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s148 << "[" << "real = " << temp148[i].real << ", " << "imaginary = " << temp148[i].imaginary << "]";
	s148 << "; ";
	}
	s148 << "[" << "real = " << temp148[15].real << ", " << "imaginary = " << temp148[15].imaginary << "]" << "]" << std::endl;
	s148 << std::endl;
	printf("%s", s148.str().c_str());
	
	std::ostringstream s149;
	s149 << "ac1: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s149 << "[" << "real = " << ac1[i].real << ", " << "imaginary = " << ac1[i].imaginary << "]";
	s149 << "; ";
	}
	s149 << "[" << "real = " << ac1[15].real << ", " << "imaginary = " << ac1[15].imaginary << "]" << "]" << std::endl;
	s149 << std::endl;
	printf("%s", s149.str().c_str());
	std::array<Complex, 16> temp150{};
	MPI_Gather(md1.data(), 4, Complex_mpi_type, temp150.data(), 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	std::ostringstream s150;
	s150 << "md1: " << std::endl << "[" << std::endl;
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s150 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s150 << "[" << "real = " << temp150[(counter_rows%2) * 2 + (counter_rows / 2) * 4 * 2 + (counter_cols/ 2) * 4 + counter_cols%2].real << ", " << "imaginary = " << temp150[(counter_rows%2) * 2 + (counter_rows / 2) * 4 * 2 + (counter_cols/ 2) * 4 + counter_cols%2].imaginary << "]";
			if(counter_cols < 3){
				s150 << "; ";
			}else{
				s150 << "]" << std::endl;
			}
		}
	}
	
	s150 << "]" << std::endl;
	printf("%s", s150.str().c_str());
	
	std::ostringstream s151;
	s151 << "mc1: " << std::endl << "[" << std::endl;
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s151 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s151 << "[" << "real = " << mc1[counter_rows * 4 + counter_cols].real << ", " << "imaginary = " << mc1[counter_rows * 4 + counter_cols].imaginary << "]";
			if(counter_cols < 3){
				s151 << "; ";
			}else{
				s151 << "]" << std::endl;
			}
		}
	}
	
	s151 << "]" << std::endl;
	printf("%s", s151.str().c_str());
	printf("Zip; expected result: values * 2.\n");
	// Zip skeleton start
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 4; ++counter){
		adr[counter] = sumComplex_functor(ad2[counter], ad1[counter]);
	}
	// Zip skeleton end
	// Zip skeleton start
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 16; ++counter){
		acr[counter] = sumComplex_functor(ac2[counter], ac1[counter]);
	}
	// Zip skeleton end
	// Zip skeleton start
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 4; ++counter){
		mdr[counter] = sumComplex_functor(md2[counter], md1[counter]);
	}
	// Zip skeleton end
	// Zip skeleton start
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 16; ++counter){
		mcr[counter] = sumComplex_functor(mc2[counter], mc1[counter]);
	}
	// Zip skeleton end
	std::array<Complex, 16> temp152{};
	MPI_Gather(adr.data(), 4, Complex_mpi_type, temp152.data(), 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	std::ostringstream s152;
	s152 << "adr: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s152 << "[" << "real = " << temp152[i].real << ", " << "imaginary = " << temp152[i].imaginary << "]";
	s152 << "; ";
	}
	s152 << "[" << "real = " << temp152[15].real << ", " << "imaginary = " << temp152[15].imaginary << "]" << "]" << std::endl;
	s152 << std::endl;
	printf("%s", s152.str().c_str());
	
	std::ostringstream s153;
	s153 << "acr: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s153 << "[" << "real = " << acr[i].real << ", " << "imaginary = " << acr[i].imaginary << "]";
	s153 << "; ";
	}
	s153 << "[" << "real = " << acr[15].real << ", " << "imaginary = " << acr[15].imaginary << "]" << "]" << std::endl;
	s153 << std::endl;
	printf("%s", s153.str().c_str());
	std::array<Complex, 16> temp154{};
	MPI_Gather(mdr.data(), 4, Complex_mpi_type, temp154.data(), 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	std::ostringstream s154;
	s154 << "mdr: " << std::endl << "[" << std::endl;
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s154 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s154 << "[" << "real = " << temp154[(counter_rows%2) * 2 + (counter_rows / 2) * 4 * 2 + (counter_cols/ 2) * 4 + counter_cols%2].real << ", " << "imaginary = " << temp154[(counter_rows%2) * 2 + (counter_rows / 2) * 4 * 2 + (counter_cols/ 2) * 4 + counter_cols%2].imaginary << "]";
			if(counter_cols < 3){
				s154 << "; ";
			}else{
				s154 << "]" << std::endl;
			}
		}
	}
	
	s154 << "]" << std::endl;
	printf("%s", s154.str().c_str());
	
	std::ostringstream s155;
	s155 << "mcr: " << std::endl << "[" << std::endl;
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s155 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s155 << "[" << "real = " << mcr[counter_rows * 4 + counter_cols].real << ", " << "imaginary = " << mcr[counter_rows * 4 + counter_cols].imaginary << "]";
			if(counter_cols < 3){
				s155 << "; ";
			}else{
				s155 << "]" << std::endl;
			}
		}
	}
	
	s155 << "]" << std::endl;
	printf("%s", s155.str().c_str());
	printf("zipInPlace; expected result: values / 2.\n");
	// Zip skeleton start
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 4; ++counter){
		adr[counter] = subComplex_functor(ad1[counter], adr[counter]);
	}
	// Zip skeleton end
	// Zip skeleton start
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 16; ++counter){
		acr[counter] = subComplex_functor(ac1[counter], acr[counter]);
	}
	// Zip skeleton end
	// Zip skeleton start
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 4; ++counter){
		mdr[counter] = subComplex_functor(md1[counter], mdr[counter]);
	}
	// Zip skeleton end
	// Zip skeleton start
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 16; ++counter){
		mcr[counter] = subComplex_functor(mc1[counter], mcr[counter]);
	}
	// Zip skeleton end
	std::array<Complex, 16> temp156{};
	MPI_Gather(adr.data(), 4, Complex_mpi_type, temp156.data(), 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	std::ostringstream s156;
	s156 << "adr: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s156 << "[" << "real = " << temp156[i].real << ", " << "imaginary = " << temp156[i].imaginary << "]";
	s156 << "; ";
	}
	s156 << "[" << "real = " << temp156[15].real << ", " << "imaginary = " << temp156[15].imaginary << "]" << "]" << std::endl;
	s156 << std::endl;
	printf("%s", s156.str().c_str());
	
	std::ostringstream s157;
	s157 << "acr: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s157 << "[" << "real = " << acr[i].real << ", " << "imaginary = " << acr[i].imaginary << "]";
	s157 << "; ";
	}
	s157 << "[" << "real = " << acr[15].real << ", " << "imaginary = " << acr[15].imaginary << "]" << "]" << std::endl;
	s157 << std::endl;
	printf("%s", s157.str().c_str());
	std::array<Complex, 16> temp158{};
	MPI_Gather(mdr.data(), 4, Complex_mpi_type, temp158.data(), 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	std::ostringstream s158;
	s158 << "mdr: " << std::endl << "[" << std::endl;
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s158 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s158 << "[" << "real = " << temp158[(counter_rows%2) * 2 + (counter_rows / 2) * 4 * 2 + (counter_cols/ 2) * 4 + counter_cols%2].real << ", " << "imaginary = " << temp158[(counter_rows%2) * 2 + (counter_rows / 2) * 4 * 2 + (counter_cols/ 2) * 4 + counter_cols%2].imaginary << "]";
			if(counter_cols < 3){
				s158 << "; ";
			}else{
				s158 << "]" << std::endl;
			}
		}
	}
	
	s158 << "]" << std::endl;
	printf("%s", s158.str().c_str());
	
	std::ostringstream s159;
	s159 << "mcr: " << std::endl << "[" << std::endl;
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s159 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s159 << "[" << "real = " << mcr[counter_rows * 4 + counter_cols].real << ", " << "imaginary = " << mcr[counter_rows * 4 + counter_cols].imaginary << "]";
			if(counter_cols < 3){
				s159 << "; ";
			}else{
				s159 << "]" << std::endl;
			}
		}
	}
	
	s159 << "]" << std::endl;
	printf("%s", s159.str().c_str());
	printf("ZipIndex:\n");
	// ZipIndexSkeleton Array Start
	
	elem_offset = 0;
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 4; ++counter){
		adr[counter] = sumComplexPlusAIndex_functor(counter + elem_offset, ad2[counter], ad1[counter]);
	}
	// ZipIndexSkeleton Array End
	// ZipIndexSkeleton Array Start
	
	elem_offset = 0;
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 16; ++counter){
		acr[counter] = sumComplexPlusAIndex_functor(counter + elem_offset, ac2[counter], ac1[counter]);
	}
	// ZipIndexSkeleton Array End
	// ZipIndexSkeleton Matrix Start
	
	row_offset = 0;
	col_offset = 0;
	
	#pragma omp parallel for 
	for(size_t counter_rows = 0; counter_rows < 2; ++counter_rows){
		#pragma omp simd
		for(size_t counter_cols = 0; counter_cols < 2; ++counter_cols){
			size_t counter = counter_rows * 2 + counter_cols;
			mdr[counter] = sumComplexPlusMIndex_functor(counter_rows + row_offset, counter_cols + col_offset, md2[counter], md1[counter]);
		}
	}
	// ZipIndexSkeleton Matrix End
	// ZipIndexSkeleton Matrix Start
	
	row_offset = 0;
	col_offset = 0;
	
	#pragma omp parallel for 
	for(size_t counter_rows = 0; counter_rows < 4; ++counter_rows){
		#pragma omp simd
		for(size_t counter_cols = 0; counter_cols < 4; ++counter_cols){
			size_t counter = counter_rows * 4 + counter_cols;
			mcr[counter] = sumComplexPlusMIndex_functor(counter_rows + row_offset, counter_cols + col_offset, mc2[counter], mc1[counter]);
		}
	}
	// ZipIndexSkeleton Matrix End
	std::array<Complex, 16> temp160{};
	MPI_Gather(adr.data(), 4, Complex_mpi_type, temp160.data(), 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	std::ostringstream s160;
	s160 << "adr: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s160 << "[" << "real = " << temp160[i].real << ", " << "imaginary = " << temp160[i].imaginary << "]";
	s160 << "; ";
	}
	s160 << "[" << "real = " << temp160[15].real << ", " << "imaginary = " << temp160[15].imaginary << "]" << "]" << std::endl;
	s160 << std::endl;
	printf("%s", s160.str().c_str());
	
	std::ostringstream s161;
	s161 << "acr: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s161 << "[" << "real = " << acr[i].real << ", " << "imaginary = " << acr[i].imaginary << "]";
	s161 << "; ";
	}
	s161 << "[" << "real = " << acr[15].real << ", " << "imaginary = " << acr[15].imaginary << "]" << "]" << std::endl;
	s161 << std::endl;
	printf("%s", s161.str().c_str());
	std::array<Complex, 16> temp162{};
	MPI_Gather(mdr.data(), 4, Complex_mpi_type, temp162.data(), 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	std::ostringstream s162;
	s162 << "mdr: " << std::endl << "[" << std::endl;
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s162 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s162 << "[" << "real = " << temp162[(counter_rows%2) * 2 + (counter_rows / 2) * 4 * 2 + (counter_cols/ 2) * 4 + counter_cols%2].real << ", " << "imaginary = " << temp162[(counter_rows%2) * 2 + (counter_rows / 2) * 4 * 2 + (counter_cols/ 2) * 4 + counter_cols%2].imaginary << "]";
			if(counter_cols < 3){
				s162 << "; ";
			}else{
				s162 << "]" << std::endl;
			}
		}
	}
	
	s162 << "]" << std::endl;
	printf("%s", s162.str().c_str());
	
	std::ostringstream s163;
	s163 << "mcr: " << std::endl << "[" << std::endl;
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s163 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s163 << "[" << "real = " << mcr[counter_rows * 4 + counter_cols].real << ", " << "imaginary = " << mcr[counter_rows * 4 + counter_cols].imaginary << "]";
			if(counter_cols < 3){
				s163 << "; ";
			}else{
				s163 << "]" << std::endl;
			}
		}
	}
	
	s163 << "]" << std::endl;
	printf("%s", s163.str().c_str());
	printf("ZipIndexInPlace:\n");
	// ZipIndexInPlaceSkeleton Array Start
	elem_offset = 0;
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 4; ++counter){
		adr[counter] = sumComplexPlusAIndex_functor(counter + elem_offset, ad2[counter], adr[counter]);
	}
	// ZipIndexInPlaceSkeleton Array End
	// ZipIndexInPlaceSkeleton Array Start
	elem_offset = 0;
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 16; ++counter){
		acr[counter] = sumComplexPlusAIndex_functor(counter + elem_offset, ac2[counter], acr[counter]);
	}
	// ZipIndexInPlaceSkeleton Array End
	// ZipIndexInPlaceSkeleton Matrix Start
	row_offset = 0;
	col_offset = 0;
	
	#pragma omp parallel for 
	for(size_t counter_rows = 0; counter_rows < 2; ++counter_rows){
		#pragma omp simd
		for(size_t counter_cols = 0; counter_cols < 2; ++counter_cols){
			size_t counter = counter_rows * 2 + counter_cols;
			mdr[counter] = sumComplexPlusMIndex_functor(counter_rows + row_offset, counter_cols + col_offset, md2[counter], mdr[counter]);
		}
	}
	// ZipIndexInPlaceSkeleton Matrix End
	// ZipIndexInPlaceSkeleton Matrix Start
	row_offset = 0;
	col_offset = 0;
	
	#pragma omp parallel for 
	for(size_t counter_rows = 0; counter_rows < 4; ++counter_rows){
		#pragma omp simd
		for(size_t counter_cols = 0; counter_cols < 4; ++counter_cols){
			size_t counter = counter_rows * 4 + counter_cols;
			mcr[counter] = sumComplexPlusMIndex_functor(counter_rows + row_offset, counter_cols + col_offset, mc2[counter], mcr[counter]);
		}
	}
	// ZipIndexInPlaceSkeleton Matrix End
	std::array<Complex, 16> temp164{};
	MPI_Gather(adr.data(), 4, Complex_mpi_type, temp164.data(), 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	std::ostringstream s164;
	s164 << "adr: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s164 << "[" << "real = " << temp164[i].real << ", " << "imaginary = " << temp164[i].imaginary << "]";
	s164 << "; ";
	}
	s164 << "[" << "real = " << temp164[15].real << ", " << "imaginary = " << temp164[15].imaginary << "]" << "]" << std::endl;
	s164 << std::endl;
	printf("%s", s164.str().c_str());
	
	std::ostringstream s165;
	s165 << "acr: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s165 << "[" << "real = " << acr[i].real << ", " << "imaginary = " << acr[i].imaginary << "]";
	s165 << "; ";
	}
	s165 << "[" << "real = " << acr[15].real << ", " << "imaginary = " << acr[15].imaginary << "]" << "]" << std::endl;
	s165 << std::endl;
	printf("%s", s165.str().c_str());
	std::array<Complex, 16> temp166{};
	MPI_Gather(mdr.data(), 4, Complex_mpi_type, temp166.data(), 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	std::ostringstream s166;
	s166 << "mdr: " << std::endl << "[" << std::endl;
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s166 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s166 << "[" << "real = " << temp166[(counter_rows%2) * 2 + (counter_rows / 2) * 4 * 2 + (counter_cols/ 2) * 4 + counter_cols%2].real << ", " << "imaginary = " << temp166[(counter_rows%2) * 2 + (counter_rows / 2) * 4 * 2 + (counter_cols/ 2) * 4 + counter_cols%2].imaginary << "]";
			if(counter_cols < 3){
				s166 << "; ";
			}else{
				s166 << "]" << std::endl;
			}
		}
	}
	
	s166 << "]" << std::endl;
	printf("%s", s166.str().c_str());
	
	std::ostringstream s167;
	s167 << "mcr: " << std::endl << "[" << std::endl;
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s167 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s167 << "[" << "real = " << mcr[counter_rows * 4 + counter_cols].real << ", " << "imaginary = " << mcr[counter_rows * 4 + counter_cols].imaginary << "]";
			if(counter_cols < 3){
				s167 << "; ";
			}else{
				s167 << "]" << std::endl;
			}
		}
	}
	
	s167 << "]" << std::endl;
	printf("%s", s167.str().c_str());
	printf("ZipLocalIndex:\n");
	// ZipLocalIndexSkeleton Array Start
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 4; ++counter){
		adr[counter] = sumComplexPlusAIndex_functor(counter, ad2[counter], ad1[counter]);
	}
	// ZipLocalIndexSkeleton Array End
	// ZipLocalIndexSkeleton Array Start
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 16; ++counter){
		acr[counter] = sumComplexPlusAIndex_functor(counter, ac2[counter], ac1[counter]);
	}
	// ZipLocalIndexSkeleton Array End
	// ZipLocalIndexSkeleton Matrix Start
	
	#pragma omp parallel for 
	for(size_t counter_rows = 0; counter_rows < 2; ++counter_rows){
		#pragma omp simd
		for(size_t counter_cols = 0; counter_cols < 2; ++counter_cols){
			size_t counter = counter_rows * 2 + counter_cols;
			mdr[counter] = sumComplexPlusMIndex_functor(counter_rows, counter_cols, md2[counter], md1[counter]);
		}
	}
	// ZipLocalIndexSkeleton Matrix End
	// ZipLocalIndexSkeleton Matrix Start
	
	#pragma omp parallel for 
	for(size_t counter_rows = 0; counter_rows < 4; ++counter_rows){
		#pragma omp simd
		for(size_t counter_cols = 0; counter_cols < 4; ++counter_cols){
			size_t counter = counter_rows * 4 + counter_cols;
			mcr[counter] = sumComplexPlusMIndex_functor(counter_rows, counter_cols, mc2[counter], mc1[counter]);
		}
	}
	// ZipLocalIndexSkeleton Matrix End
	std::array<Complex, 16> temp168{};
	MPI_Gather(adr.data(), 4, Complex_mpi_type, temp168.data(), 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	std::ostringstream s168;
	s168 << "adr: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s168 << "[" << "real = " << temp168[i].real << ", " << "imaginary = " << temp168[i].imaginary << "]";
	s168 << "; ";
	}
	s168 << "[" << "real = " << temp168[15].real << ", " << "imaginary = " << temp168[15].imaginary << "]" << "]" << std::endl;
	s168 << std::endl;
	printf("%s", s168.str().c_str());
	
	std::ostringstream s169;
	s169 << "acr: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s169 << "[" << "real = " << acr[i].real << ", " << "imaginary = " << acr[i].imaginary << "]";
	s169 << "; ";
	}
	s169 << "[" << "real = " << acr[15].real << ", " << "imaginary = " << acr[15].imaginary << "]" << "]" << std::endl;
	s169 << std::endl;
	printf("%s", s169.str().c_str());
	std::array<Complex, 16> temp170{};
	MPI_Gather(mdr.data(), 4, Complex_mpi_type, temp170.data(), 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	std::ostringstream s170;
	s170 << "mdr: " << std::endl << "[" << std::endl;
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s170 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s170 << "[" << "real = " << temp170[(counter_rows%2) * 2 + (counter_rows / 2) * 4 * 2 + (counter_cols/ 2) * 4 + counter_cols%2].real << ", " << "imaginary = " << temp170[(counter_rows%2) * 2 + (counter_rows / 2) * 4 * 2 + (counter_cols/ 2) * 4 + counter_cols%2].imaginary << "]";
			if(counter_cols < 3){
				s170 << "; ";
			}else{
				s170 << "]" << std::endl;
			}
		}
	}
	
	s170 << "]" << std::endl;
	printf("%s", s170.str().c_str());
	
	std::ostringstream s171;
	s171 << "mcr: " << std::endl << "[" << std::endl;
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s171 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s171 << "[" << "real = " << mcr[counter_rows * 4 + counter_cols].real << ", " << "imaginary = " << mcr[counter_rows * 4 + counter_cols].imaginary << "]";
			if(counter_cols < 3){
				s171 << "; ";
			}else{
				s171 << "]" << std::endl;
			}
		}
	}
	
	s171 << "]" << std::endl;
	printf("%s", s171.str().c_str());
	printf("ZipLocalIndexInPlace:\n");
	// ZipLocalIndexInPlaceSkeleton Array Start
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 4; ++counter){
		adr[counter] = sumComplexPlusAIndex_functor(counter, ad2[counter], adr[counter]);
	}
	// ZipLocalIndexInPlaceSkeleton Array End
	// ZipLocalIndexInPlaceSkeleton Array Start
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 16; ++counter){
		acr[counter] = sumComplexPlusAIndex_functor(counter, ac2[counter], acr[counter]);
	}
	// ZipLocalIndexInPlaceSkeleton Array End
	// ZipLocalIndexInPlaceSkeleton Matrix Start
	#pragma omp parallel for 
	for(size_t counter_rows = 0; counter_rows < 2; ++counter_rows){
		#pragma omp simd
		for(size_t counter_cols = 0; counter_cols < 2; ++counter_cols){
			size_t counter = counter_rows * 2 + counter_cols;
			mdr[counter] = sumComplexPlusMIndex_functor(counter_rows, counter_cols, md2[counter], mdr[counter]);
		}
	}
	// ZipLocalIndexInPlaceSkeleton Matrix End
	// ZipLocalIndexInPlaceSkeleton Matrix Start
	#pragma omp parallel for 
	for(size_t counter_rows = 0; counter_rows < 4; ++counter_rows){
		#pragma omp simd
		for(size_t counter_cols = 0; counter_cols < 4; ++counter_cols){
			size_t counter = counter_rows * 4 + counter_cols;
			mcr[counter] = sumComplexPlusMIndex_functor(counter_rows, counter_cols, mc2[counter], mcr[counter]);
		}
	}
	// ZipLocalIndexInPlaceSkeleton Matrix End
	std::array<Complex, 16> temp172{};
	MPI_Gather(adr.data(), 4, Complex_mpi_type, temp172.data(), 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	std::ostringstream s172;
	s172 << "adr: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s172 << "[" << "real = " << temp172[i].real << ", " << "imaginary = " << temp172[i].imaginary << "]";
	s172 << "; ";
	}
	s172 << "[" << "real = " << temp172[15].real << ", " << "imaginary = " << temp172[15].imaginary << "]" << "]" << std::endl;
	s172 << std::endl;
	printf("%s", s172.str().c_str());
	
	std::ostringstream s173;
	s173 << "acr: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s173 << "[" << "real = " << acr[i].real << ", " << "imaginary = " << acr[i].imaginary << "]";
	s173 << "; ";
	}
	s173 << "[" << "real = " << acr[15].real << ", " << "imaginary = " << acr[15].imaginary << "]" << "]" << std::endl;
	s173 << std::endl;
	printf("%s", s173.str().c_str());
	std::array<Complex, 16> temp174{};
	MPI_Gather(mdr.data(), 4, Complex_mpi_type, temp174.data(), 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	std::ostringstream s174;
	s174 << "mdr: " << std::endl << "[" << std::endl;
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s174 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s174 << "[" << "real = " << temp174[(counter_rows%2) * 2 + (counter_rows / 2) * 4 * 2 + (counter_cols/ 2) * 4 + counter_cols%2].real << ", " << "imaginary = " << temp174[(counter_rows%2) * 2 + (counter_rows / 2) * 4 * 2 + (counter_cols/ 2) * 4 + counter_cols%2].imaginary << "]";
			if(counter_cols < 3){
				s174 << "; ";
			}else{
				s174 << "]" << std::endl;
			}
		}
	}
	
	s174 << "]" << std::endl;
	printf("%s", s174.str().c_str());
	
	std::ostringstream s175;
	s175 << "mcr: " << std::endl << "[" << std::endl;
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s175 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s175 << "[" << "real = " << mcr[counter_rows * 4 + counter_cols].real << ", " << "imaginary = " << mcr[counter_rows * 4 + counter_cols].imaginary << "]";
			if(counter_cols < 3){
				s175 << "; ";
			}else{
				s175 << "]" << std::endl;
			}
		}
	}
	
	s175 << "]" << std::endl;
	printf("%s", s175.str().c_str());
	
	printf("Threads: %i\n", omp_get_max_threads());
	printf("Processes: %i\n", mpi_world_size);
	
	MPI_Finalize();
	return EXIT_SUCCESS;
}
