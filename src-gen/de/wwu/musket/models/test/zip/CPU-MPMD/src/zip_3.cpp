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
#include "../include/zip_3.hpp"

const size_t number_of_processes = 4;
const size_t process_id = 3;
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
	
	elem_offset = 12;
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 4; ++counter){
		ad1[counter] = initIndexA1_functor(elem_offset + counter, ad1[counter]);
	}
	elem_offset = 12;
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
	row_offset = 2;
	col_offset = 2;
	#pragma omp parallel for 
	for(size_t counter_rows = 0; counter_rows < 2; ++counter_rows){
		#pragma omp simd
		for(size_t counter_cols = 0; counter_cols < 2; ++counter_cols){
			size_t counter = counter_rows * 2 + counter_cols;
			md1[counter] = initIndexM1_functor(row_offset + counter_rows, col_offset + counter_cols, md1[counter]);
		}
	}
	row_offset = 2;
	col_offset = 2;
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
	MPI_Gather(ad1.data(), 4, Complex_mpi_type, nullptr, 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	
	MPI_Gather(md1.data(), 4, Complex_mpi_type, nullptr, 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	
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
	MPI_Gather(adr.data(), 4, Complex_mpi_type, nullptr, 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	
	MPI_Gather(mdr.data(), 4, Complex_mpi_type, nullptr, 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	
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
	MPI_Gather(adr.data(), 4, Complex_mpi_type, nullptr, 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	
	MPI_Gather(mdr.data(), 4, Complex_mpi_type, nullptr, 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	
	// ZipIndexSkeleton Array Start
	
	elem_offset = 12;
	
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
	
	row_offset = 2;
	col_offset = 2;
	
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
	MPI_Gather(adr.data(), 4, Complex_mpi_type, nullptr, 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	
	MPI_Gather(mdr.data(), 4, Complex_mpi_type, nullptr, 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	
	// ZipIndexInPlaceSkeleton Array Start
	elem_offset = 12;
	
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
	row_offset = 2;
	col_offset = 2;
	
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
	MPI_Gather(adr.data(), 4, Complex_mpi_type, nullptr, 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	
	MPI_Gather(mdr.data(), 4, Complex_mpi_type, nullptr, 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	
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
	MPI_Gather(adr.data(), 4, Complex_mpi_type, nullptr, 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	
	MPI_Gather(mdr.data(), 4, Complex_mpi_type, nullptr, 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	
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
	MPI_Gather(adr.data(), 4, Complex_mpi_type, nullptr, 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	
	MPI_Gather(mdr.data(), 4, Complex_mpi_type, nullptr, 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	
	
	
	MPI_Finalize();
	return EXIT_SUCCESS;
}
