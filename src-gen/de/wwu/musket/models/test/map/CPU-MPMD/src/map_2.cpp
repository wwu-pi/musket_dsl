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
#include "../include/map_2.hpp"

const size_t number_of_processes = 4;
const size_t process_id = 2;
int mpi_rank = -1;
int mpi_world_size = 0;

size_t tmp_size_t = 0;


const int dimA = 16;
const int dimM = 4;
std::vector<Complex> ad(4);
std::vector<Complex> adr(4);
std::vector<Complex> ac(16);
std::vector<Complex> acr(16);
std::vector<Complex> md(4);
std::vector<Complex> mdr(4);
std::vector<Complex> mc(16);
std::vector<Complex> mcr(16);

//Complex::Complex() : real(), imaginary() {}



struct PlusIndexA_functor{
	auto operator()(int i, Complex x) const{
		Complex result;
		result.real = ((x).real + (i));
		result.imaginary = ((x).imaginary + ((i) / 2));
		return (result);
	}
};
struct PlusIndexM_functor{
	auto operator()(int r, int c, Complex x) const{
		Complex result;
		result.real = ((x).real + (r));
		result.imaginary = ((x).imaginary + (c));
		return (result);
	}
};
struct TimesY_functor{
	auto operator()(float y, Complex x) const{
		Complex result;
		result.real = ((x).real * (y));
		result.imaginary = ((x).imaginary * (y));
		return (result);
	}
};
struct Init_functor{
	auto operator()(Complex x) const{
		x.real = 42.0f;
		x.imaginary = 17.0f;
		return (x);
	}
};
struct InitIndexA_functor{
	auto operator()(int i, Complex x) const{
		x.real = static_cast<float>((i));
		x.imaginary = (static_cast<float>((i)) / 2.0f);
		return (x);
	}
};
struct InitIndexM_functor{
	auto operator()(int r, int c, Complex x) const{
		x.real = static_cast<float>((r));
		x.imaginary = static_cast<float>((c));
		return (x);
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
	
	
	
	PlusIndexA_functor plusIndexA_functor{};
	PlusIndexM_functor plusIndexM_functor{};
	TimesY_functor timesY_functor{};
	Init_functor init_functor{};
	InitIndexA_functor initIndexA_functor{};
	InitIndexM_functor initIndexM_functor{};
	
	
	
	
	MPI_Datatype Complex_mpi_type_temp, Complex_mpi_type;
	MPI_Type_create_struct(2, (std::array<int,2>{1, 1}).data(), (std::array<MPI_Aint,2>{static_cast<MPI_Aint>(offsetof(struct Complex, real)), static_cast<MPI_Aint>(offsetof(struct Complex, imaginary))}).data(), (std::array<MPI_Datatype,2>{MPI_FLOAT, MPI_FLOAT}).data(), &Complex_mpi_type_temp);
	MPI_Type_create_resized(Complex_mpi_type_temp, 0, sizeof(Complex), &Complex_mpi_type);
	MPI_Type_free(&Complex_mpi_type_temp);
	MPI_Type_commit(&Complex_mpi_type);
	
	
	MPI_Datatype md_partition_type, md_partition_type_resized;
	MPI_Type_vector(2, 2, 4, Complex_mpi_type, &md_partition_type);
	MPI_Type_create_resized(md_partition_type, 0, sizeof(Complex) * 2, &md_partition_type_resized);
	MPI_Type_free(&md_partition_type);
	MPI_Type_commit(&md_partition_type_resized);
	MPI_Datatype mdr_partition_type, mdr_partition_type_resized;
	MPI_Type_vector(2, 2, 4, Complex_mpi_type, &mdr_partition_type);
	MPI_Type_create_resized(mdr_partition_type, 0, sizeof(Complex) * 2, &mdr_partition_type_resized);
	MPI_Type_free(&mdr_partition_type);
	MPI_Type_commit(&mdr_partition_type_resized);

	
	
	size_t row_offset = 0;size_t col_offset = 0;size_t elem_offset = 0;
	
	MPI_Gather(ad.data(), 4, Complex_mpi_type, nullptr, 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	
	MPI_Gather(md.data(), 4, Complex_mpi_type, nullptr, 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 4; ++counter){
		ad[counter] = init_functor(ad[counter]);
	}
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 16; ++counter){
		ac[counter] = init_functor(ac[counter]);
	}
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 4; ++counter){
		md[counter] = init_functor(md[counter]);
	}
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 16; ++counter){
		mc[counter] = init_functor(mc[counter]);
	}
	MPI_Gather(ad.data(), 4, Complex_mpi_type, nullptr, 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	
	MPI_Gather(md.data(), 4, Complex_mpi_type, nullptr, 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 4; ++counter){
		adr[counter] = timesY_functor(2.0f, ad[counter]);
	}
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 16; ++counter){
		acr[counter] = timesY_functor(2.0f, ac[counter]);
	}
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 4; ++counter){
		mdr[counter] = timesY_functor(2.0f, md[counter]);
	}
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 16; ++counter){
		mcr[counter] = timesY_functor(2.0f, mc[counter]);
	}
	MPI_Gather(adr.data(), 4, Complex_mpi_type, nullptr, 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	
	MPI_Gather(mdr.data(), 4, Complex_mpi_type, nullptr, 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	
	elem_offset = 8;
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 4; ++counter){
		ad[counter] = initIndexA_functor(elem_offset + counter, ad[counter]);
	}
	elem_offset = 0;
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 16; ++counter){
		ac[counter] = initIndexA_functor(elem_offset + counter, ac[counter]);
	}
	row_offset = 2;
	col_offset = 0;
	#pragma omp parallel for 
	for(size_t counter_rows = 0; counter_rows < 2; ++counter_rows){
		#pragma omp simd
		for(size_t counter_cols = 0; counter_cols < 2; ++counter_cols){
			size_t counter = counter_rows * 2 + counter_cols;
			md[counter] = initIndexM_functor(row_offset + counter_rows, col_offset + counter_cols, md[counter]);
		}
	}
	row_offset = 0;
	col_offset = 0;
	#pragma omp parallel for 
	for(size_t counter_rows = 0; counter_rows < 4; ++counter_rows){
		#pragma omp simd
		for(size_t counter_cols = 0; counter_cols < 4; ++counter_cols){
			size_t counter = counter_rows * 4 + counter_cols;
			mc[counter] = initIndexM_functor(row_offset + counter_rows, col_offset + counter_cols, mc[counter]);
		}
	}
	MPI_Gather(ad.data(), 4, Complex_mpi_type, nullptr, 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	
	MPI_Gather(md.data(), 4, Complex_mpi_type, nullptr, 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	
	// MapIndexSkeleton Array Start
	
	elem_offset = 8;
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 4; ++counter){
		adr[counter] = plusIndexA_functor(counter + elem_offset, ad[counter]);
	}
	// MapIndexSkeleton Array End
	// MapIndexSkeleton Array Start
	
	elem_offset = 0;
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 16; ++counter){
		acr[counter] = plusIndexA_functor(counter + elem_offset, ac[counter]);
	}
	// MapIndexSkeleton Array End
	// MapIndexSkeleton Matrix Start
	
	row_offset = 2;
	col_offset = 0;
	
	#pragma omp parallel for 
	for(size_t counter_rows = 0; counter_rows < 2; ++counter_rows){
		#pragma omp simd
		for(size_t counter_cols = 0; counter_cols < 2; ++counter_cols){
			size_t counter = counter_rows * 2 + counter_cols;
			mdr[counter] = plusIndexM_functor(counter_rows + row_offset, counter_cols + col_offset, md[counter]);
		}
	}
	// MapIndexSkeleton Matrix End
	// MapIndexSkeleton Matrix Start
	
	row_offset = 0;
	col_offset = 0;
	
	#pragma omp parallel for 
	for(size_t counter_rows = 0; counter_rows < 4; ++counter_rows){
		#pragma omp simd
		for(size_t counter_cols = 0; counter_cols < 4; ++counter_cols){
			size_t counter = counter_rows * 4 + counter_cols;
			mcr[counter] = plusIndexM_functor(counter_rows + row_offset, counter_cols + col_offset, mc[counter]);
		}
	}
	// MapIndexSkeleton Matrix End
	MPI_Gather(adr.data(), 4, Complex_mpi_type, nullptr, 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	
	MPI_Gather(mdr.data(), 4, Complex_mpi_type, nullptr, 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 4; ++counter){
		ad[counter] = initIndexA_functor(counter, ad[counter]);
	}
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 16; ++counter){
		ac[counter] = initIndexA_functor(counter, ac[counter]);
	}
	#pragma omp parallel for 
	for(size_t counter_rows = 0; counter_rows < 2; ++counter_rows){
		#pragma omp simd
		for(size_t counter_cols = 0; counter_cols < 2; ++counter_cols){
			size_t counter = counter_rows * 2 + counter_cols;
			md[counter] = initIndexM_functor(counter_rows, counter_cols, md[counter]);
		}
	}
	#pragma omp parallel for 
	for(size_t counter_rows = 0; counter_rows < 4; ++counter_rows){
		#pragma omp simd
		for(size_t counter_cols = 0; counter_cols < 4; ++counter_cols){
			size_t counter = counter_rows * 4 + counter_cols;
			mc[counter] = initIndexM_functor(counter_rows, counter_cols, mc[counter]);
		}
	}
	MPI_Gather(ad.data(), 4, Complex_mpi_type, nullptr, 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	
	MPI_Gather(md.data(), 4, Complex_mpi_type, nullptr, 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 4; ++counter){
		adr[counter] = plusIndexA_functor(counter, ad[counter]);
	}
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 16; ++counter){
		acr[counter] = plusIndexA_functor(counter, ac[counter]);
	}
	#pragma omp parallel for 
	for(size_t counter_rows = 0; counter_rows < 2; ++counter_rows){
		#pragma omp simd
		for(size_t counter_cols = 0; counter_cols < 2; ++counter_cols){
			size_t counter = counter_rows * 2 + counter_cols;
			mdr[counter] = plusIndexM_functor(counter_rows, counter_cols, md[counter]);
		}
	}
	#pragma omp parallel for 
	for(size_t counter_rows = 0; counter_rows < 4; ++counter_rows){
		#pragma omp simd
		for(size_t counter_cols = 0; counter_cols < 4; ++counter_cols){
			size_t counter = counter_rows * 4 + counter_cols;
			mcr[counter] = plusIndexM_functor(counter_rows, counter_cols, mc[counter]);
		}
	}
	MPI_Gather(adr.data(), 4, Complex_mpi_type, nullptr, 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	
	MPI_Gather(mdr.data(), 4, Complex_mpi_type, nullptr, 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	
	
	
	MPI_Finalize();
	return EXIT_SUCCESS;
}
