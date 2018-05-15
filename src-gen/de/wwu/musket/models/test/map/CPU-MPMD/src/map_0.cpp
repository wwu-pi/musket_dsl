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
#include "../include/map_0.hpp"

const size_t number_of_processes = 4;
const size_t process_id = 0;
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
	
	
	printf("Run Map\n\n");			
	
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
	
	printf("Inital:\n");
	std::array<Complex, 16> temp0{};
	MPI_Gather(ad.data(), 4, Complex_mpi_type, temp0.data(), 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	std::ostringstream s0;
	s0 << "ad: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s0 << "[" << "real = " << temp0[i].real << ", " << "imaginary = " << temp0[i].imaginary << "]";
	s0 << "; ";
	}
	s0 << "[" << "real = " << temp0[15].real << ", " << "imaginary = " << temp0[15].imaginary << "]" << "]" << std::endl;
	s0 << std::endl;
	printf("%s", s0.str().c_str());
	
	std::ostringstream s1;
	s1 << "ac: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s1 << "[" << "real = " << ac[i].real << ", " << "imaginary = " << ac[i].imaginary << "]";
	s1 << "; ";
	}
	s1 << "[" << "real = " << ac[15].real << ", " << "imaginary = " << ac[15].imaginary << "]" << "]" << std::endl;
	s1 << std::endl;
	printf("%s", s1.str().c_str());
	std::array<Complex, 16> temp2{};
	MPI_Gather(md.data(), 4, Complex_mpi_type, temp2.data(), 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	std::ostringstream s2;
	s2 << "md: " << std::endl << "[" << std::endl;
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s2 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s2 << "[" << "real = " << temp2[(counter_rows%2) * 2 + (counter_rows / 2) * 4 * 2 + (counter_cols/ 2) * 4 + counter_cols%2].real << ", " << "imaginary = " << temp2[(counter_rows%2) * 2 + (counter_rows / 2) * 4 * 2 + (counter_cols/ 2) * 4 + counter_cols%2].imaginary << "]";
			if(counter_cols < 3){
				s2 << "; ";
			}else{
				s2 << "]" << std::endl;
			}
		}
	}
	
	s2 << "]" << std::endl;
	printf("%s", s2.str().c_str());
	
	std::ostringstream s3;
	s3 << "mc: " << std::endl << "[" << std::endl;
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s3 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s3 << "[" << "real = " << mc[counter_rows * 4 + counter_cols].real << ", " << "imaginary = " << mc[counter_rows * 4 + counter_cols].imaginary << "]";
			if(counter_cols < 3){
				s3 << "; ";
			}else{
				s3 << "]" << std::endl;
			}
		}
	}
	
	s3 << "]" << std::endl;
	printf("%s", s3.str().c_str());
	printf("MapInPlace:\n");
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
	std::array<Complex, 16> temp4{};
	MPI_Gather(ad.data(), 4, Complex_mpi_type, temp4.data(), 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	std::ostringstream s4;
	s4 << "ad: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s4 << "[" << "real = " << temp4[i].real << ", " << "imaginary = " << temp4[i].imaginary << "]";
	s4 << "; ";
	}
	s4 << "[" << "real = " << temp4[15].real << ", " << "imaginary = " << temp4[15].imaginary << "]" << "]" << std::endl;
	s4 << std::endl;
	printf("%s", s4.str().c_str());
	
	std::ostringstream s5;
	s5 << "ac: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s5 << "[" << "real = " << ac[i].real << ", " << "imaginary = " << ac[i].imaginary << "]";
	s5 << "; ";
	}
	s5 << "[" << "real = " << ac[15].real << ", " << "imaginary = " << ac[15].imaginary << "]" << "]" << std::endl;
	s5 << std::endl;
	printf("%s", s5.str().c_str());
	std::array<Complex, 16> temp6{};
	MPI_Gather(md.data(), 4, Complex_mpi_type, temp6.data(), 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	std::ostringstream s6;
	s6 << "md: " << std::endl << "[" << std::endl;
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s6 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s6 << "[" << "real = " << temp6[(counter_rows%2) * 2 + (counter_rows / 2) * 4 * 2 + (counter_cols/ 2) * 4 + counter_cols%2].real << ", " << "imaginary = " << temp6[(counter_rows%2) * 2 + (counter_rows / 2) * 4 * 2 + (counter_cols/ 2) * 4 + counter_cols%2].imaginary << "]";
			if(counter_cols < 3){
				s6 << "; ";
			}else{
				s6 << "]" << std::endl;
			}
		}
	}
	
	s6 << "]" << std::endl;
	printf("%s", s6.str().c_str());
	
	std::ostringstream s7;
	s7 << "mc: " << std::endl << "[" << std::endl;
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s7 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s7 << "[" << "real = " << mc[counter_rows * 4 + counter_cols].real << ", " << "imaginary = " << mc[counter_rows * 4 + counter_cols].imaginary << "]";
			if(counter_cols < 3){
				s7 << "; ";
			}else{
				s7 << "]" << std::endl;
			}
		}
	}
	
	s7 << "]" << std::endl;
	printf("%s", s7.str().c_str());
	printf("Map:\n");
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
	std::array<Complex, 16> temp8{};
	MPI_Gather(adr.data(), 4, Complex_mpi_type, temp8.data(), 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	std::ostringstream s8;
	s8 << "adr: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s8 << "[" << "real = " << temp8[i].real << ", " << "imaginary = " << temp8[i].imaginary << "]";
	s8 << "; ";
	}
	s8 << "[" << "real = " << temp8[15].real << ", " << "imaginary = " << temp8[15].imaginary << "]" << "]" << std::endl;
	s8 << std::endl;
	printf("%s", s8.str().c_str());
	
	std::ostringstream s9;
	s9 << "acr: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s9 << "[" << "real = " << acr[i].real << ", " << "imaginary = " << acr[i].imaginary << "]";
	s9 << "; ";
	}
	s9 << "[" << "real = " << acr[15].real << ", " << "imaginary = " << acr[15].imaginary << "]" << "]" << std::endl;
	s9 << std::endl;
	printf("%s", s9.str().c_str());
	std::array<Complex, 16> temp10{};
	MPI_Gather(mdr.data(), 4, Complex_mpi_type, temp10.data(), 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	std::ostringstream s10;
	s10 << "mdr: " << std::endl << "[" << std::endl;
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s10 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s10 << "[" << "real = " << temp10[(counter_rows%2) * 2 + (counter_rows / 2) * 4 * 2 + (counter_cols/ 2) * 4 + counter_cols%2].real << ", " << "imaginary = " << temp10[(counter_rows%2) * 2 + (counter_rows / 2) * 4 * 2 + (counter_cols/ 2) * 4 + counter_cols%2].imaginary << "]";
			if(counter_cols < 3){
				s10 << "; ";
			}else{
				s10 << "]" << std::endl;
			}
		}
	}
	
	s10 << "]" << std::endl;
	printf("%s", s10.str().c_str());
	
	std::ostringstream s11;
	s11 << "mcr: " << std::endl << "[" << std::endl;
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s11 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s11 << "[" << "real = " << mcr[counter_rows * 4 + counter_cols].real << ", " << "imaginary = " << mcr[counter_rows * 4 + counter_cols].imaginary << "]";
			if(counter_cols < 3){
				s11 << "; ";
			}else{
				s11 << "]" << std::endl;
			}
		}
	}
	
	s11 << "]" << std::endl;
	printf("%s", s11.str().c_str());
	printf("MapIndexInPlace:\n");
	elem_offset = 0;
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 4; ++counter){
		ad[counter] = initIndexA_functor(elem_offset + counter, ad[counter]);
	}
	elem_offset = 0;
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 16; ++counter){
		ac[counter] = initIndexA_functor(elem_offset + counter, ac[counter]);
	}
	row_offset = 0;
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
	std::array<Complex, 16> temp12{};
	MPI_Gather(ad.data(), 4, Complex_mpi_type, temp12.data(), 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	std::ostringstream s12;
	s12 << "ad: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s12 << "[" << "real = " << temp12[i].real << ", " << "imaginary = " << temp12[i].imaginary << "]";
	s12 << "; ";
	}
	s12 << "[" << "real = " << temp12[15].real << ", " << "imaginary = " << temp12[15].imaginary << "]" << "]" << std::endl;
	s12 << std::endl;
	printf("%s", s12.str().c_str());
	
	std::ostringstream s13;
	s13 << "ac: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s13 << "[" << "real = " << ac[i].real << ", " << "imaginary = " << ac[i].imaginary << "]";
	s13 << "; ";
	}
	s13 << "[" << "real = " << ac[15].real << ", " << "imaginary = " << ac[15].imaginary << "]" << "]" << std::endl;
	s13 << std::endl;
	printf("%s", s13.str().c_str());
	std::array<Complex, 16> temp14{};
	MPI_Gather(md.data(), 4, Complex_mpi_type, temp14.data(), 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	std::ostringstream s14;
	s14 << "md: " << std::endl << "[" << std::endl;
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s14 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s14 << "[" << "real = " << temp14[(counter_rows%2) * 2 + (counter_rows / 2) * 4 * 2 + (counter_cols/ 2) * 4 + counter_cols%2].real << ", " << "imaginary = " << temp14[(counter_rows%2) * 2 + (counter_rows / 2) * 4 * 2 + (counter_cols/ 2) * 4 + counter_cols%2].imaginary << "]";
			if(counter_cols < 3){
				s14 << "; ";
			}else{
				s14 << "]" << std::endl;
			}
		}
	}
	
	s14 << "]" << std::endl;
	printf("%s", s14.str().c_str());
	
	std::ostringstream s15;
	s15 << "mc: " << std::endl << "[" << std::endl;
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s15 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s15 << "[" << "real = " << mc[counter_rows * 4 + counter_cols].real << ", " << "imaginary = " << mc[counter_rows * 4 + counter_cols].imaginary << "]";
			if(counter_cols < 3){
				s15 << "; ";
			}else{
				s15 << "]" << std::endl;
			}
		}
	}
	
	s15 << "]" << std::endl;
	printf("%s", s15.str().c_str());
	printf("MapIndex:\n");
	// MapIndexSkeleton Array Start
	
	elem_offset = 0;
	
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
	
	row_offset = 0;
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
	std::array<Complex, 16> temp16{};
	MPI_Gather(adr.data(), 4, Complex_mpi_type, temp16.data(), 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	std::ostringstream s16;
	s16 << "adr: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s16 << "[" << "real = " << temp16[i].real << ", " << "imaginary = " << temp16[i].imaginary << "]";
	s16 << "; ";
	}
	s16 << "[" << "real = " << temp16[15].real << ", " << "imaginary = " << temp16[15].imaginary << "]" << "]" << std::endl;
	s16 << std::endl;
	printf("%s", s16.str().c_str());
	
	std::ostringstream s17;
	s17 << "acr: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s17 << "[" << "real = " << acr[i].real << ", " << "imaginary = " << acr[i].imaginary << "]";
	s17 << "; ";
	}
	s17 << "[" << "real = " << acr[15].real << ", " << "imaginary = " << acr[15].imaginary << "]" << "]" << std::endl;
	s17 << std::endl;
	printf("%s", s17.str().c_str());
	std::array<Complex, 16> temp18{};
	MPI_Gather(mdr.data(), 4, Complex_mpi_type, temp18.data(), 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	std::ostringstream s18;
	s18 << "mdr: " << std::endl << "[" << std::endl;
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s18 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s18 << "[" << "real = " << temp18[(counter_rows%2) * 2 + (counter_rows / 2) * 4 * 2 + (counter_cols/ 2) * 4 + counter_cols%2].real << ", " << "imaginary = " << temp18[(counter_rows%2) * 2 + (counter_rows / 2) * 4 * 2 + (counter_cols/ 2) * 4 + counter_cols%2].imaginary << "]";
			if(counter_cols < 3){
				s18 << "; ";
			}else{
				s18 << "]" << std::endl;
			}
		}
	}
	
	s18 << "]" << std::endl;
	printf("%s", s18.str().c_str());
	
	std::ostringstream s19;
	s19 << "mcr: " << std::endl << "[" << std::endl;
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s19 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s19 << "[" << "real = " << mcr[counter_rows * 4 + counter_cols].real << ", " << "imaginary = " << mcr[counter_rows * 4 + counter_cols].imaginary << "]";
			if(counter_cols < 3){
				s19 << "; ";
			}else{
				s19 << "]" << std::endl;
			}
		}
	}
	
	s19 << "]" << std::endl;
	printf("%s", s19.str().c_str());
	printf("MapLocalIndexInPlace:\n");
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
	std::array<Complex, 16> temp20{};
	MPI_Gather(ad.data(), 4, Complex_mpi_type, temp20.data(), 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	std::ostringstream s20;
	s20 << "ad: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s20 << "[" << "real = " << temp20[i].real << ", " << "imaginary = " << temp20[i].imaginary << "]";
	s20 << "; ";
	}
	s20 << "[" << "real = " << temp20[15].real << ", " << "imaginary = " << temp20[15].imaginary << "]" << "]" << std::endl;
	s20 << std::endl;
	printf("%s", s20.str().c_str());
	
	std::ostringstream s21;
	s21 << "ac: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s21 << "[" << "real = " << ac[i].real << ", " << "imaginary = " << ac[i].imaginary << "]";
	s21 << "; ";
	}
	s21 << "[" << "real = " << ac[15].real << ", " << "imaginary = " << ac[15].imaginary << "]" << "]" << std::endl;
	s21 << std::endl;
	printf("%s", s21.str().c_str());
	std::array<Complex, 16> temp22{};
	MPI_Gather(md.data(), 4, Complex_mpi_type, temp22.data(), 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	std::ostringstream s22;
	s22 << "md: " << std::endl << "[" << std::endl;
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s22 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s22 << "[" << "real = " << temp22[(counter_rows%2) * 2 + (counter_rows / 2) * 4 * 2 + (counter_cols/ 2) * 4 + counter_cols%2].real << ", " << "imaginary = " << temp22[(counter_rows%2) * 2 + (counter_rows / 2) * 4 * 2 + (counter_cols/ 2) * 4 + counter_cols%2].imaginary << "]";
			if(counter_cols < 3){
				s22 << "; ";
			}else{
				s22 << "]" << std::endl;
			}
		}
	}
	
	s22 << "]" << std::endl;
	printf("%s", s22.str().c_str());
	
	std::ostringstream s23;
	s23 << "mc: " << std::endl << "[" << std::endl;
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s23 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s23 << "[" << "real = " << mc[counter_rows * 4 + counter_cols].real << ", " << "imaginary = " << mc[counter_rows * 4 + counter_cols].imaginary << "]";
			if(counter_cols < 3){
				s23 << "; ";
			}else{
				s23 << "]" << std::endl;
			}
		}
	}
	
	s23 << "]" << std::endl;
	printf("%s", s23.str().c_str());
	printf("MapLocalIndex:\n");
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
	std::array<Complex, 16> temp24{};
	MPI_Gather(adr.data(), 4, Complex_mpi_type, temp24.data(), 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	std::ostringstream s24;
	s24 << "adr: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s24 << "[" << "real = " << temp24[i].real << ", " << "imaginary = " << temp24[i].imaginary << "]";
	s24 << "; ";
	}
	s24 << "[" << "real = " << temp24[15].real << ", " << "imaginary = " << temp24[15].imaginary << "]" << "]" << std::endl;
	s24 << std::endl;
	printf("%s", s24.str().c_str());
	
	std::ostringstream s25;
	s25 << "acr: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s25 << "[" << "real = " << acr[i].real << ", " << "imaginary = " << acr[i].imaginary << "]";
	s25 << "; ";
	}
	s25 << "[" << "real = " << acr[15].real << ", " << "imaginary = " << acr[15].imaginary << "]" << "]" << std::endl;
	s25 << std::endl;
	printf("%s", s25.str().c_str());
	std::array<Complex, 16> temp26{};
	MPI_Gather(mdr.data(), 4, Complex_mpi_type, temp26.data(), 4, Complex_mpi_type, 0, MPI_COMM_WORLD);
	
	std::ostringstream s26;
	s26 << "mdr: " << std::endl << "[" << std::endl;
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s26 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s26 << "[" << "real = " << temp26[(counter_rows%2) * 2 + (counter_rows / 2) * 4 * 2 + (counter_cols/ 2) * 4 + counter_cols%2].real << ", " << "imaginary = " << temp26[(counter_rows%2) * 2 + (counter_rows / 2) * 4 * 2 + (counter_cols/ 2) * 4 + counter_cols%2].imaginary << "]";
			if(counter_cols < 3){
				s26 << "; ";
			}else{
				s26 << "]" << std::endl;
			}
		}
	}
	
	s26 << "]" << std::endl;
	printf("%s", s26.str().c_str());
	
	std::ostringstream s27;
	s27 << "mcr: " << std::endl << "[" << std::endl;
	
	for(int counter_rows = 0; counter_rows < 4; ++counter_rows){
		s27 << "[";
		for(int counter_cols = 0; counter_cols < 4; ++counter_cols){
			s27 << "[" << "real = " << mcr[counter_rows * 4 + counter_cols].real << ", " << "imaginary = " << mcr[counter_rows * 4 + counter_cols].imaginary << "]";
			if(counter_cols < 3){
				s27 << "; ";
			}else{
				s27 << "]" << std::endl;
			}
		}
	}
	
	s27 << "]" << std::endl;
	printf("%s", s27.str().c_str());
	
	printf("Threads: %i\n", omp_get_max_threads());
	printf("Processes: %i\n", mpi_world_size);
	
	MPI_Finalize();
	return EXIT_SUCCESS;
}
