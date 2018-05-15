#include <mpi.h>

#include <omp.h>
#include <array>
#include <sstream>
#include <chrono>
#include <random>
#include <limits>
#include <memory>
#include "../include/matmult_float-n-16-c-18.hpp"

const size_t number_of_processes = 16;
int process_id = -1;
size_t tmp_size_t = 0;


std::vector<float> as(16777216);
std::vector<float> bs(16777216);
std::vector<float> cs(16777216);


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
	printf("Run Matmult_float-n-16-c-18\n\n");			
	}
	
	
	
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter  < 16777216; ++counter){
		as[counter] = 1.0f;
	}
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter  < 16777216; ++counter){
		bs[counter] = 0.001f;
	}
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter  < 16777216; ++counter){
		cs[counter] = 0.0f;
	}
	
	size_t row_offset = 0;size_t col_offset = 0;
	
	switch(process_id){
	case 0: {
		row_offset = 0;
		col_offset = 0;
		break;
	}
	case 1: {
		row_offset = 0;
		col_offset = 4096;
		break;
	}
	case 2: {
		row_offset = 0;
		col_offset = 8192;
		break;
	}
	case 3: {
		row_offset = 0;
		col_offset = 12288;
		break;
	}
	case 4: {
		row_offset = 4096;
		col_offset = 0;
		break;
	}
	case 5: {
		row_offset = 4096;
		col_offset = 4096;
		break;
	}
	case 6: {
		row_offset = 4096;
		col_offset = 8192;
		break;
	}
	case 7: {
		row_offset = 4096;
		col_offset = 12288;
		break;
	}
	case 8: {
		row_offset = 8192;
		col_offset = 0;
		break;
	}
	case 9: {
		row_offset = 8192;
		col_offset = 4096;
		break;
	}
	case 10: {
		row_offset = 8192;
		col_offset = 8192;
		break;
	}
	case 11: {
		row_offset = 8192;
		col_offset = 12288;
		break;
	}
	case 12: {
		row_offset = 12288;
		col_offset = 0;
		break;
	}
	case 13: {
		row_offset = 12288;
		col_offset = 4096;
		break;
	}
	case 14: {
		row_offset = 12288;
		col_offset = 8192;
		break;
	}
	case 15: {
		row_offset = 12288;
		col_offset = 12288;
		break;
	}
	}		
	#pragma omp parallel for 
	for(size_t counter_rows = 0; counter_rows < 4096; ++counter_rows){
		#pragma omp simd
		for(size_t counter_cols = 0; counter_cols < 4096; ++counter_cols){
			
			as[counter_rows * 4096 + counter_cols] = ((static_cast<float>(((row_offset + counter_rows))) * 4) + ((col_offset + counter_cols)));
		}
	}
	switch(process_id){
	case 0: {
		row_offset = 0;
		col_offset = 0;
		break;
	}
	case 1: {
		row_offset = 0;
		col_offset = 4096;
		break;
	}
	case 2: {
		row_offset = 0;
		col_offset = 8192;
		break;
	}
	case 3: {
		row_offset = 0;
		col_offset = 12288;
		break;
	}
	case 4: {
		row_offset = 4096;
		col_offset = 0;
		break;
	}
	case 5: {
		row_offset = 4096;
		col_offset = 4096;
		break;
	}
	case 6: {
		row_offset = 4096;
		col_offset = 8192;
		break;
	}
	case 7: {
		row_offset = 4096;
		col_offset = 12288;
		break;
	}
	case 8: {
		row_offset = 8192;
		col_offset = 0;
		break;
	}
	case 9: {
		row_offset = 8192;
		col_offset = 4096;
		break;
	}
	case 10: {
		row_offset = 8192;
		col_offset = 8192;
		break;
	}
	case 11: {
		row_offset = 8192;
		col_offset = 12288;
		break;
	}
	case 12: {
		row_offset = 12288;
		col_offset = 0;
		break;
	}
	case 13: {
		row_offset = 12288;
		col_offset = 4096;
		break;
	}
	case 14: {
		row_offset = 12288;
		col_offset = 8192;
		break;
	}
	case 15: {
		row_offset = 12288;
		col_offset = 12288;
		break;
	}
	}		
	#pragma omp parallel for 
	for(size_t counter_rows = 0; counter_rows < 4096; ++counter_rows){
		#pragma omp simd
		for(size_t counter_cols = 0; counter_cols < 4096; ++counter_cols){
			
			bs[counter_rows * 4096 + counter_cols] = ((static_cast<float>(16) + (((row_offset + counter_rows)) * 4)) + ((col_offset + counter_cols)));
		}
	}
	std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
	switch(process_id){
	case 0:	{
		int shift_source = 0;
		int shift_target = 0;
		int 
		shift_steps = -((0));
		
		shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) + 0;
		shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) + 0;
		
		if(shift_target != 0){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_816 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_816, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_817 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_817, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 1:	{
		int shift_source = 1;
		int shift_target = 1;
		int 
		shift_steps = -((0));
		
		shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) + 0;
		shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) + 0;
		
		if(shift_target != 1){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_818 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_818, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_819 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_819, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 2:	{
		int shift_source = 2;
		int shift_target = 2;
		int 
		shift_steps = -((0));
		
		shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) + 0;
		shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) + 0;
		
		if(shift_target != 2){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_820 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_820, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_821 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_821, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 3:	{
		int shift_source = 3;
		int shift_target = 3;
		int 
		shift_steps = -((0));
		
		shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) + 0;
		shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) + 0;
		
		if(shift_target != 3){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_822 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_822, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_823 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_823, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 4:	{
		int shift_source = 4;
		int shift_target = 4;
		int 
		shift_steps = -((1));
		
		shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) + 4;
		shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) + 4;
		
		if(shift_target != 4){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_824 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_824, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_825 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_825, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 5:	{
		int shift_source = 5;
		int shift_target = 5;
		int 
		shift_steps = -((1));
		
		shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) + 4;
		shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) + 4;
		
		if(shift_target != 5){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_826 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_826, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_827 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_827, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 6:	{
		int shift_source = 6;
		int shift_target = 6;
		int 
		shift_steps = -((1));
		
		shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) + 4;
		shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) + 4;
		
		if(shift_target != 6){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_828 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_828, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_829 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_829, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 7:	{
		int shift_source = 7;
		int shift_target = 7;
		int 
		shift_steps = -((1));
		
		shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) + 4;
		shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) + 4;
		
		if(shift_target != 7){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_830 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_830, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_831 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_831, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 8:	{
		int shift_source = 8;
		int shift_target = 8;
		int 
		shift_steps = -((2));
		
		shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) + 8;
		shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) + 8;
		
		if(shift_target != 8){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_832 = ((shift_source + 8) * (shift_source + 8 + 1)) / 2 + 8;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_832, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_833 = ((8 + shift_target) * (8 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_833, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 9:	{
		int shift_source = 9;
		int shift_target = 9;
		int 
		shift_steps = -((2));
		
		shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) + 8;
		shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) + 8;
		
		if(shift_target != 9){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_834 = ((shift_source + 9) * (shift_source + 9 + 1)) / 2 + 9;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_834, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_835 = ((9 + shift_target) * (9 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_835, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 10:	{
		int shift_source = 10;
		int shift_target = 10;
		int 
		shift_steps = -((2));
		
		shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) + 8;
		shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) + 8;
		
		if(shift_target != 10){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_836 = ((shift_source + 10) * (shift_source + 10 + 1)) / 2 + 10;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_836, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_837 = ((10 + shift_target) * (10 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_837, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 11:	{
		int shift_source = 11;
		int shift_target = 11;
		int 
		shift_steps = -((2));
		
		shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) + 8;
		shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) + 8;
		
		if(shift_target != 11){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_838 = ((shift_source + 11) * (shift_source + 11 + 1)) / 2 + 11;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_838, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_839 = ((11 + shift_target) * (11 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_839, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 12:	{
		int shift_source = 12;
		int shift_target = 12;
		int 
		shift_steps = -((3));
		
		shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) + 12;
		shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) + 12;
		
		if(shift_target != 12){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_840 = ((shift_source + 12) * (shift_source + 12 + 1)) / 2 + 12;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_840, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_841 = ((12 + shift_target) * (12 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_841, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 13:	{
		int shift_source = 13;
		int shift_target = 13;
		int 
		shift_steps = -((3));
		
		shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) + 12;
		shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) + 12;
		
		if(shift_target != 13){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_842 = ((shift_source + 13) * (shift_source + 13 + 1)) / 2 + 13;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_842, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_843 = ((13 + shift_target) * (13 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_843, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 14:	{
		int shift_source = 14;
		int shift_target = 14;
		int 
		shift_steps = -((3));
		
		shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) + 12;
		shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) + 12;
		
		if(shift_target != 14){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_844 = ((shift_source + 14) * (shift_source + 14 + 1)) / 2 + 14;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_844, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_845 = ((14 + shift_target) * (14 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_845, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 15:	{
		int shift_source = 15;
		int shift_target = 15;
		int 
		shift_steps = -((3));
		
		shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) + 12;
		shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) + 12;
		
		if(shift_target != 15){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_846 = ((shift_source + 15) * (shift_source + 15 + 1)) / 2 + 15;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_846, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_847 = ((15 + shift_target) * (15 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_847, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	}
	switch(process_id){
	case 0:	{
		int shift_source = 0;
		int shift_target = 0;
		int 
		shift_steps = -((0));
		
		shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) * 4 + 0;
		shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) * 4 + 0;
	
		if(shift_target != 0){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_848 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_848, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_849 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_849, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 1:	{
		int shift_source = 1;
		int shift_target = 1;
		int 
		shift_steps = -((1));
		
		shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) * 4 + 1;
		shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) * 4 + 1;
	
		if(shift_target != 1){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_850 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_850, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_851 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_851, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 2:	{
		int shift_source = 2;
		int shift_target = 2;
		int 
		shift_steps = -((2));
		
		shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) * 4 + 2;
		shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) * 4 + 2;
	
		if(shift_target != 2){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_852 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_852, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_853 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_853, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 3:	{
		int shift_source = 3;
		int shift_target = 3;
		int 
		shift_steps = -((3));
		
		shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) * 4 + 3;
		shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) * 4 + 3;
	
		if(shift_target != 3){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_854 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_854, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_855 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_855, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 4:	{
		int shift_source = 4;
		int shift_target = 4;
		int 
		shift_steps = -((0));
		
		shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) * 4 + 0;
		shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) * 4 + 0;
	
		if(shift_target != 4){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_856 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_856, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_857 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_857, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 5:	{
		int shift_source = 5;
		int shift_target = 5;
		int 
		shift_steps = -((1));
		
		shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) * 4 + 1;
		shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) * 4 + 1;
	
		if(shift_target != 5){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_858 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_858, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_859 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_859, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 6:	{
		int shift_source = 6;
		int shift_target = 6;
		int 
		shift_steps = -((2));
		
		shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) * 4 + 2;
		shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) * 4 + 2;
	
		if(shift_target != 6){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_860 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_860, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_861 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_861, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 7:	{
		int shift_source = 7;
		int shift_target = 7;
		int 
		shift_steps = -((3));
		
		shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) * 4 + 3;
		shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) * 4 + 3;
	
		if(shift_target != 7){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_862 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_862, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_863 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_863, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 8:	{
		int shift_source = 8;
		int shift_target = 8;
		int 
		shift_steps = -((0));
		
		shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) * 4 + 0;
		shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) * 4 + 0;
	
		if(shift_target != 8){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_864 = ((shift_source + 8) * (shift_source + 8 + 1)) / 2 + 8;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_864, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_865 = ((8 + shift_target) * (8 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_865, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 9:	{
		int shift_source = 9;
		int shift_target = 9;
		int 
		shift_steps = -((1));
		
		shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) * 4 + 1;
		shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) * 4 + 1;
	
		if(shift_target != 9){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_866 = ((shift_source + 9) * (shift_source + 9 + 1)) / 2 + 9;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_866, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_867 = ((9 + shift_target) * (9 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_867, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 10:	{
		int shift_source = 10;
		int shift_target = 10;
		int 
		shift_steps = -((2));
		
		shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) * 4 + 2;
		shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) * 4 + 2;
	
		if(shift_target != 10){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_868 = ((shift_source + 10) * (shift_source + 10 + 1)) / 2 + 10;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_868, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_869 = ((10 + shift_target) * (10 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_869, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 11:	{
		int shift_source = 11;
		int shift_target = 11;
		int 
		shift_steps = -((3));
		
		shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) * 4 + 3;
		shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) * 4 + 3;
	
		if(shift_target != 11){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_870 = ((shift_source + 11) * (shift_source + 11 + 1)) / 2 + 11;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_870, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_871 = ((11 + shift_target) * (11 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_871, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 12:	{
		int shift_source = 12;
		int shift_target = 12;
		int 
		shift_steps = -((0));
		
		shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) * 4 + 0;
		shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) * 4 + 0;
	
		if(shift_target != 12){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_872 = ((shift_source + 12) * (shift_source + 12 + 1)) / 2 + 12;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_872, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_873 = ((12 + shift_target) * (12 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_873, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 13:	{
		int shift_source = 13;
		int shift_target = 13;
		int 
		shift_steps = -((1));
		
		shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) * 4 + 1;
		shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) * 4 + 1;
	
		if(shift_target != 13){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_874 = ((shift_source + 13) * (shift_source + 13 + 1)) / 2 + 13;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_874, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_875 = ((13 + shift_target) * (13 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_875, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 14:	{
		int shift_source = 14;
		int shift_target = 14;
		int 
		shift_steps = -((2));
		
		shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) * 4 + 2;
		shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) * 4 + 2;
	
		if(shift_target != 14){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_876 = ((shift_source + 14) * (shift_source + 14 + 1)) / 2 + 14;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_876, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_877 = ((14 + shift_target) * (14 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_877, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 15:	{
		int shift_source = 15;
		int shift_target = 15;
		int 
		shift_steps = -((3));
		
		shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) * 4 + 3;
		shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) * 4 + 3;
	
		if(shift_target != 15){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_878 = ((shift_source + 15) * (shift_source + 15 + 1)) / 2 + 15;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_878, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_879 = ((15 + shift_target) * (15 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_879, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	}
	for(int i = 0; ((i) < 4); ++i){
		#pragma omp parallel for 
		for(size_t counter_rows = 0; counter_rows < 4096; ++counter_rows){
			#pragma omp simd
			for(size_t counter_cols = 0; counter_cols < 4096; ++counter_cols){
				
				float sum = (cs[counter_rows * 4096 + counter_cols]);
				for(int k = 0; ((k) < 4096); k++){
					sum += ((as)[(counter_rows) * 4096 + (k)] * (bs)[(k) * 4096 + (counter_cols)]);
				}
				cs[counter_rows * 4096 + counter_cols] = (sum);
			}
		}
		switch(process_id){
		case 0:	{
			int shift_source = 0;
			int shift_target = 0;
			int 
			shift_steps = -(1);
			
			shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) + 0;
			shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) + 0;
			
			if(shift_target != 0){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_880 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_880, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_881 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_881, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
			}
			break;
		}
		case 1:	{
			int shift_source = 1;
			int shift_target = 1;
			int 
			shift_steps = -(1);
			
			shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) + 0;
			shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) + 0;
			
			if(shift_target != 1){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_882 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_882, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_883 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_883, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
			}
			break;
		}
		case 2:	{
			int shift_source = 2;
			int shift_target = 2;
			int 
			shift_steps = -(1);
			
			shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) + 0;
			shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) + 0;
			
			if(shift_target != 2){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_884 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_884, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_885 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_885, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
			}
			break;
		}
		case 3:	{
			int shift_source = 3;
			int shift_target = 3;
			int 
			shift_steps = -(1);
			
			shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) + 0;
			shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) + 0;
			
			if(shift_target != 3){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_886 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_886, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_887 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_887, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
			}
			break;
		}
		case 4:	{
			int shift_source = 4;
			int shift_target = 4;
			int 
			shift_steps = -(1);
			
			shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) + 4;
			shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) + 4;
			
			if(shift_target != 4){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_888 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_888, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_889 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_889, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
			}
			break;
		}
		case 5:	{
			int shift_source = 5;
			int shift_target = 5;
			int 
			shift_steps = -(1);
			
			shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) + 4;
			shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) + 4;
			
			if(shift_target != 5){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_890 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_890, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_891 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_891, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
			}
			break;
		}
		case 6:	{
			int shift_source = 6;
			int shift_target = 6;
			int 
			shift_steps = -(1);
			
			shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) + 4;
			shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) + 4;
			
			if(shift_target != 6){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_892 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_892, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_893 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_893, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
			}
			break;
		}
		case 7:	{
			int shift_source = 7;
			int shift_target = 7;
			int 
			shift_steps = -(1);
			
			shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) + 4;
			shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) + 4;
			
			if(shift_target != 7){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_894 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_894, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_895 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_895, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
			}
			break;
		}
		case 8:	{
			int shift_source = 8;
			int shift_target = 8;
			int 
			shift_steps = -(1);
			
			shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) + 8;
			shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) + 8;
			
			if(shift_target != 8){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_896 = ((shift_source + 8) * (shift_source + 8 + 1)) / 2 + 8;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_896, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_897 = ((8 + shift_target) * (8 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_897, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
			}
			break;
		}
		case 9:	{
			int shift_source = 9;
			int shift_target = 9;
			int 
			shift_steps = -(1);
			
			shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) + 8;
			shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) + 8;
			
			if(shift_target != 9){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_898 = ((shift_source + 9) * (shift_source + 9 + 1)) / 2 + 9;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_898, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_899 = ((9 + shift_target) * (9 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_899, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
			}
			break;
		}
		case 10:	{
			int shift_source = 10;
			int shift_target = 10;
			int 
			shift_steps = -(1);
			
			shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) + 8;
			shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) + 8;
			
			if(shift_target != 10){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_900 = ((shift_source + 10) * (shift_source + 10 + 1)) / 2 + 10;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_900, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_901 = ((10 + shift_target) * (10 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_901, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
			}
			break;
		}
		case 11:	{
			int shift_source = 11;
			int shift_target = 11;
			int 
			shift_steps = -(1);
			
			shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) + 8;
			shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) + 8;
			
			if(shift_target != 11){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_902 = ((shift_source + 11) * (shift_source + 11 + 1)) / 2 + 11;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_902, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_903 = ((11 + shift_target) * (11 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_903, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
			}
			break;
		}
		case 12:	{
			int shift_source = 12;
			int shift_target = 12;
			int 
			shift_steps = -(1);
			
			shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) + 12;
			shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) + 12;
			
			if(shift_target != 12){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_904 = ((shift_source + 12) * (shift_source + 12 + 1)) / 2 + 12;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_904, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_905 = ((12 + shift_target) * (12 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_905, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
			}
			break;
		}
		case 13:	{
			int shift_source = 13;
			int shift_target = 13;
			int 
			shift_steps = -(1);
			
			shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) + 12;
			shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) + 12;
			
			if(shift_target != 13){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_906 = ((shift_source + 13) * (shift_source + 13 + 1)) / 2 + 13;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_906, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_907 = ((13 + shift_target) * (13 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_907, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
			}
			break;
		}
		case 14:	{
			int shift_source = 14;
			int shift_target = 14;
			int 
			shift_steps = -(1);
			
			shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) + 12;
			shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) + 12;
			
			if(shift_target != 14){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_908 = ((shift_source + 14) * (shift_source + 14 + 1)) / 2 + 14;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_908, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_909 = ((14 + shift_target) * (14 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_909, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
			}
			break;
		}
		case 15:	{
			int shift_source = 15;
			int shift_target = 15;
			int 
			shift_steps = -(1);
			
			shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) + 12;
			shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) + 12;
			
			if(shift_target != 15){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_910 = ((shift_source + 15) * (shift_source + 15 + 1)) / 2 + 15;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_910, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_911 = ((15 + shift_target) * (15 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_911, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
			}
			break;
		}
		}
		switch(process_id){
		case 0:	{
			int shift_source = 0;
			int shift_target = 0;
			int 
			shift_steps = -(1);
			
			shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) * 4 + 0;
			shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) * 4 + 0;
		
			if(shift_target != 0){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_912 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_912, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_913 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_913, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
			}
			break;
		}
		case 1:	{
			int shift_source = 1;
			int shift_target = 1;
			int 
			shift_steps = -(1);
			
			shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) * 4 + 1;
			shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) * 4 + 1;
		
			if(shift_target != 1){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_914 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_914, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_915 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_915, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
			}
			break;
		}
		case 2:	{
			int shift_source = 2;
			int shift_target = 2;
			int 
			shift_steps = -(1);
			
			shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) * 4 + 2;
			shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) * 4 + 2;
		
			if(shift_target != 2){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_916 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_916, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_917 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_917, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
			}
			break;
		}
		case 3:	{
			int shift_source = 3;
			int shift_target = 3;
			int 
			shift_steps = -(1);
			
			shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) * 4 + 3;
			shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) * 4 + 3;
		
			if(shift_target != 3){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_918 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_918, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_919 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_919, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
			}
			break;
		}
		case 4:	{
			int shift_source = 4;
			int shift_target = 4;
			int 
			shift_steps = -(1);
			
			shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) * 4 + 0;
			shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) * 4 + 0;
		
			if(shift_target != 4){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_920 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_920, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_921 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_921, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
			}
			break;
		}
		case 5:	{
			int shift_source = 5;
			int shift_target = 5;
			int 
			shift_steps = -(1);
			
			shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) * 4 + 1;
			shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) * 4 + 1;
		
			if(shift_target != 5){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_922 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_922, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_923 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_923, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
			}
			break;
		}
		case 6:	{
			int shift_source = 6;
			int shift_target = 6;
			int 
			shift_steps = -(1);
			
			shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) * 4 + 2;
			shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) * 4 + 2;
		
			if(shift_target != 6){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_924 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_924, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_925 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_925, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
			}
			break;
		}
		case 7:	{
			int shift_source = 7;
			int shift_target = 7;
			int 
			shift_steps = -(1);
			
			shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) * 4 + 3;
			shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) * 4 + 3;
		
			if(shift_target != 7){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_926 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_926, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_927 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_927, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
			}
			break;
		}
		case 8:	{
			int shift_source = 8;
			int shift_target = 8;
			int 
			shift_steps = -(1);
			
			shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) * 4 + 0;
			shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) * 4 + 0;
		
			if(shift_target != 8){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_928 = ((shift_source + 8) * (shift_source + 8 + 1)) / 2 + 8;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_928, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_929 = ((8 + shift_target) * (8 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_929, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
			}
			break;
		}
		case 9:	{
			int shift_source = 9;
			int shift_target = 9;
			int 
			shift_steps = -(1);
			
			shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) * 4 + 1;
			shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) * 4 + 1;
		
			if(shift_target != 9){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_930 = ((shift_source + 9) * (shift_source + 9 + 1)) / 2 + 9;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_930, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_931 = ((9 + shift_target) * (9 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_931, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
			}
			break;
		}
		case 10:	{
			int shift_source = 10;
			int shift_target = 10;
			int 
			shift_steps = -(1);
			
			shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) * 4 + 2;
			shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) * 4 + 2;
		
			if(shift_target != 10){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_932 = ((shift_source + 10) * (shift_source + 10 + 1)) / 2 + 10;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_932, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_933 = ((10 + shift_target) * (10 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_933, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
			}
			break;
		}
		case 11:	{
			int shift_source = 11;
			int shift_target = 11;
			int 
			shift_steps = -(1);
			
			shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) * 4 + 3;
			shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) * 4 + 3;
		
			if(shift_target != 11){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_934 = ((shift_source + 11) * (shift_source + 11 + 1)) / 2 + 11;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_934, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_935 = ((11 + shift_target) * (11 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_935, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
			}
			break;
		}
		case 12:	{
			int shift_source = 12;
			int shift_target = 12;
			int 
			shift_steps = -(1);
			
			shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) * 4 + 0;
			shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) * 4 + 0;
		
			if(shift_target != 12){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_936 = ((shift_source + 12) * (shift_source + 12 + 1)) / 2 + 12;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_936, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_937 = ((12 + shift_target) * (12 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_937, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
			}
			break;
		}
		case 13:	{
			int shift_source = 13;
			int shift_target = 13;
			int 
			shift_steps = -(1);
			
			shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) * 4 + 1;
			shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) * 4 + 1;
		
			if(shift_target != 13){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_938 = ((shift_source + 13) * (shift_source + 13 + 1)) / 2 + 13;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_938, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_939 = ((13 + shift_target) * (13 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_939, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
			}
			break;
		}
		case 14:	{
			int shift_source = 14;
			int shift_target = 14;
			int 
			shift_steps = -(1);
			
			shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) * 4 + 2;
			shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) * 4 + 2;
		
			if(shift_target != 14){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_940 = ((shift_source + 14) * (shift_source + 14 + 1)) / 2 + 14;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_940, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_941 = ((14 + shift_target) * (14 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_941, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
			}
			break;
		}
		case 15:	{
			int shift_source = 15;
			int shift_target = 15;
			int 
			shift_steps = -(1);
			
			shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) * 4 + 3;
			shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) * 4 + 3;
		
			if(shift_target != 15){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_942 = ((shift_source + 15) * (shift_source + 15 + 1)) / 2 + 15;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_942, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_943 = ((15 + shift_target) * (15 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_943, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
			}
			break;
		}
		}
	}
	switch(process_id){
	case 0:	{
		int shift_source = 0;
		int shift_target = 0;
		int 
		shift_steps = (0);
		
		shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) + 0;
		shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) + 0;
		
		if(shift_target != 0){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_944 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_944, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_945 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_945, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 1:	{
		int shift_source = 1;
		int shift_target = 1;
		int 
		shift_steps = (0);
		
		shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) + 0;
		shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) + 0;
		
		if(shift_target != 1){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_946 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_946, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_947 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_947, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 2:	{
		int shift_source = 2;
		int shift_target = 2;
		int 
		shift_steps = (0);
		
		shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) + 0;
		shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) + 0;
		
		if(shift_target != 2){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_948 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_948, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_949 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_949, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 3:	{
		int shift_source = 3;
		int shift_target = 3;
		int 
		shift_steps = (0);
		
		shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) + 0;
		shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) + 0;
		
		if(shift_target != 3){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_950 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_950, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_951 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_951, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 4:	{
		int shift_source = 4;
		int shift_target = 4;
		int 
		shift_steps = (1);
		
		shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) + 4;
		shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) + 4;
		
		if(shift_target != 4){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_952 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_952, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_953 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_953, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 5:	{
		int shift_source = 5;
		int shift_target = 5;
		int 
		shift_steps = (1);
		
		shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) + 4;
		shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) + 4;
		
		if(shift_target != 5){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_954 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_954, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_955 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_955, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 6:	{
		int shift_source = 6;
		int shift_target = 6;
		int 
		shift_steps = (1);
		
		shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) + 4;
		shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) + 4;
		
		if(shift_target != 6){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_956 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_956, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_957 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_957, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 7:	{
		int shift_source = 7;
		int shift_target = 7;
		int 
		shift_steps = (1);
		
		shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) + 4;
		shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) + 4;
		
		if(shift_target != 7){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_958 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_958, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_959 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_959, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 8:	{
		int shift_source = 8;
		int shift_target = 8;
		int 
		shift_steps = (2);
		
		shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) + 8;
		shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) + 8;
		
		if(shift_target != 8){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_960 = ((shift_source + 8) * (shift_source + 8 + 1)) / 2 + 8;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_960, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_961 = ((8 + shift_target) * (8 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_961, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 9:	{
		int shift_source = 9;
		int shift_target = 9;
		int 
		shift_steps = (2);
		
		shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) + 8;
		shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) + 8;
		
		if(shift_target != 9){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_962 = ((shift_source + 9) * (shift_source + 9 + 1)) / 2 + 9;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_962, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_963 = ((9 + shift_target) * (9 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_963, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 10:	{
		int shift_source = 10;
		int shift_target = 10;
		int 
		shift_steps = (2);
		
		shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) + 8;
		shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) + 8;
		
		if(shift_target != 10){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_964 = ((shift_source + 10) * (shift_source + 10 + 1)) / 2 + 10;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_964, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_965 = ((10 + shift_target) * (10 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_965, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 11:	{
		int shift_source = 11;
		int shift_target = 11;
		int 
		shift_steps = (2);
		
		shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) + 8;
		shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) + 8;
		
		if(shift_target != 11){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_966 = ((shift_source + 11) * (shift_source + 11 + 1)) / 2 + 11;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_966, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_967 = ((11 + shift_target) * (11 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_967, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 12:	{
		int shift_source = 12;
		int shift_target = 12;
		int 
		shift_steps = (3);
		
		shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) + 12;
		shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) + 12;
		
		if(shift_target != 12){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_968 = ((shift_source + 12) * (shift_source + 12 + 1)) / 2 + 12;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_968, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_969 = ((12 + shift_target) * (12 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_969, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 13:	{
		int shift_source = 13;
		int shift_target = 13;
		int 
		shift_steps = (3);
		
		shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) + 12;
		shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) + 12;
		
		if(shift_target != 13){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_970 = ((shift_source + 13) * (shift_source + 13 + 1)) / 2 + 13;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_970, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_971 = ((13 + shift_target) * (13 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_971, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 14:	{
		int shift_source = 14;
		int shift_target = 14;
		int 
		shift_steps = (3);
		
		shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) + 12;
		shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) + 12;
		
		if(shift_target != 14){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_972 = ((shift_source + 14) * (shift_source + 14 + 1)) / 2 + 14;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_972, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_973 = ((14 + shift_target) * (14 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_973, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 15:	{
		int shift_source = 15;
		int shift_target = 15;
		int 
		shift_steps = (3);
		
		shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) + 12;
		shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) + 12;
		
		if(shift_target != 15){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_974 = ((shift_source + 15) * (shift_source + 15 + 1)) / 2 + 15;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_974, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_975 = ((15 + shift_target) * (15 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_975, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	}
	switch(process_id){
	case 0:	{
		int shift_source = 0;
		int shift_target = 0;
		int 
		shift_steps = (0);
		
		shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) * 4 + 0;
		shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) * 4 + 0;
	
		if(shift_target != 0){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_976 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_976, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_977 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_977, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 1:	{
		int shift_source = 1;
		int shift_target = 1;
		int 
		shift_steps = (1);
		
		shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) * 4 + 1;
		shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) * 4 + 1;
	
		if(shift_target != 1){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_978 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_978, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_979 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_979, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 2:	{
		int shift_source = 2;
		int shift_target = 2;
		int 
		shift_steps = (2);
		
		shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) * 4 + 2;
		shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) * 4 + 2;
	
		if(shift_target != 2){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_980 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_980, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_981 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_981, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 3:	{
		int shift_source = 3;
		int shift_target = 3;
		int 
		shift_steps = (3);
		
		shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) * 4 + 3;
		shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) * 4 + 3;
	
		if(shift_target != 3){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_982 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_982, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_983 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_983, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 4:	{
		int shift_source = 4;
		int shift_target = 4;
		int 
		shift_steps = (0);
		
		shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) * 4 + 0;
		shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) * 4 + 0;
	
		if(shift_target != 4){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_984 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_984, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_985 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_985, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 5:	{
		int shift_source = 5;
		int shift_target = 5;
		int 
		shift_steps = (1);
		
		shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) * 4 + 1;
		shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) * 4 + 1;
	
		if(shift_target != 5){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_986 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_986, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_987 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_987, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 6:	{
		int shift_source = 6;
		int shift_target = 6;
		int 
		shift_steps = (2);
		
		shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) * 4 + 2;
		shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) * 4 + 2;
	
		if(shift_target != 6){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_988 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_988, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_989 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_989, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 7:	{
		int shift_source = 7;
		int shift_target = 7;
		int 
		shift_steps = (3);
		
		shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) * 4 + 3;
		shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) * 4 + 3;
	
		if(shift_target != 7){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_990 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_990, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_991 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_991, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 8:	{
		int shift_source = 8;
		int shift_target = 8;
		int 
		shift_steps = (0);
		
		shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) * 4 + 0;
		shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) * 4 + 0;
	
		if(shift_target != 8){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_992 = ((shift_source + 8) * (shift_source + 8 + 1)) / 2 + 8;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_992, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_993 = ((8 + shift_target) * (8 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_993, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 9:	{
		int shift_source = 9;
		int shift_target = 9;
		int 
		shift_steps = (1);
		
		shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) * 4 + 1;
		shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) * 4 + 1;
	
		if(shift_target != 9){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_994 = ((shift_source + 9) * (shift_source + 9 + 1)) / 2 + 9;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_994, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_995 = ((9 + shift_target) * (9 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_995, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 10:	{
		int shift_source = 10;
		int shift_target = 10;
		int 
		shift_steps = (2);
		
		shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) * 4 + 2;
		shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) * 4 + 2;
	
		if(shift_target != 10){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_996 = ((shift_source + 10) * (shift_source + 10 + 1)) / 2 + 10;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_996, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_997 = ((10 + shift_target) * (10 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_997, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 11:	{
		int shift_source = 11;
		int shift_target = 11;
		int 
		shift_steps = (3);
		
		shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) * 4 + 3;
		shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) * 4 + 3;
	
		if(shift_target != 11){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_998 = ((shift_source + 11) * (shift_source + 11 + 1)) / 2 + 11;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_998, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_999 = ((11 + shift_target) * (11 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_999, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 12:	{
		int shift_source = 12;
		int shift_target = 12;
		int 
		shift_steps = (0);
		
		shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) * 4 + 0;
		shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) * 4 + 0;
	
		if(shift_target != 12){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1000 = ((shift_source + 12) * (shift_source + 12 + 1)) / 2 + 12;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1000, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1001 = ((12 + shift_target) * (12 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1001, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 13:	{
		int shift_source = 13;
		int shift_target = 13;
		int 
		shift_steps = (1);
		
		shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) * 4 + 1;
		shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) * 4 + 1;
	
		if(shift_target != 13){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1002 = ((shift_source + 13) * (shift_source + 13 + 1)) / 2 + 13;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1002, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1003 = ((13 + shift_target) * (13 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1003, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 14:	{
		int shift_source = 14;
		int shift_target = 14;
		int 
		shift_steps = (2);
		
		shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) * 4 + 2;
		shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) * 4 + 2;
	
		if(shift_target != 14){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1004 = ((shift_source + 14) * (shift_source + 14 + 1)) / 2 + 14;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1004, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1005 = ((14 + shift_target) * (14 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1005, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 15:	{
		int shift_source = 15;
		int shift_target = 15;
		int 
		shift_steps = (3);
		
		shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) * 4 + 3;
		shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) * 4 + 3;
	
		if(shift_target != 15){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1006 = ((shift_source + 15) * (shift_source + 15 + 1)) / 2 + 15;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1006, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1007 = ((15 + shift_target) * (15 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1007, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	}
	std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
	double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
	
	if(process_id == 0){
	printf("Execution time: %.5fs\n", seconds);
	printf("Threads: %i\n", omp_get_max_threads());
	printf("Processes: %i\n", mpi_world_size);
	}
	
	MPI_Finalize();
	return EXIT_SUCCESS;
}
