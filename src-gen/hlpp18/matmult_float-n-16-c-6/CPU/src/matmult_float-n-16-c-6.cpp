#include <mpi.h>

#include <omp.h>
#include <array>
#include <sstream>
#include <chrono>
#include <random>
#include <limits>
#include <memory>
#include "../include/matmult_float-n-16-c-6.hpp"

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
	printf("Run Matmult_float-n-16-c-6\n\n");			
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
			int tag_432 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_432, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_433 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_433, MPI_COMM_WORLD, &requests[0]);
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
			int tag_434 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_434, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_435 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_435, MPI_COMM_WORLD, &requests[0]);
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
			int tag_436 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_436, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_437 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_437, MPI_COMM_WORLD, &requests[0]);
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
			int tag_438 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_438, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_439 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_439, MPI_COMM_WORLD, &requests[0]);
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
			int tag_440 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_440, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_441 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_441, MPI_COMM_WORLD, &requests[0]);
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
			int tag_442 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_442, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_443 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_443, MPI_COMM_WORLD, &requests[0]);
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
			int tag_444 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_444, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_445 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_445, MPI_COMM_WORLD, &requests[0]);
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
			int tag_446 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_446, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_447 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_447, MPI_COMM_WORLD, &requests[0]);
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
			int tag_448 = ((shift_source + 8) * (shift_source + 8 + 1)) / 2 + 8;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_448, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_449 = ((8 + shift_target) * (8 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_449, MPI_COMM_WORLD, &requests[0]);
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
			int tag_450 = ((shift_source + 9) * (shift_source + 9 + 1)) / 2 + 9;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_450, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_451 = ((9 + shift_target) * (9 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_451, MPI_COMM_WORLD, &requests[0]);
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
			int tag_452 = ((shift_source + 10) * (shift_source + 10 + 1)) / 2 + 10;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_452, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_453 = ((10 + shift_target) * (10 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_453, MPI_COMM_WORLD, &requests[0]);
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
			int tag_454 = ((shift_source + 11) * (shift_source + 11 + 1)) / 2 + 11;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_454, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_455 = ((11 + shift_target) * (11 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_455, MPI_COMM_WORLD, &requests[0]);
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
			int tag_456 = ((shift_source + 12) * (shift_source + 12 + 1)) / 2 + 12;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_456, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_457 = ((12 + shift_target) * (12 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_457, MPI_COMM_WORLD, &requests[0]);
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
			int tag_458 = ((shift_source + 13) * (shift_source + 13 + 1)) / 2 + 13;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_458, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_459 = ((13 + shift_target) * (13 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_459, MPI_COMM_WORLD, &requests[0]);
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
			int tag_460 = ((shift_source + 14) * (shift_source + 14 + 1)) / 2 + 14;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_460, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_461 = ((14 + shift_target) * (14 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_461, MPI_COMM_WORLD, &requests[0]);
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
			int tag_462 = ((shift_source + 15) * (shift_source + 15 + 1)) / 2 + 15;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_462, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_463 = ((15 + shift_target) * (15 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_463, MPI_COMM_WORLD, &requests[0]);
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
			int tag_464 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_464, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_465 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_465, MPI_COMM_WORLD, &requests[0]);
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
			int tag_466 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_466, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_467 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_467, MPI_COMM_WORLD, &requests[0]);
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
			int tag_468 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_468, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_469 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_469, MPI_COMM_WORLD, &requests[0]);
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
			int tag_470 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_470, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_471 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_471, MPI_COMM_WORLD, &requests[0]);
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
			int tag_472 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_472, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_473 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_473, MPI_COMM_WORLD, &requests[0]);
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
			int tag_474 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_474, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_475 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_475, MPI_COMM_WORLD, &requests[0]);
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
			int tag_476 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_476, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_477 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_477, MPI_COMM_WORLD, &requests[0]);
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
			int tag_478 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_478, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_479 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_479, MPI_COMM_WORLD, &requests[0]);
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
			int tag_480 = ((shift_source + 8) * (shift_source + 8 + 1)) / 2 + 8;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_480, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_481 = ((8 + shift_target) * (8 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_481, MPI_COMM_WORLD, &requests[0]);
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
			int tag_482 = ((shift_source + 9) * (shift_source + 9 + 1)) / 2 + 9;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_482, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_483 = ((9 + shift_target) * (9 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_483, MPI_COMM_WORLD, &requests[0]);
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
			int tag_484 = ((shift_source + 10) * (shift_source + 10 + 1)) / 2 + 10;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_484, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_485 = ((10 + shift_target) * (10 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_485, MPI_COMM_WORLD, &requests[0]);
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
			int tag_486 = ((shift_source + 11) * (shift_source + 11 + 1)) / 2 + 11;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_486, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_487 = ((11 + shift_target) * (11 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_487, MPI_COMM_WORLD, &requests[0]);
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
			int tag_488 = ((shift_source + 12) * (shift_source + 12 + 1)) / 2 + 12;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_488, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_489 = ((12 + shift_target) * (12 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_489, MPI_COMM_WORLD, &requests[0]);
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
			int tag_490 = ((shift_source + 13) * (shift_source + 13 + 1)) / 2 + 13;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_490, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_491 = ((13 + shift_target) * (13 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_491, MPI_COMM_WORLD, &requests[0]);
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
			int tag_492 = ((shift_source + 14) * (shift_source + 14 + 1)) / 2 + 14;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_492, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_493 = ((14 + shift_target) * (14 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_493, MPI_COMM_WORLD, &requests[0]);
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
			int tag_494 = ((shift_source + 15) * (shift_source + 15 + 1)) / 2 + 15;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_494, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_495 = ((15 + shift_target) * (15 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_495, MPI_COMM_WORLD, &requests[0]);
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
				int tag_496 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_496, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_497 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_497, MPI_COMM_WORLD, &requests[0]);
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
				int tag_498 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_498, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_499 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_499, MPI_COMM_WORLD, &requests[0]);
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
				int tag_500 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_500, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_501 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_501, MPI_COMM_WORLD, &requests[0]);
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
				int tag_502 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_502, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_503 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_503, MPI_COMM_WORLD, &requests[0]);
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
				int tag_504 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_504, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_505 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_505, MPI_COMM_WORLD, &requests[0]);
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
				int tag_506 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_506, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_507 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_507, MPI_COMM_WORLD, &requests[0]);
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
				int tag_508 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_508, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_509 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_509, MPI_COMM_WORLD, &requests[0]);
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
				int tag_510 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_510, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_511 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_511, MPI_COMM_WORLD, &requests[0]);
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
				int tag_512 = ((shift_source + 8) * (shift_source + 8 + 1)) / 2 + 8;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_512, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_513 = ((8 + shift_target) * (8 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_513, MPI_COMM_WORLD, &requests[0]);
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
				int tag_514 = ((shift_source + 9) * (shift_source + 9 + 1)) / 2 + 9;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_514, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_515 = ((9 + shift_target) * (9 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_515, MPI_COMM_WORLD, &requests[0]);
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
				int tag_516 = ((shift_source + 10) * (shift_source + 10 + 1)) / 2 + 10;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_516, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_517 = ((10 + shift_target) * (10 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_517, MPI_COMM_WORLD, &requests[0]);
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
				int tag_518 = ((shift_source + 11) * (shift_source + 11 + 1)) / 2 + 11;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_518, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_519 = ((11 + shift_target) * (11 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_519, MPI_COMM_WORLD, &requests[0]);
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
				int tag_520 = ((shift_source + 12) * (shift_source + 12 + 1)) / 2 + 12;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_520, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_521 = ((12 + shift_target) * (12 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_521, MPI_COMM_WORLD, &requests[0]);
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
				int tag_522 = ((shift_source + 13) * (shift_source + 13 + 1)) / 2 + 13;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_522, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_523 = ((13 + shift_target) * (13 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_523, MPI_COMM_WORLD, &requests[0]);
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
				int tag_524 = ((shift_source + 14) * (shift_source + 14 + 1)) / 2 + 14;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_524, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_525 = ((14 + shift_target) * (14 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_525, MPI_COMM_WORLD, &requests[0]);
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
				int tag_526 = ((shift_source + 15) * (shift_source + 15 + 1)) / 2 + 15;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_526, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_527 = ((15 + shift_target) * (15 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_527, MPI_COMM_WORLD, &requests[0]);
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
				int tag_528 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_528, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_529 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_529, MPI_COMM_WORLD, &requests[0]);
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
				int tag_530 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_530, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_531 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_531, MPI_COMM_WORLD, &requests[0]);
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
				int tag_532 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_532, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_533 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_533, MPI_COMM_WORLD, &requests[0]);
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
				int tag_534 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_534, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_535 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_535, MPI_COMM_WORLD, &requests[0]);
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
				int tag_536 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_536, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_537 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_537, MPI_COMM_WORLD, &requests[0]);
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
				int tag_538 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_538, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_539 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_539, MPI_COMM_WORLD, &requests[0]);
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
				int tag_540 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_540, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_541 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_541, MPI_COMM_WORLD, &requests[0]);
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
				int tag_542 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_542, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_543 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_543, MPI_COMM_WORLD, &requests[0]);
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
				int tag_544 = ((shift_source + 8) * (shift_source + 8 + 1)) / 2 + 8;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_544, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_545 = ((8 + shift_target) * (8 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_545, MPI_COMM_WORLD, &requests[0]);
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
				int tag_546 = ((shift_source + 9) * (shift_source + 9 + 1)) / 2 + 9;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_546, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_547 = ((9 + shift_target) * (9 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_547, MPI_COMM_WORLD, &requests[0]);
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
				int tag_548 = ((shift_source + 10) * (shift_source + 10 + 1)) / 2 + 10;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_548, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_549 = ((10 + shift_target) * (10 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_549, MPI_COMM_WORLD, &requests[0]);
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
				int tag_550 = ((shift_source + 11) * (shift_source + 11 + 1)) / 2 + 11;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_550, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_551 = ((11 + shift_target) * (11 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_551, MPI_COMM_WORLD, &requests[0]);
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
				int tag_552 = ((shift_source + 12) * (shift_source + 12 + 1)) / 2 + 12;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_552, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_553 = ((12 + shift_target) * (12 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_553, MPI_COMM_WORLD, &requests[0]);
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
				int tag_554 = ((shift_source + 13) * (shift_source + 13 + 1)) / 2 + 13;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_554, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_555 = ((13 + shift_target) * (13 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_555, MPI_COMM_WORLD, &requests[0]);
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
				int tag_556 = ((shift_source + 14) * (shift_source + 14 + 1)) / 2 + 14;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_556, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_557 = ((14 + shift_target) * (14 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_557, MPI_COMM_WORLD, &requests[0]);
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
				int tag_558 = ((shift_source + 15) * (shift_source + 15 + 1)) / 2 + 15;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_558, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_559 = ((15 + shift_target) * (15 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_559, MPI_COMM_WORLD, &requests[0]);
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
			int tag_560 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_560, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_561 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_561, MPI_COMM_WORLD, &requests[0]);
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
			int tag_562 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_562, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_563 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_563, MPI_COMM_WORLD, &requests[0]);
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
			int tag_564 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_564, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_565 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_565, MPI_COMM_WORLD, &requests[0]);
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
			int tag_566 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_566, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_567 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_567, MPI_COMM_WORLD, &requests[0]);
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
			int tag_568 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_568, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_569 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_569, MPI_COMM_WORLD, &requests[0]);
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
			int tag_570 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_570, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_571 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_571, MPI_COMM_WORLD, &requests[0]);
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
			int tag_572 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_572, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_573 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_573, MPI_COMM_WORLD, &requests[0]);
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
			int tag_574 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_574, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_575 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_575, MPI_COMM_WORLD, &requests[0]);
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
			int tag_576 = ((shift_source + 8) * (shift_source + 8 + 1)) / 2 + 8;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_576, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_577 = ((8 + shift_target) * (8 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_577, MPI_COMM_WORLD, &requests[0]);
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
			int tag_578 = ((shift_source + 9) * (shift_source + 9 + 1)) / 2 + 9;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_578, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_579 = ((9 + shift_target) * (9 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_579, MPI_COMM_WORLD, &requests[0]);
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
			int tag_580 = ((shift_source + 10) * (shift_source + 10 + 1)) / 2 + 10;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_580, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_581 = ((10 + shift_target) * (10 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_581, MPI_COMM_WORLD, &requests[0]);
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
			int tag_582 = ((shift_source + 11) * (shift_source + 11 + 1)) / 2 + 11;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_582, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_583 = ((11 + shift_target) * (11 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_583, MPI_COMM_WORLD, &requests[0]);
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
			int tag_584 = ((shift_source + 12) * (shift_source + 12 + 1)) / 2 + 12;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_584, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_585 = ((12 + shift_target) * (12 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_585, MPI_COMM_WORLD, &requests[0]);
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
			int tag_586 = ((shift_source + 13) * (shift_source + 13 + 1)) / 2 + 13;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_586, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_587 = ((13 + shift_target) * (13 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_587, MPI_COMM_WORLD, &requests[0]);
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
			int tag_588 = ((shift_source + 14) * (shift_source + 14 + 1)) / 2 + 14;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_588, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_589 = ((14 + shift_target) * (14 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_589, MPI_COMM_WORLD, &requests[0]);
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
			int tag_590 = ((shift_source + 15) * (shift_source + 15 + 1)) / 2 + 15;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_590, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_591 = ((15 + shift_target) * (15 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_591, MPI_COMM_WORLD, &requests[0]);
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
			int tag_592 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_592, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_593 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_593, MPI_COMM_WORLD, &requests[0]);
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
			int tag_594 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_594, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_595 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_595, MPI_COMM_WORLD, &requests[0]);
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
			int tag_596 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_596, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_597 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_597, MPI_COMM_WORLD, &requests[0]);
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
			int tag_598 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_598, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_599 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_599, MPI_COMM_WORLD, &requests[0]);
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
			int tag_600 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_600, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_601 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_601, MPI_COMM_WORLD, &requests[0]);
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
			int tag_602 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_602, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_603 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_603, MPI_COMM_WORLD, &requests[0]);
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
			int tag_604 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_604, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_605 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_605, MPI_COMM_WORLD, &requests[0]);
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
			int tag_606 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_606, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_607 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_607, MPI_COMM_WORLD, &requests[0]);
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
			int tag_608 = ((shift_source + 8) * (shift_source + 8 + 1)) / 2 + 8;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_608, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_609 = ((8 + shift_target) * (8 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_609, MPI_COMM_WORLD, &requests[0]);
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
			int tag_610 = ((shift_source + 9) * (shift_source + 9 + 1)) / 2 + 9;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_610, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_611 = ((9 + shift_target) * (9 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_611, MPI_COMM_WORLD, &requests[0]);
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
			int tag_612 = ((shift_source + 10) * (shift_source + 10 + 1)) / 2 + 10;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_612, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_613 = ((10 + shift_target) * (10 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_613, MPI_COMM_WORLD, &requests[0]);
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
			int tag_614 = ((shift_source + 11) * (shift_source + 11 + 1)) / 2 + 11;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_614, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_615 = ((11 + shift_target) * (11 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_615, MPI_COMM_WORLD, &requests[0]);
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
			int tag_616 = ((shift_source + 12) * (shift_source + 12 + 1)) / 2 + 12;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_616, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_617 = ((12 + shift_target) * (12 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_617, MPI_COMM_WORLD, &requests[0]);
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
			int tag_618 = ((shift_source + 13) * (shift_source + 13 + 1)) / 2 + 13;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_618, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_619 = ((13 + shift_target) * (13 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_619, MPI_COMM_WORLD, &requests[0]);
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
			int tag_620 = ((shift_source + 14) * (shift_source + 14 + 1)) / 2 + 14;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_620, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_621 = ((14 + shift_target) * (14 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_621, MPI_COMM_WORLD, &requests[0]);
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
			int tag_622 = ((shift_source + 15) * (shift_source + 15 + 1)) / 2 + 15;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_622, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_623 = ((15 + shift_target) * (15 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_623, MPI_COMM_WORLD, &requests[0]);
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
