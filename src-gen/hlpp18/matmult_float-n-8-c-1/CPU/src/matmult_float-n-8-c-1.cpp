#include <mpi.h>

#include <array>
#include <sstream>
#include <chrono>
#include <random>
#include <limits>
#include <memory>
#include "../include/matmult_float-n-8-c-1.hpp"

const size_t number_of_processes = 8;
int process_id = -1;
size_t tmp_size_t = 0;


std::vector<float> as(33554432);
std::vector<float> bs(33554432);
std::vector<float> cs(33554432);


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
	printf("Run Matmult_float-n-8-c-1\n\n");			
	}
	
	
	
	
	for(size_t counter = 0; counter  < 33554432; ++counter){
		as[counter] = 1.0f;
	}
	
	for(size_t counter = 0; counter  < 33554432; ++counter){
		bs[counter] = 0.001f;
	}
	
	for(size_t counter = 0; counter  < 33554432; ++counter){
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
		col_offset = 8192;
		break;
	}
	case 2: {
		row_offset = 8192;
		col_offset = 0;
		break;
	}
	case 3: {
		row_offset = 8192;
		col_offset = 8192;
		break;
	}
	case 4: {
		row_offset = 16384;
		col_offset = 0;
		break;
	}
	case 5: {
		row_offset = 16384;
		col_offset = 8192;
		break;
	}
	case 6: {
		row_offset = 24576;
		col_offset = 0;
		break;
	}
	case 7: {
		row_offset = 24576;
		col_offset = 8192;
		break;
	}
	}		
	for(size_t counter_rows = 0; counter_rows < 8192; ++counter_rows){
		for(size_t counter_cols = 0; counter_cols < 8192; ++counter_cols){
			
			as[counter_rows * 8192 + counter_cols] = ((static_cast<float>(((row_offset + counter_rows))) * 4) + ((col_offset + counter_cols)));
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
		col_offset = 8192;
		break;
	}
	case 2: {
		row_offset = 8192;
		col_offset = 0;
		break;
	}
	case 3: {
		row_offset = 8192;
		col_offset = 8192;
		break;
	}
	case 4: {
		row_offset = 16384;
		col_offset = 0;
		break;
	}
	case 5: {
		row_offset = 16384;
		col_offset = 8192;
		break;
	}
	case 6: {
		row_offset = 24576;
		col_offset = 0;
		break;
	}
	case 7: {
		row_offset = 24576;
		col_offset = 8192;
		break;
	}
	}		
	for(size_t counter_rows = 0; counter_rows < 8192; ++counter_rows){
		for(size_t counter_cols = 0; counter_cols < 8192; ++counter_cols){
			
			bs[counter_rows * 8192 + counter_cols] = ((static_cast<float>(16) + (((row_offset + counter_rows)) * 4)) + ((col_offset + counter_cols)));
		}
	}
	std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
	switch(process_id){
	case 0:	{
		int shift_source = 0;
		int shift_target = 0;
		int 
		shift_steps = -((0));
		
		shift_target = ((((0 + shift_steps) % 2) + 2 ) % 2) + 0;
		shift_source = ((((0 - shift_steps) % 2) + 2 ) % 2) + 0;
		
		if(shift_target != 0){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_360 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_360, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_361 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_361, MPI_COMM_WORLD, &requests[0]);
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
		
		shift_target = ((((1 + shift_steps) % 2) + 2 ) % 2) + 0;
		shift_source = ((((1 - shift_steps) % 2) + 2 ) % 2) + 0;
		
		if(shift_target != 1){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_362 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_362, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_363 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_363, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
					
		}
		break;
	}
	case 2:	{
		int shift_source = 2;
		int shift_target = 2;
		int 
		shift_steps = -((1));
		
		shift_target = ((((0 + shift_steps) % 2) + 2 ) % 2) + 2;
		shift_source = ((((0 - shift_steps) % 2) + 2 ) % 2) + 2;
		
		if(shift_target != 2){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_364 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_364, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_365 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_365, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
					
		}
		break;
	}
	case 3:	{
		int shift_source = 3;
		int shift_target = 3;
		int 
		shift_steps = -((1));
		
		shift_target = ((((1 + shift_steps) % 2) + 2 ) % 2) + 2;
		shift_source = ((((1 - shift_steps) % 2) + 2 ) % 2) + 2;
		
		if(shift_target != 3){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_366 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_366, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_367 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_367, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
					
		}
		break;
	}
	case 4:	{
		int shift_source = 4;
		int shift_target = 4;
		int 
		shift_steps = -((2));
		
		shift_target = ((((0 + shift_steps) % 2) + 2 ) % 2) + 4;
		shift_source = ((((0 - shift_steps) % 2) + 2 ) % 2) + 4;
		
		if(shift_target != 4){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_368 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_368, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_369 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_369, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
					
		}
		break;
	}
	case 5:	{
		int shift_source = 5;
		int shift_target = 5;
		int 
		shift_steps = -((2));
		
		shift_target = ((((1 + shift_steps) % 2) + 2 ) % 2) + 4;
		shift_source = ((((1 - shift_steps) % 2) + 2 ) % 2) + 4;
		
		if(shift_target != 5){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_370 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_370, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_371 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_371, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
					
		}
		break;
	}
	case 6:	{
		int shift_source = 6;
		int shift_target = 6;
		int 
		shift_steps = -((3));
		
		shift_target = ((((0 + shift_steps) % 2) + 2 ) % 2) + 6;
		shift_source = ((((0 - shift_steps) % 2) + 2 ) % 2) + 6;
		
		if(shift_target != 6){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_372 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_372, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_373 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_373, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
					
		}
		break;
	}
	case 7:	{
		int shift_source = 7;
		int shift_target = 7;
		int 
		shift_steps = -((3));
		
		shift_target = ((((1 + shift_steps) % 2) + 2 ) % 2) + 6;
		shift_source = ((((1 - shift_steps) % 2) + 2 ) % 2) + 6;
		
		if(shift_target != 7){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_374 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_374, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_375 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_375, MPI_COMM_WORLD, &requests[0]);
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
		
		shift_target = ((((0 + shift_steps) % 2) + 2 ) % 2) * 2 + 0;
		shift_source = ((((0 - shift_steps) % 2) + 2 ) % 2) * 2 + 0;
	
		if(shift_target != 0){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_376 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_376, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_377 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_377, MPI_COMM_WORLD, &requests[0]);
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
		
		shift_target = ((((0 + shift_steps) % 2) + 2 ) % 2) * 2 + 1;
		shift_source = ((((0 - shift_steps) % 2) + 2 ) % 2) * 2 + 1;
	
		if(shift_target != 1){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_378 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_378, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_379 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_379, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 2:	{
		int shift_source = 2;
		int shift_target = 2;
		int 
		shift_steps = -((0));
		
		shift_target = ((((1 + shift_steps) % 2) + 2 ) % 2) * 2 + 0;
		shift_source = ((((1 - shift_steps) % 2) + 2 ) % 2) * 2 + 0;
	
		if(shift_target != 2){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_380 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_380, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_381 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_381, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 3:	{
		int shift_source = 3;
		int shift_target = 3;
		int 
		shift_steps = -((1));
		
		shift_target = ((((1 + shift_steps) % 2) + 2 ) % 2) * 2 + 1;
		shift_source = ((((1 - shift_steps) % 2) + 2 ) % 2) * 2 + 1;
	
		if(shift_target != 3){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_382 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_382, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_383 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_383, MPI_COMM_WORLD, &requests[0]);
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
		
		shift_target = ((((2 + shift_steps) % 2) + 2 ) % 2) * 2 + 0;
		shift_source = ((((2 - shift_steps) % 2) + 2 ) % 2) * 2 + 0;
	
		if(shift_target != 4){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_384 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_384, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_385 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_385, MPI_COMM_WORLD, &requests[0]);
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
		
		shift_target = ((((2 + shift_steps) % 2) + 2 ) % 2) * 2 + 1;
		shift_source = ((((2 - shift_steps) % 2) + 2 ) % 2) * 2 + 1;
	
		if(shift_target != 5){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_386 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_386, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_387 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_387, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 6:	{
		int shift_source = 6;
		int shift_target = 6;
		int 
		shift_steps = -((0));
		
		shift_target = ((((3 + shift_steps) % 2) + 2 ) % 2) * 2 + 0;
		shift_source = ((((3 - shift_steps) % 2) + 2 ) % 2) * 2 + 0;
	
		if(shift_target != 6){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_388 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_388, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_389 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_389, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 7:	{
		int shift_source = 7;
		int shift_target = 7;
		int 
		shift_steps = -((1));
		
		shift_target = ((((3 + shift_steps) % 2) + 2 ) % 2) * 2 + 1;
		shift_source = ((((3 - shift_steps) % 2) + 2 ) % 2) * 2 + 1;
	
		if(shift_target != 7){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_390 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_390, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_391 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_391, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	}
	for(int i = 0; ((i) < 2); ++i){
		for(size_t counter_rows = 0; counter_rows < 8192; ++counter_rows){
			for(size_t counter_cols = 0; counter_cols < 8192; ++counter_cols){
				
				float sum = (cs[counter_rows * 8192 + counter_cols]);
				for(int k = 0; ((k) < 8192); k++){
					sum += ((as)[(counter_rows) * 8192 + (k)] * (bs)[(k) * 8192 + (counter_cols)]);
				}
				cs[counter_rows * 8192 + counter_cols] = (sum);
			}
		}
		switch(process_id){
		case 0:	{
			int shift_source = 0;
			int shift_target = 0;
			int 
			shift_steps = -(1);
			
			shift_target = ((((0 + shift_steps) % 2) + 2 ) % 2) + 0;
			shift_source = ((((0 - shift_steps) % 2) + 2 ) % 2) + 0;
			
			if(shift_target != 0){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_392 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_392, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_393 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_393, MPI_COMM_WORLD, &requests[0]);
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
			
			shift_target = ((((1 + shift_steps) % 2) + 2 ) % 2) + 0;
			shift_source = ((((1 - shift_steps) % 2) + 2 ) % 2) + 0;
			
			if(shift_target != 1){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_394 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_394, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_395 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_395, MPI_COMM_WORLD, &requests[0]);
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
			
			shift_target = ((((0 + shift_steps) % 2) + 2 ) % 2) + 2;
			shift_source = ((((0 - shift_steps) % 2) + 2 ) % 2) + 2;
			
			if(shift_target != 2){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_396 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_396, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_397 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_397, MPI_COMM_WORLD, &requests[0]);
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
			
			shift_target = ((((1 + shift_steps) % 2) + 2 ) % 2) + 2;
			shift_source = ((((1 - shift_steps) % 2) + 2 ) % 2) + 2;
			
			if(shift_target != 3){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_398 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_398, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_399 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_399, MPI_COMM_WORLD, &requests[0]);
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
			
			shift_target = ((((0 + shift_steps) % 2) + 2 ) % 2) + 4;
			shift_source = ((((0 - shift_steps) % 2) + 2 ) % 2) + 4;
			
			if(shift_target != 4){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_400 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_400, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_401 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_401, MPI_COMM_WORLD, &requests[0]);
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
			
			shift_target = ((((1 + shift_steps) % 2) + 2 ) % 2) + 4;
			shift_source = ((((1 - shift_steps) % 2) + 2 ) % 2) + 4;
			
			if(shift_target != 5){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_402 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_402, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_403 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_403, MPI_COMM_WORLD, &requests[0]);
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
			
			shift_target = ((((0 + shift_steps) % 2) + 2 ) % 2) + 6;
			shift_source = ((((0 - shift_steps) % 2) + 2 ) % 2) + 6;
			
			if(shift_target != 6){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_404 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_404, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_405 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_405, MPI_COMM_WORLD, &requests[0]);
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
			
			shift_target = ((((1 + shift_steps) % 2) + 2 ) % 2) + 6;
			shift_source = ((((1 - shift_steps) % 2) + 2 ) % 2) + 6;
			
			if(shift_target != 7){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_406 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_406, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_407 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_407, MPI_COMM_WORLD, &requests[0]);
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
			
			shift_target = ((((0 + shift_steps) % 2) + 2 ) % 2) * 2 + 0;
			shift_source = ((((0 - shift_steps) % 2) + 2 ) % 2) * 2 + 0;
		
			if(shift_target != 0){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_408 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_408, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_409 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_409, MPI_COMM_WORLD, &requests[0]);
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
			
			shift_target = ((((0 + shift_steps) % 2) + 2 ) % 2) * 2 + 1;
			shift_source = ((((0 - shift_steps) % 2) + 2 ) % 2) * 2 + 1;
		
			if(shift_target != 1){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_410 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_410, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_411 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_411, MPI_COMM_WORLD, &requests[0]);
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
			
			shift_target = ((((1 + shift_steps) % 2) + 2 ) % 2) * 2 + 0;
			shift_source = ((((1 - shift_steps) % 2) + 2 ) % 2) * 2 + 0;
		
			if(shift_target != 2){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_412 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_412, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_413 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_413, MPI_COMM_WORLD, &requests[0]);
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
			
			shift_target = ((((1 + shift_steps) % 2) + 2 ) % 2) * 2 + 1;
			shift_source = ((((1 - shift_steps) % 2) + 2 ) % 2) * 2 + 1;
		
			if(shift_target != 3){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_414 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_414, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_415 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_415, MPI_COMM_WORLD, &requests[0]);
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
			
			shift_target = ((((2 + shift_steps) % 2) + 2 ) % 2) * 2 + 0;
			shift_source = ((((2 - shift_steps) % 2) + 2 ) % 2) * 2 + 0;
		
			if(shift_target != 4){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_416 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_416, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_417 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_417, MPI_COMM_WORLD, &requests[0]);
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
			
			shift_target = ((((2 + shift_steps) % 2) + 2 ) % 2) * 2 + 1;
			shift_source = ((((2 - shift_steps) % 2) + 2 ) % 2) * 2 + 1;
		
			if(shift_target != 5){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_418 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_418, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_419 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_419, MPI_COMM_WORLD, &requests[0]);
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
			
			shift_target = ((((3 + shift_steps) % 2) + 2 ) % 2) * 2 + 0;
			shift_source = ((((3 - shift_steps) % 2) + 2 ) % 2) * 2 + 0;
		
			if(shift_target != 6){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_420 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_420, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_421 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_421, MPI_COMM_WORLD, &requests[0]);
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
			
			shift_target = ((((3 + shift_steps) % 2) + 2 ) % 2) * 2 + 1;
			shift_source = ((((3 - shift_steps) % 2) + 2 ) % 2) * 2 + 1;
		
			if(shift_target != 7){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_422 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_422, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_423 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_423, MPI_COMM_WORLD, &requests[0]);
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
		
		shift_target = ((((0 + shift_steps) % 2) + 2 ) % 2) + 0;
		shift_source = ((((0 - shift_steps) % 2) + 2 ) % 2) + 0;
		
		if(shift_target != 0){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_424 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_424, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_425 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_425, MPI_COMM_WORLD, &requests[0]);
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
		
		shift_target = ((((1 + shift_steps) % 2) + 2 ) % 2) + 0;
		shift_source = ((((1 - shift_steps) % 2) + 2 ) % 2) + 0;
		
		if(shift_target != 1){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_426 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_426, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_427 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_427, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
					
		}
		break;
	}
	case 2:	{
		int shift_source = 2;
		int shift_target = 2;
		int 
		shift_steps = (1);
		
		shift_target = ((((0 + shift_steps) % 2) + 2 ) % 2) + 2;
		shift_source = ((((0 - shift_steps) % 2) + 2 ) % 2) + 2;
		
		if(shift_target != 2){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_428 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_428, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_429 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_429, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
					
		}
		break;
	}
	case 3:	{
		int shift_source = 3;
		int shift_target = 3;
		int 
		shift_steps = (1);
		
		shift_target = ((((1 + shift_steps) % 2) + 2 ) % 2) + 2;
		shift_source = ((((1 - shift_steps) % 2) + 2 ) % 2) + 2;
		
		if(shift_target != 3){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_430 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_430, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_431 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_431, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
					
		}
		break;
	}
	case 4:	{
		int shift_source = 4;
		int shift_target = 4;
		int 
		shift_steps = (2);
		
		shift_target = ((((0 + shift_steps) % 2) + 2 ) % 2) + 4;
		shift_source = ((((0 - shift_steps) % 2) + 2 ) % 2) + 4;
		
		if(shift_target != 4){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_432 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_432, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_433 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_433, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
					
		}
		break;
	}
	case 5:	{
		int shift_source = 5;
		int shift_target = 5;
		int 
		shift_steps = (2);
		
		shift_target = ((((1 + shift_steps) % 2) + 2 ) % 2) + 4;
		shift_source = ((((1 - shift_steps) % 2) + 2 ) % 2) + 4;
		
		if(shift_target != 5){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_434 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_434, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_435 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_435, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
					
		}
		break;
	}
	case 6:	{
		int shift_source = 6;
		int shift_target = 6;
		int 
		shift_steps = (3);
		
		shift_target = ((((0 + shift_steps) % 2) + 2 ) % 2) + 6;
		shift_source = ((((0 - shift_steps) % 2) + 2 ) % 2) + 6;
		
		if(shift_target != 6){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_436 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_436, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_437 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_437, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
					
		}
		break;
	}
	case 7:	{
		int shift_source = 7;
		int shift_target = 7;
		int 
		shift_steps = (3);
		
		shift_target = ((((1 + shift_steps) % 2) + 2 ) % 2) + 6;
		shift_source = ((((1 - shift_steps) % 2) + 2 ) % 2) + 6;
		
		if(shift_target != 7){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_438 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_438, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_439 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_439, MPI_COMM_WORLD, &requests[0]);
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
		
		shift_target = ((((0 + shift_steps) % 2) + 2 ) % 2) * 2 + 0;
		shift_source = ((((0 - shift_steps) % 2) + 2 ) % 2) * 2 + 0;
	
		if(shift_target != 0){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_440 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_440, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_441 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_441, MPI_COMM_WORLD, &requests[0]);
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
		
		shift_target = ((((0 + shift_steps) % 2) + 2 ) % 2) * 2 + 1;
		shift_source = ((((0 - shift_steps) % 2) + 2 ) % 2) * 2 + 1;
	
		if(shift_target != 1){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_442 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_442, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_443 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_443, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 2:	{
		int shift_source = 2;
		int shift_target = 2;
		int 
		shift_steps = (0);
		
		shift_target = ((((1 + shift_steps) % 2) + 2 ) % 2) * 2 + 0;
		shift_source = ((((1 - shift_steps) % 2) + 2 ) % 2) * 2 + 0;
	
		if(shift_target != 2){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_444 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_444, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_445 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_445, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 3:	{
		int shift_source = 3;
		int shift_target = 3;
		int 
		shift_steps = (1);
		
		shift_target = ((((1 + shift_steps) % 2) + 2 ) % 2) * 2 + 1;
		shift_source = ((((1 - shift_steps) % 2) + 2 ) % 2) * 2 + 1;
	
		if(shift_target != 3){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_446 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_446, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_447 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_447, MPI_COMM_WORLD, &requests[0]);
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
		
		shift_target = ((((2 + shift_steps) % 2) + 2 ) % 2) * 2 + 0;
		shift_source = ((((2 - shift_steps) % 2) + 2 ) % 2) * 2 + 0;
	
		if(shift_target != 4){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_448 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_448, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_449 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_449, MPI_COMM_WORLD, &requests[0]);
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
		
		shift_target = ((((2 + shift_steps) % 2) + 2 ) % 2) * 2 + 1;
		shift_source = ((((2 - shift_steps) % 2) + 2 ) % 2) * 2 + 1;
	
		if(shift_target != 5){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_450 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_450, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_451 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_451, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 6:	{
		int shift_source = 6;
		int shift_target = 6;
		int 
		shift_steps = (0);
		
		shift_target = ((((3 + shift_steps) % 2) + 2 ) % 2) * 2 + 0;
		shift_source = ((((3 - shift_steps) % 2) + 2 ) % 2) * 2 + 0;
	
		if(shift_target != 6){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_452 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_452, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_453 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_453, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 7:	{
		int shift_source = 7;
		int shift_target = 7;
		int 
		shift_steps = (1);
		
		shift_target = ((((3 + shift_steps) % 2) + 2 ) % 2) * 2 + 1;
		shift_source = ((((3 - shift_steps) % 2) + 2 ) % 2) * 2 + 1;
	
		if(shift_target != 7){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(33554432);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_454 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_454, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_455 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_455, MPI_COMM_WORLD, &requests[0]);
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
	printf("Threads: %i\n", 1);
	printf("Processes: %i\n", mpi_world_size);
	}
	
	MPI_Finalize();
	return EXIT_SUCCESS;
}
