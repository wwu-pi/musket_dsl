#include <mpi.h>

#include <omp.h>
#include <array>
#include <sstream>
#include <chrono>
#include <random>
#include <limits>
#include <memory>
#include "../include/matmult_float-n-8-c-18.hpp"

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
	printf("Run Matmult_float-n-8-c-18\n\n");			
	}
	
	
	
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter  < 33554432; ++counter){
		as[counter] = 1.0f;
	}
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter  < 33554432; ++counter){
		bs[counter] = 0.001f;
	}
	
	#pragma omp parallel for simd
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
	#pragma omp parallel for
	for(size_t counter_rows = 0; counter_rows < 8192; ++counter_rows){
		#pragma omp simd
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
	#pragma omp parallel for
	for(size_t counter_rows = 0; counter_rows < 8192; ++counter_rows){
		#pragma omp simd
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
			int tag_648 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_648, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_649 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_649, MPI_COMM_WORLD, &requests[0]);
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
			int tag_650 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_650, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_651 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_651, MPI_COMM_WORLD, &requests[0]);
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
			int tag_652 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_652, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_653 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_653, MPI_COMM_WORLD, &requests[0]);
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
			int tag_654 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_654, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_655 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_655, MPI_COMM_WORLD, &requests[0]);
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
			int tag_656 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_656, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_657 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_657, MPI_COMM_WORLD, &requests[0]);
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
			int tag_658 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_658, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_659 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_659, MPI_COMM_WORLD, &requests[0]);
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
			int tag_660 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_660, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_661 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_661, MPI_COMM_WORLD, &requests[0]);
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
			int tag_662 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_662, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_663 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_663, MPI_COMM_WORLD, &requests[0]);
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
			int tag_664 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_664, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_665 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_665, MPI_COMM_WORLD, &requests[0]);
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
			int tag_666 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_666, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_667 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_667, MPI_COMM_WORLD, &requests[0]);
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
			int tag_668 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_668, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_669 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_669, MPI_COMM_WORLD, &requests[0]);
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
			int tag_670 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_670, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_671 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_671, MPI_COMM_WORLD, &requests[0]);
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
			int tag_672 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_672, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_673 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_673, MPI_COMM_WORLD, &requests[0]);
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
			int tag_674 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_674, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_675 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_675, MPI_COMM_WORLD, &requests[0]);
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
			int tag_676 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_676, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_677 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_677, MPI_COMM_WORLD, &requests[0]);
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
			int tag_678 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_678, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_679 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_679, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	}
	for(int i = 0; ((i) < 2); ++i){
		#pragma omp parallel for
		for(size_t counter_rows = 0; counter_rows < 8192; ++counter_rows){
			#pragma omp simd
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
				int tag_680 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_680, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_681 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_681, MPI_COMM_WORLD, &requests[0]);
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
				int tag_682 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_682, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_683 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_683, MPI_COMM_WORLD, &requests[0]);
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
				int tag_684 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_684, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_685 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_685, MPI_COMM_WORLD, &requests[0]);
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
				int tag_686 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_686, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_687 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_687, MPI_COMM_WORLD, &requests[0]);
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
				int tag_688 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_688, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_689 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_689, MPI_COMM_WORLD, &requests[0]);
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
				int tag_690 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_690, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_691 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_691, MPI_COMM_WORLD, &requests[0]);
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
				int tag_692 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_692, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_693 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_693, MPI_COMM_WORLD, &requests[0]);
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
				int tag_694 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_694, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_695 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_695, MPI_COMM_WORLD, &requests[0]);
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
				int tag_696 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_696, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_697 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_697, MPI_COMM_WORLD, &requests[0]);
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
				int tag_698 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_698, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_699 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_699, MPI_COMM_WORLD, &requests[0]);
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
				int tag_700 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_700, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_701 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_701, MPI_COMM_WORLD, &requests[0]);
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
				int tag_702 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_702, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_703 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_703, MPI_COMM_WORLD, &requests[0]);
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
				int tag_704 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_704, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_705 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_705, MPI_COMM_WORLD, &requests[0]);
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
				int tag_706 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_706, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_707 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_707, MPI_COMM_WORLD, &requests[0]);
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
				int tag_708 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_708, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_709 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_709, MPI_COMM_WORLD, &requests[0]);
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
				int tag_710 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_710, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_711 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_711, MPI_COMM_WORLD, &requests[0]);
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
			int tag_712 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_712, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_713 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_713, MPI_COMM_WORLD, &requests[0]);
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
			int tag_714 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_714, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_715 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_715, MPI_COMM_WORLD, &requests[0]);
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
			int tag_716 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_716, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_717 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_717, MPI_COMM_WORLD, &requests[0]);
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
			int tag_718 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_718, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_719 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_719, MPI_COMM_WORLD, &requests[0]);
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
			int tag_720 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_720, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_721 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_721, MPI_COMM_WORLD, &requests[0]);
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
			int tag_722 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_722, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_723 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_723, MPI_COMM_WORLD, &requests[0]);
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
			int tag_724 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_724, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_725 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_725, MPI_COMM_WORLD, &requests[0]);
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
			int tag_726 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_726, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_727 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_727, MPI_COMM_WORLD, &requests[0]);
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
			int tag_728 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_728, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_729 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_729, MPI_COMM_WORLD, &requests[0]);
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
			int tag_730 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_730, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_731 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_731, MPI_COMM_WORLD, &requests[0]);
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
			int tag_732 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_732, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_733 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_733, MPI_COMM_WORLD, &requests[0]);
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
			int tag_734 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_734, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_735 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_735, MPI_COMM_WORLD, &requests[0]);
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
			int tag_736 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_736, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_737 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_737, MPI_COMM_WORLD, &requests[0]);
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
			int tag_738 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_738, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_739 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_739, MPI_COMM_WORLD, &requests[0]);
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
			int tag_740 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_740, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_741 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_741, MPI_COMM_WORLD, &requests[0]);
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
			int tag_742 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_742, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_743 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_743, MPI_COMM_WORLD, &requests[0]);
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
