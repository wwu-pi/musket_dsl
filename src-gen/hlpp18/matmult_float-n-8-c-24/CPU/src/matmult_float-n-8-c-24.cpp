#include <mpi.h>

#include <omp.h>
#include <array>
#include <sstream>
#include <chrono>
#include <random>
#include <limits>
#include <memory>
#include "../include/matmult_float-n-8-c-24.hpp"

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
	printf("Run Matmult_float-n-8-c-24\n\n");			
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
			int tag_744 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_744, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_745 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_745, MPI_COMM_WORLD, &requests[0]);
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
			int tag_746 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_746, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_747 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_747, MPI_COMM_WORLD, &requests[0]);
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
			int tag_748 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_748, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_749 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_749, MPI_COMM_WORLD, &requests[0]);
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
			int tag_750 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_750, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_751 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_751, MPI_COMM_WORLD, &requests[0]);
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
			int tag_752 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_752, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_753 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_753, MPI_COMM_WORLD, &requests[0]);
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
			int tag_754 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_754, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_755 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_755, MPI_COMM_WORLD, &requests[0]);
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
			int tag_756 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_756, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_757 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_757, MPI_COMM_WORLD, &requests[0]);
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
			int tag_758 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_758, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_759 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_759, MPI_COMM_WORLD, &requests[0]);
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
			int tag_760 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_760, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_761 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_761, MPI_COMM_WORLD, &requests[0]);
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
			int tag_762 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_762, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_763 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_763, MPI_COMM_WORLD, &requests[0]);
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
			int tag_764 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_764, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_765 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_765, MPI_COMM_WORLD, &requests[0]);
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
			int tag_766 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_766, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_767 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_767, MPI_COMM_WORLD, &requests[0]);
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
			int tag_768 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_768, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_769 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_769, MPI_COMM_WORLD, &requests[0]);
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
			int tag_770 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_770, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_771 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_771, MPI_COMM_WORLD, &requests[0]);
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
			int tag_772 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_772, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_773 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_773, MPI_COMM_WORLD, &requests[0]);
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
			int tag_774 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_774, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_775 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_775, MPI_COMM_WORLD, &requests[0]);
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
				int tag_776 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_776, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_777 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_777, MPI_COMM_WORLD, &requests[0]);
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
				int tag_778 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_778, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_779 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_779, MPI_COMM_WORLD, &requests[0]);
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
				int tag_780 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_780, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_781 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_781, MPI_COMM_WORLD, &requests[0]);
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
				int tag_782 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_782, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_783 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_783, MPI_COMM_WORLD, &requests[0]);
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
				int tag_784 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_784, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_785 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_785, MPI_COMM_WORLD, &requests[0]);
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
				int tag_786 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_786, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_787 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_787, MPI_COMM_WORLD, &requests[0]);
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
				int tag_788 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_788, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_789 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_789, MPI_COMM_WORLD, &requests[0]);
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
				int tag_790 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_790, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_791 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_791, MPI_COMM_WORLD, &requests[0]);
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
				int tag_792 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_792, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_793 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_793, MPI_COMM_WORLD, &requests[0]);
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
				int tag_794 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_794, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_795 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_795, MPI_COMM_WORLD, &requests[0]);
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
				int tag_796 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_796, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_797 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_797, MPI_COMM_WORLD, &requests[0]);
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
				int tag_798 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_798, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_799 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_799, MPI_COMM_WORLD, &requests[0]);
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
				int tag_800 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_800, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_801 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_801, MPI_COMM_WORLD, &requests[0]);
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
				int tag_802 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_802, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_803 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_803, MPI_COMM_WORLD, &requests[0]);
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
				int tag_804 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_804, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_805 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_805, MPI_COMM_WORLD, &requests[0]);
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
				int tag_806 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_806, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 33554432 * sizeof(float);
				int tag_807 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_807, MPI_COMM_WORLD, &requests[0]);
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
			int tag_808 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_808, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_809 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_809, MPI_COMM_WORLD, &requests[0]);
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
			int tag_810 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_810, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_811 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_811, MPI_COMM_WORLD, &requests[0]);
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
			int tag_812 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_812, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_813 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_813, MPI_COMM_WORLD, &requests[0]);
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
			int tag_814 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_814, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_815 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_815, MPI_COMM_WORLD, &requests[0]);
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
			int tag_816 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_816, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_817 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_817, MPI_COMM_WORLD, &requests[0]);
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
			int tag_818 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_818, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_819 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_819, MPI_COMM_WORLD, &requests[0]);
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
			int tag_820 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_820, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_821 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_821, MPI_COMM_WORLD, &requests[0]);
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
			int tag_822 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_822, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_823 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_823, MPI_COMM_WORLD, &requests[0]);
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
			int tag_824 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_824, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_825 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_825, MPI_COMM_WORLD, &requests[0]);
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
			int tag_826 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_826, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_827 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_827, MPI_COMM_WORLD, &requests[0]);
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
			int tag_828 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_828, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_829 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_829, MPI_COMM_WORLD, &requests[0]);
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
			int tag_830 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_830, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_831 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_831, MPI_COMM_WORLD, &requests[0]);
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
			int tag_832 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_832, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_833 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_833, MPI_COMM_WORLD, &requests[0]);
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
			int tag_834 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_834, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_835 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_835, MPI_COMM_WORLD, &requests[0]);
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
			int tag_836 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_836, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_837 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_837, MPI_COMM_WORLD, &requests[0]);
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
			int tag_838 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_838, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 33554432 * sizeof(float);
			int tag_839 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_839, MPI_COMM_WORLD, &requests[0]);
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
