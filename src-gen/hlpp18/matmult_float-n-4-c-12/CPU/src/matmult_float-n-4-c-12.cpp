#include <mpi.h>

#include <omp.h>
#include <array>
#include <sstream>
#include <chrono>
#include <random>
#include <limits>
#include <memory>
#include "../include/matmult_float-n-4-c-12.hpp"

const size_t number_of_processes = 4;
int process_id = -1;
size_t tmp_size_t = 0;


std::vector<float> as(67108864);
std::vector<float> bs(67108864);
std::vector<float> cs(67108864);


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
	printf("Run Matmult_float-n-4-c-12\n\n");			
	}
	
	
	
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter  < 67108864; ++counter){
		as[counter] = 1.0f;
	}
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter  < 67108864; ++counter){
		bs[counter] = 0.001f;
	}
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter  < 67108864; ++counter){
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
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(67108864);
			tmp_size_t = 67108864 * sizeof(float);
			int tag_96 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_96, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 67108864 * sizeof(float);
			int tag_97 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_97, MPI_COMM_WORLD, &requests[0]);
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
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(67108864);
			tmp_size_t = 67108864 * sizeof(float);
			int tag_98 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_98, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 67108864 * sizeof(float);
			int tag_99 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_99, MPI_COMM_WORLD, &requests[0]);
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
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(67108864);
			tmp_size_t = 67108864 * sizeof(float);
			int tag_100 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_100, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 67108864 * sizeof(float);
			int tag_101 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_101, MPI_COMM_WORLD, &requests[0]);
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
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(67108864);
			tmp_size_t = 67108864 * sizeof(float);
			int tag_102 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_102, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 67108864 * sizeof(float);
			int tag_103 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_103, MPI_COMM_WORLD, &requests[0]);
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
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(67108864);
			tmp_size_t = 67108864 * sizeof(float);
			int tag_104 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_104, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 67108864 * sizeof(float);
			int tag_105 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_105, MPI_COMM_WORLD, &requests[0]);
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
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(67108864);
			tmp_size_t = 67108864 * sizeof(float);
			int tag_106 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_106, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 67108864 * sizeof(float);
			int tag_107 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_107, MPI_COMM_WORLD, &requests[0]);
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
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(67108864);
			tmp_size_t = 67108864 * sizeof(float);
			int tag_108 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_108, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 67108864 * sizeof(float);
			int tag_109 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_109, MPI_COMM_WORLD, &requests[0]);
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
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(67108864);
			tmp_size_t = 67108864 * sizeof(float);
			int tag_110 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_110, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 67108864 * sizeof(float);
			int tag_111 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_111, MPI_COMM_WORLD, &requests[0]);
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
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(67108864);
				tmp_size_t = 67108864 * sizeof(float);
				int tag_112 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_112, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 67108864 * sizeof(float);
				int tag_113 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_113, MPI_COMM_WORLD, &requests[0]);
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
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(67108864);
				tmp_size_t = 67108864 * sizeof(float);
				int tag_114 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_114, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 67108864 * sizeof(float);
				int tag_115 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_115, MPI_COMM_WORLD, &requests[0]);
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
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(67108864);
				tmp_size_t = 67108864 * sizeof(float);
				int tag_116 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_116, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 67108864 * sizeof(float);
				int tag_117 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_117, MPI_COMM_WORLD, &requests[0]);
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
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(67108864);
				tmp_size_t = 67108864 * sizeof(float);
				int tag_118 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_118, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 67108864 * sizeof(float);
				int tag_119 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_119, MPI_COMM_WORLD, &requests[0]);
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
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(67108864);
				tmp_size_t = 67108864 * sizeof(float);
				int tag_120 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_120, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 67108864 * sizeof(float);
				int tag_121 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_121, MPI_COMM_WORLD, &requests[0]);
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
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(67108864);
				tmp_size_t = 67108864 * sizeof(float);
				int tag_122 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_122, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 67108864 * sizeof(float);
				int tag_123 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_123, MPI_COMM_WORLD, &requests[0]);
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
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(67108864);
				tmp_size_t = 67108864 * sizeof(float);
				int tag_124 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_124, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 67108864 * sizeof(float);
				int tag_125 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_125, MPI_COMM_WORLD, &requests[0]);
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
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(67108864);
				tmp_size_t = 67108864 * sizeof(float);
				int tag_126 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_126, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 67108864 * sizeof(float);
				int tag_127 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_127, MPI_COMM_WORLD, &requests[0]);
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
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(67108864);
			tmp_size_t = 67108864 * sizeof(float);
			int tag_128 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_128, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 67108864 * sizeof(float);
			int tag_129 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_129, MPI_COMM_WORLD, &requests[0]);
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
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(67108864);
			tmp_size_t = 67108864 * sizeof(float);
			int tag_130 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_130, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 67108864 * sizeof(float);
			int tag_131 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_131, MPI_COMM_WORLD, &requests[0]);
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
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(67108864);
			tmp_size_t = 67108864 * sizeof(float);
			int tag_132 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_132, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 67108864 * sizeof(float);
			int tag_133 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_133, MPI_COMM_WORLD, &requests[0]);
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
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(67108864);
			tmp_size_t = 67108864 * sizeof(float);
			int tag_134 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_134, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 67108864 * sizeof(float);
			int tag_135 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_135, MPI_COMM_WORLD, &requests[0]);
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
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(67108864);
			tmp_size_t = 67108864 * sizeof(float);
			int tag_136 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_136, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 67108864 * sizeof(float);
			int tag_137 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_137, MPI_COMM_WORLD, &requests[0]);
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
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(67108864);
			tmp_size_t = 67108864 * sizeof(float);
			int tag_138 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_138, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 67108864 * sizeof(float);
			int tag_139 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_139, MPI_COMM_WORLD, &requests[0]);
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
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(67108864);
			tmp_size_t = 67108864 * sizeof(float);
			int tag_140 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_140, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 67108864 * sizeof(float);
			int tag_141 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_141, MPI_COMM_WORLD, &requests[0]);
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
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(67108864);
			tmp_size_t = 67108864 * sizeof(float);
			int tag_142 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_142, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 67108864 * sizeof(float);
			int tag_143 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_143, MPI_COMM_WORLD, &requests[0]);
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
