#include <mpi.h>

#include <array>
#include <sstream>
#include <chrono>
#include <random>
#include <limits>
#include <memory>
#include "../include/matmult_float-n-16-c-1.hpp"

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
	printf("Run Matmult_float-n-16-c-1\n\n");			
	}
	
	
	
	
	for(size_t counter = 0; counter  < 16777216; ++counter){
		as[counter] = 1.0f;
	}
	
	for(size_t counter = 0; counter  < 16777216; ++counter){
		bs[counter] = 0.001f;
	}
	
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
	#pragma omp simd 
	for(size_t counter_rows = 0; counter_rows < 4096; ++counter_rows){
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
	#pragma omp simd 
	for(size_t counter_rows = 0; counter_rows < 4096; ++counter_rows){
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
			int tag_240 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_240, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_241 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_241, MPI_COMM_WORLD, &requests[0]);
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
			int tag_242 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_242, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_243 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_243, MPI_COMM_WORLD, &requests[0]);
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
			int tag_244 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_244, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_245 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_245, MPI_COMM_WORLD, &requests[0]);
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
			int tag_246 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_246, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_247 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_247, MPI_COMM_WORLD, &requests[0]);
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
			int tag_248 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_248, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_249 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_249, MPI_COMM_WORLD, &requests[0]);
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
			int tag_250 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_250, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_251 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_251, MPI_COMM_WORLD, &requests[0]);
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
			int tag_252 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_252, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_253 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_253, MPI_COMM_WORLD, &requests[0]);
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
			int tag_254 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_254, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_255 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_255, MPI_COMM_WORLD, &requests[0]);
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
			int tag_256 = ((shift_source + 8) * (shift_source + 8 + 1)) / 2 + 8;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_256, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_257 = ((8 + shift_target) * (8 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_257, MPI_COMM_WORLD, &requests[0]);
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
			int tag_258 = ((shift_source + 9) * (shift_source + 9 + 1)) / 2 + 9;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_258, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_259 = ((9 + shift_target) * (9 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_259, MPI_COMM_WORLD, &requests[0]);
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
			int tag_260 = ((shift_source + 10) * (shift_source + 10 + 1)) / 2 + 10;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_260, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_261 = ((10 + shift_target) * (10 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_261, MPI_COMM_WORLD, &requests[0]);
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
			int tag_262 = ((shift_source + 11) * (shift_source + 11 + 1)) / 2 + 11;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_262, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_263 = ((11 + shift_target) * (11 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_263, MPI_COMM_WORLD, &requests[0]);
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
			int tag_264 = ((shift_source + 12) * (shift_source + 12 + 1)) / 2 + 12;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_264, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_265 = ((12 + shift_target) * (12 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_265, MPI_COMM_WORLD, &requests[0]);
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
			int tag_266 = ((shift_source + 13) * (shift_source + 13 + 1)) / 2 + 13;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_266, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_267 = ((13 + shift_target) * (13 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_267, MPI_COMM_WORLD, &requests[0]);
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
			int tag_268 = ((shift_source + 14) * (shift_source + 14 + 1)) / 2 + 14;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_268, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_269 = ((14 + shift_target) * (14 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_269, MPI_COMM_WORLD, &requests[0]);
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
			int tag_270 = ((shift_source + 15) * (shift_source + 15 + 1)) / 2 + 15;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_270, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_271 = ((15 + shift_target) * (15 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_271, MPI_COMM_WORLD, &requests[0]);
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
			int tag_272 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_272, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_273 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_273, MPI_COMM_WORLD, &requests[0]);
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
			int tag_274 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_274, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_275 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_275, MPI_COMM_WORLD, &requests[0]);
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
			int tag_276 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_276, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_277 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_277, MPI_COMM_WORLD, &requests[0]);
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
			int tag_278 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_278, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_279 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_279, MPI_COMM_WORLD, &requests[0]);
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
			int tag_280 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_280, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_281 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_281, MPI_COMM_WORLD, &requests[0]);
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
			int tag_282 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_282, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_283 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_283, MPI_COMM_WORLD, &requests[0]);
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
			int tag_284 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_284, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_285 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_285, MPI_COMM_WORLD, &requests[0]);
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
			int tag_286 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_286, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_287 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_287, MPI_COMM_WORLD, &requests[0]);
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
			int tag_288 = ((shift_source + 8) * (shift_source + 8 + 1)) / 2 + 8;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_288, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_289 = ((8 + shift_target) * (8 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_289, MPI_COMM_WORLD, &requests[0]);
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
			int tag_290 = ((shift_source + 9) * (shift_source + 9 + 1)) / 2 + 9;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_290, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_291 = ((9 + shift_target) * (9 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_291, MPI_COMM_WORLD, &requests[0]);
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
			int tag_292 = ((shift_source + 10) * (shift_source + 10 + 1)) / 2 + 10;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_292, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_293 = ((10 + shift_target) * (10 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_293, MPI_COMM_WORLD, &requests[0]);
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
			int tag_294 = ((shift_source + 11) * (shift_source + 11 + 1)) / 2 + 11;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_294, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_295 = ((11 + shift_target) * (11 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_295, MPI_COMM_WORLD, &requests[0]);
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
			int tag_296 = ((shift_source + 12) * (shift_source + 12 + 1)) / 2 + 12;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_296, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_297 = ((12 + shift_target) * (12 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_297, MPI_COMM_WORLD, &requests[0]);
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
			int tag_298 = ((shift_source + 13) * (shift_source + 13 + 1)) / 2 + 13;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_298, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_299 = ((13 + shift_target) * (13 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_299, MPI_COMM_WORLD, &requests[0]);
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
			int tag_300 = ((shift_source + 14) * (shift_source + 14 + 1)) / 2 + 14;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_300, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_301 = ((14 + shift_target) * (14 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_301, MPI_COMM_WORLD, &requests[0]);
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
			int tag_302 = ((shift_source + 15) * (shift_source + 15 + 1)) / 2 + 15;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_302, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_303 = ((15 + shift_target) * (15 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_303, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	}
	for(int i = 0; ((i) < 4); ++i){
		#pragma omp simd 
		for(size_t counter_rows = 0; counter_rows < 4096; ++counter_rows){
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
				int tag_304 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_304, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_305 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_305, MPI_COMM_WORLD, &requests[0]);
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
				int tag_306 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_306, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_307 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_307, MPI_COMM_WORLD, &requests[0]);
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
				int tag_308 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_308, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_309 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_309, MPI_COMM_WORLD, &requests[0]);
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
				int tag_310 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_310, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_311 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_311, MPI_COMM_WORLD, &requests[0]);
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
				int tag_312 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_312, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_313 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_313, MPI_COMM_WORLD, &requests[0]);
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
				int tag_314 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_314, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_315 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_315, MPI_COMM_WORLD, &requests[0]);
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
				int tag_316 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_316, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_317 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_317, MPI_COMM_WORLD, &requests[0]);
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
				int tag_318 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_318, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_319 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_319, MPI_COMM_WORLD, &requests[0]);
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
				int tag_320 = ((shift_source + 8) * (shift_source + 8 + 1)) / 2 + 8;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_320, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_321 = ((8 + shift_target) * (8 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_321, MPI_COMM_WORLD, &requests[0]);
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
				int tag_322 = ((shift_source + 9) * (shift_source + 9 + 1)) / 2 + 9;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_322, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_323 = ((9 + shift_target) * (9 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_323, MPI_COMM_WORLD, &requests[0]);
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
				int tag_324 = ((shift_source + 10) * (shift_source + 10 + 1)) / 2 + 10;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_324, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_325 = ((10 + shift_target) * (10 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_325, MPI_COMM_WORLD, &requests[0]);
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
				int tag_326 = ((shift_source + 11) * (shift_source + 11 + 1)) / 2 + 11;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_326, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_327 = ((11 + shift_target) * (11 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_327, MPI_COMM_WORLD, &requests[0]);
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
				int tag_328 = ((shift_source + 12) * (shift_source + 12 + 1)) / 2 + 12;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_328, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_329 = ((12 + shift_target) * (12 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_329, MPI_COMM_WORLD, &requests[0]);
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
				int tag_330 = ((shift_source + 13) * (shift_source + 13 + 1)) / 2 + 13;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_330, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_331 = ((13 + shift_target) * (13 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_331, MPI_COMM_WORLD, &requests[0]);
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
				int tag_332 = ((shift_source + 14) * (shift_source + 14 + 1)) / 2 + 14;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_332, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_333 = ((14 + shift_target) * (14 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_333, MPI_COMM_WORLD, &requests[0]);
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
				int tag_334 = ((shift_source + 15) * (shift_source + 15 + 1)) / 2 + 15;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_334, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_335 = ((15 + shift_target) * (15 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_335, MPI_COMM_WORLD, &requests[0]);
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
				int tag_336 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_336, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_337 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_337, MPI_COMM_WORLD, &requests[0]);
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
				int tag_338 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_338, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_339 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_339, MPI_COMM_WORLD, &requests[0]);
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
				int tag_340 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_340, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_341 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_341, MPI_COMM_WORLD, &requests[0]);
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
				int tag_342 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_342, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_343 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_343, MPI_COMM_WORLD, &requests[0]);
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
				int tag_344 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_344, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_345 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_345, MPI_COMM_WORLD, &requests[0]);
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
				int tag_346 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_346, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_347 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_347, MPI_COMM_WORLD, &requests[0]);
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
				int tag_348 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_348, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_349 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_349, MPI_COMM_WORLD, &requests[0]);
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
				int tag_350 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_350, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_351 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_351, MPI_COMM_WORLD, &requests[0]);
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
				int tag_352 = ((shift_source + 8) * (shift_source + 8 + 1)) / 2 + 8;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_352, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_353 = ((8 + shift_target) * (8 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_353, MPI_COMM_WORLD, &requests[0]);
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
				int tag_354 = ((shift_source + 9) * (shift_source + 9 + 1)) / 2 + 9;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_354, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_355 = ((9 + shift_target) * (9 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_355, MPI_COMM_WORLD, &requests[0]);
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
				int tag_356 = ((shift_source + 10) * (shift_source + 10 + 1)) / 2 + 10;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_356, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_357 = ((10 + shift_target) * (10 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_357, MPI_COMM_WORLD, &requests[0]);
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
				int tag_358 = ((shift_source + 11) * (shift_source + 11 + 1)) / 2 + 11;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_358, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_359 = ((11 + shift_target) * (11 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_359, MPI_COMM_WORLD, &requests[0]);
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
				int tag_360 = ((shift_source + 12) * (shift_source + 12 + 1)) / 2 + 12;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_360, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_361 = ((12 + shift_target) * (12 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_361, MPI_COMM_WORLD, &requests[0]);
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
				int tag_362 = ((shift_source + 13) * (shift_source + 13 + 1)) / 2 + 13;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_362, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_363 = ((13 + shift_target) * (13 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_363, MPI_COMM_WORLD, &requests[0]);
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
				int tag_364 = ((shift_source + 14) * (shift_source + 14 + 1)) / 2 + 14;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_364, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_365 = ((14 + shift_target) * (14 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_365, MPI_COMM_WORLD, &requests[0]);
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
				int tag_366 = ((shift_source + 15) * (shift_source + 15 + 1)) / 2 + 15;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_366, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_367 = ((15 + shift_target) * (15 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_367, MPI_COMM_WORLD, &requests[0]);
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
			int tag_368 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_368, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_369 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_369, MPI_COMM_WORLD, &requests[0]);
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
			int tag_370 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_370, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_371 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_371, MPI_COMM_WORLD, &requests[0]);
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
			int tag_372 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_372, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_373 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_373, MPI_COMM_WORLD, &requests[0]);
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
			int tag_374 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_374, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_375 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_375, MPI_COMM_WORLD, &requests[0]);
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
			int tag_376 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_376, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_377 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_377, MPI_COMM_WORLD, &requests[0]);
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
			int tag_378 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_378, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_379 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_379, MPI_COMM_WORLD, &requests[0]);
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
			int tag_380 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_380, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_381 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_381, MPI_COMM_WORLD, &requests[0]);
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
			int tag_382 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_382, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_383 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_383, MPI_COMM_WORLD, &requests[0]);
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
			int tag_384 = ((shift_source + 8) * (shift_source + 8 + 1)) / 2 + 8;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_384, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_385 = ((8 + shift_target) * (8 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_385, MPI_COMM_WORLD, &requests[0]);
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
			int tag_386 = ((shift_source + 9) * (shift_source + 9 + 1)) / 2 + 9;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_386, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_387 = ((9 + shift_target) * (9 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_387, MPI_COMM_WORLD, &requests[0]);
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
			int tag_388 = ((shift_source + 10) * (shift_source + 10 + 1)) / 2 + 10;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_388, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_389 = ((10 + shift_target) * (10 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_389, MPI_COMM_WORLD, &requests[0]);
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
			int tag_390 = ((shift_source + 11) * (shift_source + 11 + 1)) / 2 + 11;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_390, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_391 = ((11 + shift_target) * (11 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_391, MPI_COMM_WORLD, &requests[0]);
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
			int tag_392 = ((shift_source + 12) * (shift_source + 12 + 1)) / 2 + 12;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_392, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_393 = ((12 + shift_target) * (12 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_393, MPI_COMM_WORLD, &requests[0]);
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
			int tag_394 = ((shift_source + 13) * (shift_source + 13 + 1)) / 2 + 13;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_394, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_395 = ((13 + shift_target) * (13 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_395, MPI_COMM_WORLD, &requests[0]);
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
			int tag_396 = ((shift_source + 14) * (shift_source + 14 + 1)) / 2 + 14;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_396, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_397 = ((14 + shift_target) * (14 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_397, MPI_COMM_WORLD, &requests[0]);
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
			int tag_398 = ((shift_source + 15) * (shift_source + 15 + 1)) / 2 + 15;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_398, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_399 = ((15 + shift_target) * (15 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_399, MPI_COMM_WORLD, &requests[0]);
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
			int tag_400 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_400, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_401 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_401, MPI_COMM_WORLD, &requests[0]);
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
			int tag_402 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_402, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_403 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_403, MPI_COMM_WORLD, &requests[0]);
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
			int tag_404 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_404, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_405 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_405, MPI_COMM_WORLD, &requests[0]);
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
			int tag_406 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_406, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_407 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_407, MPI_COMM_WORLD, &requests[0]);
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
			int tag_408 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_408, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_409 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_409, MPI_COMM_WORLD, &requests[0]);
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
			int tag_410 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_410, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_411 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_411, MPI_COMM_WORLD, &requests[0]);
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
			int tag_412 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_412, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_413 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_413, MPI_COMM_WORLD, &requests[0]);
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
			int tag_414 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_414, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_415 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_415, MPI_COMM_WORLD, &requests[0]);
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
			int tag_416 = ((shift_source + 8) * (shift_source + 8 + 1)) / 2 + 8;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_416, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_417 = ((8 + shift_target) * (8 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_417, MPI_COMM_WORLD, &requests[0]);
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
			int tag_418 = ((shift_source + 9) * (shift_source + 9 + 1)) / 2 + 9;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_418, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_419 = ((9 + shift_target) * (9 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_419, MPI_COMM_WORLD, &requests[0]);
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
			int tag_420 = ((shift_source + 10) * (shift_source + 10 + 1)) / 2 + 10;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_420, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_421 = ((10 + shift_target) * (10 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_421, MPI_COMM_WORLD, &requests[0]);
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
			int tag_422 = ((shift_source + 11) * (shift_source + 11 + 1)) / 2 + 11;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_422, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_423 = ((11 + shift_target) * (11 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_423, MPI_COMM_WORLD, &requests[0]);
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
			int tag_424 = ((shift_source + 12) * (shift_source + 12 + 1)) / 2 + 12;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_424, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_425 = ((12 + shift_target) * (12 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_425, MPI_COMM_WORLD, &requests[0]);
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
			int tag_426 = ((shift_source + 13) * (shift_source + 13 + 1)) / 2 + 13;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_426, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_427 = ((13 + shift_target) * (13 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_427, MPI_COMM_WORLD, &requests[0]);
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
			int tag_428 = ((shift_source + 14) * (shift_source + 14 + 1)) / 2 + 14;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_428, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_429 = ((14 + shift_target) * (14 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_429, MPI_COMM_WORLD, &requests[0]);
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
			int tag_430 = ((shift_source + 15) * (shift_source + 15 + 1)) / 2 + 15;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_430, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_431 = ((15 + shift_target) * (15 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_431, MPI_COMM_WORLD, &requests[0]);
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
