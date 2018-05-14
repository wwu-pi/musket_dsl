	#include <mpi.h>
	#include <array>
	#include <vector>
	#include "../include/mpi.hpp"
	
	const int number_of_processes = 4;
	const int vector_size = 4;
	int process_id = -1;	

	int main(int argc, char** argv) {

		MPI_Init(&argc, &argv);
		
		int mpi_world_size = 0;
		MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
		
		if(mpi_world_size != number_of_processes){
			if(process_id == 0){
				printf("Please run with %i processes!\n", number_of_processes);			
			}
			MPI_Finalize();
			return EXIT_FAILURE;
		}
		
		MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
		
		if(process_id == 0){
			printf("Run MPI test!\n\n");			
		}
			
		std::vector<int> input;
		std::vector<int> output(number_of_processes*vector_size, -1);
			
		switch(process_id){
		case 0 :{input.assign(vector_size, 0);
		break;}
		case 1 :{input.assign(vector_size, 1);
		break;}
		case 2 :{input.assign(vector_size, 2);
		break;}
		case 3 :{input.assign(vector_size, 3);
		break;}
		}
			
		if(process_id == 0){
			printf("input:\n");
			for(int i = 0; i < vector_size; ++i){
				printf("%i, ", input[i]);
			} 
		}

		//MPI_Allgather(input.data(), vector_size, MPI_INT, output.data(), vector_size, MPI_INT, MPI_COMM_WORLD); // [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]
			
		MPI_Datatype vec_type_tmp, vec_type;
		MPI_Type_vector(2, 2, 4, MPI_INT, &vec_type_tmp);
		MPI_Type_create_resized(vec_type_tmp, 0, sizeof(int) * 2, &vec_type);
		MPI_Type_free(&vec_type_tmp);
		MPI_Type_commit(&vec_type);

		//MPI_Datatype row_type;
		//MPI_Type_vector(2, MPI_INT, &row_type);
		//MPI_Type_commit(&row_type);	

		std::array<int, number_of_processes> counts = {1,1,1,1};
		std::array<int, number_of_processes> displs = {0,1,4,5};
		

		//MPI_Allgather(input.data(), vector_size, MPI_INT, output.data(), 1, vec_type, MPI_COMM_WORLD); // [0,0,1,1,0,0,1,1,2,2,3,3,2,2,3,3]

		MPI_Allgatherv(input.data(), vector_size, MPI_INT, output.data(), counts.data(), displs.data(), vec_type, MPI_COMM_WORLD); // [0,0,1,1,0,0,1,1,2,2,3,3,2,2,3,3]

		//MPI_Allgatherv(input.data(), 2, row_type, output.data(), counts.data(), displs.data(), vec_type, MPI_COMM_WORLD); // [0,0,1,1,0,0,1,1,2,2,3,3,2,2,3,3]

		if(process_id == 0){
			printf("\n\nOutput:\n");
			for(int i = 0; i < output.size(); ++i){
				printf("%i, ", output[i]);
			} 
		}
		MPI_Finalize();
		return EXIT_SUCCESS;
	}
