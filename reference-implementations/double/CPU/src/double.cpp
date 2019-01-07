
#include <mpi.h>
#include <omp.h>
#include <sstream>
#include <chrono>
#include "../include/double.hpp"
#include "../include/darray.hpp"

int my_sum(int a, int b){
  return a + b;
}

// // reductions


class MySum {
 public:

  int operator()(int x, int y) const {
	  return x + y; 
  }
};

class DoublePlusX {
 public:

  int operator()(int x) const {
	  return x * _y + _z; 
  }

  int _y;
  int _z;
};

template<typename T, typename Functor>
class MapInPlace {
 public:
	void operator()(mkt::DArray<T>& inout, const Functor& f) const {
    #pragma omp parallel for simd
    for (int i = 0; i < inout.get_size_local(); ++i) {
      inout.set_local(i, f(inout.get_local(i)));
    }
  }
};


template<typename T, typename R, typename Functor>
class Fold {
 public:
	R operator()(mkt::DArray<T>& in, const R& identity, const Functor& f) const {    

    #pragma omp declare reduction(my_sum_reduction : R : omp_out = my_sum(omp_out, omp_in)) initializer(omp_priv = omp_orig)

    R result = identity;

    #pragma omp parallel for simd reduction(my_sum_reduction: result)
    for (int i = 0; i < in.get_size_local(); ++i) {
      result = f(result, in.get_local(i));
    }

    return result;
  }
};


// constants and global vars
const int size = 16;
const size_t number_of_processes = 4;
int process_id = -1;




int main(int argc, char** argv) {

	// mpi setup
	MPI_Init(&argc, &argv);
	
	int mpi_world_size = 0;
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
	
	if(mpi_world_size != number_of_processes){
		MPI_Finalize();
		return EXIT_FAILURE;
	}
	
	MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
	
	if(process_id == 0){
		printf("Run Double\n\n");			
	}
	
  mkt::DArray<int> myarray(process_id,
          size, size/number_of_processes, 2, number_of_processes, process_id, 0,
          mkt::Distribution::DIST);

  DoublePlusX dp42;
  dp42._y = 2;
  dp42._z = 42;

  MapInPlace<int, DoublePlusX> map_in_place;

  MySum my_sum;
  Fold<int, int, MySum> fold;


  if(process_id == 0){
    for (int i = 0; i < myarray.get_size_local(); ++i) {
      printf("%i ", myarray.get_local(i));
    } 
    printf("\n");
  }

	// start main program
	std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
	
	map_in_place(myarray, dp42);

  int result = fold(myarray, 0, my_sum);
		
	std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
	double seconds = std::chrono::duration<double>(timer_end - timer_start).count();

  if(process_id == 0){
    for (int i = 0; i < myarray.get_size_local(); ++i) {
      printf("%i ", myarray.get_local(i));
    } 
    printf("\nResult = %i\n", result);
  }

	// final output
	if(process_id == 0){
		printf("Execution time: %.5fs\n", seconds);
		printf("Threads: %i\n", omp_get_max_threads());
		printf("Processes: %i\n", mpi_world_size);
	}
	
	MPI_Finalize();
	return EXIT_SUCCESS;
}
