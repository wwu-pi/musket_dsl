package de.wwu.musket.generator.cuda.lib

import org.apache.log4j.LogManager
import org.apache.log4j.Logger
import de.wwu.musket.generator.cuda.Config
import org.eclipse.emf.ecore.resource.Resource
import org.eclipse.xtext.generator.IFileSystemAccess2
import org.eclipse.xtext.generator.IGeneratorContext

class Kernel {
	private static final Logger logger = LogManager.getLogger(Kernel)

	def static void generateKernelHeaderFile(Resource resource, IFileSystemAccess2 fsa, IGeneratorContext context) {
		logger.info("Generate Kernel header file.")
		fsa.generateFile(Config.base_path + Config.include_path + 'kernel' + Config.header_extension,
			headerFileContent(resource))
		logger.info("Generation of Kernel header file done.")
	}

	def static headerFileContent(Resource resource) '''
		#include <limits.h>
		#include <float.h>
			
		namespace mkt{
		namespace kernel{
		
		«generateMapKernelDeclarations»
		«generateFoldKernelDeclarations»
		«generateReductionKernelDeclarations»
		
		} //namespace kernel
		} //namespace mkt
		
		«generateMapKernelDefinitions»
		«generateFoldKernelDefinitions»
		«generateReductionKernelDefinitions»
	'''

	def static generateMapKernelDeclarations() '''
		template <typename T, typename R, typename F>
		__global__ void map(T* in, R* out, size_t size, F func);
		
		template <typename T, typename F>
		__global__ void map_in_place(T* inout, size_t size, F func);		
		
		template <typename T, typename R, typename F>
		__global__ void map_index(T* in, R* out, size_t size, size_t offset, F func);
		
		template <typename T, typename F>
		__global__ void map_index_in_place(T* inout, size_t size, size_t offset, F func);
		
		template <typename T, typename R, typename F>
		__global__ void map_index(T* in, R* out, size_t rows, size_t columns, size_t row_offset, size_t column_offset, F func);
		
		template <typename T, typename F>
		__global__ void map_index_in_place(T* inout, size_t rows, size_t columns, size_t row_offset, size_t column_offset, F func);
	'''
	
	def static generateFoldKernelDeclarations() '''
		template<typename T, typename F>
		void fold_call(size_t size, T* d_idata, T* d_odata, int threads, int blocks, F& f, cudaStream_t& stream, int gpu);

		template<typename T, typename F, size_t blockSize>
		__global__ void fold(T *g_idata, T *g_odata, size_t n, F func);
	'''
	
	def static generateReductionKernelDeclarations() '''
		«generateReductionCallKernelDeclaration("plus", true, "")»
		«generateReductionCallKernelDeclaration("multiply", true, "")»
		«generateReductionCallKernelDeclaration("min", false, "int")»
		«generateReductionCallKernelDeclaration("min", false, "float")»
		«generateReductionCallKernelDeclaration("min", false, "double")»
		«generateReductionCallKernelDeclaration("max", false, "int")»
		«generateReductionCallKernelDeclaration("max", false, "float")»
		«generateReductionCallKernelDeclaration("max", false, "double")»

		template<typename T, size_t blockSize>
		__global__ void reduce_plus(T *g_idata, T *g_odata, size_t n);
		
		template<typename T, size_t blockSize>
		__global__ void reduce_multiply(T *g_idata, T *g_odata, size_t n);
		
		template<size_t blockSize>
		__global__ void reduce_max(int *g_idata, int *g_odata, size_t n);
		
		template<size_t blockSize>
		__global__ void reduce_max(float *g_idata, float *g_odata, size_t n);
		
		template<size_t blockSize>
		__global__ void reduce_max(double *g_idata, double *g_odata, size_t n);
		
		template<size_t blockSize>
		__global__ void reduce_min(int *g_idata, int *g_odata, size_t n);
		
		template<size_t blockSize>
		__global__ void reduce_min(float *g_idata, float *g_odata, size_t n);
		
		template<size_t blockSize>
		__global__ void reduce_min(double *g_idata, double *g_odata, size_t n);
	'''
	
	def static generateReductionCallKernelDeclaration(String name, boolean template, String type) '''
		«IF template»template<typename T>«ENDIF»
		void reduce_«name»_call(size_t size, «IF template»T«ELSE»«type»«ENDIF»* d_idata, «IF template»T«ELSE»«type»«ENDIF»* d_odata, int threads, int blocks, cudaStream_t& stream, int gpu);»
	'''
	
	def static generateMapKernelDefinitions() '''
		template <typename T, typename R, typename F>
		__global__ void mkt::kernel::map(T* in, R* out, size_t size, F func)
		{
		  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
		
		  if (x < size) {
		    out[x] = func(in[x]);
		  }
		}
		
		template <typename T, typename F>
		__global__ void mkt::kernel::map_in_place(T* inout, size_t size, F func)
		{
		  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
		
		  if (x < size) {
		    func(inout[x]);
		  }
		}
		
		template <typename T, typename R, typename F>
		__global__ void mkt::kernel::map_index(T* in, R* out, size_t size, size_t offset, F func)
		{
		  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
		
		  if (x < size) {
		    out[x] = func(x + offset, in[x]);
		  }
		}
		
		template <typename T, typename F>
		__global__ void mkt::kernel::map_index_in_place(T* inout, size_t size, size_t offset, F func)
		{
		  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
		  
		  if (x < size) {
		    func(x + offset, inout[x]);
		  }
		}
		
		template <typename T, typename R, typename F>
		__global__ void mkt::kernel::map_index(T* in, R* out, size_t rows, size_t columns, size_t row_offset, size_t column_offset, F func)
		{
		  size_t y = blockIdx.y * blockDim.y + threadIdx.y;
		  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
		
		  if (y < rows) {
		    if (x < columns) {
		      out[y * columns + x] = func(y + rowOffset, x + colOffset, in[y * columns + x]);
		    }
		  }
		}
		
		template <typename T, typename R, typename F>
		__global__ void mkt::kernel::map_index_in_place(T* inout, size_t rows, size_t columns, size_t row_offset, size_t column_offset, F func)
		{
		  size_t y = blockIdx.y * blockDim.y + threadIdx.y;
		  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
		
		  if (y < rows) {
		    if (x < columns) {
		      func(y + rowOffset, x + colOffset, inout[y * columns + x]);
		    }
		  }
		}
	'''
	
	def static generateFoldKernelDefinitions() '''
	template<typename T, typename F>
	void mkt::kernel::fold_call(size_t size, T* d_idata, T* d_odata, int threads, int blocks, T identity, F& f, cudaStream_t& stream, int gpu) {
	  cudaSetDevice(gpu);
	  dim3 dimBlock(threads, 1, 1);
	  dim3 dimGrid(blocks, 1, 1);
	  // when there is only one warp per block, we need to allocate two warps
	  // worth of shared memory so that we don't index shared memory out of bounds
	  size_t smemSize =
	      (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
	
	    switch (threads) {
	      case 1024:
	        foldKernel<T, F, 1024> <<<dimGrid, dimBlock, smemSize, stream>>>(
	            d_idata, d_odata, size, identity, f);
	        break;
	      case 512:
	        foldKernel<T, F, 512> <<<dimGrid, dimBlock, smemSize, stream>>>(
	            d_idata, d_odata, size, identity, f);
	        break;
	      case 256:
	        foldKernel<T, F, 256> <<<dimGrid, dimBlock, smemSize, stream>>>(
	            d_idata, d_odata, size, identity, f);
	        break;
	      case 128:
	        foldKernel<T, F, 128> <<<dimGrid, dimBlock, smemSize, stream>>>(
	            d_idata, d_odata, size, identity, f);
	        break;
	      case 64:
	        foldKernel<T, F, 64> <<<dimGrid, dimBlock, smemSize, stream>>>(
	            d_idata, d_odata, size, identity, f);
	        break;
	      case 32:
	        foldKernel<T, F, 32> <<<dimGrid, dimBlock, smemSize, stream>>>(
	            d_idata, d_odata, size, identity, f);
	        break;
	      case 16:
	        foldKernel<T, F, 16> <<<dimGrid, dimBlock, smemSize, stream>>>(
	            d_idata, d_odata, size, identity, f);
	        break;
	      case 8:
	        foldKernel<T, F, 8> <<<dimGrid, dimBlock, smemSize, stream>>>(
	            d_idata, d_odata, size, identity, f);
	        break;
	      case 4:
	        foldKernel<T, F, 4> <<<dimGrid, dimBlock, smemSize, stream>>>(
	            d_idata, d_odata, size, identity, f);
	        break;
	      case 2:
	        foldKernel<T, F, 2> <<<dimGrid, dimBlock, smemSize, stream>>>(
	            d_idata, d_odata, size, identity, f);
	        break;
	      case 1:
	        foldKernel<T, F, 1> <<<dimGrid, dimBlock, smemSize, stream>>>(
	            d_idata, d_odata, size, identity, f);
	        break;
	    }
	
		template<typename T, typename F, size_t blockSize>
		__global__ void mkt::kernel::fold(T *g_idata, T *g_odata, size_t n, T identity, F func) {
		  extern __shared__ T sdata[];
		
		  // perform first level of reduction,
		  // reading from global memory, writing to shared memory
		  size_t tid = threadIdx.x;
		  size_t i = blockIdx.x * blockSize + threadIdx.x;
		  size_t gridSize = blockSize * gridDim.x;
		
		  // we reduce multiple elements per thread.  The number is determined by the
		  // number of active thread blocks (via gridDim). More blocks will result
		  // in a larger gridSize and therefore fewer elements per thread.
		  sdata[tid] = identity;
		
		  while (i < n) {
		    sdata[tid] = func(sdata[tid], g_idata[i]);
		    i += gridSize;
		  }
		  __syncthreads();
		
		  // perform reduction in shared memory
		  if ((blockSize >= 1024) && (tid < 512)) {
		    sdata[tid] = func(sdata[tid], sdata[tid + 512]);
		  }
		  __syncthreads();
		
		  if ((blockSize >= 512) && (tid < 256)) {
		    sdata[tid] = func(sdata[tid], sdata[tid + 256]);
		  }
		  __syncthreads();
		
		  if ((blockSize >= 256) && (tid < 128)) {
		    sdata[tid] = func(sdata[tid], sdata[tid + 128]);
		  }
		  __syncthreads();
		
		  if ((blockSize >= 128) && (tid < 64)) {
		    sdata[tid] = func(sdata[tid], sdata[tid + 64]);
		  }
		  __syncthreads();
		
		  if ((blockSize >= 64) && (tid < 32)) {
		    sdata[tid] = func(sdata[tid], sdata[tid + 32]);
		  }
		  __syncthreads();
		
		  if ((blockSize >= 32) && (tid < 16)) {
		    sdata[tid] = func(sdata[tid], sdata[tid + 16]);
		  }
		  __syncthreads();
		
		  if ((blockSize >= 16) && (tid < 8)) {
		    sdata[tid] = func(sdata[tid], sdata[tid + 8]);
		  }
		  __syncthreads();
		
		  if ((blockSize >= 8) && (tid < 4)) {
		    sdata[tid] = func(sdata[tid], sdata[tid + 4]);
		  }
		  __syncthreads();
		
		  if ((blockSize >= 4) && (tid < 2)) {
		    sdata[tid] = func(sdata[tid], sdata[tid + 2]);
		  }
		  __syncthreads();
		
		  if ((blockSize >= 2) && (tid < 1)) {
		    sdata[tid] = func(sdata[tid], sdata[tid + 1]);
		  }
		  __syncthreads();
		
		  // write result for this block to global mem
		  if (tid == 0) {
		    g_odata[blockIdx.x] = sdata[0];
		  }
		}
	'''
	
	def static generateReductionKernelDefinitions() '''
		«generateReduceCallDefinition("plus", "", "reduce_plus", true)»
		«generateReduceCallDefinition("multiply", "", "reduce_multiply", true)»
		«generateReduceCallDefinition("min", "int", "reduce_min", false)»
		«generateReduceCallDefinition("min", "float", "reduce_min", false)»
		«generateReduceCallDefinition("min", "double", "reduce_min", false)»
		«generateReduceCallDefinition("max", "int", "reduce_max", false)»
		«generateReduceCallDefinition("max", "float", "reduce_max", false)»
		«generateReduceCallDefinition("max", "double", "reduce_max", false)»
		
		    
		template<typename T, size_t blockSize>
		__global__ void mkt::kernel::reduce_plus(T *g_idata, T *g_odata, size_t n) {
		  extern __shared__ T sdata[];
		
		  // perform first level of reduction,
		  // reading from global memory, writing to shared memory
		  size_t tid = threadIdx.x;
		  size_t i = blockIdx.x * blockSize + threadIdx.x;
		  size_t gridSize = blockSize * gridDim.x;
		
		  // we reduce multiple elements per thread.  The number is determined by the
		  // number of active thread blocks (via gridDim). More blocks will result
		  // in a larger gridSize and therefore fewer elements per thread.
		  sdata[tid] = static_cast<T>(0);
		
		  while (i < n) {
		    sdata[tid] += g_idata[i];
		    i += gridSize;
		  }
		  __syncthreads();
		
		  // perform reduction in shared memory
		  if ((blockSize >= 1024) && (tid < 512)) {
		    sdata[tid] += sdata[tid + 512];
		  }
		  __syncthreads();
		
		  if ((blockSize >= 512) && (tid < 256)) {
		    sdata[tid] += sdata[tid + 256];
		  }
		  __syncthreads();
		
		  if ((blockSize >= 256) && (tid < 128)) {
		    sdata[tid] += sdata[tid + 128];
		  }
		  __syncthreads();
		
		  if ((blockSize >= 128) && (tid < 64)) {
		    sdata[tid] += sdata[tid + 64];
		  }
		  __syncthreads();
		
		  if ((blockSize >= 64) && (tid < 32)) {
		    sdata[tid] += sdata[tid + 32];
		  }
		  __syncthreads();
		
		  if ((blockSize >= 32) && (tid < 16)) {
		    sdata[tid] += sdata[tid + 16];
		  }
		  __syncthreads();
		
		  if ((blockSize >= 16) && (tid < 8)) {
		    sdata[tid] += sdata[tid + 8];
		  }
		  __syncthreads();
		
		  if ((blockSize >= 8) && (tid < 4)) {
		    sdata[tid] += sdata[tid + 4];
		  }
		  __syncthreads();
		
		  if ((blockSize >= 4) && (tid < 2)) {
		    sdata[tid] += sdata[tid + 2];
		  }
		  __syncthreads();
		
		  if ((blockSize >= 2) && (tid < 1)) {
		    sdata[tid] += sdata[tid + 1];
		  }
		  __syncthreads();
		
		  // write result for this block to global mem
		  if (tid == 0) {
		    g_odata[blockIdx.x] = sdata[0];
		  }
		}
		
		template<typename T, size_t blockSize>
		__global__ void mkt::kernel::reduce_multiply(T *g_idata, T *g_odata, size_t n) {
		  extern __shared__ T sdata[];
		
		  // perform first level of reduction,
		  // reading from global memory, writing to shared memory
		  size_t tid = threadIdx.x;
		  size_t i = blockIdx.x * blockSize + threadIdx.x;
		  size_t gridSize = blockSize * gridDim.x;
		
		  // we reduce multiple elements per thread.  The number is determined by the
		  // number of active thread blocks (via gridDim). More blocks will result
		  // in a larger gridSize and therefore fewer elements per thread.
		  sdata[tid] = static_cast<T>(1);
		
		  while (i < n) {
		    sdata[tid] *= g_idata[i];
		    i += gridSize;
		  }
		  __syncthreads();
		
		  // perform reduction in shared memory
		  if ((blockSize >= 1024) && (tid < 512)) {
		    sdata[tid] *= sdata[tid + 512];
		  }
		  __syncthreads();
		
		  if ((blockSize >= 512) && (tid < 256)) {
		    sdata[tid] *= sdata[tid + 256];
		  }
		  __syncthreads();
		
		  if ((blockSize >= 256) && (tid < 128)) {
		    sdata[tid] *= sdata[tid + 128];
		  }
		  __syncthreads();
		
		  if ((blockSize >= 128) && (tid < 64)) {
		    sdata[tid] *= sdata[tid + 64];
		  }
		  __syncthreads();
		
		  if ((blockSize >= 64) && (tid < 32)) {
		    sdata[tid] *= sdata[tid + 32];
		  }
		  __syncthreads();
		
		  if ((blockSize >= 32) && (tid < 16)) {
		    sdata[tid] *= sdata[tid + 16];
		  }
		  __syncthreads();
		
		  if ((blockSize >= 16) && (tid < 8)) {
		    sdata[tid] *= sdata[tid + 8];
		  }
		  __syncthreads();
		
		  if ((blockSize >= 8) && (tid < 4)) {
		    sdata[tid] *= sdata[tid + 4];
		  }
		  __syncthreads();
		
		  if ((blockSize >= 4) && (tid < 2)) {
		    sdata[tid] *= sdata[tid + 2];
		  }
		  __syncthreads();
		
		  if ((blockSize >= 2) && (tid < 1)) {
		    sdata[tid] *= sdata[tid + 1];
		  }
		  __syncthreads();
		
		  // write result for this block to global mem
		  if (tid == 0) {
		    g_odata[blockIdx.x] = sdata[0];
		  }
		}
		
		generateReduceWithFunctionAndType("max", "max", "int", "INT_MIN");
		generateReduceWithFunctionAndType("max", "fmaxf", "float", "FLT_MIN");
		generateReduceWithFunctionAndType("max", "fmax", "double", "DBL_MIN");
		
		generateReduceWithFunctionAndType("min", "min", "int", "INT_MAX");
		generateReduceWithFunctionAndType("min", "fminf", "float", "FLT_MAX");
		generateReduceWithFunctionAndType("min", "fmin", "double", "DBL_MAX");
	'''
	
	def static generateReduceWithFunctionAndType(String reduction_name, String function, String type, String identity)'''
		template<size_t blockSize>
		__global__ void mkt::kernel::reduce_«reduction_name»(«type» *g_idata, «type» *g_odata, size_t n) {
		  extern __shared__ «type» sdata[];
		
		  // perform first level of reduction,
		  // reading from global memory, writing to shared memory
		  size_t tid = threadIdx.x;
		  size_t i = blockIdx.x * blockSize + threadIdx.x;
		  size_t gridSize = blockSize * gridDim.x;
		
		  // we reduce multiple elements per thread.  The number is determined by the
		  // number of active thread blocks (via gridDim). More blocks will result
		  // in a larger gridSize and therefore fewer elements per thread.
		  sdata[tid] = «identity»;
		
		  while (i < n) {
		    sdata[tid] = «function»(sdata[tid], g_idata[i]);
		    i += gridSize;
		  }
		  __syncthreads();
		
		  // perform reduction in shared memory
		  if ((blockSize >= 1024) && (tid < 512)) {
		    sdata[tid] = «function»(sdata[tid], sdata[tid + 512]);
		  }
		  __syncthreads();
		
		  if ((blockSize >= 512) && (tid < 256)) {
		    sdata[tid] = «function»(sdata[tid], sdata[tid + 256]);
		  }
		  __syncthreads();
		
		  if ((blockSize >= 256) && (tid < 128)) {
		    sdata[tid] = «function»(sdata[tid], sdata[tid + 128]);
		  }
		  __syncthreads();
		
		  if ((blockSize >= 128) && (tid < 64)) {
		    sdata[tid] = «function»(sdata[tid], sdata[tid + 64]);
		  }
		  __syncthreads();
		
		  if ((blockSize >= 64) && (tid < 32)) {
		    sdata[tid] = «function»(sdata[tid], sdata[tid + 32]);
		  }
		  __syncthreads();
		
		  if ((blockSize >= 32) && (tid < 16)) {
		    sdata[tid] = «function»(sdata[tid], sdata[tid + 16]);
		  }
		  __syncthreads();
		
		  if ((blockSize >= 16) && (tid < 8)) {
		    sdata[tid] = «function»(sdata[tid], sdata[tid + 8]);
		  }
		  __syncthreads();
		
		  if ((blockSize >= 8) && (tid < 4)) {
		    sdata[tid] = «function»(sdata[tid], sdata[tid + 4]);
		  }
		  __syncthreads();
		
		  if ((blockSize >= 4) && (tid < 2)) {
		    sdata[tid] = «function»(sdata[tid], sdata[tid + 2]);
		  }
		  __syncthreads();
		
		  if ((blockSize >= 2) && (tid < 1)) {
		    sdata[tid] = «function»(sdata[tid], sdata[tid + 1]);
		  }
		  __syncthreads();
		
		  // write result for this block to global mem
		  if (tid == 0) {
		    g_odata[blockIdx.x] = sdata[0];
		  }
		}
	'''
	
	def static generateReduceCallDefinition(String name, String type, String function, boolean template)'''
		«IF template»template<typename T>«ENDIF»
		void reduce_«name»_call(size_t size, «IF template»T«ELSE»«type»«ENDIF»* d_idata, «IF template»T«ELSE»«type»«ENDIF»* d_odata, int threads, int blocks, cudaStream_t& stream, int gpu) {
		  cudaSetDevice(gpu);
		  dim3 dimBlock(threads, 1, 1);
		  dim3 dimGrid(blocks, 1, 1);
		  // when there is only one warp per block, we need to allocate two warps
		  // worth of shared memory so that we don't index shared memory out of bounds
		  size_t smemSize = (threads <= 32) ? 2 * threads * sizeof(«IF template»T«ELSE»«type»«ENDIF») : threads * sizeof(«IF template»T«ELSE»«type»«ENDIF»);
		
		    switch (threads) {
		      case 1024:
		        «function»<«IF template»T, «ENDIF»1024> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
		        break;
		      case 512:
		        «function»<«IF template»T, «ENDIF»512> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
		        break;
		      case 256:
		        «function»<«IF template»T, «ENDIF»256> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
		        break;
		      case 128:
		        «function»<«IF template»T, «ENDIF»128> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
		        break;
		      case 64:
		        «function»<«IF template»T, «ENDIF»64> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
		        break;
		      case 32:
		        «function»<«IF template»T, «ENDIF»32> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
		        break;
		      case 16:
		        «function»<«IF template»T, «ENDIF»16> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
		        break;
		      case 8:
		        «function»<«IF template»T, «ENDIF»8> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
		        break;
		      case 4:
		        «function»<«IF template»T, «ENDIF»4> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
		        break;
		      case 2:
		        «function»<«IF template»T, «ENDIF»2> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
		        break;
		      case 1:
		        «function»<«IF template»T, «ENDIF»1> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
		        break;
		    }
	'''
}