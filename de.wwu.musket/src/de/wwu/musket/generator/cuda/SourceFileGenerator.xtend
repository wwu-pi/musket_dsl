package de.wwu.musket.generator.cuda

import de.wwu.musket.musket.DistributionMode
import de.wwu.musket.musket.MusketFunctionCall
import de.wwu.musket.musket.MusketFunctionName
import de.wwu.musket.musket.StructArrayType
import org.apache.log4j.LogManager
import org.apache.log4j.Logger
import org.eclipse.emf.ecore.resource.Resource
import org.eclipse.xtext.generator.IFileSystemAccess2
import org.eclipse.xtext.generator.IGeneratorContext

import static de.wwu.musket.generator.cuda.FoldSkeletonGenerator.*
import static de.wwu.musket.generator.cuda.LogicGenerator.*
import static de.wwu.musket.generator.cuda.MapSkeletonGenerator.*
import static de.wwu.musket.generator.cuda.ReductionSkeletonGenerator.*
import static de.wwu.musket.generator.cuda.RngGenerator.*
import static extension de.wwu.musket.generator.cuda.FunctorGenerator.*
import static de.wwu.musket.generator.cuda.ShiftSkeletonGenerator.generateMPIVectorType
import static de.wwu.musket.generator.cuda.ShiftSkeletonGenerator.generateShiftSkeletonVariables
import static de.wwu.musket.generator.cuda.MusketFunctionCalls.generateAllTimerGlobalVars
import static extension de.wwu.musket.generator.cuda.MPIRoutines.*

import static de.wwu.musket.generator.cuda.ShiftSkeletonGenerator.*

import static extension de.wwu.musket.generator.cuda.DataGenerator.*
import static extension de.wwu.musket.generator.cuda.StructGenerator.*
import static extension de.wwu.musket.generator.cuda.util.DataHelper.*
import static extension de.wwu.musket.util.CollectionHelper.*
import static extension de.wwu.musket.generator.extensions.ModelElementAccess.*
import static extension de.wwu.musket.util.MusketHelper.*
import de.wwu.musket.musket.ShiftPartitionsHorizontallySkeleton
import de.wwu.musket.musket.ShiftPartitionsVerticallySkeleton
import de.wwu.musket.musket.MatrixType
import java.util.List
import de.wwu.musket.musket.Skeleton
import de.wwu.musket.musket.Function
import de.wwu.musket.musket.SkeletonParameterInput
import de.wwu.musket.musket.MapFoldSkeleton
import de.wwu.musket.musket.MapReductionSkeleton
import de.wwu.musket.musket.ArrayType

/** 
 * Generates the source file of the project.
 * <p>
 * The generator is split into smaller functions that generates sections of the source file, such as generateGlobalConstants.
 * The entry point is the function generateSourceFile(Resource resource, IFileSystemAccess2 fsa, IGeneratorContext context), which is called by the CPU generator.
 */
class SourceFileGenerator {
	private static final Logger logger = LogManager.getLogger(HeaderFileGenerator)

	/**
	 * Creates the source file in the source folder of the project.
	 */
	def static void generateSourceFile(Resource resource, IFileSystemAccess2 fsa, IGeneratorContext context,
		int processId) {
		logger.info("Generate source file.")
		fsa.generateFile(Config.base_path + Config.source_path + resource.ProjectName + '_' + processId +
			Config.source_extension, sourceFileContent(resource, processId))
		logger.info("Generation of source file done.")
	}

	/**
	 * Generates the content of the source file. It calls several smaller functions that generate certain parts.
	 * @param resource the resource object
	 * @return content of the source file
	 */
	def static sourceFileContent(Resource resource, int processId) '''
			«generateIncludes»
			#include "../«Config.include_path + resource.ProjectName + "_" + processId + Config.header_extension»"
			
			«generateGlobalConstants(processId)»
			«generateGlobalVariables(resource, processId)»
			
			«FOR d : resource.Data»
				«d.generateObjectDefinitionGlobal(processId)»
			«ENDFOR»
			
			«FOR s : resource.Structs»
				«s.generateStructDefaultConstructor»
			«ENDFOR»
			
			«val fold_functions = (resource.FoldSkeletons.map[it.param.toFunction] + resource.MapFoldSkeletons.map[it.param.toFunction]).toSet»
		
			«FOR f : fold_functions»
				«f.generateFunction(processId)»
			«ENDFOR»
			
			«generateFunctors(resource, processId)»
			
			«IF Config.processes > 1»
				«generateMPIFoldFunction(resource.SkeletonExpressions, processId)»
			«ENDIF»
			
			«generateReductionDeclarations(resource, processId)»
			
			«generateFoldFunctionDefinitions(resource, processId)»
			«generateMapFoldFunctionDefinitions(resource, processId)»
			
			«generateReductionSkeletonFunctionDefinitions(resource)»
			«generateMapReductionSkeletonFunctionDefinitions(resource)»
			
			«IF resource.ShiftPartitionsHorizontallySkeletons.size() > 0  && Config.processes > 1»
				«generateShiftHorizontallyFunctionDefinitions(resource)»
			«ENDIF»
			
			«IF resource.ShiftPartitionsVerticallySkeletons.size() > 0  && Config.processes > 1»
				«generateShiftVerticallyFunctionDefinitions(resource)»
			«ENDIF»
			
			«generateMainFunction(resource, processId)»
	'''

	/** 
	 * Generate required imports.
	 * 
	 * TODO: find a more sophisticated way to generate imports. At the moment here are simple all imports, which could be required, but they might not actyally be.
	 */
	def static generateIncludes() '''
		«IF Config.processes > 1»
			#include <mpi.h>
		«ENDIF»
		#include <cuda.h>
		#include <omp.h>
		#include <stdlib.h>
		#include <math.h>
		#include <array>
		#include <vector>
		#include <sstream>
		#include <chrono>
		#include <curand_kernel.h>
		#include <limits>
		#include <memory>
		#include <cstddef>
		#include <type_traits>
		
		
		#include "../include/musket«Config.header_extension»"
	'''

	/**
	 * Generates global constants, which are required but not in the model.
	 */
	def static generateGlobalConstants(int processId) '''
		«IF Config.processes > 1»
			const size_t «Config.var_np» = «Config.processes»;
			const size_t «Config.var_pid» = «processId»;
		«ENDIF»
	'''

	/**
	 * Generates global variables, which are required but not in the model.
	 */
	def static generateGlobalVariables(Resource resource, int processId) '''
		«IF Config.processes > 1»
			int «Config.var_mpi_rank» = -1;
			int «Config.var_mpi_procs» = 0;
		«ENDIF»
		
		«generateAllTimerGlobalVars(resource.MusketFunctionCalls, processId)»
	'''

	def static generateFunctors(Resource resource, int processId) {
		var result = ""
		var List<Pair<String, String>> generated = newArrayList
		// all skeleton expressions but those without function such as gather and scatter
		for (skeletonExpression : resource.SkeletonExpressions.reject[it.skeleton.param.toFunction === null]) {
			val skel = skeletonExpression.skeleton

			if (!(Config.processes < 2 &&
				(skel instanceof ShiftPartitionsHorizontallySkeleton ||
					skel instanceof ShiftPartitionsVerticallySkeleton))) {

				val func = skeletonExpression.skeleton.param.toFunction
				val skelContainerName = skel.skeletonName.toString + "_" +
					skeletonExpression.obj.collectionContainerName.toString

				if (!generated.contains(skelContainerName -> func.name)) {
					generated.add(skelContainerName -> func.name)
					var colname = skeletonExpression.obj.collectionContainerName.toString;
					if ((skeletonExpression.obj.type as ArrayType).getView().literal == 'no'){
						colname = 'GPUArray'
					}
					result +=
						FunctorGenerator.generateFunctor(skel, func, skel.skeletonName.toString,
							colname,
							skeletonExpression.getNumberOfFreeParameters(func), processId)
				}

				if (skel instanceof MapFoldSkeleton) {
					val mapFunction = (skel as MapFoldSkeleton).mapFunction.toFunction
					if (!generated.contains(skelContainerName -> mapFunction.name)) {
						generated.add(skelContainerName -> mapFunction.name)
						var colname = skeletonExpression.obj.collectionContainerName.toString;
						if ((skeletonExpression.obj.type as ArrayType).getView().literal == 'no'){
							colname = 'GPUArray'
						}
						result +=
							FunctorGenerator.generateFunctor(skel, mapFunction, skel.skeletonName.toString,
								colname,
								skeletonExpression.getNumberOfFreeParameters(mapFunction), processId)
					}
				}
			}
		}
		
		//special treatment for mapReduction sekelton
		for (skeletonExpression : resource.SkeletonExpressions.filter[it.skeleton instanceof MapReductionSkeleton]) {
			val skel = skeletonExpression.skeleton
			val skelContainerName = skel.skeletonName.toString + "_" + skeletonExpression.obj.collectionContainerName.toString
			val mapFunction = (skel as MapReductionSkeleton).mapFunction.toFunction
			if (!generated.contains(skelContainerName -> mapFunction.name)) {
				generated.add(skelContainerName -> mapFunction.name)
				var colname = skeletonExpression.obj.collectionContainerName.toString;
				if ((skeletonExpression.obj.type as ArrayType).getView().literal == 'no'){
					colname = 'GPUArray'
				}				
				result +=
					FunctorGenerator.generateFunctor(skel, mapFunction, skel.skeletonName.toString,
						colname,
						skeletonExpression.getNumberOfFreeParameters(mapFunction), processId)
			}
		}
		return result
	}

	def static generateFunctorInstantiations(Resource resource, int processId) {
		var result = ""
		var List<Pair<String, String>> generated = newArrayList
		// all skeleton expressions but those without function such as gather and scatter
		for (skeletonExpression : resource.SkeletonExpressions.reject[it.skeleton.param.toFunction === null]) {
			val skel = skeletonExpression.skeleton
			if (!(Config.processes < 2 &&
				(skel instanceof ShiftPartitionsHorizontallySkeleton ||
					skel instanceof ShiftPartitionsVerticallySkeleton))) {
				val func = skeletonExpression.skeleton.param.toFunction
				val skelContainerName = skel.skeletonName.toString + "_" +
					skeletonExpression.obj.collectionContainerName.toString
				if (!generated.contains(skelContainerName -> func.name)) {
					generated.add(skelContainerName -> func.name)
					result += generateFunctorInstantiation(skeletonExpression, skel.param, processId)
				}

				if (skel instanceof MapFoldSkeleton) {
					val mfunc = (skel as MapFoldSkeleton).mapFunction.toFunction
					if (!generated.contains(skelContainerName -> mfunc.name)) {
						generated.add(skelContainerName -> mfunc.name)
						result += generateFunctorInstantiation(skeletonExpression, skel.mapFunction, processId)
					}
				}
			}
		}
		//special treatment for mapReduction sekelton
		for (skeletonExpression : resource.SkeletonExpressions.filter[it.skeleton instanceof MapReductionSkeleton]) {
			val skel = skeletonExpression.skeleton as MapReductionSkeleton
			val skelContainerName = skel.skeletonName.toString + "_" + skeletonExpression.obj.collectionContainerName.toString
			val mfunc = (skel as MapReductionSkeleton).mapFunction.toFunction
			if (!generated.contains(skelContainerName -> mfunc.name)) {
				generated.add(skelContainerName -> mfunc.name)
				result += generateFunctorInstantiation(skeletonExpression, skel.mapFunction, processId)
			}
		}
		return result
	}

	/** 
	 * Generate content of the main function in the cpp source file.
	 * 
	 * @param resource the resource object
	 */
	def static generateMainFunction(Resource resource, int processId) '''
		int main(int argc, char** argv) {
			«generateInitialization»
			
			«IF Config.processes > 1 && processId == 0»
				printf("Run «resource.ProjectName.toFirstUpper»\n\n");
			«ENDIF»
			
			mkt::sync_streams();
			«IF processId == 0»
				std::chrono::high_resolution_clock::time_point complete_timer_start = std::chrono::high_resolution_clock::now();
			«ENDIF»
			
			«FOR d : resource.Data»
				«d.generateObjectDefinitionMain(processId)»
			«ENDFOR»
			
«««			functor instantiation
			«generateFunctorInstantiations(resource, processId)»
			
			«val rcs = resource.MusketFunctionCalls.filter[it.value == MusketFunctionName.RAND].toList»
			
			«generateInitializeDataStructures(resource, processId)»
					
			«IF Config.processes > 1»
				«FOR s : resource.Structs»
					«s.generateCreateDatatypeStruct»
				«ENDFOR»
				
				«val dist_matrices = resource.Matrices.filter[it.type.distributionMode == DistributionMode.DIST]»
				
				«FOR m : dist_matrices»
					«generateMPIVectorType(m.type as MatrixType, processId)»
				«ENDFOR»
			
				«generateMPIFoldOperators(resource)»
				
			«««				«IF resource.SkeletonExpressions.exists[it.skeleton instanceof ShiftPartitionsHorizontallySkeleton || it.skeleton instanceof ShiftPartitionsVerticallySkeleton]»
«««					«generateShiftSkeletonVariables(processId)»
«««					ENDIF»

			«ENDIF»
			
			«generateLogic(resource.Model.main, processId)»
			
			mkt::sync_streams();
			«IF processId == 0»
				std::chrono::high_resolution_clock::time_point complete_timer_end = std::chrono::high_resolution_clock::now();
				double complete_seconds = std::chrono::duration<double>(complete_timer_end - complete_timer_start).count();
				printf("Complete execution time: %.5fs\n", complete_seconds);
			«ENDIF»
			
			«IF processId == 0»		
				«IF resource.Model.main.content.exists[it instanceof MusketFunctionCall && (it as MusketFunctionCall).value == MusketFunctionName.ROI_START]»
					printf("Execution time: %.5fs\n", seconds);
				«ENDIF»
				printf("Threads: %i\n", «IF Config.cores > 1»omp_get_max_threads()«ELSE»«Config.cores»«ENDIF»);
				printf("Processes: %i\n", «IF Config.processes > 1»«Config.var_mpi_procs»«ELSE»«Config.processes»«ENDIF»);
			«ENDIF»
			
			«generateFinalization(resource)»
			}
	'''

	/**
	 * Generates boilerplate code, which is required for initialization.
	 */
	def static generateInitialization() '''
		«IF Config.processes > 1»
			MPI_Init(&argc, &argv);
			
			MPI_Comm_size(MPI_COMM_WORLD, &«Config.var_mpi_procs»);
			MPI_Comm_rank(MPI_COMM_WORLD, &«Config.var_mpi_rank»);
			
			if(«Config.var_mpi_procs» != «Config.var_np» || «Config.var_mpi_rank» != «Config.var_pid»){
				MPI_Finalize();
				return EXIT_FAILURE;
			}				
		«ENDIF»
		mkt::init();
	'''

	/**
	 * Generates boilerplate code, which is required for finalization.
	 */
	def static generateFinalization(Resource resource) '''
		«IF Config.processes > 1»
			MPI_Finalize();
		«ENDIF»
		return EXIT_SUCCESS;
	'''

	/** 
	 * Generates the initialization of data structures. The method generates an if-clause that checks for the process id.
	 * Within each case all data structures of the respective process are initialized. 
	 * Struct arrays and matrices are rejected, since std::arrays, which are used, are always initialized with calls to the default constructor.
	 * 
	 * @param resource the resource object
	 * @return generated code
	 */
	def static String generateInitializeDataStructures(Resource resource, int processId) {
		var result = ""

		// init distributed arrays with values
		if (Config.processes > 1) {
//		distributed
			for (a : resource.Arrays.reject[it.type instanceof StructArrayType].filter [
				it.type.distributionMode == DistributionMode.DIST
			]) {
				val values = a.ValuesAsString
				if (values.size > 1) {
					val sizeLocal = a.type.sizeLocal(processId)
					result +=
						a.generateArrayInitializationForProcess(
							values.drop((sizeLocal * processId) as int).take(sizeLocal as int))
				}
			}
//		copy distributed
			for (a : resource.Arrays.reject[it.type instanceof StructArrayType].filter [
				it.type.distributionMode == DistributionMode.COPY
			]) {
				val values = a.ValuesAsString
				if (values.size > 1) {
					result += a.generateArrayInitializationForProcess(values)
				}
			}
		} else {
			for (a : resource.Arrays.reject [
				it.type instanceof StructArrayType || it.type.distributionMode == DistributionMode.LOC
			]) {
				val values = a.ValuesAsString
				if (values.size > 1) {
					result += a.generateArrayInitializationForProcess(values)
				}
			}
		}

		return result
	}

}
