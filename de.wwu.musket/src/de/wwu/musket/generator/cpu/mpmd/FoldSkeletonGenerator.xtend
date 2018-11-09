package de.wwu.musket.generator.cpu.mpmd

import de.wwu.musket.musket.CollectionObject
import de.wwu.musket.musket.Expression
import de.wwu.musket.musket.FoldSkeleton
import de.wwu.musket.musket.FoldSkeletonVariants
import de.wwu.musket.musket.MapFoldSkeleton
import de.wwu.musket.musket.SkeletonExpression
import java.util.HashMap
import java.util.List
import java.util.Map
import org.eclipse.emf.ecore.resource.Resource

import static extension de.wwu.musket.generator.cpu.mpmd.ExpressionGenerator.*
import static extension de.wwu.musket.generator.extensions.ModelElementAccess.*
import static extension de.wwu.musket.generator.extensions.StringExtension.*
import static extension de.wwu.musket.generator.cpu.mpmd.util.DataHelper.*
import static extension de.wwu.musket.util.MusketHelper.*
import static extension de.wwu.musket.util.TypeHelper.*
import static extension de.wwu.musket.util.CollectionHelper.*
import de.wwu.musket.musket.DistributionMode
import de.wwu.musket.musket.Assignment
import de.wwu.musket.musket.MusketAssignment
import de.wwu.musket.generator.cpu.mpmd.lib.Musket

/**
 * Generate everything required for the fold skeleton, except for the actual fold skeleton call.
 * That is done in the skeleton generator.
 * <p>
 * The fold skeleton requires two things:
 * First, the OpenMP reduction declaration.
 * http://scc.ustc.edu.cn/zlsc/tc4600/intel/2016.0.109/compiler_c/common/core/GUID-7312910C-D175-4544-99C5-29C12D980744.htm
 * 
 * Second, the MPI fold function and the operator. 
 * http://mpi-forum.org/docs/mpi-2.2/mpi22-report/node107.htm
 * 
 * TODO: some functions take the resource object, some a list of skeleton calls --> unify
 * 
 * The methods are called by the source file generator.
 */
class FoldSkeletonGenerator {

// OpenMP part
	/**
	 * Generates OpenMP reduction declaration.
	 * It iterates through all skeleton statements and generates the reduction declaration
	 * only once for each function that is used in a fold skeleton.
	 * 
	 * @param sleketons list of all skeleton expressions.
	 * @return generated code
	 */
	def static generateReductionDeclarations(Resource resource, int processId) {
		var result = ""
		var List<SkeletonExpression> processed = newArrayList
		for (SkeletonExpression se : resource.SkeletonExpressions) {
			if (se.skeleton instanceof FoldSkeletonVariants) {
				val alreadyProcessed = processed.exists [
					it.skeleton.param.functionName == se.skeleton.param.functionName
				]

				if (!alreadyProcessed) {
					result += generateReductionDeclaration(se.skeleton as FoldSkeletonVariants, se.obj, processId)
					processed.add(se)
				}
			}
		}

		return result
	}

	/**
	 * Generate single reduction declaration.
	 * 
	 * @param s the fold skeleton
	 * @param a the collection object on which the fold skeleton is used
	 * @return generated code
	 */
	def static generateReductionDeclaration(FoldSkeletonVariants s, CollectionObject a, int processId) '''
		#pragma omp declare reduction(«s.param.functionName.toFirstLower»_reduction : «s.identity.calculateType.cppType.replace("0", s.identity.calculateType.collectionType?.size.toString)» : omp_out = «s.param.functionName»_function(«FOR p : s.param.functionArguments SEPARATOR ", " AFTER ", "»«p.generateExpression(null, processId)»«ENDFOR»omp_in, omp_out)) initializer(omp_priv = omp_orig)
	'''

// MPI part
	/**
	 * Generates MPI fold functions. It iterates through all skeleton statements and generates the MPI fold function
	 * only once for each function that is used in a fold skeleton.
	 * 
	 * @param sleketons list of all skeleton expressions.
	 * @return generated code
	 */
	def static generateMPIFoldFunction(Iterable<SkeletonExpression> skeletons, int processId) {
		var result = ""
		var List<SkeletonExpression> processed = newArrayList
		for (SkeletonExpression se : skeletons) {
			if (se.skeleton instanceof FoldSkeleton || se.skeleton instanceof MapFoldSkeleton) {
				val alreadyProcessed = processed.exists [
					it.skeleton.param.functionName == se.skeleton.param.functionName
				]

				if (!alreadyProcessed) {
					result += generateMPIFoldFunction(se.skeleton as FoldSkeletonVariants, se.obj, processId)
					processed.add(se)
				}
			}
		}

		return result
	}

	/**
	 * Generates a single MPI fold function.
	 * 
	 * @param foldSkeleton the fold skeleton
	 * @param a the collection object on which the fold skeleton is used.
	 * @return generated code
	 */
	def static generateMPIFoldFunction(FoldSkeletonVariants foldSkeleton, CollectionObject a, int processId) '''
		«««		«val type = if(foldSkeleton.identity.calculateType.collection) foldSkeleton.identity.calculateType.calculateCollectionType.cppType else foldSkeleton.identity.calculateType.cppType»
		«val type = foldSkeleton.identity.calculateType.cppType.replace("0", foldSkeleton.identity.calculateType.collectionType?.size.toString)»
		void «foldSkeleton.param.functionName»(void* in, void* inout, int *len, MPI_Datatype *dptr){
			«type»* inv = static_cast<«type»*>(in);
			«type»* inoutv = static_cast<«type»*>(inout);
			*inoutv = «foldSkeleton.param.functionName»_function(«FOR arg : foldSkeleton.param.functionArguments SEPARATOR ", " AFTER ", "»«arg.generateExpression(null, processId)»«ENDFOR»*inv, *inoutv);
		} 
	'''

	def static generateMPIFoldOperatorDeclarations(Resource resource) {
		var result = ""
		var List<SkeletonExpression> processed = newArrayList
		for (SkeletonExpression se : resource.SkeletonExpressions) {
			if (se.skeleton instanceof FoldSkeleton || se.skeleton instanceof MapFoldSkeleton) {
				val alreadyProcessed = processed.exists [
					it.skeleton.param.functionName == se.skeleton.param.functionName
				]

				if (!alreadyProcessed) {
					result += generateMPIFoldOperatorDeclaration(se.skeleton as FoldSkeletonVariants)
					processed.add(se)
				}
			}
		}

		return result
	}

	def static generateMPIFoldOperatorDeclaration(FoldSkeletonVariants s) '''
		«val name = s.param.functionName»
		MPI_Op «name»_reduction«Config.mpi_op_suffix»;
	'''

	/**
	 * Generates MPI fold Operators. It iterates through all skeleton statements and generates the MPI fold operator
	 * only once for each function that is used in a fold skeleton.
	 * 
	 * @param resource the resource object
	 * @return generated code
	 */
	def static generateMPIFoldOperators(Resource resource) {
		var result = ""
		var List<SkeletonExpression> processed = newArrayList
		for (SkeletonExpression se : resource.SkeletonExpressions) {
			if (se.skeleton instanceof FoldSkeleton || se.skeleton instanceof MapFoldSkeleton) {
				val alreadyProcessed = processed.exists [
					it.skeleton.param.functionName == se.skeleton.param.functionName
				]

				if (!alreadyProcessed) {
					result += generateMPIFoldOperator(se.skeleton as FoldSkeletonVariants)
					processed.add(se)
				}
			}
		}

		return result
	}

	/**
	 * Generates a single MPI fold operator.
	 * 
	 * @param foldSkeleton the fold skeleton
	 * @return generated code
	 */
	def static generateMPIFoldOperator(FoldSkeletonVariants s) '''
		«val name = s.param.functionName»
		MPI_Op_create( «name», 0, &«name»_reduction«Config.mpi_op_suffix» );
	'''

	/**
	 * Generates required temporary variables used to store intermediate fold results.
	 * 
	 * @param resource the resource object
	 * @return generated code
	 */
	def static generateTmpFoldResults(Resource resource) {
		var result = ""
		var List<SkeletonExpression> processed = newArrayList
		for (SkeletonExpression se : resource.SkeletonExpressions) {
			if (se.skeleton instanceof FoldSkeleton || se.skeleton instanceof MapFoldSkeleton) {
				val alreadyProcessed = processed.exists [
					(it.skeleton as FoldSkeletonVariants).identity.calculateType.cppType ==
						(se.skeleton as FoldSkeletonVariants).identity.calculateType.cppType
				]

				if (!alreadyProcessed) {
					val type = (se.skeleton as FoldSkeletonVariants).identity.calculateType.cppType.replace("0",
						(se.skeleton as FoldSkeletonVariants).identity.calculateType.collectionType?.size.toString)

					result += '''«type» «Config.var_fold_result»_«type.toCXXIdentifier»;'''
				}
				processed.add(se)
			}
		}

		return result
	}

	def static generateFoldFunctionDefinitions(Resource resource, int processId) {
		var result = ""
		var List<SkeletonExpression> arrayProcessed = newArrayList
		var List<SkeletonExpression> matrixProcessed = newArrayList
		for (SkeletonExpression se : resource.SkeletonExpressions) {
			if (se.skeleton instanceof FoldSkeleton) {
				if (se.obj.calculateType.isArray) {
					val alreadyProcessed = arrayProcessed.exists [
						(it.skeleton as FoldSkeletonVariants).identity.calculateType.cppType ==
							(se.skeleton as FoldSkeletonVariants).identity.calculateType.cppType
					]
					if (!alreadyProcessed) {
						result += generateArrayFoldFunctionDefinition(se)
						result += generateArrayFoldCopyFunctionDefinition(se)
					}
					arrayProcessed.add(se)
				} else if (se.obj.calculateType.isMatrix) {
					val alreadyProcessed = matrixProcessed.exists [
						(it.skeleton as FoldSkeletonVariants).identity.calculateType.cppType ==
							(se.skeleton as FoldSkeletonVariants).identity.calculateType.cppType
					]
					if (!alreadyProcessed) {
						result += generateMatrixFoldFunctionDefinition(se)
						result += generateMatrixFoldCopyFunctionDefinition(se)
					}
					matrixProcessed.add(se)
				}
			}
		}

		return result
	}

	def static generateArrayFoldFunctionDefinition(SkeletonExpression se) '''
		«val cpptype = se.obj.calculateCollectionType.cppType»
		«val FoldSkeleton fs = se.skeleton as FoldSkeleton»
		«val foldName = fs.param.functionName + "_reduction"»
		«var local_result = ""»
		template<>
		void mkt::fold<«cpptype», «se.getFunctorName(se.skeleton.param)»>(const mkt::DArray<«cpptype»>& in, «cpptype»& out, const «cpptype» identity, const «se.getFunctorName(se.skeleton.param)»& f){
		  «IF Config.processes > 1»  		  	
		  	«cpptype» «local_result = "local_result"» = identity;
		  «ELSE»
		  	«local_result = "out"» = identity;
		  «ENDIF»		  
		  
		  const int size_local = in.get_size_local();
		  
		  #pragma omp parallel for simd reduction(«foldName»:«local_result»)
		  for(int «Config.var_loop_counter» = 0; «Config.var_loop_counter» < size_local; ++«Config.var_loop_counter»){
		    f(«local_result», in.get_local(«Config.var_loop_counter»));
		  }
		  
		  «IF Config.processes > 1»
		  	MPI_Allreduce(&local_result, &out, 1, «fs.identity.calculateType.MPIType», «foldName»«Config.mpi_op_suffix», MPI_COMM_WORLD); 
		  «ENDIF»
		}
	'''

	def static generateArrayFoldCopyFunctionDefinition(SkeletonExpression se) '''
		«val cpptype = se.obj.calculateCollectionType.cppType»
		«val FoldSkeleton fs = se.skeleton as FoldSkeleton»
		«val foldName = fs.param.functionName + "_reduction"»
		template<>
		void mkt::fold_copy<«cpptype», «se.getFunctorName(se.skeleton.param)»>(const mkt::DArray<«cpptype»>& in, «cpptype»& out, const «cpptype» identity, const «se.getFunctorName(se.skeleton.param)»& f){
		  out = identity;
		  
		  const int size = in.get_size();
		  
		  #pragma omp parallel for simd reduction(«foldName»:out)
		  for(int «Config.var_loop_counter» = 0; «Config.var_loop_counter» < size; ++«Config.var_loop_counter»){
		    f(out, in.get_local(«Config.var_loop_counter»));
		  }
		}
	'''

	def static generateMatrixFoldFunctionDefinition(SkeletonExpression se) '''
		«val cpptype = se.obj.calculateCollectionType.cppType»
		«val FoldSkeleton fs = se.skeleton as FoldSkeleton»
		«val foldName = fs.param.functionName + "_reduction"»
		«var local_result = ""»
		template<>
		void mkt::fold<«cpptype», «se.getFunctorName(se.skeleton.param)»>(const mkt::DMatrix<«cpptype»>& in, «cpptype»& out, const «cpptype» identity, const «se.getFunctorName(se.skeleton.param)»& f){
		  «IF Config.processes > 1»  		  	
		  	«cpptype» «local_result = "local_result"» = identity;
		  «ELSE»
		  	«local_result = "out"» = identity;
		  «ENDIF»		  
		  
		  const int size_local = in.get_size_local();
		  
		  #pragma omp parallel for simd reduction(«foldName»:«local_result»)
		  for(int «Config.var_loop_counter» = 0; «Config.var_loop_counter» < size_local; ++«Config.var_loop_counter»){
		    f(«local_result», in.get_local(«Config.var_loop_counter»));
		  }
		  
		  «IF Config.processes > 1»
		  	MPI_Allreduce(&local_result, &out, 1, «fs.identity.calculateType.MPIType», «foldName»«Config.mpi_op_suffix», MPI_COMM_WORLD); 
		  «ENDIF»
		}
	'''

	def static generateMatrixFoldCopyFunctionDefinition(SkeletonExpression se) '''
		«val cpptype = se.obj.calculateCollectionType.cppType»
		«val FoldSkeleton fs = se.skeleton as FoldSkeleton»
		«val foldName = fs.param.functionName + "_reduction"»
		template<>
		void mkt::fold_copy<«cpptype», «se.getFunctorName(se.skeleton.param)»>(const mkt::DMatrix<«cpptype»>& in, «cpptype»& out, const «cpptype» identity, const «se.getFunctorName(se.skeleton.param)»& f){
		  out = identity;
		  
		  const int size = in.get_size();
		  
		  #pragma omp parallel for simd reduction(«foldName»:out)
		  for(int «Config.var_loop_counter» = 0; «Config.var_loop_counter» < size; ++«Config.var_loop_counter»){
		    f(out, in.get_local(«Config.var_loop_counter»));
		  }
		}
	'''

	def static generateMapFoldFunctionDefinitions(Resource resource, int processId) {
		var result = ""
		var List<SkeletonExpression> arrayProcessed = newArrayList
		var List<SkeletonExpression> matrixProcessed = newArrayList
		for (SkeletonExpression se : resource.SkeletonExpressions) {
			if (se.skeleton instanceof MapFoldSkeleton) {
				if (se.obj.calculateType.isArray) {
					if (se.eContainer instanceof MusketAssignment &&
						(se.eContainer as MusketAssignment).^var.calculateType.isArray) {
						val alreadyProcessed = arrayProcessed.exists [
							(it.skeleton as MapFoldSkeleton).identity.calculateType.cppType ==
								(se.skeleton as MapFoldSkeleton).identity.calculateType.cppType &&
								it.obj.calculateCollectionType.cppType == se.obj.calculateCollectionType.cppType &&
								it.skeleton.param.functionName == se.skeleton.param.functionName &&
								(it.skeleton as MapFoldSkeleton).mapFunction.functionName ==
									(se.skeleton as MapFoldSkeleton).mapFunction.functionName &&
								(it.eContainer as MusketAssignment).^var.calculateCollectionType.cppType ==
									(se.eContainer as MusketAssignment).^var.calculateCollectionType.cppType
						]
						if (!alreadyProcessed) {
							result += generateArrayMapFoldFunctionDefinition(se)
							result += generateArrayMapFoldCopyFunctionDefinition(se)
						}
						arrayProcessed.add(se)
					} else {
						val alreadyProcessed = arrayProcessed.exists [
							(it.skeleton as MapFoldSkeleton).identity.calculateType.cppType ==
								(se.skeleton as MapFoldSkeleton).identity.calculateType.cppType &&
								it.obj.calculateCollectionType.cppType == se.obj.calculateCollectionType.cppType &&
								it.skeleton.param.functionName == se.skeleton.param.functionName &&
								(it.skeleton as MapFoldSkeleton).mapFunction.functionName ==
									(se.skeleton as MapFoldSkeleton).mapFunction.functionName
						]
						if (!alreadyProcessed) {
							result += generateArrayMapFoldFunctionDefinition(se)
							result += generateArrayMapFoldCopyFunctionDefinition(se)
						}
						arrayProcessed.add(se)
					}

				} else if (se.obj.calculateType.isMatrix) {
					val alreadyProcessed = matrixProcessed.exists [
						(it.skeleton as FoldSkeletonVariants).identity.calculateType.cppType ==
							(se.skeleton as FoldSkeletonVariants).identity.calculateType.cppType
					]
					if (!alreadyProcessed) {
						result += generateMatrixMapFoldFunctionDefinition(se)
						result += generateMatrixMapFoldCopyFunctionDefinition(se)
					}
					matrixProcessed.add(se)
				}
			}
		}

		return result
	}

	def static generateArrayMapFoldFunctionDefinition(SkeletonExpression se) '''		
		«val fs = se.skeleton as MapFoldSkeleton»
		«val mapcpptype = se.obj.calculateCollectionType.cppType»
		«val foldcpptype = fs.identity.calculateType.cppType»
		«val foldName = fs.param.functionName + "_reduction"»
		«var local_result = ""»
		«val resultIsArray = se.eContainer instanceof MusketAssignment && (se.eContainer as MusketAssignment).^var.calculateType.isArray»
		«val resultcpptype = try{ (se.eContainer as MusketAssignment).^var.calculateCollectionType.cppType } catch(ClassCastException e) {""}»
		template<>
		void mkt::map_fold<«mapcpptype», «IF resultIsArray»«resultcpptype», «ENDIF»«foldcpptype», «se.getFunctorName(fs.mapFunction)», «se.getFunctorName(fs.param)»>(const mkt::DArray<«mapcpptype»>& in, «IF resultIsArray»mkt::DArray<«resultcpptype»>&«ELSE»«foldcpptype»&«ENDIF» out, const «se.getFunctorName(fs.mapFunction)»& mf, const «foldcpptype»& identity, const «se.getFunctorName(fs.param)»& ff){
		«IF Config.processes > 1»  		  	
			«foldcpptype» «local_result = "local_result"» = identity;
			«ELSE»
			«IF resultIsArray»
				«foldcpptype» «local_result = "local_result"» = identity;
			«ELSE»
				«local_result = "out"» = identity;
			«ENDIF»
			«ENDIF»		  
			
			const int size_local = in.get_size_local();
			
			#pragma omp parallel for simd reduction(«foldName»:«local_result»)
			for(int «Config.var_loop_counter» = 0; «Config.var_loop_counter» < size_local; ++«Config.var_loop_counter»){
			«foldcpptype» mapped_value = mf(in.get_local(«Config.var_loop_counter»));
		      ff(«local_result», mapped_value);
		    }
		 
			«IF Config.processes > 1»
				MPI_Allreduce(&local_result, «IF se.eContainer instanceof MusketAssignment && (se.eContainer as MusketAssignment).^var.calculateType.isArray»out.get_data(), out.get_size(), «(se.eContainer as MusketAssignment).^var.calculateCollectionType.MPIType»«ELSE»&out, 1, «fs.identity.calculateType.MPIType»«ENDIF», «foldName»«Config.mpi_op_suffix», MPI_COMM_WORLD); 
			«ELSE»
				«IF resultIsArray»
					#pragma omp parallel for simd
					for(int «Config.var_loop_counter» = 0; «Config.var_loop_counter» < out.get_size_local(); ++«Config.var_loop_counter»){
						out.set_local(«Config.var_loop_counter», «local_result»[«Config.var_loop_counter»]);
					}
				«ENDIF»
			«ENDIF»
			}
	'''

	def static generateArrayMapFoldCopyFunctionDefinition(SkeletonExpression se) '''
		«val fs = se.skeleton as MapFoldSkeleton»
		«val mapcpptype = se.obj.calculateCollectionType.cppType»
		«val foldcpptype = fs.identity.calculateType.cppType»
		«val foldName = fs.param.functionName + "_reduction"»
		template<>
		void mkt::map_fold_copy<«mapcpptype», «foldcpptype», «se.getFunctorName(fs.mapFunction)», «se.getFunctorName(fs.param)»>(const mkt::DArray<«mapcpptype»>& in, «foldcpptype»& out, const «se.getFunctorName(fs.mapFunction)»& mf, const «foldcpptype»& identity, const «se.getFunctorName(fs.param)»& ff){
		  out = identity;
		  
		  const int size = in.get_size();
		  
		  #pragma omp parallel for simd reduction(«foldName»:out)
		  for(int «Config.var_loop_counter» = 0; «Config.var_loop_counter» < size; ++«Config.var_loop_counter»){
		  	«foldcpptype» mapped_value = mf(in.get_local(«Config.var_loop_counter»));
		  	 ff(out, mapped_value);
		  }
		}
	'''

	def static generateMatrixMapFoldFunctionDefinition(SkeletonExpression se) '''
		«val fs = se.skeleton as MapFoldSkeleton»
		«val mapcpptype = se.obj.calculateCollectionType.cppType»
		«val foldcpptype = fs.identity.calculateType.cppType»
		«val foldName = fs.param.functionName + "_reduction"»
		«var local_result = ""»
		template<>
		void mkt::map_fold<«mapcpptype», «foldcpptype», «se.getFunctorName(fs.mapFunction)», «se.getFunctorName(fs.param)»>(const mkt::DMatrix<«mapcpptype»>& in, «foldcpptype»& out, const «se.getFunctorName(fs.mapFunction)»& mf, const «foldcpptype»& identity, const «se.getFunctorName(fs.param)»& ff){
		  «IF Config.processes > 1»  		  	
		  	«foldcpptype» «local_result = "local_result"» = identity;
		  «ELSE»
		  	«local_result = "out"» = identity;
		  «ENDIF»		  
		  
		  const int size_local = in.get_size_local();
		  
		  #pragma omp parallel for simd reduction(«foldName»:«local_result»)
		  for(int «Config.var_loop_counter» = 0; «Config.var_loop_counter» < size_local; ++«Config.var_loop_counter»){
		  	«foldcpptype» mapped_value = mf(in.get_local(«Config.var_loop_counter»));
		  	 ff(«local_result», mapped_value);
		  }
		  
		  «IF Config.processes > 1»
		  	MPI_Allreduce(&local_result, &out, 1, «fs.identity.calculateType.MPIType», «foldName»«Config.mpi_op_suffix», MPI_COMM_WORLD); 
		  «ENDIF»
		}
	'''

	def static generateMatrixMapFoldCopyFunctionDefinition(SkeletonExpression se) '''
		«val fs = se.skeleton as MapFoldSkeleton»
		«val mapcpptype = se.obj.calculateCollectionType.cppType»
		«val foldcpptype = fs.identity.calculateType.cppType»
		«val foldName = fs.param.functionName + "_reduction"»
		template<>
		void mkt::map_fold_copy<«mapcpptype», «foldcpptype», «se.getFunctorName(fs.mapFunction)», «se.getFunctorName(fs.param)»>(const mkt::DMatrix<«mapcpptype»>& in, «foldcpptype»& out, const «se.getFunctorName(fs.mapFunction)»& mf, const «foldcpptype»& identity, const «se.getFunctorName(fs.param)»& ff){
		  out = identity;
		  
		  const int size = in.get_size();
		  
		  #pragma omp parallel for simd reduction(«foldName»:out)
		  for(int «Config.var_loop_counter» = 0; «Config.var_loop_counter» < size; ++«Config.var_loop_counter»){
		  	«foldcpptype» mapped_value = mf(in.get_local(«Config.var_loop_counter»));
		  	 ff(out, mapped_value);
		  }
		}
	'''
}
