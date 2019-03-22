package de.wwu.musket.generator.extensions

import de.wwu.musket.musket.ArrayType
import de.wwu.musket.musket.CollectionObject
import de.wwu.musket.musket.ConfigBlock
import de.wwu.musket.musket.FoldSkeleton
import de.wwu.musket.musket.Model
import de.wwu.musket.musket.MusketFunctionCall
import de.wwu.musket.musket.SkeletonExpression
import org.eclipse.emf.ecore.resource.Resource
import de.wwu.musket.musket.MatrixType
import de.wwu.musket.musket.Struct
import de.wwu.musket.musket.RegularFunction
import de.wwu.musket.musket.LambdaFunction
import de.wwu.musket.musket.Function
import de.wwu.musket.musket.MapFoldSkeleton
import de.wwu.musket.musket.CollectionFunctionCall
import de.wwu.musket.musket.CollectionFunctionName
import java.util.List
import de.wwu.musket.musket.Skeleton
import de.wwu.musket.musket.ShiftPartitionsVerticallySkeleton
import de.wwu.musket.musket.ShiftPartitionsHorizontallySkeleton
import de.wwu.musket.musket.ReductionSkeleton
import de.wwu.musket.musket.MapReductionSkeleton

/**
 * Helper methods to access certain elements of the model faster.
 * <p>
 * All methods take the resource object as input and return the requested object, objects or value.
 */
class ModelElementAccess {

	// get general info about resource
	def static ProjectName(Resource resource) {
		resource.URI.trimFileExtension.trimFragment.trimQuery.segment(resource.URI.segmentCount - 1)
	}
	
	def static Processes(Resource resource) {
		resource.ConfigBlock.processes
	}

	// getter for certain elements
	def static Model(Resource resource) {
		resource.allContents.filter(Model).head
	}

	def static ConfigBlock(Resource resource) {
		resource.allContents.filter(ConfigBlock).head
	}
	
	def static Functions(Resource resource) {
		resource.Model.functions
	}
	
	def static FunctionsAndLambdas(Resource resource) {
		resource.allContents.filter(Function).toIterable
	}
	
	//def static SkeletonParamInputPairs(Resource resource) {
	//	val skeletonExpressions = resource.SkeletonExpressions
	//	var List<Pair<Skeleton, Function>> result = newArrayList
	//	for(skelExpr : skeletonExpressions){
	//		result.add(skelExpr.skeleton -> skelExpr.skeleton.param.)	
	//	}
	//	return result
	//}
	
	def static Data(Resource resource) {
		resource.Model.data
	}
	
	def static Arrays(Resource resource) {
		resource.Model.data.filter(CollectionObject).filter[it.type instanceof ArrayType]
	}
	
	def static Matrices(Resource resource) {
		resource.Model.data.filter(CollectionObject).filter[it.type instanceof MatrixType]
	}
	
	def static CollectionObjects(Resource resource) {
		resource.Model.data.filter(CollectionObject)
	}
	
	def static Structs(Resource resource) {
		resource.Model.data.filter(Struct)
	}
	
	def static ShiftPartitionsHorizontallySkeletons(Resource resource) {
		resource.allContents.filter(ShiftPartitionsHorizontallySkeleton).toIterable
	}
	
	def static ShiftPartitionsVerticallySkeletons(Resource resource) {
		resource.allContents.filter(ShiftPartitionsVerticallySkeleton).toIterable
	}
	
	def static MapFoldSkeletons(Resource resource) {
		resource.allContents.filter(MapFoldSkeleton).toIterable
	}
	
	def static FoldSkeletons(Resource resource) {
		resource.allContents.filter(FoldSkeleton).toIterable
	}
	
	def static ReductionSkeletons(Resource resource) {
		resource.allContents.filter(ReductionSkeleton).toIterable
	}
	
	def static MapReductionSkeletons(Resource resource) {
		resource.allContents.filter(MapReductionSkeleton).toIterable
	}
	
	def static SkeletonExpressions(Resource resource) {
		resource.allContents.filter(SkeletonExpression).toList
	}
	
	def static MusketFunctionCalls(Resource resource) {
		resource.allContents.filter(MusketFunctionCall).toList
	}
	
	def static ShowCalls(Resource resource) {
		resource.allContents.filter(CollectionFunctionCall).filter[it.function == CollectionFunctionName.SHOW].toList
	}
	
	def static ShowLocalCalls(Resource resource) {
		resource.allContents.filter(CollectionFunctionCall).filter[it.function == CollectionFunctionName.SHOW_LOCAL].toList
	}

	def static isPlatformCPU(Resource resource) {
		resource.allContents.filter(ConfigBlock).head.platformCPU
	}
	
	def static isPlatformCPUMPMD(Resource resource) {
		resource.allContents.filter(ConfigBlock).head.platformCPUMPMD
	}
	
	def static isPlatformGPU(Resource resource) {
		resource.allContents.filter(ConfigBlock).head.platformGPU
	}
}
