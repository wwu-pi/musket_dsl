package de.wwu.musket.util

import de.wwu.musket.musket.IntRef
import de.wwu.musket.musket.Function
import de.wwu.musket.musket.RegularFunction
import java.util.Map
import de.wwu.musket.musket.LambdaFunction
import java.util.LinkedHashMap
import de.wwu.musket.musket.InternalFunctionCall
import de.wwu.musket.musket.SkeletonParameterInput
import de.wwu.musket.musket.Expression
import org.eclipse.emf.common.util.BasicEList
import de.wwu.musket.musket.Type
import de.wwu.musket.musket.IntArrayType
import de.wwu.musket.musket.IntMatrixType
import de.wwu.musket.musket.PrimitiveTypeLiteral
import de.wwu.musket.musket.BoolMatrixType
import de.wwu.musket.musket.BoolArrayType
import de.wwu.musket.musket.FloatArrayType
import de.wwu.musket.musket.DoubleArrayType
import de.wwu.musket.musket.DoubleMatrixType
import de.wwu.musket.musket.FloatMatrixType
import de.wwu.musket.musket.PrimitiveType
import static extension de.wwu.musket.generator.extensions.ObjectExtension.*
import de.wwu.musket.musket.CollectionType
import de.wwu.musket.musket.Struct
import de.wwu.musket.musket.Skeleton
import de.wwu.musket.musket.MapSkeleton
import de.wwu.musket.musket.MapInPlaceSkeleton
import de.wwu.musket.musket.MapIndexSkeleton
import de.wwu.musket.musket.MapLocalIndexSkeleton
import de.wwu.musket.musket.MapIndexInPlaceSkeleton
import de.wwu.musket.musket.MapLocalIndexInPlaceSkeleton
import de.wwu.musket.musket.FoldSkeleton
import de.wwu.musket.musket.FoldLocalSkeleton
import de.wwu.musket.musket.MapFoldSkeleton
import de.wwu.musket.musket.ZipSkeleton
import de.wwu.musket.musket.ZipInPlaceSkeleton
import de.wwu.musket.musket.ZipIndexSkeleton
import de.wwu.musket.musket.ZipLocalIndexSkeleton
import de.wwu.musket.musket.ZipIndexInPlaceSkeleton
import de.wwu.musket.musket.ZipLocalIndexInPlaceSkeleton
import de.wwu.musket.musket.ShiftPartitionsHorizontallySkeleton
import de.wwu.musket.musket.ShiftPartitionsVerticallySkeleton
import de.wwu.musket.musket.GatherSkeleton
import de.wwu.musket.musket.ScatterSkeleton
import de.wwu.musket.musket.SkeletonExpression

import static extension de.wwu.musket.util.CollectionHelper.*
import static extension de.wwu.musket.util.MusketType.*
import static extension de.wwu.musket.util.TypeHelper.*

class MusketHelper {
	// Resolve concrete values from references
	static def getConcreteValue(IntRef ref) {
		if (ref.ref !== null) {
			return ref.ref.value
		}
		return ref.value
	}

	static def getCXXPrimitiveDefaultValue(Type t) {
		switch (t) {
			IntArrayType,
			IntMatrixType,
			PrimitiveType case t.type == PrimitiveTypeLiteral.INT: '''0'''
			DoubleArrayType,
			DoubleMatrixType,
			PrimitiveType case t.type == PrimitiveTypeLiteral.DOUBLE: '''0.0'''
			FloatArrayType,
			FloatMatrixType,
			PrimitiveType case t.type == PrimitiveTypeLiteral.FLOAT: '''0.0f'''
			BoolArrayType,
			BoolMatrixType,
			PrimitiveType case t.type == PrimitiveTypeLiteral.BOOL: '''false'''
			PrimitiveType case t.type == PrimitiveTypeLiteral.STRING: ''''''
			default:
				null
		}
	}
	
	static def getCXXDefaultValue(Type t) {
		switch (t) {
			IntArrayType,
			IntMatrixType,
			PrimitiveType case t.type == PrimitiveTypeLiteral.INT: '''0'''
			DoubleArrayType,
			DoubleMatrixType,
			PrimitiveType case t.type == PrimitiveTypeLiteral.DOUBLE: '''0.0'''
			FloatArrayType,
			FloatMatrixType,
			PrimitiveType case t.type == PrimitiveTypeLiteral.FLOAT: '''0.0f'''
			BoolArrayType,
			BoolMatrixType,
			PrimitiveType case t.type == PrimitiveTypeLiteral.BOOL: '''false'''
			PrimitiveType case t.type == PrimitiveTypeLiteral.STRING: ''''''
			Struct: '''«t.name»{}'''
			default:
				null
		}
	}

	static def getCXXDefaultConstructorValue(Type t) {
		switch (t) {
			CollectionType: '''(«t.sizeLocal», «t.CXXPrimitiveDefaultValue»)'''
			PrimitiveType: '''(«t.CXXPrimitiveDefaultValue»)'''
			default: '''()'''
		}
	}
	
	static def toMPIPrimitiveType(Type t){
		switch(t){
			PrimitiveType case t.type == PrimitiveTypeLiteral.BOOL: '''MPI_BOOL'''
			PrimitiveType case t.type == PrimitiveTypeLiteral.INT: '''MPI_INT'''
			PrimitiveType case t.type == PrimitiveTypeLiteral.FLOAT: '''MPI_FLOAT'''
			PrimitiveType case t.type == PrimitiveTypeLiteral.DOUBLE: '''MPI_DOUBLE'''
			PrimitiveType case t.type == PrimitiveTypeLiteral.STRING: '''MPI_CHAR'''
			default: ''''''
		}
	}
	
	static def getFunctorName(SkeletonExpression se, SkeletonParameterInput spi) {
		val skel = se.skeleton
		val container = se.obj.collectionContainerName
		spi.functionName.toFirstUpper + "_" + skel.skeletonName + "_" + container + '_functor'
	}
	
	static def getFunctorObjectName(SkeletonExpression se, SkeletonParameterInput spi) {
		val skel = se.skeleton
		val container = se.obj.collectionContainerName
		spi.functionName.toFirstLower + "_" + skel.skeletonName + "_" + container + '_functor'
	}
	
	static def getSkeletonName(Skeleton skel) {
		switch skel {
			MapSkeleton: '''map'''
			MapInPlaceSkeleton: '''map_in_place'''
			MapIndexSkeleton: '''map_index'''
			MapLocalIndexSkeleton: '''map_local_index'''
			MapIndexInPlaceSkeleton: '''map_index_in_place'''
			MapLocalIndexInPlaceSkeleton: '''map_local_index_in_place'''
			FoldSkeleton: '''fold'''
			FoldLocalSkeleton: '''fold_local'''
			MapFoldSkeleton: '''map_fold'''
			ZipSkeleton:  '''zip'''
			ZipInPlaceSkeleton:  '''zip_in_place'''
			ZipIndexSkeleton: '''zip_index'''	
			ZipLocalIndexSkeleton: '''zip_local_index'''
			ZipIndexInPlaceSkeleton: '''zip_index_in_place'''
			ZipLocalIndexInPlaceSkeleton: '''zip_local_index_in_place'''
			ShiftPartitionsHorizontallySkeleton: '''shift_partitions_horizontally'''
			ShiftPartitionsVerticallySkeleton: '''shift_partitions_vertically'''
			GatherSkeleton: '''gather'''
			ScatterSkeleton: '''scatter'''
			default: '''// error switch: default case skeleton name'''
		}
	}
	
	static def int getNumberOfFixedParameters(SkeletonExpression se, Function f) {
		switch se.skeleton {
			MapSkeleton: 1
			MapInPlaceSkeleton: 1
			MapIndexSkeleton: if (se.obj.calculateType.isArray) 2 else 3
			MapLocalIndexSkeleton: if (se.obj.calculateType.isArray) 2 else 3
			MapIndexInPlaceSkeleton: if (se.obj.calculateType.isArray) 2 else 3
			MapLocalIndexInPlaceSkeleton: if (se.obj.calculateType.isArray) 2 else 3
			FoldSkeleton: 2
			FoldLocalSkeleton: -1
			MapFoldSkeleton: if((se.skeleton as MapFoldSkeleton).mapFunction.functionName == f.name) 1 else 2
			ZipSkeleton: 2
			ZipInPlaceSkeleton:  2
			ZipIndexSkeleton: if (se.obj.calculateType.isArray) 3 else 4
			ZipLocalIndexSkeleton: if (se.obj.calculateType.isArray) 3 else 4
			ZipIndexInPlaceSkeleton: if (se.obj.calculateType.isArray) 3 else 4
			ZipLocalIndexInPlaceSkeleton: if (se.obj.calculateType.isArray) 3 else 4
			ShiftPartitionsHorizontallySkeleton: 0
			ShiftPartitionsVerticallySkeleton: 0
			GatherSkeleton: 0
			ScatterSkeleton: 0
			default: -1
		}
	}
	
	static def int getNumberOfFreeParameters(SkeletonExpression se, Function f) {
		if(f === null){
			return 0
		}
		
		f.params.size - getNumberOfFixedParameters(se, f)
	}
	
	static def toFunction(SkeletonParameterInput spi) {
		switch spi {
			InternalFunctionCall:
				spi.value
			LambdaFunction:
				spi
		}
	}

	static def getFunctionArguments(SkeletonParameterInput spi) {
		switch spi {
			InternalFunctionCall:
				spi.params
			LambdaFunction:
				new BasicEList<Expression>()
		}
	}

	static def getFunctionParameters(SkeletonParameterInput spi) {
		switch spi {
			InternalFunctionCall:
				spi.value.params
			LambdaFunction:
				spi.params
		}
	}

	static def getFunctionName(SkeletonParameterInput spi) {
		switch spi {
			InternalFunctionCall:
				spi.value.name
			LambdaFunction:
				spi.name
		}
	}

	static def getName(Function func) {
		switch func {
			RegularFunction:
				func.name
			LambdaFunction: {
				if (State.getLambdaNames.containsKey(func))
					State.getLambdaNames.get(func)
				else {
					val name = "lambda" + State.nextLambdaCounter
					State.addLambdaName(func, name)
					name
				}
			}
		}
	}

	static class State {
		private static int lambdaCounter = 0;

		private static Map<LambdaFunction, String> lambdaNames = new LinkedHashMap<LambdaFunction, String>();

		def static getNextLambdaCounter() {
			lambdaCounter++
		}

		def static getLambdaNames() {
			return lambdaNames
		}

		def static addLambdaName(LambdaFunction lf, String name) {
			lambdaNames.put(lf, name)
		}
	}
}
