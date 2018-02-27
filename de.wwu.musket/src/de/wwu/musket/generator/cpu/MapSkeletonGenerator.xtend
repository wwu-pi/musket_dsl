package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.SkeletonExpression
import de.wwu.musket.musket.MatrixType
import de.wwu.musket.musket.MapIndexSkeleton
import de.wwu.musket.musket.MapIndexInPlaceSkeleton
import de.wwu.musket.musket.ArrayType

/**
 * Generate everything required for the map skeleton, except for the actual map skeleton call.
 * That is done in the skeleton generator.
 * <p>
 * At the moment that is the offset calculation for mapIndex skeletons.
 */
class MapSkeletonGenerator {

	/**
	 * Generates the offset variables for the mapIndex skeleton.
	 * There are two variables for matrices and one for arrays.
	 * The value is set in the SkeletonGenerator for the mapIndex calls.
	 * 
	 * @param skeletons all skeleton expressions
	 * @return generated code
	 */
	def static generateOffsetVariableDeclarations(Iterable<SkeletonExpression> skeletons) {
		var result = ""

		if (skeletons.exists [
			(it.skeleton instanceof MapIndexSkeleton || it.skeleton instanceof MapIndexInPlaceSkeleton) &&
				it.obj.type instanceof MatrixType
		]) {
			result += "size_t " + Config.var_row_offset + " = 0;"
			result += "size_t " + Config.var_col_offset + " = 0;"
		}

		if (skeletons.exists [
			(it.skeleton instanceof MapIndexSkeleton || it.skeleton instanceof MapIndexInPlaceSkeleton) &&
				it.obj.type instanceof ArrayType
		]) {
			result += "size_t " + Config.var_elem_offset + " = 0;"
		}
		return result
	}

}
