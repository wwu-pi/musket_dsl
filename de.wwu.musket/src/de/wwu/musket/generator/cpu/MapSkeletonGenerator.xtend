package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.SkeletonExpression
import de.wwu.musket.musket.Matrix
import de.wwu.musket.musket.MapIndexSkeleton
import de.wwu.musket.musket.MapIndexInPlaceSkeleton
import de.wwu.musket.musket.Array

class MapSkeletonGenerator {
	def static generateOffsetVariableDeclarations(Iterable<SkeletonExpression> skeletons) {
		var result = ""

		if (skeletons.exists [
			(it.skeleton instanceof MapIndexSkeleton || it.skeleton instanceof MapIndexInPlaceSkeleton) && it.obj instanceof Matrix
		]) {
			result += "size_t " + Config.var_row_offset + " = 0;"
			result += "size_t " + Config.var_col_offset + " = 0;"
		}

		if (skeletons.exists [
			(it.skeleton instanceof MapIndexSkeleton || it.skeleton instanceof MapIndexInPlaceSkeleton) && it.obj instanceof Array
		]) {
			result += "size_t " + Config.var_elem_offset + " = 0;"
		}
		return result
	}

}
