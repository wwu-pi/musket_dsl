package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.SkeletonStatement

import static extension de.wwu.musket.generator.extensions.ObjectExtension.*
import de.wwu.musket.musket.Array
import de.wwu.musket.musket.InternalFunctionCall

class SkeletonGenerator {
	def static generateSkeletonStatement(SkeletonStatement s) {
		switch s.function {
			case MAP_IN_PLACE: generateMapInPlaceSkeleton(s)
			default: ''''''
		}
	}

	def static generateMapInPlaceSkeleton(SkeletonStatement s) {
		switch s.obj {
			Array: generateArrayMapInPlaceSkeleton(s, s.obj as Array)
		}
	}

	def static generateArrayMapInPlaceSkeleton(SkeletonStatement s, Array a) '''
		#pragma omp parallel for
		for(int i = 0; i < «a.sizeLocal»; ++i){
			«a.name»[i] = «(s.param as InternalFunctionCall).value.name»(«a.name»[i]);
		}
	'''

}
