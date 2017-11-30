package de.wwu.musket.generator.extensions

import de.wwu.musket.musket.ConfigBlock
import de.wwu.musket.musket.Model
import org.eclipse.emf.ecore.resource.Resource
import de.wwu.musket.musket.Array
import de.wwu.musket.musket.FoldSkeleton

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
	
	def static Data(Resource resource) {
		resource.Model.data
	}
	
	def static Arrays(Resource resource) {
		resource.Model.data.filter(Array)
	}
	
	def static FoldSkeletons(Resource resource) {
		resource.allContents.filter(FoldSkeleton).toIterable
	}

	def static isPlatformCPU(Resource resource) {
		resource.allContents.filter(ConfigBlock).head.platformCPU
	}
}
