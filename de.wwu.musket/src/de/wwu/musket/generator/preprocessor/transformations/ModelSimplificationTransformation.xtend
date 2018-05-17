package de.wwu.musket.generator.preprocessor.transformations

import de.wwu.musket.generator.preprocessor.util.MusketComplexElementFactory
import org.eclipse.emf.ecore.resource.Resource
import de.wwu.musket.musket.CollectionObject
import static de.wwu.musket.generator.preprocessor.util.PreprocessorUtil.*
import de.wwu.musket.musket.Model
import de.wwu.musket.musket.MapSkeleton
import de.wwu.musket.musket.MapOption
import org.eclipse.emf.ecore.util.EcoreUtil
import de.wwu.musket.musket.FoldSkeleton
import de.wwu.musket.musket.FoldOption
import de.wwu.musket.musket.ZipSkeleton
import de.wwu.musket.musket.ZipOption

class ModelSimplificationTransformation extends PreprocessorTransformation {
	
	new(MusketComplexElementFactory factory) {
		super(factory)
	}
	
	override run(Resource input) {
		replaceMultiCollectionObjects(input)
		
		replaceGenericMapSkeleton(input)
	}
	
	/**
	 * Simplify combined definition of CollectionObjects by creating single definitions
	 */
	def replaceMultiCollectionObjects(Resource resource) {
		val model = resource.allContents.filter(Model).head
		
		// Select data objects defined as multiple variable declarations
		val multiCollections = resource.allContents.filter(CollectionObject).filter[it.vars.size > 0]
		
		multiCollections.forEach[
			val multiCollection = it
			multiCollection.vars.forEach[
				// Create new element with same properties
				val newObj = factory.createCollectionObject
				newObj.type = copyElement(multiCollection.type)
				newObj.name = it.name
				newObj.values.addAll(multiCollection.values.map[copyElement(it)].toList)
				
				// Add to model
				model.data.add(newObj)
			]
			// Clear original list in multi object
			multiCollection.vars.clear
		]
	}
	
	def replaceGenericMapSkeleton(Resource resource) {
		val maps = resource.allContents.filter(MapSkeleton)
		
		maps.forEach[
			if(it.options.contains(MapOption.IN_PLACE) && (it.options.contains(MapOption.INDEX))){
				// MapIndexInPlace
				val skel = factory.createMapIndexInPlaceSkeleton
				skel.param = it.param
				EcoreUtil.replace(it, skel)
			} else if(it.options.contains(MapOption.IN_PLACE) && (it.options.contains(MapOption.LOCAL_INDEX))){
				// MapLocalIndexInPlace
				val skel = factory.createMapLocalIndexInPlaceSkeleton
				skel.param = it.param
				EcoreUtil.replace(it, skel)
			} else if(it.options.contains(MapOption.IN_PLACE)) {
				// MapInPlace
				val skel = factory.createMapInPlaceSkeleton
				skel.param = it.param
				EcoreUtil.replace(it, skel)
			} else if(it.options.contains(MapOption.INDEX)){
				// MapIndex
				val skel = factory.createMapIndexSkeleton
				skel.param = it.param
				EcoreUtil.replace(it, skel)
			} else if(it.options.contains(MapOption.LOCAL_INDEX)){
				// MapLocalIndex
				val skel = factory.createMapLocalIndexSkeleton
				skel.param = it.param
				EcoreUtil.replace(it, skel)
			}
		]
	}
	
	def replaceGenericFoldSkeleton(Resource resource) {
		val maps = resource.allContents.filter(FoldSkeleton)
		
		maps.forEach[
			if(it.options.contains(FoldOption.INDEX)){
				// FoldIndex
				val skel = factory.createFoldLocalSkeleton
				skel.param = it.param
				EcoreUtil.replace(it, skel)
			}
		]
	}
	
	def replaceGenericZipSkeleton(Resource resource) {
		val maps = resource.allContents.filter(ZipSkeleton)
		
		maps.forEach[
			if(it.options.contains(ZipOption.IN_PLACE) && (it.options.contains(ZipOption.INDEX))){
				// ZipIndexInPlace
				val skel = factory.createZipIndexInPlaceSkeleton
				skel.param = it.param
				EcoreUtil.replace(it, skel)
			} else if(it.options.contains(ZipOption.IN_PLACE) && (it.options.contains(ZipOption.LOCAL_INDEX))){
				// ZipLocalIndexInPlace
				val skel = factory.createZipLocalIndexInPlaceSkeleton
				skel.param = it.param
				EcoreUtil.replace(it, skel)
			} else if(it.options.contains(ZipOption.IN_PLACE)) {
				// ZipInPlace
				val skel = factory.createZipInPlaceSkeleton
				skel.param = it.param
				EcoreUtil.replace(it, skel)
			} else if(it.options.contains(ZipOption.INDEX)){
				// ZipIndex
				val skel = factory.createZipIndexSkeleton
				skel.param = it.param
				EcoreUtil.replace(it, skel)
			} else if(it.options.contains(ZipOption.LOCAL_INDEX)){
				// ZipLocalIndex
				val skel = factory.createZipLocalIndexSkeleton
				skel.param = it.param
				EcoreUtil.replace(it, skel)
			}
		]
	}
}