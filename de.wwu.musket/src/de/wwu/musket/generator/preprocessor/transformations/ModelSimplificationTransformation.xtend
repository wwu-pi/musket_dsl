package de.wwu.musket.generator.preprocessor.transformations

import de.wwu.musket.generator.preprocessor.util.MusketComplexElementFactory
import org.eclipse.emf.ecore.resource.Resource
import de.wwu.musket.musket.CollectionObject
import static de.wwu.musket.generator.preprocessor.util.PreprocessorUtil.*
import de.wwu.musket.musket.Model
import de.wwu.musket.musket.MapSkeleton
import de.wwu.musket.musket.MapOption
import org.eclipse.emf.ecore.util.EcoreUtil

class ModelSimplificationTransformation extends PreprocessorTransformation {
	
	new(MusketComplexElementFactory factory) {
		super(factory)
	}
	
	override run(Resource input) {
		replaceMultiCollectionObjects(input)
		
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
	
}