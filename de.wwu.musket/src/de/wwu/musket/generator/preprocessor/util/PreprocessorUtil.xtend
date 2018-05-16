package de.wwu.musket.generator.preprocessor.util

import org.eclipse.emf.ecore.resource.Resource
import org.eclipse.emf.ecore.resource.impl.ResourceImpl
import org.eclipse.emf.ecore.util.EcoreUtil
import org.eclipse.emf.ecore.EObject

class PreprocessorUtil {
	
	def static Resource copyModel(Resource input) {
		val Resource workingInput = new ResourceImpl
		val copier = new EcoreUtil.Copier
		
		workingInput.URI = input.URI
		workingInput.contents.addAll((copier.copyAll(input.contents)))
		copier.copyReferences
		
		return workingInput
	}
	
	def static <T extends EObject> T copyElement(T elem) {
		val copier = new EcoreUtil.Copier
		val newElem = copier.copy(elem) as T
		copier.copyReferences
		return newElem
	}
}