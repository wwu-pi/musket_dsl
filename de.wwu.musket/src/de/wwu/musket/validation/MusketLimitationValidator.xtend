package de.wwu.musket.validation

import org.eclipse.xtext.validation.Check
import de.wwu.musket.musket.CollectionObject
import de.wwu.musket.musket.DistributionMode
import de.wwu.musket.musket.MusketPackage
import de.wwu.musket.musket.Struct

class MusketLimitationValidator extends AbstractMusketValidator {
	
	public static val INVALID_OPTION = 'invalidOption'
	
	// Check that collections in structs are copy-distributed
	@Check
	def checkCollectionsInStructsAreCopyDistributed(CollectionObject coll) {
		if(coll.eContainer instanceof Struct && coll.distributionMode !== DistributionMode.COPY){
			error('Collections in structs must be copy distributed!', 
				MusketPackage.eINSTANCE.collectionObject_DistributionMode,
				INVALID_OPTION)
		}
	}
}