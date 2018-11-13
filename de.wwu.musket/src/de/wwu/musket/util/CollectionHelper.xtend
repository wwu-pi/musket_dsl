package de.wwu.musket.util

import de.wwu.musket.musket.Ref

class CollectionHelper {
	
	/**
	 * Shortcut to check for collection elements with either local or global index.
	 */
	static def isCollectionElementRef(Ref ref){
		return ref.localCollectionIndex?.size > 0 || ref.globalCollectionIndex?.size > 0
	}
}