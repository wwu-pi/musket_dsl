package de.wwu.musket.util

import de.wwu.musket.musket.Ref

class CollectionHelper {
	static def isCollectionElementRef(Ref ref){
		return ref.localCollectionIndex?.size > 0 || ref.globalCollectionIndex?.size > 0
	}
}