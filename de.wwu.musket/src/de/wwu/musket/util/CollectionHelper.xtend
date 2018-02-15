package de.wwu.musket.util

import de.wwu.musket.musket.ObjectRef

class CollectionHelper {
	
	static def <T> collectK(Iterable<T> iterable, int k){
		var result = newArrayList();
		val iter = iterable.iterator
		for(var i=0; i < k; i++){
			if(iter.hasNext){
				result.add(iter.next)
			}
		}
		return result;
	}
	
	static def isCollectionRef(ObjectRef ref){
		return ref.localCollectionIndex?.size > 0 || ref.globalCollectionIndex?.size > 0
	}
}