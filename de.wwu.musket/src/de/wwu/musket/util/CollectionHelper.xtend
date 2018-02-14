package de.wwu.musket.util

class CollectionHelper<T> {
	
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
}