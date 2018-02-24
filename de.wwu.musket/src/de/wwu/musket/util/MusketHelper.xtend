package de.wwu.musket.util

import de.wwu.musket.musket.IntRef

class MusketHelper {
	// Resolve concrete values from references
	static def getConcreteValue(IntRef ref){
		if(ref.ref !== null) {
			return ref.ref.value
		}
		return ref.value
	}	
}