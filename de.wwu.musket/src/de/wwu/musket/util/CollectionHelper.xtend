package de.wwu.musket.util

import de.wwu.musket.musket.Ref
import de.wwu.musket.musket.CollectionObject
import de.wwu.musket.musket.ArrayType
import de.wwu.musket.musket.MatrixType

class CollectionHelper {
	static def isCollectionElementRef(Ref ref){
		return ref.localCollectionIndex?.size > 0 || ref.globalCollectionIndex?.size > 0
	}
	
	static def getCollectionContainerName(CollectionObject co){
		switch co.type {
			ArrayType: '''array'''
			MatrixType: '''matrix'''
			default: '''//CollectionHelper: getCollectionContainerName: default case'''
		}
	}
}