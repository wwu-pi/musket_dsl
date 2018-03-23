package de.wwu.musket.validation

import de.wwu.musket.musket.CollectionObject
import de.wwu.musket.musket.DistributionMode
import de.wwu.musket.musket.FoldSkeletonVariants
import de.wwu.musket.musket.Function
import de.wwu.musket.musket.MusketAssignment
import de.wwu.musket.musket.MusketPackage
import de.wwu.musket.musket.ReturnStatement
import de.wwu.musket.musket.SkeletonExpression
import de.wwu.musket.musket.Struct
import org.eclipse.xtext.validation.Check

class MusketLimitationValidator extends AbstractMusketValidator {
	
	public static val INVALID_OPTION = 'invalidOption'
	
	// Check that collections in structs are copy-distributed
	@Check
	def checkCollectionsInStructsAreCopyDistributed(CollectionObject coll) {
		if(coll.eContainer instanceof Struct && coll.type.distributionMode !== DistributionMode.LOC){
			error('Collections in structs must be local!', 
				MusketPackage.eINSTANCE.collectionObject_Type,
				INVALID_OPTION)
		}
	}
	
	// Only allow a single return statement at the end of a function
	@Check
	def checkFunctionHasSingleReturnStatement(ReturnStatement statement) {
		val container = statement.eContainer
		if(!(container instanceof Function) || (container as Function).statement.indexOf(statement) != (container as Function).statement.length - 1){
				error('Only one return statement is allowed and must be placed at the end of a function!', 
					statement,
					null,
					-1)
		}
	}
	
	// Only allow the fold result to be a copy-distrubted data structure
	@Check
	def checkFoldAssignmentIsCopyDistributed(MusketAssignment assignment) {
		if(assignment.value instanceof SkeletonExpression && (assignment.value as SkeletonExpression).skeleton instanceof FoldSkeletonVariants &&
			assignment.^var.value instanceof CollectionObject && (assignment.^var.value as CollectionObject).type.distributionMode != DistributionMode.COPY
		){
			error('The result of a fold operation can only be stored in copy distributed data structures!', 
				MusketPackage.eINSTANCE.musketAssignment_Var,
				INVALID_OPTION)
		}
	}
}