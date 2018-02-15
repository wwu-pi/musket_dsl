/*
 * generated by Xtext 2.11.0
 */
package de.wwu.musket.scoping

import de.wwu.musket.musket.Function
import de.wwu.musket.musket.FunctionStatement
import de.wwu.musket.musket.BoolVariable
import de.wwu.musket.musket.DoubleVariable
import de.wwu.musket.musket.IntVariable
import de.wwu.musket.musket.StructVariable
import de.wwu.musket.musket.MusketPackage
//import de.wwu.musket.musket.NestedAttributeRef
import de.wwu.musket.musket.ObjectRef
import de.wwu.musket.musket.ReferableObject
import de.wwu.musket.musket.Struct
import de.wwu.musket.musket.StructArray
import de.wwu.musket.musket.StructMatrix
import de.wwu.musket.musket.StructParameter
import java.util.Collection
import org.eclipse.emf.ecore.EObject
import org.eclipse.emf.ecore.EReference
import org.eclipse.xtext.EcoreUtil2
import org.eclipse.xtext.scoping.IScope
import org.eclipse.xtext.scoping.Scopes
import de.wwu.musket.musket.CollectionObject
//import de.wwu.musket.musket.NestedCollectionElementRef
//import de.wwu.musket.musket.CollectionElementRef
import de.wwu.musket.musket.MainRef
import de.wwu.musket.musket.Model
import de.wwu.musket.musket.MusketStructVariable

/**
 * This class contains custom scoping description.
 * 
 * See https://www.eclipse.org/Xtext/documentation/303_runtime_concepts.html#scoping
 * on how and when to use it.
 */
class MusketScopeProvider extends AbstractMusketScopeProvider {

	override getScope(EObject context, EReference reference) {
		if(context instanceof ObjectRef){
			val head = context.ref  
			switch (head) {
				MainRef: {
					println(head)
					switch (head.value) {
						MusketStructVariable: return Scopes::scopeFor((head.value as MusketStructVariable).type.attributes)
						StructVariable: return Scopes::scopeFor((head.value as StructVariable).type.attributes)
						StructParameter: return Scopes::scopeFor((head.value as StructParameter).type.attributes)
						StructArray: return Scopes::scopeFor((head.value as StructArray).type.attributes)
						StructMatrix: return Scopes::scopeFor((head.value as StructArray).type.attributes)
						default: println('unknown scope for head '+head.value)
					}
					return IScope::NULLSCOPE
				}
				ObjectRef: {
					val tail = head.tail
					println("->"+tail)
					switch (tail) {
						StructVariable: return Scopes::scopeFor(tail.type.attributes)
						MusketStructVariable: return Scopes::scopeFor(tail.type.attributes)
						StructArray: return Scopes::scopeFor(tail.type.attributes)
						StructMatrix: return Scopes::scopeFor(tail.type.attributes)
						default: println('unknown scope for tail '+tail)
					}
					return IScope::NULLSCOPE
				}
				 
				default: return getScopeFromPosition(context)
			}	
		} else if ((context instanceof Function && reference == MusketPackage.eINSTANCE.objectRef_Tail)){
			// No idea why global variable reference ends up here without MainRef object
			//println(context)
		}

		return super.getScope(context, reference);
	}
	
	def getScopeFromPosition(EObject pos){
	 	// Move to top level of nested statements to get function
		var EObject obj = pos
		var Collection<ReferableObject> inScope = newArrayList()
		while(obj !== null) {
			// collect available elements in scope on this level but exclude non-instantiable struct type definition
			// TODO exclude objects after current position
			inScope.addAll(obj.eContents.filter(ReferableObject).filter[!(it instanceof Struct)].toList)
			// Add nested names in multi attributes
			inScope.addAll(obj.eContents.filter(IntVariable).map[it.vars].flatten.toList)
			inScope.addAll(obj.eContents.filter(DoubleVariable).map[it.vars].flatten.toList)
			inScope.addAll(obj.eContents.filter(BoolVariable).map[it.vars].flatten.toList)
			inScope.addAll(obj.eContents.filter(StructVariable).map[it.vars].flatten.toList)
			inScope.addAll(obj.eContents.filter(CollectionObject).map[it.vars].flatten.toList)
			
			obj = obj.eContainer

		} 
		return Scopes.scopeFor(inScope)
	}
	
//	def getScope2(EObject context, EReference reference) {
//		// Assign statement -> allowed reference values
//		if ((context instanceof ObjectRef || context instanceof Function || context instanceof FunctionStatement) && reference == MusketPackage.eINSTANCE.objectRef_Value){
//			// P? = 9.0		| context RegularFunction  
//			// P.! = 9.0	| context ObjectRef auf 	ReferableObject 	mit container 	Compare 
//			
//			
//			// Move to top level of nested statements to get function
//			var EObject obj = context
//			var Collection<ReferableObject> inScope = newArrayList()
//			while(obj !== null) {
//				// collect available elements in scope on this level but exclude non-instantiable struct type definition
//				inScope.addAll(obj.eContents.filter(ReferableObject).filter[!(it instanceof Struct)].toList)
//				// Add nested names in multi attributes
//				inScope.addAll(obj.eContents.filter(IntVariable).map[it.vars].flatten.toList)
//				inScope.addAll(obj.eContents.filter(DoubleVariable).map[it.vars].flatten.toList)
//				inScope.addAll(obj.eContents.filter(BoolVariable).map[it.vars].flatten.toList)
//				inScope.addAll(obj.eContents.filter(StructVariable).map[it.vars].flatten.toList)
//				inScope.addAll(obj.eContents.filter(CollectionObject).map[it.vars].flatten.toList)
//				
//				obj = obj.eContainer
//
//			} 
//			return Scopes.scopeFor(inScope)
//		
//		} else if ((context instanceof Function || context instanceof FunctionStatement) && reference == MusketPackage.eINSTANCE.nestedAttributeRef_Value){
//			// P.?			| context Regular Function  
//			println('test')
//		// Nested refences -> allowed references
//		} else if (context instanceof CollectionElementRef && reference == MusketPackage.eINSTANCE.nestedAttributeRef_Value) {
//			// array[0].?   <-- initially before nestedAttributeRef is first created
//			
//			val ReferableObject containerElement = (context as CollectionElementRef).value
//			val rootElement = EcoreUtil2.getRootContainer(context)
//					
//			if(containerElement instanceof StructArray) {
//				val candidates = EcoreUtil2.getAllContentsOfType(rootElement, Struct).filter[it === (containerElement as StructArray).type].map[it.attributes].flatten
//				return Scopes.scopeFor(candidates)
//			} else if (containerElement instanceof StructMatrix) {
//				val candidates = EcoreUtil2.getAllContentsOfType(rootElement, Struct).filter[it === (containerElement as StructMatrix).type].map[it.attributes].flatten
//				return Scopes.scopeFor(candidates)
//			} else {
//				return IScope::NULLSCOPE;
//			}
//			
//		} else if (context instanceof NestedAttributeRef && reference == MusketPackage.eINSTANCE.nestedAttributeRef_Value) {
//			// 			| nested value					mit container 	
//			// P.x		| 				ReferableObject					ObjectReference 
//			// p[0].x?	| 				ReferableObj					CollectionElementRef
//			// p[0].x.?	| 				DoubleVariable					CollectionElementRef 
//			val ReferableObject containerElement = 
//				if(context.eContainer instanceof ObjectRef) {
//					// Dealing with multi-array definitions
//					if((context.eContainer as ObjectRef).value.eContainer instanceof CollectionObject){
//						(context.eContainer as ObjectRef).value.eContainer as CollectionObject
//					} else {
//						(context.eContainer as ObjectRef).value
//					}
//				} else if(context.eContainer instanceof NestedAttributeRef) {
//					// Dealing with multi-array definitions
//					if((context.eContainer as NestedAttributeRef).value.eContainer instanceof CollectionObject){
//						(context.eContainer as NestedAttributeRef).value.eContainer as CollectionObject
//					} else {
//						(context.eContainer as NestedAttributeRef).value
//					}
//				}
//			
//			val rootElement = EcoreUtil2.getRootContainer(context)
//					
//			if(containerElement instanceof StructParameter) {
//				val candidates = EcoreUtil2.getAllContentsOfType(rootElement, Struct).filter[it === containerElement.type].map[it.attributes].flatten
//				return Scopes.scopeFor(candidates)
//			} else if ((context.eContainer instanceof CollectionElementRef || context.eContainer instanceof NestedCollectionElementRef) && containerElement instanceof StructArray) {
//				val candidates = EcoreUtil2.getAllContentsOfType(rootElement, Struct).filter[it === (containerElement as StructArray).type].map[it.attributes].flatten
//				return Scopes.scopeFor(candidates)
//			} else if ((context.eContainer instanceof CollectionElementRef || context.eContainer instanceof NestedCollectionElementRef) && containerElement instanceof StructMatrix) {
//				val candidates = EcoreUtil2.getAllContentsOfType(rootElement, Struct).filter[it === (containerElement as StructMatrix).type].map[it.attributes].flatten
//				return Scopes.scopeFor(candidates)
//			} else {
//				return IScope::NULLSCOPE;
//			}
//		}
//		return super.getScope(context, reference);
//	}
}
