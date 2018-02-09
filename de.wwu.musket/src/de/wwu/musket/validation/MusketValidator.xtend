/*
 * generated by Xtext 2.12.0
 */
package de.wwu.musket.validation

import org.eclipse.xtext.validation.ComposedChecks

/**
 * This class contains custom validation rules. 
 *
 * See https://www.eclipse.org/Xtext/documentation/303_runtime_concepts.html#validation
 */
 @ComposedChecks(validators = #[MusketTypeValidator, MusketModelValidator])
class MusketValidator extends AbstractMusketValidator {
	
	// Delegates validation to other classes mentioned in the annotation above.
	
}
