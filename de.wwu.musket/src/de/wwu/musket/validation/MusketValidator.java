package de.wwu.musket.validation;

import org.eclipse.xtext.validation.ComposedChecks;
import de.wwu.musket.validation.MusketTypeValidator;

@ComposedChecks(validators = {MusketTypeValidator.class})
public class MusketValidator extends AbstractMusketValidator {
	// Validation is delegated to the individual validator classes mentioned in the annotation above
}
