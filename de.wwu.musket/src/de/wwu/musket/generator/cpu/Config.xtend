package de.wwu.musket.generator.cpu

import org.eclipse.emf.ecore.resource.Resource

import static extension de.wwu.musket.generator.extensions.ModelElementAccess.*

class Config {
	public static final String base_path = "CPU/"
	public static final String include_path = "include/"
	public static int processes;

	def static init(Resource resource) {
		processes = resource.Processes
	}
}
