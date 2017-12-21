package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.StandardFunctionCall

class StandardFunctionCalls {
	def static generateStandardFunctionCall(StandardFunctionCall sfc) {

		switch sfc.value.literal {
			case 'printf': '''printf'''
			default: '''//TODO unimplemented'''
		}
	}
}
