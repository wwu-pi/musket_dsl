package de.wwu.musket.generator.extensions

class StringExtension {
	def static removeLineBreak(String s) {
		if(s.endsWith('\n')){
			s.substring(0, s.length-1)
		}
	}
}
