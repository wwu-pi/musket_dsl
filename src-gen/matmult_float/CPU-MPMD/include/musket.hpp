#pragma once
#include <string>
#include "matmult_float.hpp"

namespace mkt {
enum Distribution {DIST, COPY};



template<typename T>
void print(std::ostringstream& stream, const T& a);


	
	


} // namespace mkt




template<typename T>
void mkt::print(std::ostringstream& stream, const T& a) {
	if(std::is_fundamental<T>::value){
		stream << a;
	}
}



	
