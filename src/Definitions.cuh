//
//  Definitions.cuh
//
//  Created by Ishbir Singh on 24/07/12.
//  Copyright (c) 2012-2014 webmaster@ishbir.com. All rights reserved.
//

#ifndef _Definitions_h
#define _Definitions_h

#include <sstream>
struct bad_conversion { };
/*
 * Convert from string to any type via streaming operations.
 */
template <class T>
void from_string(T& t,
                 const std::string& s,
                 std::ios_base& (*f)(std::ios_base&))
{
    std::istringstream iss(s);
    if((iss >> f >> t).fail())
        throw bad_conversion();
}
#endif
