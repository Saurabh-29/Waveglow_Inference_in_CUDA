#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#pragma once

/*
 This providea an inteface for logging primitive variables, objects etc into json format.
**/

#include<string>
namespace livai
{
    namespace tts
    {
        namespace common
        {
            std::string get_param_name(std::string param, size_t index);
            std::string get_res_name(std::string param, size_t i, size_t j);
        }
    }
}

#endif