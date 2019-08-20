#include<utils.hpp>

using namespace livai::tts;

std::string common::get_param_name(std::string param, size_t index)
{
    // search {} in param name and replace with index
    size_t found = param.find("{}"); 
    if(found != std::string::npos) 
    {
        param.replace(found,2,std::to_string(index));
    }

    return param;
}

std::string common::get_res_name(std::string param, size_t i, size_t j)
{
    // search {} in param name and replace with index
    size_t found = param.find("{}"); 
    if(found != std::string::npos) 
    {
        param.replace(found,2,std::to_string(i));
    }

    found = param.find("{}"); 
    if(found != std::string::npos) 
    {
        param.replace(found,2,std::to_string(j));
    }
    return param;
}