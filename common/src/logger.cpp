#include<logger.hpp>

namespace livai {

namespace tts {

namespace common {

// change it to json file later..
std::ostream out(std::cout.rdbuf());

template<>
void log<std::vector<size_t>>(const std::string& name, const std::vector<size_t>& obj)
{
    std::string valString="";
    // collect comma separated values
    for(size_t i=0;i<obj.size();++i)
    {
        valString +=  std::to_string(obj[i]) + ",";
    }

    out<<name<<" : "<<valString<<"\n";
}

}

}

}
