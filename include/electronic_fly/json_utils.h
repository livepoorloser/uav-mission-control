#pragma once

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <sstream>
#include <string>

namespace electronic_fly
{

inline bool parseJson(const std::string& json_text, boost::property_tree::ptree& tree)
{
    try
    {
        std::istringstream input(json_text);
        boost::property_tree::read_json(input, tree);
        return true;
    }
    catch (...)
    {
        tree.clear();
        return false;
    }
}

inline std::string toJson(const boost::property_tree::ptree& tree, bool pretty = false)
{
    std::ostringstream output;
    boost::property_tree::write_json(output, tree, pretty);
    std::string json_text = output.str();
    while (!json_text.empty() && (json_text.back() == '\n' || json_text.back() == '\r'))
    {
        json_text.pop_back();
    }
    return json_text;
}

inline void appendArrayItem(boost::property_tree::ptree& array_tree, const boost::property_tree::ptree& item)
{
    array_tree.push_back(std::make_pair("", item));
}

inline boost::property_tree::ptree makeStringArray(const std::vector<std::string>& values)
{
    boost::property_tree::ptree array_tree;
    for (const auto& value : values)
    {
        boost::property_tree::ptree item;
        item.put("", value);
        appendArrayItem(array_tree, item);
    }
    return array_tree;
}

}  // namespace electronic_fly
