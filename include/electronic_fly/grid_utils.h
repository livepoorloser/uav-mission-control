#pragma once

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <string>

namespace electronic_fly
{

struct GridCell
{
    int row = 0;
    int col = 0;

    bool operator==(const GridCell& other) const
    {
        return row == other.row && col == other.col;
    }

    bool operator!=(const GridCell& other) const
    {
        return !(*this == other);
    }
};

inline std::string buildGridCode(int row, int col)
{
    char buffer[32] = {0};
    std::snprintf(buffer, sizeof(buffer), "A%dB%d", col + 1, row + 1);
    return std::string(buffer);
}

inline bool parseGridCode(const std::string& raw_code, GridCell& cell)
{
    std::string code;
    code.reserve(raw_code.size());
    for (char ch : raw_code)
    {
        if (!std::isspace(static_cast<unsigned char>(ch)))
        {
            code.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(ch))));
        }
    }

    const std::size_t a_pos = code.find('A');
    const std::size_t b_pos = code.find('B');
    if (a_pos == std::string::npos || b_pos == std::string::npos || a_pos >= b_pos)
    {
        return false;
    }

    try
    {
        const int col = std::stoi(code.substr(a_pos + 1, b_pos - a_pos - 1)) - 1;
        const int row = std::stoi(code.substr(b_pos + 1)) - 1;
        if (row < 0 || col < 0)
        {
            return false;
        }
        cell.row = row;
        cell.col = col;
        return true;
    }
    catch (...)
    {
        return false;
    }
}

inline bool isInsideGrid(const GridCell& cell, int rows, int cols)
{
    return cell.row >= 0 && cell.row < rows && cell.col >= 0 && cell.col < cols;
}

}  // namespace electronic_fly
