#pragma once
#include <boost/exception/all.hpp>
#include <exception>
#include <string>

typedef boost::error_info<struct tag_errmsg, std::string> errmsg_info;

class SfMException : public boost::exception, public std::exception
{
public:
    SfMException(const std::string& msg) : err_msg(msg) {}
    const char* what() const noexcept { return err_msg.c_str(); }
private:
    std::string err_msg;
};