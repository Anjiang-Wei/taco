#ifndef TACO_LEGION_STRINGS_H
#define TACO_LEGION_STRINGS_H

#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <limits>

// Split a string into components based on delim.
std::vector<std::string> split(const std::string &str, const std::string &delim, bool keepDelim = false);

// Check if a string ends with another.
bool endsWith(std::string const &fullString, std::string const &ending);

/// Join the elements between begin and end in a sep-separated string.
template <typename Iterator>
std::string join(Iterator begin, Iterator end, const std::string &sep=", ") {
  std::ostringstream result;
  if (begin != end) {
    result << *begin++;
  }
  while (begin != end) {
    result << sep << *begin++;
  }
  return result.str();
}

/// Join the elements in the collection in a sep-separated string.
template <typename Collection>
std::string join(const Collection &collection, const std::string &sep=", ") {
  return join(collection.begin(), collection.end(), sep);
}

/// Turn anything except floating points that can be written to a stream
/// into a string.
template <class T>
typename std::enable_if<!std::is_floating_point<T>::value, std::string>::type
toString(const T &val) {
  std::stringstream sstream;
  sstream << val;
  return sstream.str();
}
/// Turn any floating point that can be written to a stream into a string,
/// forcing full precision and inclusion of the decimal point.
template <class T>
typename std::enable_if<std::is_floating_point<T>::value, std::string>::type
toString(const T &val) {
  std::stringstream sstream;
  sstream << std::setprecision(std::numeric_limits<T>::max_digits10) << std::showpoint << val;
  return sstream.str();
}

#endif // TACO_LEGION_STRINGS_H