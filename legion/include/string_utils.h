#ifndef TACO_LEGION_STRINGS_H
#define TACO_LEGION_STRINGS_H

#include <string>
#include <vector>
#include <sstream>

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

#endif // TACO_LEGION_STRINGS_H