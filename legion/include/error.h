#ifndef TACO_ERROR_H
#define TACO_ERROR_H

#include <string>
#include <iostream>
#include <sstream>
#include <ostream>

class TacoException : public std::exception{

public:
  explicit TacoException(std::string message);
  const char * what() const noexcept;

private:
  std::string message;

};

/// Error report (based on Halide's Error.h)
struct ErrorReport {
  enum Kind { User, Internal, Temporary };

  std::ostringstream *msg;
  const char *file;
  const char *func;
  int line;

  bool condition;
  const char *conditionString;

  Kind kind;
  bool warning;

  ErrorReport(const char *file, const char *func, int line, bool condition,
              const char *conditionString, Kind kind, bool warning);

  template<typename T>
  ErrorReport &operator<<(T x) {
    if (condition) {
      return *this;
    }
    (*msg) << x;
    return *this;
  }

  ErrorReport &operator<<(std::ostream& (*manip)(std::ostream&)) {
    if (condition) {
      return *this;
    }
    (*msg) << manip;
    return *this;
  }

  ~ErrorReport() noexcept(false) {
    if (condition) {
      return;
    }
    std::cout << (this->msg->str()) << std::endl;
    explodeWithException();
  }

  void explodeWithException();
};

#define taco_iassert(c)                                                     \
  ErrorReport(__FILE__, __FUNCTION__, __LINE__, (c), #c,              \
                    ErrorReport::Internal, false)
#define taco_ierror                                                         \
  ErrorReport(__FILE__, __FUNCTION__, __LINE__, false, NULL,          \
                    ErrorReport::Internal, false)

#define taco_unreachable                                                       \
  taco_ierror << "reached unreachable location"

// User asserts
#define taco_uassert(c)                                                        \
  ErrorReport(__FILE__,__FUNCTION__,__LINE__, (c), #c,                   \
                    ErrorReport::User, false)
#define taco_uerror                                                            \
  ErrorReport(__FILE__,__FUNCTION__,__LINE__, false, nullptr,            \
                    ErrorReport::User, false)
#define taco_uwarning                                                          \
  taco::ErrorReport(__FILE__,__FUNCTION__,__LINE__, false, nullptr,            \
                    ErrorReport::User, true)

// Temporary assertions (planned for the future)
#define taco_tassert(c)                                                        \
  ErrorReport(__FILE__, __FUNCTION__, __LINE__, (c), #c,                 \
                    ErrorReport::Temporary, false)
#define taco_terror                                                            \
  ErrorReport(__FILE__, __FUNCTION__, __LINE__, false, NULL,             \
                    ErrorReport::Temporary, false)

#define taco_not_supported_yet taco_uerror << "Not supported yet"

#endif
