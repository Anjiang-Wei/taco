# FindTCI
# -------
#
# TCI is a dependency of TBLIS
#
# Defines:
#  - TCI_FOUND: if FALSE, do not try to use TCI
#  - TCI_INCLUDE_DIRS
#  - TCI_LIBRARIES
#  - TCI::TCI target

# TCI is usually bundled with TBLIS, so search in TBLIS_ROOT as well.
find_path(
    TCI_INCLUDE_DIR tci.h
    HINTS ENV TBLIS_ROOT ENV TCI_ROOT
    PATH_SUFFIXES tci)

find_library(
    TCI_LIBRARY tci
    HINTS ENV TBLIS_ROOT ENV TCI_ROOT)

mark_as_advanced(TCI_LIBRARY TCI_INCLUDE_DIR)

set(TCI_LIBRARIES ${TCI_LIBRARY} )
set(TCI_INCLUDE_DIRS ${TCI_INCLUDE_DIR} )

if(NOT TARGET TCI::TCI)
  add_library(TCI::TCI UNKNOWN IMPORTED)
  set_target_properties(TCI::TCI PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${TCI_INCLUDE_DIRS}")

  if(EXISTS "${TCI_LIBRARY}")
    set_target_properties(TCI::TCI PROPERTIES
        IMPORTED_LOCATION "${TCI_LIBRARY}")
    set_property(TARGET TCI::TCI
        PROPERTY IMPORTED_LINK_INTERFACE_LANGUAGES "C" "CXX")
  endif()

endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    TCI
    DEFAULT_MSG
    TCI_LIBRARY TCI_INCLUDE_DIR)
