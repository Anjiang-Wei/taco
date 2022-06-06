# FindTBLIS
# ---------
#
# Find TBLIS, https://github.com/devinamatthews/tblis
#
# Defines:
#  - TBLIS_FOUND: if FALSE, do not try to use TBLIS
#  - TBLIS_INCLUDE_DIRS
#  - TBLIS_LIBRARIES
#  - TBLIS::TBLIS target

if(TBLIS_FIND_QUIETLY)
  set(_FIND_TCI_ARG QUIET)
endif()
find_package(TCI ${_FIND_TCI_ARG})

if(TCI_FOUND)

  find_path(
      TBLIS_INCLUDE_DIR tblis/tblis.h
      HINTS ENV TBLIS_ROOT
      PATH_SUFFIXES tblis)

  find_library(
      TBLIS_LIBRARY tblis
      HINTS ENV TBLIS_ROOT)

  mark_as_advanced(TBLIS_LIBRARY TBLIS_INCLUDE_DIR)

  set(TBLIS_LIBRARIES ${TBLIS_LIBRARY} ${TCI_LIBRARIES})
  set(TBLIS_INCLUDE_DIRS ${TBLIS_INCLUDE_DIR} ${TCI_INCLUDE_DIRS})

  if(NOT TARGET TBLIS::TBLIS)
    add_library(TBLIS::TBLIS UNKNOWN IMPORTED)
    set_target_properties(TBLIS::TBLIS PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${TBLIS_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES TCI::TCI)

    if(EXISTS "${TBLIS_LIBRARY}")
      set_target_properties(TBLIS::TBLIS PROPERTIES
          IMPORTED_LOCATION "${TBLIS_LIBRARY}")
      set_property(TARGET TBLIS::TBLIS PROPERTY
          IMPORTED_LINK_INTERFACE_LANGUAGES "C" "CXX")
    endif()

  endif()

endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    TBLIS
    DEFAULT_MSG
    TBLIS_LIBRARY TBLIS_INCLUDE_DIR)
