lib_sources = files(
  'lbm_solver.cpp',
  'hello.cpp'
)

lib = static_library(
  'core',
  lib_sources,
  include_directories: include_directories('.'),
  dependencies: [mpi, kokkoscore]
)

lib_incdirs = include_directories('.')

lib_dep = declare_dependency(
  include_directories: lib_incdirs,
  link_with: lib,
  dependencies: [kokkoscore, mpi]  
)
