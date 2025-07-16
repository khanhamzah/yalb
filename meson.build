# https://mesonbuild.com/
project(
    'my-fantastic-simulation-code',
    ['c', 'cpp'],
    default_options : ['cpp_std=c++17'],
    version : '0.1'
)

# Optional MPI include path (not strictly required if MPI is found via pkg-config)
mpi_inc = include_directories('/usr/lib/x86_64-linux-gnu/openmpi/include')

# Optional dependency: Kokkos
cmake = import('cmake')
kokkos_options = cmake.subproject_options()
kokkos_options.add_cmake_defines({
  'Kokkos_ENABLE_SERIAL': true,
  'Kokkos_ENABLE_THREADS': true,
  'Kokkos_ENABLE_OPENMP': false,
  'Kokkos_ENABLE_CUDA': false,
  'Kokkos_ENABLE_HIP': false,
})
kokkos = cmake.subproject('kokkos', options: kokkos_options)
kokkoscore = kokkos.dependency('kokkoscore')

# Optional dependency: MPI
mpi = dependency('mpi', language: 'cpp', required: false)

if mpi.found()
    add_project_arguments('-DWITH_MPI', language : ['c', 'cpp'])
    message('MPI found')
else
    message('MPI not found, disabling MPI support')
endif

subdir('src')
subdir('executables')
subdir('tests')
