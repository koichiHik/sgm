#    This file is part of sgm. (https://github.com/dhernandez0/sgm).
#
#    Copyright (c) 2016 Daniel Hernandez Juarez.

#    sgm is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    sgm is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with sgm.  If not, see <http://www.gnu.org/licenses/>.

cmake_minimum_required(VERSION 2.4)
project(sgm)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -lineinfo
    -gencode=arch=compute_61,code=sm_61
    )

#cuda_add_executable(
#    sgm
#    main.cu median_filter.cu hamming_cost.cu disparity_method.cu debug.cu costs.cu)
#target_link_libraries( sgm ${OpenCV_LIBS} )

cuda_add_library(
    sgm_cuda
    median_filter.cu
    hamming_cost.cu
    disparity_method.cu
    debug.cu
    costs.cu
    sgm_cuda.cu
)

#target_include_directories(
#    sgm_cuda PRIVATE
#    ${CUDA_INCLUDE_DIRS}
#)

target_link_libraries(
    sgm_cuda
    ${OpenCV_LIBS}
)