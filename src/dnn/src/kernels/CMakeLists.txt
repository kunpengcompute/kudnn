set(ALL_SUPPORTED_PRECISIONS "float" "__fp16" "__bf16" "int8_t" "uint8_t" "int32_t")
set(FLOAT_SUPPORTED_PRECISIONS "float" "__fp16" "__bf16")
set(COMPUTE_PRECISIONS "float")
set(SHORT_NAME_float f32)
set(SHORT_NAME___fp16 f16)
set(SHORT_NAME___bf16 bf16)
set(SHORT_NAME_int8_t s8)
set(SHORT_NAME_uint8_t u8)
set(SHORT_NAME_int32_t s32)

# ============================================================================
# Merged template instantiation strategy.
#
# Problem: each kernel had an independent OBJECT target per (IN_TYPE, OUT_TYPE)
# pair, producing 70+ OBJECT targets that each parsed the identical heavy
# header tree (arm_sve.h → vector_wrappers.hpp → kudnn_reorder_def.hpp) via
# -DIN_TYPE=... -DOUT_TYPE=... macros.
#
# Solution: generate one source file per (direction × IN_TYPE) that contains
# ALL template instantiations for that combination. The header tree is parsed
# exactly once per merged TU instead of once per precision pair.
#
# TU reduction:
#   Reorder: 36 → 6  (-83%)
#   GNorm:    9 → 2  (-78%)
#   LNorm:    9 → 4  (-56%)
#   RMSNorm:  6 → 2  (-67%)
#   Softmax: 11 → 4  (-64%)
#   Total:  ~71 → ~18 (-75%)
#
# The final .o files are identical: same template instantiations, same function
# symbols, only the compile-time boundary changed. Benchmark-recommended.
# ============================================================================

set(KERNS_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/kernels_merged")
file(MAKE_DIRECTORY "${KERNS_GEN_DIR}")

# ----------------------------------------------------------------------------
# Reorder — 36 OBJECT → 6 OBJECT (one per IN_TYPE)
# ----------------------------------------------------------------------------
set(REORDER_KERNELS_LIST "")
set(REORDER_ALL_OUT_TYPES ${ALL_SUPPORTED_PRECISIONS})
foreach(IN_PREC ${ALL_SUPPORTED_PRECISIONS})
    set(KUDNN_REORDER_INSTANTIATIONS "")
    foreach(OUT_PREC ${REORDER_ALL_OUT_TYPES})
        set(KUDNN_REORDER_INSTANTIATIONS "${KUDNN_REORDER_INSTANTIATIONS}template void EvaluateReorder<${IN_PREC}, ${OUT_PREC}>(const Shape &start, const Shape &end, const ${IN_PREC} *typedSrc, ${OUT_PREC} *typedDst, const Shape &reorderedStridesSrc, const Shape &reorderedStridesDst, bool fastReorder, SizeType strideIn, SizeType strideOut) noexcept;\n")
    endforeach()
    set(GEN_FILE "${KERNS_GEN_DIR}/kudnn_reorder_merged_${SHORT_NAME_${IN_PREC}}.cpp")
    configure_file(
        "${CMAKE_CURRENT_LIST_DIR}/reorder/kudnn_reorder_merged.cpp.in"
        "${GEN_FILE}" @ONLY)
    add_library(reorder_merged_${SHORT_NAME_${IN_PREC}} OBJECT "${GEN_FILE}")
    list(APPEND REORDER_KERNELS_LIST
        $<TARGET_OBJECTS:reorder_merged_${SHORT_NAME_${IN_PREC}}>)
endforeach()
set(REORDER_KERNELS_LIST "${REORDER_KERNELS_LIST}" CACHE INTERNAL "")

# Convolution — unchanged (code-generated .cpp files, not amenable to merging)
include(convolution/conv_gen.cmake)

add_custom_command(
    OUTPUT ${CONVOLUTION_KERNELS_F32}
    COMMAND python3 ${CMAKE_SOURCE_DIR}/src/kernels/convolution/convcodegen.py f32
)

add_custom_command(
    OUTPUT ${CONVOLUTION_KERNELS_F16}
    COMMAND python3 ${CMAKE_SOURCE_DIR}/src/kernels/convolution/convcodegen.py f16
)

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/convolution/ CONVOLUTION_PIPELINE)

add_library(convolution_kernels_pipeline OBJECT ${CONVOLUTION_PIPELINE})
add_library(convolution_kernels_f32 OBJECT ${CONVOLUTION_KERNELS_F32})
add_library(convolution_kernels_f16 OBJECT ${CONVOLUTION_KERNELS_F16})

# STL-only PCH for convolution kernels.
#
# Convolution generates ~2000 .cpp files (layout × kernel_size × exec_size ×
# sw_size × precision). Each file includes vector_wrappers.hpp → arm_sve.h →
# kudnn.hpp → types headers, so the STL transitive closure (from kudnn.hpp /
# internal_funcs.hpp) is re-parsed ~2000 times.
#
# arm_sve.h cannot enter the PCH (known GCC/Clang ICE with SVE intrinsics),
# so this PCH is STL-only. The heavy SVE header is still parsed per TU, but
# all STL headers (<vector>, <string>, <memory>, <algorithm>, <cmath>, etc.)
# are parsed once. Estimated ~5-10% cold-build reduction on convolution.
target_precompile_headers(convolution_kernels_f32 PRIVATE
    <vector>
    <string>
    <memory>
    <algorithm>
    <cmath>
    <cstddef>
    <cstdint>
    <cstring>
    <array>
    <utility>
    <limits>
    <type_traits>
)
target_precompile_headers(convolution_kernels_f16 PRIVATE
    <vector>
    <string>
    <memory>
    <algorithm>
    <cmath>
    <cstddef>
    <cstdint>
    <cstring>
    <array>
    <utility>
    <limits>
    <type_traits>
)

list(APPEND CONV_KERNELS_LIST
    $<TARGET_OBJECTS:convolution_kernels_pipeline>
    $<TARGET_OBJECTS:convolution_kernels_f16>
    $<TARGET_OBJECTS:convolution_kernels_f32>)

set(CONV_KERNELS_LIST "${CONV_KERNELS_LIST}" CACHE INTERNAL "")

# ----------------------------------------------------------------------------
# GNorm — 9 OBJECT → 2 OBJECT (fwd / bwd merged)
# ----------------------------------------------------------------------------
set(GNORM_KERNELS_LIST "")
set(KUDNN_GNORM_FWD_INSTANTIATIONS "")
foreach(IN_OUT_PREC ${ALL_SUPPORTED_PRECISIONS})
    set(KUDNN_GNORM_FWD_INSTANTIATIONS "${KUDNN_GNORM_FWD_INSTANTIATIONS}template void EvaluateCrossAxis<${COMPUTE_PRECISIONS}, ${IN_OUT_PREC}, ${IN_OUT_PREC}, ${IN_OUT_PREC}>(SizeType outerSize, const ${IN_OUT_PREC} *src, ${IN_OUT_PREC} *dst, SizeType groupSize, const ${IN_OUT_PREC} *scale, const ${IN_OUT_PREC} *shift, float *mean, float *variance, bool saveStats, float eps, NormalizationFlags flags, SizeType axisSize, SizeType innerSize) noexcept;\n")
    set(KUDNN_GNORM_FWD_INSTANTIATIONS "${KUDNN_GNORM_FWD_INSTANTIATIONS}template void EvaluateLastAxis<${COMPUTE_PRECISIONS}, ${IN_OUT_PREC}, ${IN_OUT_PREC}, ${IN_OUT_PREC}>(SizeType outerSize, SizeType batchSize, const ${IN_OUT_PREC} *src, ${IN_OUT_PREC} *dst, SizeType groupSize, const ${IN_OUT_PREC} *scale, const ${IN_OUT_PREC} *shift, float *mean, float *variance, bool saveStats, float eps, NormalizationFlags flags, SizeType axisSize) noexcept;\n")
endforeach()
configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/gnorm/kudnn_gnorm_merged_fwd.cpp.in"
    "${KERNS_GEN_DIR}/kudnn_gnorm_merged_fwd.cpp" @ONLY)
add_library(gnorm_merged_fwd OBJECT "${KERNS_GEN_DIR}/kudnn_gnorm_merged_fwd.cpp")
list(APPEND GNORM_KERNELS_LIST $<TARGET_OBJECTS:gnorm_merged_fwd>)

set(KUDNN_GNORM_BWD_INSTANTIATIONS "")
foreach(IN_OUT_PREC ${FLOAT_SUPPORTED_PRECISIONS})
    set(KUDNN_GNORM_BWD_INSTANTIATIONS "${KUDNN_GNORM_BWD_INSTANTIATIONS}template void EvaluteAbx<${COMPUTE_PRECISIONS}, ${IN_OUT_PREC}, ${IN_OUT_PREC}, ${IN_OUT_PREC}>(SizeType batchNum, SizeType eltPerGroup, const ${IN_OUT_PREC} *src, const float *mean, const float *variance, const ${IN_OUT_PREC} *diffDst, const ${IN_OUT_PREC} *scale, ${IN_OUT_PREC} *diffSrc, float *diffScale, float *diffShift, float eps, NormalizationFlags flags, SizeType channelSize, SizeType channelPerGroup) noexcept;\n")
    set(KUDNN_GNORM_BWD_INSTANTIATIONS "${KUDNN_GNORM_BWD_INSTANTIATIONS}template void EvaluteAxb<${COMPUTE_PRECISIONS}, ${IN_OUT_PREC}, ${IN_OUT_PREC}, ${IN_OUT_PREC}>(SizeType batchNum, SizeType eltPerGroup, const ${IN_OUT_PREC} *src, const float *mean, const float *variance, const ${IN_OUT_PREC} *diffDst, const ${IN_OUT_PREC} *scale, ${IN_OUT_PREC} *diffSrc, float *diffScale, float *diffShift, float eps, NormalizationFlags flags, SizeType channelSize, SizeType channelPerGroup) noexcept;\n")
endforeach()
configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/gnorm/kudnn_gnorm_merged_bwd.cpp.in"
    "${KERNS_GEN_DIR}/kudnn_gnorm_merged_bwd.cpp" @ONLY)
add_library(gnorm_merged_bwd OBJECT "${KERNS_GEN_DIR}/kudnn_gnorm_merged_bwd.cpp")
list(APPEND GNORM_KERNELS_LIST $<TARGET_OBJECTS:gnorm_merged_bwd>)

set(GNORM_KERNELS_LIST "${GNORM_KERNELS_LIST}" CACHE INTERNAL "")

# ----------------------------------------------------------------------------
# LNorm — 9 OBJECT → 4 OBJECT (fwd_cross / fwd_last / bwd_cross / bwd_last)
# ----------------------------------------------------------------------------
set(LNORM_KERNELS_LIST "")

set(KUDNN_LNORM_FWD_CROSS_INSTANTIATIONS "")
foreach(IN_OUT_PREC ${ALL_SUPPORTED_PRECISIONS})
    set(KUDNN_LNORM_FWD_CROSS_INSTANTIATIONS "${KUDNN_LNORM_FWD_CROSS_INSTANTIATIONS}template void EvaluateCrossAxis<${COMPUTE_PRECISIONS}, ${IN_OUT_PREC}, ${IN_OUT_PREC}, ${IN_OUT_PREC}>(SizeType outerSize, const ${IN_OUT_PREC} *src, ${IN_OUT_PREC} *dst, const ${IN_OUT_PREC} *scale, const ${IN_OUT_PREC} *shift, float *mean, float *variance, bool saveStats, float eps, NormalizationFlags flags, SizeType axisSize, SizeType innerSize) noexcept;\n")
endforeach()
configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/lnorm/kudnn_lnorm_merged_fwd_cross.cpp.in"
    "${KERNS_GEN_DIR}/kudnn_lnorm_merged_fwd_cross.cpp" @ONLY)
add_library(lnorm_merged_fwd_cross OBJECT "${KERNS_GEN_DIR}/kudnn_lnorm_merged_fwd_cross.cpp")
list(APPEND LNORM_KERNELS_LIST $<TARGET_OBJECTS:lnorm_merged_fwd_cross>)

set(KUDNN_LNORM_FWD_LAST_INSTANTIATIONS "")
foreach(IN_OUT_PREC ${ALL_SUPPORTED_PRECISIONS})
    set(KUDNN_LNORM_FWD_LAST_INSTANTIATIONS "${KUDNN_LNORM_FWD_LAST_INSTANTIATIONS}template void EvaluateLastAxis<${COMPUTE_PRECISIONS}, ${IN_OUT_PREC}, ${IN_OUT_PREC}, ${IN_OUT_PREC}>(SizeType execAcrossAxis, const ${IN_OUT_PREC} *src, ${IN_OUT_PREC} *dst, const ${IN_OUT_PREC} *scale, const ${IN_OUT_PREC} *shift, float *mean, float *variance, bool saveStats, float eps, NormalizationFlags flags, SizeType axisSize) noexcept;\n")
endforeach()
configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/lnorm/kudnn_lnorm_merged_fwd_last.cpp.in"
    "${KERNS_GEN_DIR}/kudnn_lnorm_merged_fwd_last.cpp" @ONLY)
add_library(lnorm_merged_fwd_last OBJECT "${KERNS_GEN_DIR}/kudnn_lnorm_merged_fwd_last.cpp")
list(APPEND LNORM_KERNELS_LIST $<TARGET_OBJECTS:lnorm_merged_fwd_last>)

set(KUDNN_LNORM_BWD_CROSS_INSTANTIATIONS "")
foreach(IN_OUT_PREC ${FLOAT_SUPPORTED_PRECISIONS})
    set(KUDNN_LNORM_BWD_CROSS_INSTANTIATIONS "${KUDNN_LNORM_BWD_CROSS_INSTANTIATIONS}template void EvaluateScaleShiftCrossAxis<${COMPUTE_PRECISIONS}, ${IN_OUT_PREC}, ${IN_OUT_PREC}, ${IN_OUT_PREC}>(SizeType outerSize, const ${IN_OUT_PREC} *src, const float *mean, const float *variance, const ${IN_OUT_PREC} *diffDst, ${IN_OUT_PREC} *diffScale, ${IN_OUT_PREC} *diffShift, float *bufScale, float *bufShift, float eps, NormalizationFlags flags, SizeType axisSizeConst, SizeType innerSize) noexcept;\n")
    set(KUDNN_LNORM_BWD_CROSS_INSTANTIATIONS "${KUDNN_LNORM_BWD_CROSS_INSTANTIATIONS}template void EvaluateCrossAxis<${COMPUTE_PRECISIONS}, ${IN_OUT_PREC}, ${IN_OUT_PREC}, ${IN_OUT_PREC}>(SizeType outerSize, const ${IN_OUT_PREC} *src, const float *mean, const float *variance, const ${IN_OUT_PREC} *diffDst, const ${IN_OUT_PREC} *scale, ${IN_OUT_PREC} *diffSrc, float eps, NormalizationFlags flags, SizeType axisSize, SizeType innerSize) noexcept;\n")
endforeach()
configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/lnorm/kudnn_lnorm_merged_bwd_cross.cpp.in"
    "${KERNS_GEN_DIR}/kudnn_lnorm_merged_bwd_cross.cpp" @ONLY)
add_library(lnorm_merged_bwd_cross OBJECT "${KERNS_GEN_DIR}/kudnn_lnorm_merged_bwd_cross.cpp")
list(APPEND LNORM_KERNELS_LIST $<TARGET_OBJECTS:lnorm_merged_bwd_cross>)

set(KUDNN_LNORM_BWD_LAST_INSTANTIATIONS "")
foreach(IN_OUT_PREC ${FLOAT_SUPPORTED_PRECISIONS})
    set(KUDNN_LNORM_BWD_LAST_INSTANTIATIONS "${KUDNN_LNORM_BWD_LAST_INSTANTIATIONS}template void EvaluateScaleShiftLastAxis<${COMPUTE_PRECISIONS}, ${IN_OUT_PREC}, ${IN_OUT_PREC}, ${IN_OUT_PREC}>(SizeType outerSize, const ${IN_OUT_PREC} *src, const float *mean, const float *variance, const ${IN_OUT_PREC} *diffDst, ${IN_OUT_PREC} *diffScale, ${IN_OUT_PREC} *diffShift, float *bufScale, float *bufShift, float eps, NormalizationFlags flags, SizeType axisSize, SizeType axisSizeConst) noexcept;\n")
    set(KUDNN_LNORM_BWD_LAST_INSTANTIATIONS "${KUDNN_LNORM_BWD_LAST_INSTANTIATIONS}template void EvaluateLastAxis<${COMPUTE_PRECISIONS}, ${IN_OUT_PREC}, ${IN_OUT_PREC}, ${IN_OUT_PREC}>(SizeType outerSize, const ${IN_OUT_PREC} *src, const float *mean, const float *variance, const ${IN_OUT_PREC} *diffDst, const ${IN_OUT_PREC} *scale, ${IN_OUT_PREC} *diffSrc, float eps, NormalizationFlags flags, SizeType axisSize) noexcept;\n")
    set(KUDNN_LNORM_BWD_LAST_INSTANTIATIONS "${KUDNN_LNORM_BWD_LAST_INSTANTIATIONS}template void EvaluateRewriteScaleShift<${COMPUTE_PRECISIONS}, ${IN_OUT_PREC}, ${IN_OUT_PREC}, ${IN_OUT_PREC}>( ${IN_OUT_PREC} *diffScale, ${IN_OUT_PREC} *diffShift, float *bufScale, float *bufShift, NormalizationFlags flags, SizeType axisSize, SizeType numThreads) noexcept;\n")
endforeach()
configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/lnorm/kudnn_lnorm_merged_bwd_last.cpp.in"
    "${KERNS_GEN_DIR}/kudnn_lnorm_merged_bwd_last.cpp" @ONLY)
add_library(lnorm_merged_bwd_last OBJECT "${KERNS_GEN_DIR}/kudnn_lnorm_merged_bwd_last.cpp")
list(APPEND LNORM_KERNELS_LIST $<TARGET_OBJECTS:lnorm_merged_bwd_last>)

set(LNORM_KERNELS_LIST "${LNORM_KERNELS_LIST}" CACHE INTERNAL "")

# ----------------------------------------------------------------------------
# RMSNorm — 6 OBJECT → 2 OBJECT (fwd_cross / fwd_last merged)
# ----------------------------------------------------------------------------
set(RMSNORM_KERNELS_LIST "")

set(KUDNN_RMSNORM_FWD_CROSS_INSTANTIATIONS "")
foreach(IN_OUT_PREC ${ALL_SUPPORTED_PRECISIONS})
    set(KUDNN_RMSNORM_FWD_CROSS_INSTANTIATIONS "${KUDNN_RMSNORM_FWD_CROSS_INSTANTIATIONS}template void EvaluateCrossAxis<${COMPUTE_PRECISIONS}, ${IN_OUT_PREC}, ${IN_OUT_PREC}, ${IN_OUT_PREC}>(SizeType outerSize, const ${IN_OUT_PREC} *src, ${IN_OUT_PREC} *dst, const ${IN_OUT_PREC} *scale, float *rms, bool saveStats, float eps, NormalizationFlags flags, SizeType axisSize, SizeType innerSize) noexcept;\n")
endforeach()
configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/rmsnorm/kudnn_rmsnorm_merged_fwd_cross.cpp.in"
    "${KERNS_GEN_DIR}/kudnn_rmsnorm_merged_fwd_cross.cpp" @ONLY)
add_library(rmsnorm_merged_fwd_cross OBJECT "${KERNS_GEN_DIR}/kudnn_rmsnorm_merged_fwd_cross.cpp")
list(APPEND RMSNORM_KERNELS_LIST $<TARGET_OBJECTS:rmsnorm_merged_fwd_cross>)

set(KUDNN_RMSNORM_FWD_LAST_INSTANTIATIONS "")
foreach(IN_OUT_PREC ${ALL_SUPPORTED_PRECISIONS})
    set(KUDNN_RMSNORM_FWD_LAST_INSTANTIATIONS "${KUDNN_RMSNORM_FWD_LAST_INSTANTIATIONS}template void EvaluateLastAxis<${COMPUTE_PRECISIONS}, ${IN_OUT_PREC}, ${IN_OUT_PREC}, ${IN_OUT_PREC}>(SizeType execAcrossAxis, const ${IN_OUT_PREC} *src, ${IN_OUT_PREC} *dst, const ${IN_OUT_PREC} *scale, float *rms, bool saveStats, float eps, NormalizationFlags flags, SizeType axisSize) noexcept;\n")
endforeach()
configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/rmsnorm/kudnn_rmsnorm_merged_fwd_last.cpp.in"
    "${KERNS_GEN_DIR}/kudnn_rmsnorm_merged_fwd_last.cpp" @ONLY)
add_library(rmsnorm_merged_fwd_last OBJECT "${KERNS_GEN_DIR}/kudnn_rmsnorm_merged_fwd_last.cpp")
list(APPEND RMSNORM_KERNELS_LIST $<TARGET_OBJECTS:rmsnorm_merged_fwd_last>)

set(RMSNORM_KERNELS_LIST "${RMSNORM_KERNELS_LIST}" CACHE INTERNAL "")

# ----------------------------------------------------------------------------
# Softmax — ~11 OBJECT → 4 OBJECT (fwd_cross / fwd_last / bwd_cross / bwd_last)
# ----------------------------------------------------------------------------
set(SOFTMAX_KERNELS_LIST "")

set(KUDNN_SOFTMAX_FWD_CROSS_INSTANTIATIONS "")
foreach(IN_OUT_PREC ${ALL_SUPPORTED_PRECISIONS})
    set(KUDNN_SOFTMAX_FWD_CROSS_INSTANTIATIONS "${KUDNN_SOFTMAX_FWD_CROSS_INSTANTIATIONS}template void EvaluateCrossAxisFWD<${COMPUTE_PRECISIONS}, ${IN_OUT_PREC}, ${IN_OUT_PREC}>(const ${IN_OUT_PREC} *src, ${IN_OUT_PREC} *dst, SizeType innerSize, SizeType axisSize, SizeType outerSize, SoftmaxAlgorithmKind algorithm) noexcept;\n")
    if(IN_OUT_PREC STREQUAL "__fp16")
        set(KUDNN_SOFTMAX_FWD_CROSS_INSTANTIATIONS "${KUDNN_SOFTMAX_FWD_CROSS_INSTANTIATIONS}template void EvaluateCrossAxisFWD<__fp16, ${IN_OUT_PREC}, ${IN_OUT_PREC}>(const ${IN_OUT_PREC} *src, ${IN_OUT_PREC} *dst, SizeType innerSize, SizeType axisSize, SizeType outerSize, SoftmaxAlgorithmKind algorithm) noexcept;\n")
    endif()
endforeach()
configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/softmax/kudnn_softmax_merged_fwd_cross.cpp.in"
    "${KERNS_GEN_DIR}/kudnn_softmax_merged_fwd_cross.cpp" @ONLY)
add_library(softmax_merged_fwd_cross OBJECT "${KERNS_GEN_DIR}/kudnn_softmax_merged_fwd_cross.cpp")
list(APPEND SOFTMAX_KERNELS_LIST $<TARGET_OBJECTS:softmax_merged_fwd_cross>)

set(KUDNN_SOFTMAX_FWD_LAST_INSTANTIATIONS "")
foreach(IN_OUT_PREC ${ALL_SUPPORTED_PRECISIONS})
    set(KUDNN_SOFTMAX_FWD_LAST_INSTANTIATIONS "${KUDNN_SOFTMAX_FWD_LAST_INSTANTIATIONS}template void EvaluateLastAxisFWD<${COMPUTE_PRECISIONS}, ${IN_OUT_PREC}, ${IN_OUT_PREC}>(const ${IN_OUT_PREC} *src, ${IN_OUT_PREC} *dst, SizeType axisSize, SizeType outerSize, SoftmaxAlgorithmKind algorithm) noexcept;\n")
    if(IN_OUT_PREC STREQUAL "__fp16")
        set(KUDNN_SOFTMAX_FWD_LAST_INSTANTIATIONS "${KUDNN_SOFTMAX_FWD_LAST_INSTANTIATIONS}template void EvaluateLastAxisFWD<__fp16, ${IN_OUT_PREC}, ${IN_OUT_PREC}>(const ${IN_OUT_PREC} *src, ${IN_OUT_PREC} *dst, SizeType axisSize, SizeType outerSize, SoftmaxAlgorithmKind algorithm) noexcept;\n")
    endif()
endforeach()
configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/softmax/kudnn_softmax_merged_fwd_last.cpp.in"
    "${KERNS_GEN_DIR}/kudnn_softmax_merged_fwd_last.cpp" @ONLY)
add_library(softmax_merged_fwd_last OBJECT "${KERNS_GEN_DIR}/kudnn_softmax_merged_fwd_last.cpp")
list(APPEND SOFTMAX_KERNELS_LIST $<TARGET_OBJECTS:softmax_merged_fwd_last>)

set(KUDNN_SOFTMAX_BWD_CROSS_INSTANTIATIONS "")
foreach(IN_OUT_PREC ${ALL_SUPPORTED_PRECISIONS})
    set(KUDNN_SOFTMAX_BWD_CROSS_INSTANTIATIONS "${KUDNN_SOFTMAX_BWD_CROSS_INSTANTIATIONS}template void EvaluateCrossAxisBWD<${COMPUTE_PRECISIONS}, ${IN_OUT_PREC}, ${IN_OUT_PREC}>(const ${IN_OUT_PREC} *dst, const ${IN_OUT_PREC} *dstDiff, ${IN_OUT_PREC} *srcDiff, SizeType innerSize, SizeType axisSize, SizeType outerSize, SoftmaxAlgorithmKind algorithm) noexcept;\n")
    if(IN_OUT_PREC STREQUAL "__fp16")
        set(KUDNN_SOFTMAX_BWD_CROSS_INSTANTIATIONS "${KUDNN_SOFTMAX_BWD_CROSS_INSTANTIATIONS}template void EvaluateCrossAxisBWD<__fp16, ${IN_OUT_PREC}, ${IN_OUT_PREC}>(const ${IN_OUT_PREC} *dst, const ${IN_OUT_PREC} *dstDiff, ${IN_OUT_PREC} *srcDiff, SizeType innerSize, SizeType axisSize, SizeType outerSize, SoftmaxAlgorithmKind algorithm) noexcept;\n")
    endif()
endforeach()
configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/softmax/kudnn_softmax_merged_bwd_cross.cpp.in"
    "${KERNS_GEN_DIR}/kudnn_softmax_merged_bwd_cross.cpp" @ONLY)
add_library(softmax_merged_bwd_cross OBJECT "${KERNS_GEN_DIR}/kudnn_softmax_merged_bwd_cross.cpp")
list(APPEND SOFTMAX_KERNELS_LIST $<TARGET_OBJECTS:softmax_merged_bwd_cross>)

set(KUDNN_SOFTMAX_BWD_LAST_INSTANTIATIONS "")
foreach(IN_OUT_PREC ${ALL_SUPPORTED_PRECISIONS})
    set(KUDNN_SOFTMAX_BWD_LAST_INSTANTIATIONS "${KUDNN_SOFTMAX_BWD_LAST_INSTANTIATIONS}template void EvaluateLastAxisBWD<${COMPUTE_PRECISIONS}, ${IN_OUT_PREC}, ${IN_OUT_PREC}>(const ${IN_OUT_PREC} *dst, const ${IN_OUT_PREC} *dstDiff, ${IN_OUT_PREC} *srcDiff, SizeType innerSize, SizeType axisSize, SizeType outerSize, SoftmaxAlgorithmKind algorithm) noexcept;\n")
    if(IN_OUT_PREC STREQUAL "__fp16")
        set(KUDNN_SOFTMAX_BWD_LAST_INSTANTIATIONS "${KUDNN_SOFTMAX_BWD_LAST_INSTANTIATIONS}template void EvaluateLastAxisBWD<__fp16, ${IN_OUT_PREC}, ${IN_OUT_PREC}>(const ${IN_OUT_PREC} *dst, const ${IN_OUT_PREC} *dstDiff, ${IN_OUT_PREC} *srcDiff, SizeType innerSize, SizeType axisSize, SizeType outerSize, SoftmaxAlgorithmKind algorithm) noexcept;\n")
    endif()
endforeach()
configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/softmax/kudnn_softmax_merged_bwd_last.cpp.in"
    "${KERNS_GEN_DIR}/kudnn_softmax_merged_bwd_last.cpp" @ONLY)
add_library(softmax_merged_bwd_last OBJECT "${KERNS_GEN_DIR}/kudnn_softmax_merged_bwd_last.cpp")
list(APPEND SOFTMAX_KERNELS_LIST $<TARGET_OBJECTS:softmax_merged_bwd_last>)

set(SOFTMAX_KERNELS_LIST "${SOFTMAX_KERNELS_LIST}" CACHE INTERNAL "")

add_subdirectory(vml)
aux_source_directory(${CMAKE_CURRENT_LIST_DIR} KERNELS_LIST)
add_library(kernels OBJECT ${KERNELS_LIST})
if(ENABLE_GCOV STREQUAL "on" AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(kernels PRIVATE -fprofile-instr-generate -fcoverage-mapping)
endif()
