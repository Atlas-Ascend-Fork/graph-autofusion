/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_CORE_ATTRIBUTES_TD
#define ASC_CORE_ATTRIBUTES_TD

include "Dialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

class AscendC_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<AscendC_Dialect, name, traits> {
  let cppNamespace = "::mlir::ascendc";
  let mnemonic = attrMnemonic;
}

def AscendC_AddressSpaceAttr : I32EnumAttr<"AddressSpace", "", [
  I32EnumAttrCase<"Default", 0>,
  I32EnumAttrCase<"gm", 22>,
  I32EnumAttrCase<"ca", 23>,
  I32EnumAttrCase<"cb", 24>,
  I32EnumAttrCase<"cc", 25>,
  I32EnumAttrCase<"ubuf", 26>,
  I32EnumAttrCase<"cbuf", 27>,
  I32EnumAttrCase<"fbuf", 28>,
]> {
  let cppNamespace = "::mlir::ascendc";
  let description = "CCE-C address space";
  let underlyingType = "uint8_t";
}

def AscendC_AippInputFormatAttr : I32EnumAttr<"AippInputFormat", "", [
    I32EnumAttrCase<"YUV420SP_U8", 0>,
    I32EnumAttrCase<"XRGB8888_U8", 1>,
    I32EnumAttrCase<"RGB888_U8", 4>,
    I32EnumAttrCase<"YUV400_U8", 9>
]> {
  let cppNamespace = "::mlir::ascendc";
  let underlyingType = "uint8_t";
}

def AscendC_APIAttr : AscendC_Attr<"API", "api"> {
  let parameters = (ins OptionalParameter<"StringAttr">:$group,
                        "StringAttr":$name,
                        OptionalParameter<"StringAttr">:$pipe_in,
                        OptionalParameter<"StringAttr">:$pipe_out,
                        OptionalParameter<"ArrayAttr">:$args);
  let assemblyFormat = "`<` struct(params) `>`";
  let builders = [
    AttrBuilderWithInferredContext<(ins "StringAttr":$group,
                                        "StringAttr":$name), [{
      return $_get(group.getContext(), group, name, StringAttr{},
                   StringAttr{}, ArrayAttr{});
    }]>,
  ];
}

def AscendC_BatchModeAttr : I32EnumAttr<"BatchMode", "", [
  I32EnumAttrCase<"NONE", 0>,
  I32EnumAttrCase<"BATCH_LESS_THAN_L1", 1>,
  I32EnumAttrCase<"BATCH_LARGE_THAN_L1", 2>,
  I32EnumAttrCase<"SINGLE_LARGE_THAN_L1", 3>,
]>{
  let cppNamespace = "::mlir::ascendc";
  let description = "Represents AscendC::BatchMode";
  let underlyingType = "uint8_t";
}

def AscendC_BatchOutModeAttr : I32EnumAttr<"BatchOutMode", "", [
  I32EnumAttrCase<"SINGLE_BATCH", 0>,
  I32EnumAttrCase<"MULTI_BATCH", 1>,
  I32EnumAttrCase<"DYNAMIC", 2>,
]>{
  let cppNamespace = "::mlir::ascendc";
  let description = "Represents AscendC::BatchOutMode";
  let underlyingType = "uint8_t";
}

def AscendC_CacheModeAttr : I32EnumAttr<"CacheMode", "", [
  I32EnumAttrCase<"CACHE_MODE_DISABLE", 0>,
  I32EnumAttrCase<"CACHE_MODE_NORMAL", 1>,
]> {
  let cppNamespace = "::mlir::ascendc";
  let description = "Represents AscendC::CacheMode";
  let underlyingType = "uint8_t";
}

def AscendC_CacheRwModeAttr : I32EnumAttr<"CacheRwMode", "", [
  I32EnumAttrCase<"READ", 1>,
  I32EnumAttrCase<"WRITE", 2>,
  I32EnumAttrCase<"RW", 3>,
]> {
  let cppNamespace = "::mlir::ascendc";
  let description = "Represents AscendC::CacheRwMode";
  let underlyingType = "uint8_t";
}

def AscendC_CMPMODEAttr : I64EnumAttr<"CMPMODE", "", [
  I64EnumAttrCase<"LT", 0>,
  I64EnumAttrCase<"GT", 1>,
  I64EnumAttrCase<"EQ", 2>,
  I64EnumAttrCase<"LE", 3>,
  I64EnumAttrCase<"GE", 4>,
  I64EnumAttrCase<"NE", 5>,
]> {
  let cppNamespace = "::mlir::ascendc";
}

def AscendC_CO2LayoutAttr : I32EnumAttr<"CO2Layout", "", [
  I32EnumAttrCase<"NZ", 0>,
  I32EnumAttrCase<"ROW_MAJOR", 1>,
  I32EnumAttrCase<"COLUMN_MAJOR", 2>,
]> {
  let cppNamespace = "::mlir::ascendc";
  let description = "Represents AscendC::CO2Layout";
  let underlyingType = "uint8_t";
}

def AscendC_CacheLineAttr : I32EnumAttr<"CacheLine", "", [
  I32EnumAttrCase<"SINGLE_CACHE_LINE", 0, "single_cache_line">,
  I32EnumAttrCase<"ENTIRE_DATA_CACHE", 1, "entire_data_cache">
]> {
  let cppNamespace = "::mlir::ascendc";
  let description = "Represents AscendC::CacheLine";
  let underlyingType = "uint8_t";
}

def AscendC_CubeFormatAttr : I32EnumAttr<"CubeFormat", "matmul tensor format", [
  I32EnumAttrCase<"ND", 0, "nd">,
  I32EnumAttrCase<"NZ", 1, "nz">,
  I32EnumAttrCase<"ZN", 2, "zn">,
  I32EnumAttrCase<"ZZ", 3, "zz">,
  I32EnumAttrCase<"NN", 4, "nn">,
  I32EnumAttrCase<"ND_ALIGN", 5, "nd_align">,
  I32EnumAttrCase<"SCALAR", 6, "scalar">,
  I32EnumAttrCase<"VECTOR", 7, "vector">,
]> {
  let cppNamespace = "::mlir::ascendc";
  let description = "Represents AscendC::CubeFormat";
  let underlyingType = "uint8_t";
}

def AscendC_DcciDstAttr : I32EnumAttr<"DcciDst", "", [
  I32EnumAttrCase<"CACHELINE_ALL", 0, "cacheline_all">,
  I32EnumAttrCase<"CACHELINE_UB", 1, "cacheline_ub">,
  I32EnumAttrCase<"CACHELINE_OUT", 2, "cacheline_out">,
  I32EnumAttrCase<"CACHELINE_ATOMIC", 3, "cacheline_atomic">
]> {
  let cppNamespace = "::mlir::ascendc";
  let description = "Represents AscendC::DcciDst";
  let underlyingType = "uint8_t";
}

def AscendC_FmatrixModeAttr : I32EnumAttr<"FmatrixMode", "", [
  I32EnumAttrCase<"FMATRIX_LEFT", 0, "fmatrix_left">,
  I32EnumAttrCase<"FMATRIX_RIGHT", 1, "fmatrix_right">,
]> {
  let cppNamespace = "::mlir::ascendc";
  let description = "Represents AscendC::FmatrixMode";
  let underlyingType = "uint8_t";
}

def AscendC_GatherMaskModeAttr : I32EnumAttr<"GatherMaskMode", "", [
  I32EnumAttrCase<"DEFAULT", 0, "default">,
]> {
  let cppNamespace = "::mlir::ascendc";
  let description = "Represents AscendC::GatherMaskMode";
  let underlyingType = "uint8_t";
}

def AscendC_HardEventAttr : I32EnumAttr<"HardEvent", "", [
  I32EnumAttrCase<"MTE2_MTE1", 0, "mte2_mte1">,
  I32EnumAttrCase<"MTE1_MTE2", 1, "mte1_mte2">,
  I32EnumAttrCase<"MTE1_M", 2, "mte1_m">,
  I32EnumAttrCase<"M_MTE1", 3, "m_mte1">,
  I32EnumAttrCase<"MTE2_V", 4, "mte2_v">,
  I32EnumAttrCase<"V_MTE2", 5, "v_mte2">,
  I32EnumAttrCase<"MTE3_V", 6, "mte3_v">,
  I32EnumAttrCase<"V_MTE3", 7, "v_mte3">,
  I32EnumAttrCase<"M_V", 8, "m_v">,
  I32EnumAttrCase<"V_M", 9, "v_m">,
  I32EnumAttrCase<"V_V", 10, "v_v">,
  I32EnumAttrCase<"MTE3_MTE1", 11, "mte3_mte1">,
  I32EnumAttrCase<"MTE1_MTE3", 12, "mte1_mte3">,
  I32EnumAttrCase<"MTE1_V", 13, "mte1_v">,
  I32EnumAttrCase<"MTE2_M", 14, "mte2_m">,
  I32EnumAttrCase<"M_MTE2", 15, "m_mte2">,
  I32EnumAttrCase<"V_MTE1", 16, "v_mte1">,
  I32EnumAttrCase<"M_FIX", 17, "m_fix">,
  I32EnumAttrCase<"FIX_M", 18, "fix_m">,
  I32EnumAttrCase<"MTE3_MTE2", 19, "mte3_mte2">,
  I32EnumAttrCase<"MTE2_MTE3", 20, "mte2_mte3">,
  I32EnumAttrCase<"S_V", 21, "s_v">,
  I32EnumAttrCase<"V_S", 22, "v_s">,
  I32EnumAttrCase<"S_MTE2", 23, "s_mte2">,
  I32EnumAttrCase<"MTE2_S", 24, "mte2_s">,
  I32EnumAttrCase<"S_MTE3", 25, "s_mte3">,
  I32EnumAttrCase<"MTE3_S", 26, "mte3_s">,
  I32EnumAttrCase<"MTE2_FIX", 27, "mte2_fix">,
  I32EnumAttrCase<"FIX_MTE2", 28, "fix_mte2">,
  I32EnumAttrCase<"FIX_S", 29, "fix_s">,
  I32EnumAttrCase<"M_S", 30, "m_s">,
  I32EnumAttrCase<"FIX_MTE3", 31, "fix_mte3">,
  I32EnumAttrCase<"MTE1_FIX", 32, "mte1_fix">,
  I32EnumAttrCase<"FIX_MTE1", 33, "fix_mte1">,
  I32EnumAttrCase<"FIX_FIX", 34, "fix_fix">,
  I32EnumAttrCase<"MAX", 35, "max">,
]> {
  let cppNamespace = "::mlir::ascendc";
  let description = "Represents AscendC::HardEvent";
  let underlyingType = "uint8_t";
}

def AscendC_HardwareAttr : I32EnumAttr<"Hardware", "Hardware memory location", [
  I32EnumAttrCase<"GM", 0, "gm">,
  I32EnumAttrCase<"UB", 1, "ub">,
  I32EnumAttrCase<"L1", 2, "l1">,
  I32EnumAttrCase<"L0A", 3, "l0a">,
  I32EnumAttrCase<"L0B", 4, "l0b">,
  I32EnumAttrCase<"L0C", 5, "l0c">,
  I32EnumAttrCase<"BIAS", 6, "bias">,
  I32EnumAttrCase<"FIXBUF", 7, "fixbuf">,
  I32EnumAttrCase<"MAX", 8, "max">,
]> {
  let cppNamespace = "::mlir::ascendc";
  let description = "Represents AscendC::Hardware";
  let underlyingType = "uint8_t";
}

def AscendC_IterateModeAttr : I32EnumAttr<"IterateMode", "", [
  I32EnumAttrCase<"ITERATE_MODE_NORMAL", 0b00000001>,
  I32EnumAttrCase<"ITERATE_MODE_ALL", 0b00000010>,
  I32EnumAttrCase<"ITERATE_MODE_BATCH", 0b00000100>,
  I32EnumAttrCase<"ITERATE_MODE_N_BATCH", 0b00001000>,
  I32EnumAttrCase<"ITERATE_MODE_DEFAULT", 0b11111111>,
]>{
  let cppNamespace = "::mlir::ascendc";
  let description = "Represents AscendC::IterateMode";
  let underlyingType = "uint8_t";
}

def AscendC_IterateOrderAttr : I32EnumAttr<"IterateOrder", "", [
  I32EnumAttrCase<"ORDER_M", 0>,
  I32EnumAttrCase<"ORDER_N", 1>,
  I32EnumAttrCase<"UNDEF", 2>,
]>{
  let cppNamespace = "::mlir::ascendc";
  let description = "Represents AscendC::IterateOrder";
  let underlyingType = "uint8_t";
}

def AscendC_LayoutModeAttr : I32EnumAttr<"LayoutMode", "matmul tensor layout", [
  I32EnumAttrCase<"NONE", 0, "none">,
  I32EnumAttrCase<"NORMAL", 1, "normal">,
  I32EnumAttrCase<"BSNGD", 2, "bsngd">,
  I32EnumAttrCase<"SBNGD", 3, "sbngd">,
  I32EnumAttrCase<"BNGS1S2", 4, "bngs1s2">,
]> {
  let cppNamespace = "::mlir::ascendc";
  let description = "Represents AscendC::LayoutMode";
  let underlyingType = "uint8_t";
}

def AscendC_MaskModeAttr : I32EnumAttr<"MaskMode", "", [
  I32EnumAttrCase<"NORMAL", 0>,
  I32EnumAttrCase<"COUNTER", 1>,
]> {
  let cppNamespace = "::mlir::ascendc";
  let description = "Represents AscendC::MaskMode";
  let underlyingType = "uint8_t";
}

def AscendC_MatmulConfigAttr : AscendC_Attr<"MatmulConfig", "matmul_config">{
  let parameters = (ins OptionalParameter<"BoolAttr">:$do_norm,
                        OptionalParameter<"BoolAttr">:$do_basic_block,
                        OptionalParameter<"BoolAttr">:$do_multi_data_load,
                        OptionalParameter<"IntegerAttr">:$basic_m,
                        OptionalParameter<"IntegerAttr">:$basic_n,
                        OptionalParameter<"IntegerAttr">:$basic_k,
                        OptionalParameter<"BoolAttr">:$intrinsics_check,
                        OptionalParameter<"BoolAttr">:$is_n_batch,
                        OptionalParameter<"BoolAttr">:$en_vec_nd2nz,
                        OptionalParameter<"BoolAttr">:$do_special_basic_block,
                        OptionalParameter<"IntegerAttr">:$do_mte2_preload,
                        OptionalParameter<"IntegerAttr">:$single_core_m,
                        OptionalParameter<"IntegerAttr">:$single_core_n,
                        OptionalParameter<"IntegerAttr">:$single_core_k,
                        OptionalParameter<"IntegerAttr">:$step_m,
                        OptionalParameter<"IntegerAttr">:$step_n,
                        OptionalParameter<"IntegerAttr">:$base_mn,
                        OptionalParameter<"IntegerAttr">:$single_core_mn,
                        OptionalParameter<"BoolAttr">:$en_unit_flag,
                        OptionalParameter<"BoolAttr">:$is_per_tensor,
                        OptionalParameter<"BoolAttr">:$has_anti_quant_offset,
                        OptionalParameter<"BoolAttr">:$do_ib_share_norm,
                        OptionalParameter<"BoolAttr">:$do_special_mdl,
                        OptionalParameter<"BoolAttr">:$enable_init,
                        OptionalParameter<"IntegerAttr">:$batch_mode,
                        OptionalParameter<"BoolAttr">:$enable_end,
                        OptionalParameter<"BoolAttr">:$enable_get_tensor_c,
                        OptionalParameter<"BoolAttr">:$enable_set_org_shape,
                        OptionalParameter<"BoolAttr">:$enable_set_bias,
                        OptionalParameter<"BoolAttr">:$enable_set_tail,
                        OptionalParameter<"BoolAttr">:$enable_quant_vector,
                        OptionalParameter<"BoolAttr">:$enable_set_define_data,
                        OptionalParameter<"IntegerAttr">:$iterate_mode,
                        OptionalParameter<"BoolAttr">:$enable_reuse,
                        OptionalParameter<"BoolAttr">:$enable_ub_reuse,
                        OptionalParameter<"BoolAttr">:$enable_l1_cache_ub,
                        OptionalParameter<"BoolAttr">:$intra_block_part_sum,
                        OptionalParameter<"IntegerAttr">:$iterate_order,
                        OptionalParameter<"IntegerAttr">:$schedule_type,
                        OptionalParameter<"BoolAttr">:$enable_double_cache,
                        OptionalParameter<"BoolAttr">:$is_bias_batch,
                        OptionalParameter<"BoolAttr">:$enable_static_pad_zeros,
                        OptionalParameter<"BoolAttr">:$is_partial_output,
                        OptionalParameter<"BoolAttr">:$enable_mix_dual_master,
                        OptionalParameter<"BoolAttr">:$is_a2b2_shared,
                        OptionalParameter<"BoolAttr">:$is_enable_channel_split,
                        OptionalParameter<"BoolAttr">:$enable_kdim_reorder_load,
                        OptionalParameter<"BoolAttr">:$is_co1_shared,
                        OptionalParameter<"IntegerAttr">:$shared_co1_buffer_size,
                        OptionalParameter<"IntegerAttr">:$batch_out_mode);
  let assemblyFormat = "`<` struct(params) `>`";
}

def AscendC_MaskPattern : I32EnumAttr<"MaskPattern", "", [
  I32EnumAttrCase<"ALL", 0>,
  I32EnumAttrCase<"VL1", 1>,
  I32EnumAttrCase<"VL2", 2>,
  I32EnumAttrCase<"VL3", 3>,
  I32EnumAttrCase<"VL4", 4>,
  I32EnumAttrCase<"VL8", 5>,
  I32EnumAttrCase<"VL16", 6>,
  I32EnumAttrCase<"VL32", 7>,
  I32EnumAttrCase<"VL64", 8>,
  I32EnumAttrCase<"VL128", 9>,
  I32EnumAttrCase<"M3", 10>,
  I32EnumAttrCase<"M4", 11>,
  I32EnumAttrCase<"H", 12>,
  I32EnumAttrCase<"Q", 13>,
  I32EnumAttrCase<"ALLF", 15>,
]>{
  let cppNamespace = "::mlir::ascendc";
  let description = "Represents AscendC::Reg::MaskPattern";
  let underlyingType = "uint8_t";
}

def AscendC_LoadDist : I32EnumAttr<"LoadDist", "", [
    I32EnumAttrCase<"DIST_NORM",  0>,
    I32EnumAttrCase<"DIST_BRC_B8", 1>,
    I32EnumAttrCase<"DIST_BRC_B16", 2>,
    I32EnumAttrCase<"DIST_BRC_B32", 3>,
    I32EnumAttrCase<"DIST_US_B8", 6>,
    I32EnumAttrCase<"DIST_US_B16", 7>,
    I32EnumAttrCase<"DIST_DS_B8", 8>,
    I32EnumAttrCase<"DIST_DS_B16", 9>,
    I32EnumAttrCase<"DIST_BDINTLV", 10>,
    I32EnumAttrCase<"DIST_DINTLV_B8", 11>,
    I32EnumAttrCase<"DIST_DINTLV_B16", 12>,
    I32EnumAttrCase<"DIST_UNPACK_B8", 13>,
    I32EnumAttrCase<"DIST_UNPACK_B16", 14>,
    I32EnumAttrCase<"DIST_BLK", 15>,
    I32EnumAttrCase<"DIST_E2B_B16", 16>,
    I32EnumAttrCase<"DIST_E2B_B32", 17>,
    I32EnumAttrCase<"DIST_UNPACK_B32", 18>,
    I32EnumAttrCase<"DIST_DINTLV_B32", 19>,
    I32EnumAttrCase<"DIST_UNPACK4_B8", 20>,
    I32EnumAttrCase<"DIST_SPLT4CHN_B8", 21>,
    I32EnumAttrCase<"DIST_SPLT2CHN_B8", 22>,
    I32EnumAttrCase<"DIST_SPLT2CHN_B16", 23>,
]>{
  let cppNamespace = "::mlir::ascendc";
  let description = "Represents AscendC::Reg::LoadDist";
  let underlyingType = "uint8_t";
}

def AscendC_StoreDist : I32EnumAttr<"StoreDist", "", [
    I32EnumAttrCase<"DIST_NORM_B8", 0>,
    I32EnumAttrCase<"DIST_NORM_B16", 1>,
    I32EnumAttrCase<"DIST_NORM_B32", 2>,
    I32EnumAttrCase<"DIST_FIRST_ELEMENT_B8", 3>,
    I32EnumAttrCase<"DIST_FIRST_ELEMENT_B16", 4>,
    I32EnumAttrCase<"DIST_FIRST_ELEMENT_B32", 5>,
    I32EnumAttrCase<"DIST_PACK_B16", 6>,
    I32EnumAttrCase<"DIST_PACK_B32", 7>,
    I32EnumAttrCase<"DIST_INTLV_B8", 8>,
    I32EnumAttrCase<"DIST_INTLV_B16", 9>,
    I32EnumAttrCase<"DIST_PACK_B64", 10>,
    I32EnumAttrCase<"DIST_INTLV_B32", 11>,
    I32EnumAttrCase<"DIST_PACK4_B32", 12>,
    I32EnumAttrCase<"DIST_MRG4CHN_B8", 13>,
    I32EnumAttrCase<"DIST_MRG2CHN_B8", 14>,
    I32EnumAttrCase<"DIST_MRG2CHN_B16", 15>,
    I32EnumAttrCase<"DIST_NORM", 16>,
]>{
  let cppNamespace = "::mlir::ascendc";
  let description = "Represents AscendC::Reg::StoreDist";
  let underlyingType = "uint8_t";
}

def AscendC_MatmulPolicyAttr : I32EnumAttr<"MatmulPolicy", "", [
  I32EnumAttrCase<"MatmulPolicy", 0>,
  I32EnumAttrCase<"TrianUpperMatmulPolicy", 1>,
  I32EnumAttrCase<"TrianLowerMatmulPolicy", 2>,
  I32EnumAttrCase<"NBuffer33MatmulPolicy", 3>,
]>{
  let cppNamespace = "::Detail::Impl::AscendC";
  let description = "Represents AscendC::MatmulPolicy";
  let underlyingType = "uint8_t";
}

def AscendC_MemType : I32EnumAttr<"MemType", "", [
  I32EnumAttrCase<"VEC_STORE", 0>,
  I32EnumAttrCase<"VEC_LOAD", 1>,
  I32EnumAttrCase<"SCALAR_STORE", 2>,
  I32EnumAttrCase<"SCALAR_LOAD", 3>,
  I32EnumAttrCase<"VEC_ALL", 4>,
  I32EnumAttrCase<"SCALAR_ALL", 5>,
]>{
  let cppNamespace = "::mlir::ascendc";
  let description = "Represents AscendC::Reg::MemType";
  let underlyingType = "uint8_t";
}

def AscendC_PipeAttr : I32EnumAttr<"Pipe", "", [
  I32EnumAttrCase<"PIPE_S", 0, "pipe_s">,
  I32EnumAttrCase<"PIPE_V", 1, "pipe_v">,
  I32EnumAttrCase<"PIPE_M", 2, "pipe_m">,
  I32EnumAttrCase<"PIPE_MTE1", 3, "pipe_mte1">,
  I32EnumAttrCase<"PIPE_MTE2", 4, "pipe_mte2">,
  I32EnumAttrCase<"PIPE_MTE3", 5, "pipe_mte3">,
  I32EnumAttrCase<"PIPE_ALL", 6, "pipe_all">,
  I32EnumAttrCase<"PIPE_MTE4", 7, "pipe_mte4">,
  I32EnumAttrCase<"PIPE_MTE5", 8, "pipe_mte5">,
  I32EnumAttrCase<"PIPE_V2", 9, "pipe_v2">,
  I32EnumAttrCase<"PIPE_FIX", 10, "pipe_fix">,
]> {
  let cppNamespace = "::mlir::ascendc";
  let description = "Represents pipe_t";
  let underlyingType = "uint8_t";
}

def AscendC_TransposeTypeAttr : I32EnumAttr<"TransposeType", "Transpose type.", [
    I32EnumAttrCase<"TRANSPOSE_TYPE_NONE", 0>,
    I32EnumAttrCase<"TRANSPOSE_NZ2ND_0213", 1>,
    I32EnumAttrCase<"TRANSPOSE_NZ2NZ_0213", 2>,
    I32EnumAttrCase<"TRANSPOSE_NZ2NZ_012_WITH_N", 3>,
    I32EnumAttrCase<"TRANSPOSE_NZ2ND_012_WITH_N", 4>,
    I32EnumAttrCase<"TRANSPOSE_NZ2ND_012_WITHOUT_N", 5>,
    I32EnumAttrCase<"TRANSPOSE_NZ2NZ_012_WITHOUT_N", 6>,
    I32EnumAttrCase<"TRANSPOSE_ND2ND_ONLY", 7>,
    I32EnumAttrCase<"TRANSPOSE_ND_UB_GM", 8>,
    I32EnumAttrCase<"TRANSPOSE_GRAD_ND_UB_GM", 9>,
    I32EnumAttrCase<"TRANSPOSE_ND2ND_B16", 10>,
    I32EnumAttrCase<"TRANSPOSE_NCHW2NHWC", 11>,
    I32EnumAttrCase<"TRANSPOSE_NHWC2NCHW", 12>
]> {
  let cppNamespace = "::mlir::ascendc";
  let description = "Represents AscendC::TransposeType";
  let underlyingType = "uint8_t";
}

def AscendC_TPositionAttr : I32EnumAttr<"TPosition", "queue/buffer position", [
  I32EnumAttrCase<"GM", 0, "gm">,
  I32EnumAttrCase<"A1", 1, "a1">,
  I32EnumAttrCase<"A2", 2, "a2">,
  I32EnumAttrCase<"B1", 3, "b1">,
  I32EnumAttrCase<"B2", 4, "b2">,
  I32EnumAttrCase<"C1", 5, "c1">,
  I32EnumAttrCase<"C2", 6, "c2">,
  I32EnumAttrCase<"CO1", 7, "co1">,
  I32EnumAttrCase<"CO2", 8, "co2">,
  I32EnumAttrCase<"VECIN", 9, "vecin">,
  I32EnumAttrCase<"VECOUT", 10, "vecout">,
  I32EnumAttrCase<"VECCALC", 11, "veccalc">,
  I32EnumAttrCase<"MAX", 12, "max">,
]> {
  let cppNamespace = "::mlir::ascendc";
  let description = "Represents AscendC::TPosition";
  let underlyingType = "uint8_t";
}

def AscendC_QuantModeAttr : I32EnumAttr<"QuantMode", "", [
  I32EnumAttrCase<"NoQuant", 0, "no_quant">,
  I32EnumAttrCase<"F322F16", 1, "f322f16">,
  I32EnumAttrCase<"VQF322HIF8_PRE", 2, "vqf322hif8_pre">,
  I32EnumAttrCase<"QF322HIF8_PRE", 3, "qf322hif8_pre">,
  I32EnumAttrCase<"VQF322HIF8_PRE_HYBRID", 4, "vqf322hif8_pre_hybrid">,
  I32EnumAttrCase<"QF322HIF8_PRE_HYBRID", 5, "qf322hif8_pre_hybrid">,
  I32EnumAttrCase<"AttachF16Mul", 6, "attachf16mul">,
  I32EnumAttrCase<"VREQ8", 8, "vreq8">,
  I32EnumAttrCase<"REQ8", 9, "req8">,
  I32EnumAttrCase<"VDEQF16", 10, "vdeqf16">,
  I32EnumAttrCase<"DEQF16", 11, "deqf16">,
  I32EnumAttrCase<"VQF322FP8_PRE", 12, "vqf322fp8_pre">,
  I32EnumAttrCase<"QF322FP8_PRE", 13, "qf322fp8_pre">,
  I32EnumAttrCase<"VQF322F32_PRE", 14, "vqf322f32_pre">,
  I32EnumAttrCase<"QF322F32_PRE", 15, "qf322f32_pre">,
  I32EnumAttrCase<"F322BF16", 16, "f322bf16">,
  I32EnumAttrCase<"VQF162B8_PRE", 17, "vqf162b8_pre">,
  I32EnumAttrCase<"QF162B8_PRE", 18, "qf162b8_pre">,
  I32EnumAttrCase<"VQF162S4_PRE", 19, "vqf162s4_pre">,
  I32EnumAttrCase<"QF162S4_PRE", 20, "qf162s4_pre">,
  I32EnumAttrCase<"VREQ4", 21, "vreq4">,
  I32EnumAttrCase<"REQ4", 22, "req4">,
  I32EnumAttrCase<"VQF322B8_PRE", 23, "vqf322b8_pre">,
  I32EnumAttrCase<"QF322B8_PRE", 24, "qf322b8_pre">,
  I32EnumAttrCase<"VQF322S4_PRE", 25, "vqf322s4_pre">,
  I32EnumAttrCase<"QF322S4_PRE", 26, "qf322s4_pre">,
  I32EnumAttrCase<"VDEQS16", 27, "vdeqs16">,
  I32EnumAttrCase<"DEQS16", 28, "deqs16">,
  I32EnumAttrCase<"VQF162S16_PRE", 29, "vqf162s16_pre">,
  I32EnumAttrCase<"QF162S16_PRE", 30, "qf162s16_pre">,
  I32EnumAttrCase<"VQF322F16_PRE", 31, "vqf322f16_pre">,
  I32EnumAttrCase<"QF322F16_PRE", 32, "qf322f16_pre">,
  I32EnumAttrCase<"VQF322BF16_PRE", 33, "vqf322bf16_pre">,
  I32EnumAttrCase<"QF322BF16_PRE", 34, "qf322bf16_pre">,
  I32EnumAttrCase<"VQS322BF16_PRE", 35, "vqs322bf16_pre">,
  I32EnumAttrCase<"QS322BF16_PRE", 36, "qs322bf16_pre">,
]> {
  let cppNamespace = "::mlir::ascendc";
  let description = "Represents AscendC::QuantMode_t";
  let underlyingType = "uint32_t";
}

def AscendC_ReduceOrderAttr : I32EnumAttr<"ReduceOrder", "reduce result order and behavior", [
  I32EnumAttrCase<"ORDER_VALUE_INDEX", 0, "order_value_index">,
  I32EnumAttrCase<"ORDER_INDEX_VALUE", 1, "order_index_value">,
  I32EnumAttrCase<"ORDER_ONLY_VALUE", 2, "order_only_value">,
  I32EnumAttrCase<"ORDER_ONLY_INDEX", 3, "order_only_index">,
]> {
  let cppNamespace = "::mlir::ascendc";
  let description = "Represents AscendC::ReduceOrder";
  let underlyingType = "uint8_t";
}

def AscendC_RoundModeAttr : I32EnumAttr<"RoundMode", "", [
  I32EnumAttrCase<"CAST_NONE", 0>,
  I32EnumAttrCase<"CAST_RINT", 1>,
  I32EnumAttrCase<"CAST_FLOOR", 2>,
  I32EnumAttrCase<"CAST_CEIL", 3>,
  I32EnumAttrCase<"CAST_ROUND", 4>,
  I32EnumAttrCase<"CAST_TRUNC", 5>,
  I32EnumAttrCase<"CAST_ODD", 6>,
]> {
  let cppNamespace = "::mlir::ascendc";
  let description = "Represents AscendC::RoundMode";
  let underlyingType = "uint8_t";
}

def AscendC_SampleAttr : AscendC_Attr<"Sample", "sample"> {
  let parameters = (ins "IntegerAttr":$number);
  let assemblyFormat = "`<` $number `>`";
}

def AscendC_ScheduleTypeAttr : I32EnumAttr<"ScheduleType", "", [
  I32EnumAttrCase<"INNER_PRODUCT", 0>,
  I32EnumAttrCase<"OUTER_PRODUCT", 1>,
]>{
  let cppNamespace = "::mlir::ascendc";
  let description = "Represents AscendC::ScheduleType";
  let underlyingType = "uint8_t";
}

def AscendC_SelModeAttr : I32EnumAttr<"SELMODE", "", [
  I32EnumAttrCase<"VSEL_CMPMASK_SPR", 0>,
  I32EnumAttrCase<"VSEL_TENSOR_SCALAR_MODE", 1>,
  I32EnumAttrCase<"VSEL_TENSOR_TENSOR_MODE", 2>,
]> {
  let cppNamespace = "::mlir::ascendc";
  let description = "Represents AscendC::SELMODE";
  let underlyingType = "uint8_t";
}

def AscendC_MemDsbAttr : I32EnumAttr<"MemDsbT", "", [
  I32EnumAttrCase<"ALL", 0, "all">,
  I32EnumAttrCase<"DDR", 1, "ddr">,
  I32EnumAttrCase<"UB", 2, "ub">,
  I32EnumAttrCase<"SEQ", 3, "seq">
]> {
  let cppNamespace = "::mlir::ascendc";
  let description = "Represents AscendC::MemDsbT, indicating memory positions to wait for in DataSyncBarrier";
  let underlyingType = "uint8_t";
}


def AscendC_AtomicDtypeAttr : I32EnumAttr<"AtomicDtype", "", [
  I32EnumAttrCase<"ATOMIC_NONE", 0, "atomic_none">,
  I32EnumAttrCase<"ATOMIC_F32", 1, "atomic_f32">,
  I32EnumAttrCase<"ATOMIC_F16", 2, "atomic_f16">,
  I32EnumAttrCase<"ATOMIC_S16", 3, "atomic_s16">,
  I32EnumAttrCase<"ATOMIC_S32", 4, "atomic_s32">,
  I32EnumAttrCase<"ATOMIC_S8", 5, "atomic_s8">,
  I32EnumAttrCase<"ATOMIC_BF16", 6, "atomic_bf16">
]> {
  let cppNamespace = "::mlir::ascendc";
  let description = "Represents AscendC::AtomicDtype";
  let underlyingType = "uint8_t";
}

def AscendC_AtomicOpAttr : I32EnumAttr<"AtomicOp", "", [
  I32EnumAttrCase<"ATOMIC_SUM", 0, "atomic_sum">,
]> {
  let cppNamespace = "::mlir::ascendc";
  let description = "Represents AscendC::AtomicOp";
  let underlyingType = "uint8_t";
}

def AscendC_ReducePatternAttr : I32EnumAttr<"ReducePattern", "", [
  I32EnumAttrCase<"R", 0>,
  I32EnumAttrCase<"AR", 1>,
  I32EnumAttrCase<"RA", 2>,
  I32EnumAttrCase<"ARA", 3>,
  I32EnumAttrCase<"ARAR", 4>,
  I32EnumAttrCase<"ARARA", 5>,
  I32EnumAttrCase<"ARARAR", 6>,
  I32EnumAttrCase<"ARARARA", 7>,
  I32EnumAttrCase<"ARARARAR", 8>,
  I32EnumAttrCase<"ARARARARA", 9>,
  I32EnumAttrCase<"RAR", 10>,
  I32EnumAttrCase<"RARA", 11>,
  I32EnumAttrCase<"RARAR", 12>,
  I32EnumAttrCase<"RARARA", 13>,
  I32EnumAttrCase<"RARARAR", 14>,
  I32EnumAttrCase<"RARARARA", 15>,
]>{
  let cppNamespace = "::mlir::ascendc";
  let description = "Describes reduction pattern from reduce_common.h";
  let underlyingType = "uint8_t";
}

def AscendC_DataCopyMVTypeAttr: I32EnumAttr<"DataCopyMVType", "", [
    I32EnumAttrCase<"UB_TO_OUT", 0>,
    I32EnumAttrCase<"OUT_TO_UB", 1>,
]>{
  let cppNamespace = "::mlir::ascendc";
  let description = "Used in SetLoopModePara/ResetLoopModePara";
  let underlyingType = "uint8_t";
}

def AscendC_CopyDirectionAttr : AscendC_Attr<"CopyDirection", "copy_direction"> {
  let parameters = (ins "TPosition":$src, "TPosition":$dst);
  let assemblyFormat = "`<` $src `,` $dst `>`";
}

include "Autofuse/Attributes.td"

#endif // ASC_CORE_ATTRIBUTES_TD
