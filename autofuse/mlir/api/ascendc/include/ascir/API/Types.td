/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef API_TYPES_TD
#define API_TYPES_TD

// Ascend C API type
class APIType<string mlirTypeName = ""> {
  // IR mnemonic
  string mnemonic = ?;

  // Ascend C API class name
  string apiName = ?;

  // IR type class name
  string typeName = mlirTypeName;

  // Generate MLIR type definition for this record
  bit genTypedef = !not(!empty(mlirTypeName));

  // Generate MLIR type emitter for this record
  bit genEmitter = !not(!empty(mlirTypeName));
}

def AippChannelPaddingParams : APIType<"AippChannelPaddingParams"> {
  let mnemonic = "aipp_cpadding_params";
  let apiName = "AscendC::AippChannelPaddingParams";
}

def AippColorSpaceConvParams : APIType<"AippColorSpaceConvParams"> {
  let mnemonic = "aipp_csc_params";
  let apiName = "AscendC::AippColorSpaceConvParams";
}

def AippDataTypeConvParams : APIType<"AippDataTypeConvParams"> {
  let mnemonic = "aipp_dtc_params";
  let apiName = "AscendC::AippDataTypeConvParams";
}

def AippInputFormat : APIType<"AippInputFormat"> {
  let mnemonic = "aipp_input_format";
  let apiName = "AscendC::AippInputFormat";
}

def AippPaddingParams : APIType<"AippPaddingParams"> {
  let mnemonic = "aipp_padding_params";
  let apiName = "AscendC::AippPaddingParams";
}

def AippParams : APIType<"AippParams"> {
  let mnemonic = "aipp_params";
  let apiName = "AscendC::AippParams";
}

def AippSingleLineParams : APIType<"AippSingleLineParams"> {
  let mnemonic = "aipp_single_line_params";
  let apiName = "AscendC::AippSingleLineParams";
}

def AippSwapParams : APIType<"AippSwapParams"> {
  let mnemonic = "aipp_swap_params";
  let apiName = "AscendC::AippSwapParams";
}

def AscendQuantConfig : APIType<"AscendQuantConfig"> {
  let mnemonic = "ascend_quant_config";
  let apiName = "AscendC::AscendQuantConfig";
}

def BaseGlobalTensor : APIType {
  let mnemonic = "base_global_tensor";
  let apiName = "AscendC::BaseGlobalTensor";
}

def BaseLocalTensor : APIType {
  let mnemonic = "base_local_tensor";
  let apiName = "AscendC::BaseLocalTensor";
}

def BatchMode : APIType<"BatchMode"> {
  let mnemonic = "batch_mode";
  let apiName = "BatchMode";
}

def BatchOutMode : APIType<"BatchOutMode"> {
  let mnemonic = "batch_out_mode";
  let apiName = "BatchOutMode";
}

def BinaryRepeatParams : APIType<"BinaryRepeatParams"> {
  let mnemonic = "binary_repeat_params";
  let apiName = "AscendC::BinaryRepeatParams";
}

def BlockMode : APIType<"BlockMode"> {
  let mnemonic = "block_mode";
  let apiName = "AscendC::BlockMode";
}

def BrcbRepeatParams : APIType<"BrcbRepeatParams"> {
  let mnemonic = "brcb_repeat_params";
  let apiName = "AscendC::BrcbRepeatParams";
}

def CheckLocalMemoryIAParam : APIType<"CheckLocalMemoryIAParam"> {
  let mnemonic = "check_local_memory_ia_param";
  let apiName = "AscendC::CheckLocalMemoryIAParam";
}

def CopyRepeatParams : APIType<"CopyRepeatParams"> {
  let mnemonic = "copy_repeat_params";
  let apiName = "AscendC::CopyRepeatParams";
}

def CO2Layout : APIType<"CO2Layout"> {
  let mnemonic = "co2_layout";
  let apiName = "AscendC::CO2Layout";
}

def DataCopyCO12DstParams : APIType<"DataCopyCO12DstParams"> {
  let mnemonic = "data_copy_co12dst_params";
  let apiName = "AscendC::DataCopyCO12DstParams";
}

def DataCopyEnhancedParams : APIType<"DataCopyEnhancedParams"> {
  let mnemonic = "data_copy_enhanced_params";
  let apiName = "AscendC::DataCopyEnhancedParams";
}

def DataCopyParams : APIType<"DataCopyParams"> {
  let mnemonic = "data_copy_params";
  let apiName = "AscendC::DataCopyParams";
}

def DataCopyExtParams : APIType<"DataCopyExtParams"> {
  let mnemonic = "data_copy_ext_params";
  let apiName = "AscendC::DataCopyExtParams";
}

def DataCopyPadExtParams : APIType {
  let mnemonic = "data_copy_pad_ext_params";
  let apiName = "AscendC::DataCopyPadExtParams";
}

def DataCopyPadParams : APIType<"DataCopyPadParams"> {
  let mnemonic = "data_copy_pad_params";
  let apiName = "AscendC::DataCopyPadParams";
}

def DataFormat : APIType<"DataFormat"> {
  let mnemonic = "data_format";
  let apiName = "AscendC::DataFormat";
}

def DeqScale : APIType<"DeqScale"> {
  let mnemonic = "deq_scale";
  let apiName = "AscendC::DeqScale";
}

def Dn2NzParams : APIType<"Dn2NzParams"> {
  let mnemonic = "dn2nz_params";
  let apiName = "AscendC::Dn2NzParams";
}

def FixpipeConfig : APIType<"FixpipeConfig"> {
  let mnemonic = "fixpipe_config";
  let apiName = "AscendC::FixpipeConfig";
}

def FixpipeParamsV220 : APIType<"FixpipeParamsV220"> {
  let mnemonic = "fixpipe_params_v220";
  let apiName = "AscendC::FixpipeParamsV220";
}

def FixpipeParams : APIType {
  let mnemonic = "fixpipe_params";
  let apiName = "AscendC::FixpipeParams";
}

def FmatrixMode : APIType<"FmatrixMode"> {
  let mnemonic = "fmatrix_mode";
  let apiName = "AscendC::FmatrixMode";
}

def GatherMaskParams : APIType<"GatherMaskParams"> {
  let mnemonic = "gather_mask_params";
  let apiName = "AscendC::GatherMaskParams";
}

def GatherRepeatParams : APIType<"GatherRepeatParams"> {
  let mnemonic = "gather_repeat_params";
  let apiName = "AscendC::GatherRepeatParams";
}

def GlobalTensor : APIType {
  let mnemonic = "global_tensor";
  let apiName = "AscendC::GlobalTensor";
}

def InitConstValueParams : APIType<"InitConstValueParams"> {
  let mnemonic = "init_const_value_params";
  let apiName = "AscendC::InitConstValueParams";
}

def IsResetLoad3dConfig : APIType<"IsResetLoad3dConfig"> {
  let mnemonic = "is_reset_load_3d_config";
  let apiName = "AscendC::IsResetLoad3dConfig";
}

def IterateMode : APIType<"IterateMode"> {
  let mnemonic = "iterate_mode";
  let apiName = "IterateMode";
}

def IterateOrder : APIType<"IterateOrder"> {
  let mnemonic = "iterate_order";
  let apiName = "IterateOrder";
}

def KfcServer : APIType<"KfcServer"> {
  let mnemonic = "kfc_server";
  let apiName = "AscendC::KfcServer";
}

def LayerNormTiling : APIType<"LayerNormTiling"> {
  let mnemonic = "layer_norm_tiling";
  let apiName = "LayerNormTiling";
}

def LayerNormPara : APIType<"LayerNormPara"> {
  let mnemonic = "layer_norm_para";
  let apiName = "AscendC::LayerNormPara";
}

def LayerNormSeparateTiling : APIType<"LayerNormSeparateTiling"> {
  let mnemonic = "layer_norm_separate_tiling";
  let apiName = "LayerNormSeparateTiling";
}

def ListTensorDesc : APIType<"ListTensorDesc"> {
  let mnemonic = "list_tensor_desc";
  let apiName = "AscendC::ListTensorDesc";
}

def LoadData2DParams : APIType<"LoadData2DParams"> {
  let mnemonic = "load_data_2d_params";
  let apiName = "AscendC::LoadData2DParams";
}

def LoadData2DParamsV2 : APIType<"LoadData2DParamsV2"> {
  let mnemonic = "load_data_2d_params_v2";
  let apiName = "AscendC::LoadData2DParamsV2";
}

def LoadData2dTransposeParams : APIType<"LoadData2dTransposeParams"> {
  let mnemonic = "load_data_2d_transpose_params";
  let apiName = "AscendC::LoadData2dTransposeParams";
}

def LoadData2dTransposeParamsV2 : APIType<"LoadData2dTransposeParamsV2"> {
  let mnemonic = "load_data_2d_transpose_params_v2";
  let apiName = "AscendC::LoadData2dTransposeParamsV2";
}

def LoadData3DParamsV1 : APIType<"LoadData3DParamsV1"> {
  let mnemonic = "load_data_3d_params_v1";
  let apiName = "AscendC::LoadData3DParamsV1";
}

def LoadData3DParamsV2 : APIType<"LoadData3DParamsV2"> {
  let mnemonic = "load_data_3d_params_v2";
  let apiName = "AscendC::LoadData3DParamsV2";
}

def LoadData3DParamsV2Pro : APIType<"LoadData3DParamsV2Pro"> {
  let mnemonic = "load_data_3d_params_v2_pro";
  let apiName = "AscendC::LoadData3DParamsV2Pro";
}

def LoadDataRepeatParam : APIType<"LoadDataRepeatParam"> {
  let mnemonic = "load_data_repeat_param";
  let apiName = "AscendC::LoadDataRepeatParam";
}

def LoadImageToLocalParams : APIType<"LoadImageToLocalParams"> {
  let mnemonic = "load_image_to_local_params";
  let apiName = "AscendC::LoadImageToLocalParams";
}

def LocalMemAllocator : APIType {
  let mnemonic = "local_mem_allocator";
  let apiName = "AscendC::LocalMemAllocator";
}

def LocalTensor : APIType {
  let mnemonic = "local_tensor";
  let apiName = "AscendC::LocalTensor";
}

def LoopModeParams : APIType<"LoopModeParams"> {
  let mnemonic = "loop_mode_params";
  let apiName = "AscendC::LoopModeParams";
}

def MaskMode : APIType<"MaskMode"> {
  let mnemonic = "mask_mode";
  let apiName = "AscendC::MaskMode";
}

def Matmul : APIType {
  let mnemonic = "matmul";
  let apiName = "matmul::Matmul";
}

def MatmulApiStaticTiling : APIType<"MatmulApiStaticTiling"> {
  let mnemonic = "matmul_api_static_tiling";
  let apiName = "MatmulApiStaticTiling";
}

def MatmulConfig : APIType<"MatmulConfig"> {
  let mnemonic = "matmul_config";
  let apiName = "MatmulConfig";
}

def MatrixOffset : APIType<"MatrixOffset"> {
  let mnemonic = "matrix_offset";
  let apiName = "MatrixOffset";
}

def MmadParams : APIType<"MmadParams"> {
  let mnemonic = "mmad_params";
  let apiName = "AscendC::MmadParams";
}

def MrgSortSrcList : APIType {
  let mnemonic = "mrg_sort_src_list";
  let apiName = "AscendC::MrgSortSrcList";
}

def MrgSort4Info : APIType<"MrgSort4Info"> {
  let mnemonic = "mrg_sort4_info";
  let apiName = "AscendC::MrgSort4Info";
}

def Nd2NzParams : APIType<"Nd2NzParams"> {
  let mnemonic = "nd2nz_params";
  let apiName = "AscendC::Nd2NzParams";
}

def Nz2NdParams : APIType<"Nz2NdParams"> {
  let mnemonic = "nz2nd_params";
  let apiName = "AscendC::Nz2NdParams";
}

def Nz2NdParamsFull : APIType<"Nz2NdParamsFull"> {
  let mnemonic = "nz2nd_params_full";
  let apiName = "AscendC::Nz2NdParamsFull";
}

def PadT : APIType<"pad_t"> {
  let mnemonic = "pad_t";
  let apiName = "AscendC::pad_t";
}

def QuantModes : APIType<"QuantModes"> {
  let mnemonic = "quant_mode_t";
  let apiName = "QuantMode_t";
}

def QuantParams : APIType<"QuantParams"> {
  let mnemonic = "quant_params";
  let apiName = "AscendC::QuantParams";
}

def ReduceOrder : APIType<"ReduceOrder"> {
  let mnemonic = "reduce_order";
  let apiName = "AscendC::ReduceOrder";
}

def RmsNormTiling : APIType<"RmsNormTiling"> {
  let mnemonic = "rmsnorm_tiling";
  let apiName = "RmsNormTiling";
}

def ScheduleType : APIType<"ScheduleType"> {
  let mnemonic = "schedule_type";
  let apiName = "ScheduleType";
}

def ShapeInfo : APIType<"ShapeInfo"> {
  let mnemonic = "shape_info";
  let apiName = "AscendC::ShapeInfo";
}

def SliceInfo : APIType<"SliceInfo"> {
  let mnemonic = "slice_info";
  let apiName = "AscendC::SliceInfo";
}

def SoftMaxTiling : APIType<"SoftMaxTiling"> {
  let mnemonic = "softmax_tiling";
  let apiName = "SoftMaxTiling";
}

def SoftMaxShapeInfo : APIType<"SoftMaxShapeInfo"> {
  let mnemonic = "softmax_shape_info";
  let apiName = "AscendC::SoftMaxShapeInfo";
}

def SoftmaxConfig : APIType<"SoftmaxConfig"> {
  let mnemonic = "softmax_config";
  let apiName = "AscendC::SoftmaxConfig";
}

def SoftmaxMode : APIType<"SoftmaxMode"> {
  let mnemonic = "softmax_mode";
  let apiName = "AscendC::SoftmaxMode";
}

def TBuf : APIType {
  let mnemonic = "tbuf";
  let apiName = "AscendC::TBuf";
}

def TBuffAddr : APIType<"BufAddr"> {
  let mnemonic = "buf_addr";
  let apiName = "AscendC::TBuffAddr";
}

def TCubeTiling : APIType<"TCubeTiling"> {
  let mnemonic = "cube_tiling";
  let apiName = "TCubeTiling";
}

def TensorDesc : APIType<"TensorDesc"> {
  let mnemonic = "tensor_desc";
  let apiName = "AscendC::TensorDesc";
}

def TPipe : APIType<"Pipe"> {
  let mnemonic = "pipe";
  let apiName = "AscendC::TPipe";
}

def TQue : APIType {
  let mnemonic = "queue";
  let apiName = "AscendC::TQue";
}

def TQueBind : APIType {
  let mnemonic = "que_bind";
  let apiName = "AscendC::TQueBind";
}

def TransDataTo5HDParams : APIType<"TransDataTo5HDParams"> {
  let mnemonic = "trans_data_to_5hd_params";
  let apiName = "AscendC::TransDataTo5HDParams";
}

def TransposeParamsExt : APIType<"TransposeParamsExt"> {
  let mnemonic = "transpose_params_ext";
  let apiName = "AscendC::TransposeParamsExt";
}

def TransposeType : APIType<"TransposeType"> {
  let mnemonic = "transpose_type";
  let apiName = "AscendC::TransposeType";
}

def UnaryRepeatParams : APIType<"UnaryRepeatParams"> {
  let mnemonic = "unary_repeat_params";
  let apiName = "AscendC::UnaryRepeatParams";
}

def VdeqInfo : APIType<"VdeqInfo"> {
  let mnemonic = "vdeq_info";
  let apiName = "AscendC::VdeqInfo";
}

#endif // API_TYPES_TD
