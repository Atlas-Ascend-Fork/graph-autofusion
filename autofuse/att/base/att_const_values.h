/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_BASE_ATT_CONST_VALUE_H_
#define ATT_BASE_ATT_CONST_VALUE_H_
#include <string>
#include <cstdint>
#include "base_types.h"

namespace att {
const int32_t kCaseDefault = 0;
const int32_t kCaseOne = 1;
const int32_t kCaseTwo = 2;
const int32_t kCaseThree = 3;
const int32_t kCaseFour = 4;
const int32_t kBaseTwo = 2;
const int32_t kMinDimLength = 1;
const auto kLoadExprThres = CreateExpr(25000U);
const auto kInitA = CreateExpr(1U);
const auto kInitB = CreateExpr(1U);
const auto kTempBufSize = CreateExpr(8192U);
const auto kBlkSize = CreateExpr(32U);
const auto kMaxRepeatTime = CreateExpr(255U);
const auto kSymOne = CreateExpr(1U);
const auto kSymTwo = CreateExpr(2U);
const auto kSymThree = CreateExpr(3U);
const auto kSymFour = CreateExpr(4U);
const auto kSymFive = CreateExpr(5U);
const auto kRptSizeHalf = CreateExpr(128);
const auto kRptSizeFloat = CreateExpr(64);
const auto kRptSizeInt64 = CreateExpr(32);
const auto kSymEight = CreateExpr(8U);
const auto kSymTen = CreateExpr(10U);
const auto kSymEleven = CreateExpr(11U);
const auto kSymFifteen = CreateExpr(15U);
const auto kSymSixteen = CreateExpr(16U);
const auto kSymPowerofSeven = CreateExpr(128U);
const auto kSymPowerofEight = CreateExpr(256U);
const auto kMte2UbStrideAlignSize = CreateExpr(512U);
const uint64_t kNumZero = 0U;
const uint64_t kNumOne = 1U;
const uint64_t kNumTwo = 2U;
const uint64_t kNumThree = 3U;
const uint64_t kNumFour = 4U;
const uint64_t kNumFive = 5U;
const uint64_t kNumSix = 6U;
const uint64_t kNumSeven = 7U;
const uint64_t kNumEight = 8U;
const uint64_t kDmaMaxLen = 2U;
const double kCompareInt64EqNeAdjustmentFactor = 1.276;
const double kCompareInt64GtGeLeAdjustmentFactor = 1.18;
const double kCompareNormalAdjustmentFactor = 1.245;

// 数据类型字符串定义
inline const std::string kInt8 = "int8";
inline const std::string kUInt8 = "uint8";
inline const std::string kInt4 = "int4";
inline const std::string kFloat16 = "float16";
inline const std::string kBfloat16 = "bfloat16";
inline const std::string kUInt16 = "uint16";
inline const std::string kInt16 = "int16";
inline const std::string kFloat32 = "float32";
inline const std::string kUInt32 = "uint32";
inline const std::string kInt32 = "int32";
inline const std::string kUInt64 = "uint64";
inline const std::string kInt64 = "int64";
inline const std::string kBool = "bool";
inline const std::string kDefault = "default";
inline const std::map<std::string, Expr> kBlkEleMap = {
    {kInt8, CreateExpr(32)},   {kUInt8, CreateExpr(32)}, {kFloat16, CreateExpr(16)}, {kBfloat16, CreateExpr(16)},
    {kUInt16, CreateExpr(16)}, {kInt16, CreateExpr(16)}, {kFloat32, CreateExpr(8)},  {kUInt32, CreateExpr(8)},
    {kInt32, CreateExpr(8)},   {kUInt64, CreateExpr(4)}, {kInt64, CreateExpr(4)},    {kBool, CreateExpr(32)},
};

inline const std::map<std::string, Expr> kDataTypeSizeMap = {
    {kInt4, CreateExpr(1)},     {kInt8, CreateExpr(1)},   {kUInt8, CreateExpr(1)},  {kFloat16, CreateExpr(2)},
    {kBfloat16, CreateExpr(2)}, {kUInt16, CreateExpr(2)}, {kInt16, CreateExpr(2)},  {kFloat32, CreateExpr(4)},
    {kUInt32, CreateExpr(4)},   {kInt32, CreateExpr(4)},  {kUInt64, CreateExpr(8)}, {kInt64, CreateExpr(8)},
    {kBool, CreateExpr(1)},
};

inline const std::map<std::string, Expr> kRptEleMap = {
    {kInt8, CreateExpr(256)},   {kUInt8, CreateExpr(256)}, {kFloat16, CreateExpr(128)}, {kBfloat16, CreateExpr(128)},
    {kUInt16, CreateExpr(128)}, {kInt16, CreateExpr(128)}, {kFloat32, CreateExpr(64)},  {kUInt32, CreateExpr(64)},
    {kInt32, CreateExpr(64)},   {kUInt64, CreateExpr(32)}, {kInt64, CreateExpr(32)},    {kBool, CreateExpr(256)},
};

inline const std::map<std::string, Expr> kBrcbRepeatMap = {
    {kInt8, CreateExpr(254U)},   {kUInt8, CreateExpr(254U)}, {kFloat16, CreateExpr(254U)}, {kFloat32, CreateExpr(255U)},
    {kUInt16, CreateExpr(254U)}, {kInt16, CreateExpr(254U)}, {kUInt32, CreateExpr(255U)},  {kInt32, CreateExpr(255U)},
    {kUInt64, CreateExpr(255U)}, {kInt64, CreateExpr(255U)}, {kBool, CreateExpr(254U)},
};

// options
const std::string kOutputFilePath = "output_file_path";
const std::string kTilingDataTypeName = "tiling_data_type_name";
const std::string kGenExtraInfo = "gen_extra_info";
const std::string kVariableReplace = "do_variable_replace";
const std::string kDumpDebugInfo = "dump_debug_info";
const std::string kGenTilingDataDef = "gen_tiling_data_def";
const std::string kDefaultFilePath = "./";
const std::string kDefaultTilingDataTypeName = "TilingData";
const std::string kIsTrue = "1";
const std::string kIsFalse = "0";
const std::string kTilingFuncIdentify = "TilingFunc";
const std::string kTilingHeadIdentify = "TilingHead";
const std::string kTilingStateHeaderIdentify = "TilingStateHeader";
const std::string kTilingLogHeaderIdentify = "TilingLogHeader";
const std::string kTilingPgoHeaderIdentify = "TilingPgoHeader";
const std::string kTilingSolverHeaderIdentify = "TilingSolverHeader";
const std::string kTilingApiHeaderIdentify = "TilingApiHeader";
const std::string kTilingSolverIdentify = "solver_func";
const std::string kTilingScheduleGroupTailIdentify = "schedule_group_tail";
const std::string kDefaultTilingDataFileName = "tiling_data.h";
const std::string kDefaultTilingHeadFileName = "autofuse_tiling_func_common.h";
const std::string kTilingStateHeaderFileName = "autofuse_tiling_func_state.h";
const std::string kTilingLogHeaderFileName = "autofuse_tiling_func_log.h";
const std::string kTilingPgoHeaderFileName = "autofuse_tiling_func_pgo.h";
const std::string kTilingSolverHeaderFileName = "autofuse_tiling_func_solver.h";
const std::string kTilingApiHeaderFileName = "autofuse_tiling_func_api.h";
const std::string kDefaultTilingFuncFileName = "tiling_func.cpp";
const std::string kHighPrecision = "high_precision";
const std::string kDurationLevelName = "duration_level";
const std::string kDurationLevelDefault = "0";
const std::string kRegisterNoDefault = "";
const std::string kGenConfigType = "solver_type";
const std::string kGenConfigTypeDefault = "UNKNOWN";

inline const std::string kVectorFunc = "VectorFunc";  // 表示VF function类型
inline const std::string kConstant = "Constant";
inline const std::string kFlashSoftmax = "FlashSoftmax";
inline const std::string kDropOut = "Dropout";
// 各pipe的描述
inline const std::string kUnitMTE1 = "UnitMTE1";
inline const std::string kUnitMTE2 = "UnitMTE2";
inline const std::string kUnitMTE3 = "UnitMTE3";
inline const std::string kUnitCube = "UnitCube";
inline const std::string kUnitVector = "UnitVector";
// 不需要建模的ASCIR
inline const std::string kData = "Data";
inline const std::string kScalar = "Scalar";
inline const std::string kScalarData = "ScalarData";
inline const std::string kIndexExpr = "IndexExpr";
inline const std::string kWorkspace = "Workspace";
inline const std::string kOutput = "Output";
inline const std::string kTbufData = "TbufData";
inline const std::string kLoad = "Load";
inline const std::string kStore = "Store";
// mte
inline const std::string kMoveGmToL1 = "T_LoadTscm";
inline const std::string kMoveL2ToL1 = "CopyL2ToL1";
inline const std::string kMoveL1ToL0a = "T_LoadA";
inline const std::string kMoveL1ToL0b = "T_LoadB";
inline const std::string kMoveL0cToL2 = "CopyL0CToL2";
inline const std::string kMoveL0cToGm = "T_FixPipeTrans";
inline const std::string kMoveGmToUb = "Load";   // ASCIR
inline const std::string kMoveUbToGm = "Store";  // ASCIR
// cube
inline const std::string kComputeCube = "T_Mmad";
inline const std::string kMatMul = "MatMul";
// vec
inline const std::string kComputeVector = "VectorCompute";  // 默认Vector性能评估公式
// ascendc api名
inline const std::string kAdds = "Adds";
inline const std::string kAnd = "And";
inline const std::string kBrcb = "Brcb";
inline const std::string kBlockReduceMax = "BlockReduceMax";
inline const std::string kBlockReduceMin = "BlockReduceMin";
inline const std::string kCopy = "Copy";
inline const std::string kCompareEQ = "CompareEQ";
inline const std::string kCompareScalarEQ = "CompareScalarEQ";
inline const std::string kCompareGE = "CompareGE";
inline const std::string kCompareScalarGE = "CompareScalarGE";
inline const std::string kCompareGT = "CompareGT";
inline const std::string kCompareScalarGT = "CompareScalarGT";
inline const std::string kCompareLE = "CompareLE";
inline const std::string kCompareScalarLE = "CompareScalarLE";
inline const std::string kCompareLT = "CompareLT";
inline const std::string kCompareScalarLT = "CompareScalarLT";
inline const std::string kCompareNE = "CompareNE";
inline const std::string kCompareScalarNE = "CompareScalarNE";
inline const std::string kPower = "Power";
inline const std::string kDuplicate = "Duplicate";
inline const std::string kGatherMask = "GatherMask";
inline const std::string kMaxs = "Maxs";
inline const std::string kMax = "Max";
inline const std::string kSoftmax = "Softmax";
inline const std::string kArgMax = "ArgMax";
inline const std::string kArgMaxMultiRPhase1 = "ArgMaxMultiRPhase1";
inline const std::string kArgMaxMultiRPhase2 = "ArgMaxMultiRPhase2";
inline const std::string kArange = "Arange";
inline const std::string kMins = "Mins";
inline const std::string kMuls = "Muls";
inline const std::string kNot = "Not";
inline const std::string kOr = "Or";
inline const std::string kPairReduceSum = "PairReduceSum";
inline const std::string kSetVectorMask = "SetVectorMask";
inline const std::string kSigmoid = "Sigmoid";
inline const std::string kWholeReduceMax = "WholeReduceMax";
inline const std::string kWholeReduceMin = "WholeReduceMin";
inline const std::string kWholeReduceSum = "WholeReduceSum";
inline const std::string kDefaultApi = "Load";  // 用于获取默认的AscIrAtt实现，用于获取和具体IR无关的信息
inline const std::string kTruncate = "Truncate";
inline const std::string kMulAddDst = "MulAddDst";
inline const std::string kUpdateMask = "UpdateMask";
inline const std::string kPack = "Pack";
inline const std::string kUnPack = "UnPack";
inline const std::string kInterleave = "Interleave";
inline const std::string kDeInterleave = "DeInterleave";
inline const std::string kMaskPack = "MaskPack";
inline const std::string kMaskUnPack = "MaskUnPack";
inline const std::string kMaskOr = "kMaskOr";
inline const std::string kMaskAnd = "kMaskAnd";
inline const std::string kMaskSel = "MaskSel";
// 下面均为vf instruct定义（暂无对应的MicroApi）
inline const std::string kVcadd = "vcadd";
inline const std::string kVsqz = "vsqz";
inline const std::string kXor = "Xor";
inline const std::string kVshrs = "Vshrs";
// 下面均为ASCIR定义（已有性能评估）
inline const std::string kGather = "Gather";
inline const std::string kGather2 = "Gather2";
inline const std::string kIndirectLoad = "IndirectLoad";
inline const std::string kAbs = "Abs";
inline const std::string kAdd = "Add";
inline const std::string kBroadcast = "Broadcast";
inline const std::string kCast = "Cast";
inline const std::string kDiv = "Div";
inline const std::string kErf = "Erf";
inline const std::string kExp = "Exp";
inline const std::string kExp2 = "Exp2";
inline const std::string kFloor = "Floor";
inline const std::string kFma = "Fma";
inline const std::string kBitwiseNot = "BitwiseNot";
inline const std::string kBitwiseOr = "BitwiseOr";
inline const std::string kBitwiseXor = "BitwiseXor";
inline const std::string kCeil = "Ceil";
inline const std::string kCos = "Cos";
inline const std::string kAcos = "Acos";
inline const std::string kCosh = "Cosh";
inline const std::string kDigamma = "Digamma";
inline const std::string kErfc = "Erfc";
inline const std::string kErfcx = "Erfcx";
inline const std::string kAtan2 = "Atan2";
inline const std::string kCopySign = "CopySign";
inline const std::string kCeil2Int = "Ceil2Int";
inline const std::string kLogicalAnd = "LogicalAnd";
inline const std::string kLogicalOr = "LogicalOr";
inline const std::string kLogicalNot = "LogicalNot";
inline const std::string kMaximum = "Maximum";
inline const std::string kMinimum = "Minimum";
inline const std::string kMin = "Min";
inline const std::string kMul = "Mul";
inline const std::string kNeg = "Neg";
inline const std::string kReciprocal = "Reciprocal";
inline const std::string kRelu = "Relu";
inline const std::string kReduceAll = "ReduceAll";          // All
inline const std::string kReduceAny = "ReduceAny";          // Any
inline const std::string kReduceMax = "ReduceMax";          // Max
inline const std::string kReduceSoftmax = "ReduceSoftmax";  // Softmax
inline const std::string kReduceMean = "ReduceMean";        // Mean
inline const std::string kReduceMin = "ReduceMin";          // Min
inline const std::string kReduceSum = "ReduceSum";          // Sum
inline const std::string kReduceProd = "ReduceProd";        // Prod
inline const std::string kAll = "All";
inline const std::string kProd = "Prod";
inline const std::string kAny = "Any";
inline const std::string kMean = "Mean";
inline const std::string kRemovePad = "RemovePad";
inline const std::string kRsqrt = "Rsqrt";
inline const std::string kSelect = "Select";
inline const std::string kSign = "Sign";
inline const std::string kSqrt = "Sqrt";
inline const std::string kSub = "Sub";
inline const std::string kSum = "Sum";
inline const std::string kTanh = "Tanh";
inline const std::string kWhere = "Where";
inline const std::string kZerosLike = "ZerosLike";
inline const std::string kUb2ub = "Ub2ub";
inline const std::string kGe = "Ge";
inline const std::string kEq = "Eq";
inline const std::string kNe = "Ne";
inline const std::string kGt = "Gt";
inline const std::string kLe = "Le";
inline const std::string kLt = "Lt";

inline const std::string kPow = "Pow";
inline const std::string kPad = "Pad";
inline const std::string kRound = "Round";
inline const std::string kLn = "Ln";
inline const std::string kFloorToInt = "FloorToInt";
inline const std::string kFmod = "Fmod";
inline const std::string kHypot = "Hypot";
inline const std::string kLgamma = "Lgamma";
inline const std::string kLog10 = "Log10";
inline const std::string kLogicalXor = "LogicalXor";
inline const std::string kLog1p = "Log1p";
inline const std::string kLog2 = "Log2";
inline const std::string kExpm1 = "Expm1";
inline const std::string kLShift = "LShift";
inline const std::string kMod = "Mod";
inline const std::string kIsnan = "Isnan";
inline const std::string kSin = "Sin";
inline const std::string kAcosh = "Acosh";
inline const std::string kAsin = "Asin";
inline const std::string kAsinh = "Asinh";
inline const std::string kAtan = "Atan";
inline const std::string kAtanh = "Atanh";
inline const std::string kRShift = "RShift";
inline const std::string kIsFinite = "IsFinite";
inline const std::string kIsInf = "IsInf";
inline const std::string kMaskedFill = "MaskedFill";
inline const std::string kTrueDiv = "TrueDiv";
inline const std::string kRemainder = "Remainder";
inline const std::string kI0 = "I0";
inline const std::string kI0e = "I0e";
inline const std::string kI1e = "I1e";
inline const std::string kI1 = "I1";
inline const std::string kIdentity = "Identity";
inline const std::string kDeviceAssertAsync = "DeviceAssertAsync";
inline const std::string kInlineAsmElementwise = "InlineAsmElementwise";
inline const std::string kMasked = "Masked";
inline const std::string kRandint64 = "Randint64";
inline const std::string kXorSum = "XorSum";
inline const std::string kWelford = "Welford";
inline const std::string kScan = "Scan";
inline const std::string kStoreReduction = "StoreReduction";
inline const std::string kModifiedBesselI0 = "ModifiedBesselI0";
inline const std::string kModifiedBesselI1 = "ModifiedBesselI1";
inline const std::string kModifiedBesselK0 = "ModifiedBesselK0";
inline const std::string kModifiedBesselK1 = "ModifiedBesselK1";
inline const std::string kBesselJ0 = "BesselJ0";
inline const std::string kBesselJ1 = "BesselJ1";
inline const std::string kBesselY0 = "BesselY0";
inline const std::string kBesselY1 = "BesselY1";
inline const std::string kScaledModifiedBesselK0 = "ScaledModifiedBesselK0";
inline const std::string kScaledModifiedBesselK1 = "ScaledModifiedBesselK1";
inline const std::string kSphericalBesselJ0 = "SphericalBesselJ0";
inline const std::string kIgamma = "Igamma";
inline const std::string kIgammac = "Igammac";
inline const std::string kZeta = "Zeta";
inline const std::string kNdtr = "Ndtr";
inline const std::string kNdtri = "Ndtri";
inline const std::string kLogNdtr = "LogNdtr";
inline const std::string kNextAfter = "NextAfter";
inline const std::string kBucketize = "Bucketize";
inline const std::string kPolyGamma = "PolyGamma";
inline const std::string kSignBit = "SignBit";
inline const std::string kFrexp = "Frexp";
inline const std::string kShiftedChebyshevPolynomialT = "ShiftedChebyshevPolynomialT";
inline const std::string kShiftedChebyshevPolynomialU = "ShiftedChebyshevPolynomialU";
inline const std::string kShiftedChebyshevPolynomialV = "ShiftedChebyshevPolynomialV";
inline const std::string kShiftedChebyshevPolynomialW = "ShiftedChebyshevPolynomialW";
inline const std::string kChebyshevPolynomialT = "ChebyshevPolynomialT";
inline const std::string kChebyshevPolynomialU = "ChebyshevPolynomialU";
inline const std::string kChebyshevPolynomialV = "ChebyshevPolynomialV";
inline const std::string kChebyshevPolynomialW = "ChebyshevPolynomialW";
inline const std::string kHermitePolynomialH = "HermitePolynomialH";
inline const std::string kHermitePolynomialHe = "HermitePolynomialHe";
inline const std::string kLaguerrePolynomialL = "LaguerrePolynomialL";
inline const std::string kLegendrePolynomialP = "LegendrePolynomialP";
inline const std::string kAiryAi = "AiryAi";
inline const std::string kErfinv = "Erfinv";
inline const std::string kClipByValue = "ClipByValue";
inline const std::string kLeakyRelu = "LeakyRelu";
inline const std::string kBitwiseAnd = "BitwiseAnd";
inline const std::string kFloorDiv = "FloorDiv";
inline const std::string kGelu = "Gelu";
inline const std::string kConcat = "Concat";
inline const std::string kNop = "Nop";
inline const std::string kTranspose = "Transpose";
inline const std::string kSplit = "Split";
inline const std::string kTan = "Tan";
inline const std::string kSinh = "Sinh";
inline const std::string kSquare = "Square";
inline const std::string kTruncDiv = "TruncDiv";
inline const std::string kRoundToInt = "RoundToInt";
inline const std::string kTruncToInt = "TruncToInt";
inline const std::string kTrunc = "Trunc";
inline const std::string kVmAdd = "VmAdd";
inline const std::string kRand = "Rand";
inline const std::string kRandn = "Randn";
inline const std::string kPlaceholder = "Placeholder";
#define JOIN(a, b) a##b
#define JOIN_A_B_C(a, b, c) a##b##c
}  // namespace att

#endif  // ATT_BASE_ATT_CONST_VALUE_H_
