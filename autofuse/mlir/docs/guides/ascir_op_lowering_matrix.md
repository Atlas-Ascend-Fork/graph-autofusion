# AscIR op 两平台 lowering 对照表

> 能力参考 · 由 `tools/af-mlir-tools/scan_ascir_op_lowering_matrix.py` 生成，请勿手工编辑。

## 判据

按「API 名 / 扩展头 / 是否需 tmp」三项在两套实现上是否一致分档：

`dtype` 两列是该侧 `REG_ASC_IR` 收的**输入**元素类型（选择族取的是数据输入那一个，不是 ui8 掩码）。`n/a` 是起始节点、本来就没有输入；`?` 是有输入但解析不出来，要人去看——两者不是一回事。

| 档 | 含义 | AFIR 侧的代价 |
|---|---|---|
| **D0** | 三项一致，且不依赖 Autofuse 扩展头 | 一个 implementation，无平台相关代码 |
| **D1** | 三项一致，且两侧引用**同一个**扩展头 key | 一个 implementation + 一条 include |
| **D2** | 三项至少一项不同（含扩展头 key 不同） | 两个 implementation + 两个 stable id，Capability 查 feature |
| **D3** | op 只在一侧注册 | ODS 取并集，实现只在有供给的一侧注册 |

mem-based = `NpuArch 2201`（v1，soc `2201`）；reg-based = `NpuArch 3510`（v2，soc `3510` / `5102` / `9202`）。

**扩展头的 key 空间是全局且不相交的**：mem 侧是 `xxx.h`，reg-base 侧是 `xxx_reg_base.h`，cube 另有一批；三者注册进同一个 `AscendCApiRegistry` 单例。因此「头名 → 头文本」的查表**不需要知道平台**，头名本身就是判别式。

`tmp` 列给出 `CalcTmpBufSize` 委托的辅助函数名；`custom` 表示方法体自己按条件算（即该 op 的 tmp 需求是条件性的）；`—` 表示不 override，即不需要 tmp。
`标量` 列：`no` 不支持标量输入，`2nd` 仅第二个输入可为标量，`any` 任意，`all` 全标量，`+swap(...)` 表示交换操作数后另有一套约束。

## 统计

| 范围 | D0 | D1 | D2 | D3 | 合计 |
|---|---|---|---|---|---|
| (全部) | 16 | 21 | 49 | 90 | 176 |
| elementwise | 8 | 4 | 35 | 0 | 47 |
| concat | 0 | 0 | 1 | 0 | 1 |
| cube | 0 | 16 | 0 | 0 | 16 |
| datamove | 0 | 0 | 2 | 0 | 2 |
| gather | 0 | 0 | 1 | 1 | 2 |
| layout | 2 | 1 | 1 | 4 | 8 |
| other | 0 | 0 | 0 | 81 | 81 |
| reduce | 1 | 0 | 7 | 4 | 12 |
| split | 0 | 0 | 1 | 0 | 1 |
| structural | 5 | 0 | 1 | 0 | 6 |

## 逐 op 明细（全部 op）

| op | 族 | 档 | mem API | mem 扩展头 | mem tmp | mem 标量 | mem dtype | reg API | reg 扩展头 | reg tmp | reg 标量 | reg dtype | 判据 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `Abs` | elementwise | **D2** | AbsExtend | abs.h | CalcAbsTmpSize | no | f16/f32/i32 | Abs | — | — | no | f16/f32/i16/i32/ui8/bf16/i8/i64 | API 名 `AbsExtend` vs `Abs`；扩展头 abs.h vs -；tmp 要 vs 不要 |
| `Acos` | other | **D3** | — | — | — | — | — | AcosExtend | acos_reg_base.h | CalcAcosTmpSizeV2 | no | f16/f32/bf16 | 仅 reg-based |
| `Acosh` | other | **D3** | — | — | — | — | — | AcoshExtend | acosh_reg_base.h | CalcAcoshTmpSizeV2 | no | f16/f32/bf16 | 仅 reg-based |
| `Add` | elementwise | **D1** | Add | scalar_add.h | — | 2nd+swap(2nd) | f16/f32/i16/i32 | Add | scalar_add.h | — | 2nd+swap(2nd) | f16/f32/i16/i32/ui8/ui16/bf16/i8/i64/ui32/ui64 | 两侧共用扩展头 scalar_add.h |
| `AiryAi` | other | **D3** | — | — | — | — | — | AiryAiExtend | airy_ai_reg_base.h/trigonometric_function_utils_reg_base.h | CalcVoidTmpSizeV2 | no | f32 | 仅 reg-based |
| `All` | reduce | **D2** | ReduceAll | reduce.h/reduce_init.h | CalcReduceTmpSize | no | f32 | ReduceAll | reduce_extend_reg_base.h/reduce_init_reg_base.h | CalcReduceTmpSizeV2 | no | f32 | 扩展头 reduce.h/reduce_init.h vs reduce_extend_reg_base.h/reduce_init_reg_base.h |
| `Any` | reduce | **D2** | ReduceAny | reduce.h/reduce_init.h | CalcReduceTmpSize | no | f32 | ReduceAny | reduce_extend_reg_base.h/reduce_init_reg_base.h | CalcReduceTmpSizeV2 | no | f32 | 扩展头 reduce.h/reduce_init.h vs reduce_extend_reg_base.h/reduce_init_reg_base.h |
| `Arange` | layout | **D3** | — | — | — | — | — | Arange | — | — | no | n/a | 仅 reg-based |
| `ArgMax` | reduce | **D3** | ArgMax | argmax.h/argmax_with_value.h/compare.h/compare_v2.h/duplicate.h/reduce.h/reduce_init.h/where.h | CalcReduceTmpSize | no | f32/i32 | — | — | — | — | — | 仅 mem-based |
| `ArgMaxMultiRPhase1` | reduce | **D3** | ArgMaxMultiRPhase1 | argmax.h/argmax_with_value.h/compare.h/compare_v2.h/duplicate.h/reduce.h/reduce_init.h/where.h | CalcReduceTmpSize | no | f32/i32 | — | — | — | — | — | 仅 mem-based |
| `ArgMaxMultiRPhase2` | reduce | **D3** | ArgMaxMultiRPhase2 | argmax.h/argmax_with_value.h/compare.h/compare_v2.h/duplicate.h/reduce.h/reduce_init.h/where.h | CalcReduceTmpSize | no | i64 | — | — | — | — | — | 仅 mem-based |
| `Asin` | other | **D3** | — | — | — | — | — | AsinExtend | asin_reg_base.h | CalcAsinTmpSizeV2 | no | f16/f32/bf16 | 仅 reg-based |
| `Asinh` | other | **D3** | — | — | — | — | — | AsinhExtend | asinh_reg_base.h | CalcAsinhTmpSizeV2 | no | f16/f32/bf16 | 仅 reg-based |
| `Atan` | other | **D3** | — | — | — | — | — | AtanExtend | atan_reg_base.h | CalcAtanTmpSizeV2 | no | f16/f32/bf16 | 仅 reg-based |
| `Atan2` | other | **D3** | — | — | — | — | — | Atan2Extend | atan2_reg_base.h | CalcAtanTmpSizeV2 | no | f16/f32/bf16 | 仅 reg-based |
| `Atanh` | other | **D3** | — | — | — | — | — | AtanhExtend | atanh_reg_base.h | CalcAtanhTmpSizeV2 | no | f16/f32/bf16 | 仅 reg-based |
| `Axpy` | elementwise | **D1** | AxpyExtend | axpy.h | CalcAxpyTmpSize | no | f16/f32 | AxpyExtend | axpy.h | CalcAxpyTmpSize | no | f16/f32/bf16/i64/ui64 | 两侧共用扩展头 axpy.h |
| `BatchMatMul` | cube | **D1** | BatchMatMul | batch_mat_mul_v3_tiling_key.h/batch_mat_mul_v3_tiling_key_dynamic.h/batch_matmul.h/batch_matmul_dynamic.h/batch_matmul_include_headers.h/mat_mul_pingpong_basic_cmct.h/mat_mul_pingpong_basic_cmct_dynamic.h | — | no | f16/f32/bf16 | BatchMatMul | batch_mat_mul_v3_tiling_key.h/batch_mat_mul_v3_tiling_key_dynamic.h/batch_matmul.h/batch_matmul_dynamic.h/batch_matmul_include_headers.h/mat_mul_pingpong_basic_cmct.h/mat_mul_pingpong_basic_cmct_dynamic.h | — | no | f16/f32/bf16 | 两侧共用扩展头 batch_mat_mul_v3_tiling_key.h/batch_mat_mul_v3_tiling_key_dynamic.h/batch_matmul.h/batch_matmul_dynamic.h/batch_matmul_include_headers.h/mat_mul_pingpong_basic_cmct.h/mat_mul_pingpong_basic_cmct_dynamic.h |
| `BatchMatMulBias` | cube | **D1** | BatchMatMul | batch_mat_mul_v3_tiling_key.h/batch_mat_mul_v3_tiling_key_dynamic.h/batch_matmul.h/batch_matmul_dynamic.h/batch_matmul_include_headers.h/mat_mul_pingpong_basic_cmct.h/mat_mul_pingpong_basic_cmct_dynamic.h | — | no | f16/f32/bf16 | BatchMatMul | batch_mat_mul_v3_tiling_key.h/batch_mat_mul_v3_tiling_key_dynamic.h/batch_matmul.h/batch_matmul_dynamic.h/batch_matmul_include_headers.h/mat_mul_pingpong_basic_cmct.h/mat_mul_pingpong_basic_cmct_dynamic.h | — | no | f16/f32/bf16 | 两侧共用扩展头 batch_mat_mul_v3_tiling_key.h/batch_mat_mul_v3_tiling_key_dynamic.h/batch_matmul.h/batch_matmul_dynamic.h/batch_matmul_include_headers.h/mat_mul_pingpong_basic_cmct.h/mat_mul_pingpong_basic_cmct_dynamic.h |
| `BatchMatMulOffset` | cube | **D1** | BatchMatMul | batch_mat_mul_v3_tiling_key.h/batch_mat_mul_v3_tiling_key_dynamic.h/batch_matmul.h/batch_matmul_dynamic.h/batch_matmul_include_headers.h/mat_mul_pingpong_basic_cmct.h/mat_mul_pingpong_basic_cmct_dynamic.h | — | no | f16/f32/bf16 | BatchMatMul | batch_mat_mul_v3_tiling_key.h/batch_mat_mul_v3_tiling_key_dynamic.h/batch_matmul.h/batch_matmul_dynamic.h/batch_matmul_include_headers.h/mat_mul_pingpong_basic_cmct.h/mat_mul_pingpong_basic_cmct_dynamic.h | — | no | f16/f32/bf16 | 两侧共用扩展头 batch_mat_mul_v3_tiling_key.h/batch_mat_mul_v3_tiling_key_dynamic.h/batch_matmul.h/batch_matmul_dynamic.h/batch_matmul_include_headers.h/mat_mul_pingpong_basic_cmct.h/mat_mul_pingpong_basic_cmct_dynamic.h |
| `BatchMatMulOffsetBias` | cube | **D1** | BatchMatMul | batch_mat_mul_v3_tiling_key.h/batch_mat_mul_v3_tiling_key_dynamic.h/batch_matmul.h/batch_matmul_dynamic.h/batch_matmul_include_headers.h/mat_mul_pingpong_basic_cmct.h/mat_mul_pingpong_basic_cmct_dynamic.h | — | no | f16/f32/bf16 | BatchMatMul | batch_mat_mul_v3_tiling_key.h/batch_mat_mul_v3_tiling_key_dynamic.h/batch_matmul.h/batch_matmul_dynamic.h/batch_matmul_include_headers.h/mat_mul_pingpong_basic_cmct.h/mat_mul_pingpong_basic_cmct_dynamic.h | — | no | f16/f32/bf16 | 两侧共用扩展头 batch_mat_mul_v3_tiling_key.h/batch_mat_mul_v3_tiling_key_dynamic.h/batch_matmul.h/batch_matmul_dynamic.h/batch_matmul_include_headers.h/mat_mul_pingpong_basic_cmct.h/mat_mul_pingpong_basic_cmct_dynamic.h |
| `BesselJ0` | other | **D3** | — | — | — | — | — | BesselJ0Extend | bessel_j0_reg_base.h/bessel_j_utils_reg_base.h/trigonometric_function_utils_reg_base.h | CalcVoidTmpSizeV2 | no | f32 | 仅 reg-based |
| `BesselJ1` | other | **D3** | — | — | — | — | — | BesselJ1Extend | bessel_j1_reg_base.h/bessel_j_utils_reg_base.h/trigonometric_function_utils_reg_base.h | CalcVoidTmpSizeV2 | no | f32 | 仅 reg-based |
| `BesselY0` | other | **D3** | — | — | — | — | — | BesselY0Extend | bessel_j0_reg_base.h/bessel_j_utils_reg_base.h/bessel_y0_reg_base.h/trigonometric_function_utils_reg_base.h | CalcVoidTmpSizeV2 | no | f32 | 仅 reg-based |
| `BesselY1` | other | **D3** | — | — | — | — | — | BesselY1Extend | bessel_j1_reg_base.h/bessel_j_utils_reg_base.h/bessel_y1_reg_base.h/trigonometric_function_utils_reg_base.h | CalcVoidTmpSizeV2 | no | f32 | 仅 reg-based |
| `BitwiseAnd` | elementwise | **D2** | BitwiseAndExtend | bitwise_and.h | CalcDefaultTmpSize | no | i16/i32/ui8/ui16 | BitwiseAndExtend | bitwise_and_reg_base.h | — | no | i16/i32/ui8/ui16/i8/i64/ui32/ui64/bool | 扩展头 bitwise_and.h vs bitwise_and_reg_base.h；tmp 要 vs 不要 |
| `BitwiseNot` | other | **D3** | — | — | — | — | — | BitwiseNotExtend | bitwise_not_reg_base.h | — | no | i16/i32/ui8/ui16/i8/i64/ui32/ui64/bool | 仅 reg-based |
| `BitwiseOr` | other | **D3** | — | — | — | — | — | BitwiseOrExtend | bitwise_or_reg_base.h | — | no | i16/i32/ui8/ui16/i8/i64/ui32/ui64/bool | 仅 reg-based |
| `BitwiseXor` | other | **D3** | — | — | — | — | — | BitwiseXorExtend | bitwise_xor_reg_base.h | — | no | i16/i32/ui8/ui16/i8/i64/ui32/ui64/bool | 仅 reg-based |
| `Broadcast` | elementwise | **D2** | Broadcast | broadcast.h/duplicate.h | CalcBroadCastTmpSize | no | f16/f32/i16/i32/ui8/ui16/i8/i64/ui32/ui64 | BroadcastExtend | broadcast_reg_base.h/duplicate.h | — | no | f16/f32/i16/i32/ui8/ui16/i8/i64/ui32/ui64/bool | API 名 `Broadcast` vs `BroadcastExtend`；扩展头 broadcast.h/duplicate.h vs broadcast_reg_base.h/duplicate.h；tmp 要 vs 不要 |
| `Cast` | elementwise | **D2** | Cast | cast.h | CalcCastTmpSize | no | f16/f32/i16/i32/ui8/ui16/bf16/i4/i8/i64/ui32/ui64 | CastExtend | cast_reg_base.h | — | no | f16/f32/i16/i32/ui8/ui16/bf16/i8/i64/ui32/ui64 | API 名 `Cast` vs `CastExtend`；扩展头 cast.h vs cast_reg_base.h；tmp 要 vs 不要 |
| `Ceil` | other | **D3** | — | — | — | — | — | CeilExtend | ceil_reg_base.h | CalcCeilTmpSizeV2 | no | f16/f32/bf16 | 仅 reg-based |
| `Ceil2Int` | other | **D3** | — | — | — | — | — | Ceil2IntExtend | cast_reg_base.h | — | no | f16/f32/bf16 | 仅 reg-based |
| `ChebyshevPolynomialT` | other | **D3** | — | — | — | — | — | ChebyshevPolynomialTExtend | chebyshev_polynomial_t_reg_base.h/chebyshev_polynomial_utils_reg_base.h | CalcVoidTmpSizeV2 | no | f32 | 仅 reg-based |
| `ChebyshevPolynomialU` | other | **D3** | — | — | — | — | — | ChebyshevPolynomialUExtend | chebyshev_polynomial_u_reg_base.h/chebyshev_polynomial_utils_reg_base.h | CalcVoidTmpSizeV2 | no | f32 | 仅 reg-based |
| `ChebyshevPolynomialV` | other | **D3** | — | — | — | — | — | ChebyshevPolynomialVExtend | chebyshev_polynomial_utils_reg_base.h/chebyshev_polynomial_v_reg_base.h | CalcVoidTmpSizeV2 | no | f32 | 仅 reg-based |
| `ChebyshevPolynomialW` | other | **D3** | — | — | — | — | — | ChebyshevPolynomialWExtend | chebyshev_polynomial_utils_reg_base.h/chebyshev_polynomial_w_reg_base.h | CalcVoidTmpSizeV2 | no | f32 | 仅 reg-based |
| `ClipByValue` | elementwise | **D2** | ClipByValue | clipbyvalue.h | CalcClipByValueTmpSize | any | f16/f32 | ClipByValue | clipbyvalue_reg_base.h | — | any | f16/f32/i16/i32/ui8/ui16/bf16/i8/i64/ui32/ui64 | 扩展头 clipbyvalue.h vs clipbyvalue_reg_base.h；tmp 要 vs 不要 |
| `Concat` | concat | **D2** | Concat | concat.h | CalcConcatTmpSize | no | f16/f32/i16/i32/ui8/ui16/bf16/i8/ui32/ui64 | Concat | concat_reg_base.h | CalcConcatTmpSizeV2 | no | f16/f32/i16/i32/ui8/ui16/bf16/i8/i64/ui32/ui64/bool | 扩展头 concat.h vs concat_reg_base.h |
| `Conv2D` | cube | **D1** | Conv2D | conv2d.h/conv2d_include_headers.h/conv2d_v2_tilingkey_cv.h/conv_pingpong_basic_atcos.h | — | no | f16 | Conv2D | conv2d.h/conv2d_include_headers.h/conv2d_v2_tilingkey_cv.h/conv_pingpong_basic_atcos.h | — | no | f16 | 两侧共用扩展头 conv2d.h/conv2d_include_headers.h/conv2d_v2_tilingkey_cv.h/conv_pingpong_basic_atcos.h |
| `Conv2DBias` | cube | **D1** | Conv2D | conv2d.h/conv2d_include_headers.h/conv2d_v2_tilingkey_cv.h/conv_pingpong_basic_atcos.h | — | no | f16 | Conv2D | conv2d.h/conv2d_include_headers.h/conv2d_v2_tilingkey_cv.h/conv_pingpong_basic_atcos.h | — | no | f16 | 两侧共用扩展头 conv2d.h/conv2d_include_headers.h/conv2d_v2_tilingkey_cv.h/conv_pingpong_basic_atcos.h |
| `Conv2DOffset` | cube | **D1** | Conv2D | conv2d.h/conv2d_include_headers.h/conv2d_v2_tilingkey_cv.h/conv_pingpong_basic_atcos.h | — | no | f16 | Conv2D | conv2d.h/conv2d_include_headers.h/conv2d_v2_tilingkey_cv.h/conv_pingpong_basic_atcos.h | — | no | f16 | 两侧共用扩展头 conv2d.h/conv2d_include_headers.h/conv2d_v2_tilingkey_cv.h/conv_pingpong_basic_atcos.h |
| `Conv2DOffsetBias` | cube | **D1** | Conv2D | conv2d.h/conv2d_include_headers.h/conv2d_v2_tilingkey_cv.h/conv_pingpong_basic_atcos.h | — | no | f16 | Conv2D | conv2d.h/conv2d_include_headers.h/conv2d_v2_tilingkey_cv.h/conv_pingpong_basic_atcos.h | — | no | f16 | 两侧共用扩展头 conv2d.h/conv2d_include_headers.h/conv2d_v2_tilingkey_cv.h/conv_pingpong_basic_atcos.h |
| `CopySign` | other | **D3** | — | — | — | — | — | CopySignExtend | copy_sign_reg_base.h | — | no | f16/f32/bf16 | 仅 reg-based |
| `Cos` | other | **D3** | — | — | — | — | — | CosExtend | cos_reg_base.h | CalcCosTmpSizeV2 | no | f16/f32/bf16 | 仅 reg-based |
| `Cosh` | other | **D3** | — | — | — | — | — | CoshExtend | cosh_reg_base.h | CalcCoshTmpSizeV2 | no | f16/f32/bf16 | 仅 reg-based |
| `Data` | structural | **D0** | Data | — | — | no | n/a | Data | — | — | no | n/a | 两侧一致且不依赖扩展头 |
| `Digamma` | other | **D3** | — | — | — | — | — | Digamma | — | CalcDigammaTmpSizeV2 | no | f16/f32/bf16 | 仅 reg-based |
| `Div` | elementwise | **D2** | Div | scalar_div.h | CalcDivTmpSize | any | f16/f32 | DivExtend | div_reg_base.h | — | any | f16/f32 | API 名 `Div` vs `DivExtend`；扩展头 scalar_div.h vs div_reg_base.h；tmp 要 vs 不要 |
| `Eq` | elementwise | **D2** | EQ | compare.h/compare_v2.h | CalcEqTmpSize | 2nd+swap(2nd) | f16/f32/i32/i64 | EQ | compare_reg_base.h | GetCompareSizeV2 | 2nd+swap(2nd) | f16/f32/i16/i32/ui8/ui16/bf16/i8/i64/ui32/ui64/bool | 扩展头 compare.h/compare_v2.h vs compare_reg_base.h |
| `Erf` | elementwise | **D2** | Erf | — | CalcErfTmpSize | no | f16/f32 | ErfExtend | erf_reg_base.h | CalcVoidTmpSizeV2 | no | f16/f32/bf16 | API 名 `Erf` vs `ErfExtend`；扩展头 - vs erf_reg_base.h |
| `Erfc` | other | **D3** | — | — | — | — | — | Erfc | — | CalcErfcTmpSizeV2 | no | f16/f32/bf16 | 仅 reg-based |
| `Erfcx` | other | **D3** | — | — | — | — | — | ErfcxExtend | erfcx_reg_base.h | CalcErfcTmpSizeV2 | no | f16/f32/bf16 | 仅 reg-based |
| `Erfinv` | other | **D3** | — | — | — | — | — | Erfinv | erfinv_reg_base.h | — | no | f32 | 仅 reg-based |
| `Exp` | elementwise | **D0** | Exp | — | — | no | f16/f32 | Exp | — | — | no | f16/f32/bf16 | 两侧一致且不依赖扩展头 |
| `Exp2` | other | **D3** | — | — | — | — | — | Exp2 | exp2_reg_base.h | CalcExp2TmpSizeV2 | no | f16/f32/bf16 | 仅 reg-based |
| `Expm1` | other | **D3** | — | — | — | — | — | Expm1Extend | expm1_reg_base.h | — | no | f16/f32/bf16 | 仅 reg-based |
| `ExtendConv2D` | cube | **D1** | Conv2D | conv2d.h/conv2d_include_headers.h/conv2d_v2_tilingkey_cv.h/conv_pingpong_basic_atcos.h | — | no | f16 | Conv2D | conv2d.h/conv2d_include_headers.h/conv2d_v2_tilingkey_cv.h/conv_pingpong_basic_atcos.h | — | no | f16 | 两侧共用扩展头 conv2d.h/conv2d_include_headers.h/conv2d_v2_tilingkey_cv.h/conv_pingpong_basic_atcos.h |
| `ExtendConv2DBias` | cube | **D1** | Conv2D | conv2d.h/conv2d_include_headers.h/conv2d_v2_tilingkey_cv.h/conv_pingpong_basic_atcos.h | — | no | f16 | Conv2D | conv2d.h/conv2d_include_headers.h/conv2d_v2_tilingkey_cv.h/conv_pingpong_basic_atcos.h | — | no | f16 | 两侧共用扩展头 conv2d.h/conv2d_include_headers.h/conv2d_v2_tilingkey_cv.h/conv_pingpong_basic_atcos.h |
| `ExtendConv2DBiasScale` | cube | **D1** | Conv2D | conv2d.h/conv2d_include_headers.h/conv2d_v2_tilingkey_cv.h/conv_pingpong_basic_atcos.h | — | no | i8 | Conv2D | conv2d.h/conv2d_include_headers.h/conv2d_v2_tilingkey_cv.h/conv_pingpong_basic_atcos.h | — | no | i8 | 两侧共用扩展头 conv2d.h/conv2d_include_headers.h/conv2d_v2_tilingkey_cv.h/conv_pingpong_basic_atcos.h |
| `ExtendConv2DScale` | cube | **D1** | Conv2D | conv2d.h/conv2d_include_headers.h/conv2d_v2_tilingkey_cv.h/conv_pingpong_basic_atcos.h | — | no | i8 | Conv2D | conv2d.h/conv2d_include_headers.h/conv2d_v2_tilingkey_cv.h/conv_pingpong_basic_atcos.h | — | no | i8 | 两侧共用扩展头 conv2d.h/conv2d_include_headers.h/conv2d_v2_tilingkey_cv.h/conv_pingpong_basic_atcos.h |
| `FlashSoftmax` | reduce | **D0** | — | — | — | no | ? | — | — | — | no | ? | 两侧一致且不依赖扩展头 |
| `Floor` | other | **D3** | — | — | — | — | — | Floor | — | CalcVoidTmpSizeV2 | no | f16/f32/bf16 | 仅 reg-based |
| `FloorDiv` | elementwise | **D2** | FloorDivExtend | floor_div.h | GetInputDataSizeTmpBuffer | no | f16/f32 | FloorDivExtend | floor_div_reg_base.h | — | no | f16/f32/i16/i32/ui8/ui16/i8/i64/ui32/ui64 | 扩展头 floor_div.h vs floor_div_reg_base.h；tmp 要 vs 不要 |
| `FloorToInt` | other | **D3** | — | — | — | — | — | FloorToInt | — | — | no | f16/f32/bf16 | 仅 reg-based |
| `Fma` | other | **D3** | — | — | — | — | — | Fma | — | CalcVoidTmpSizeV2 | no | f16/f32/i16/ui8/bf16/i8 | 仅 reg-based |
| `Fmod` | other | **D3** | — | — | — | — | — | FmodExtend | fmod_reg_base.h | CalcVoidTmpSizeV2 | no | f16/f32/bf16 | 仅 reg-based |
| `Frexp` | other | **D3** | — | — | — | — | — | FrexpExtend | frexp_reg_base.h/utils_reg_base.h | CalcVoidTmpSizeV2 | no | f32 | 仅 reg-based |
| `Gather` | gather | **D2** | GatherExtend | gather.h | CalcGatherTmpSize | no | i32/i64 | GatherExtend | gather_reg_base.h | CalcGatherTmpSizeV2 | no | i32/i64 | 扩展头 gather.h vs gather_reg_base.h |
| `Ge` | elementwise | **D2** | GE | compare.h/compare_v2.h | CalcGeTmpSize | 2nd | f16/f32/i32/i64 | GE | compare_reg_base.h | GetCompareSizeV2 | 2nd | f16/f32/i16/i32/ui8/ui16/bf16/i8/i64/ui32/ui64/bool | 扩展头 compare.h/compare_v2.h vs compare_reg_base.h |
| `Gelu` | elementwise | **D0** | Gelu | — | GetInputDataSizeTmpBuffer | no | f16/f32 | Gelu | — | CalcVoidTmpSizeV2 | no | f16/f32 | 两侧一致且不依赖扩展头 |
| `Gt` | elementwise | **D2** | GT | compare.h/compare_v2.h | CalcGtTmpSize | 2nd | f16/f32/i32/i64 | GT | compare_reg_base.h | GetCompareSizeV2 | 2nd | f16/f32/i16/i32/ui8/ui16/bf16/i8/i64/ui32/ui64/bool | 扩展头 compare.h/compare_v2.h vs compare_reg_base.h |
| `HermitePolynomialH` | other | **D3** | — | — | — | — | — | HermitePolynomialHExtend | hermite_polynomial_h_reg_base.h | CalcVoidTmpSizeV2 | no | i32 | 仅 reg-based |
| `HermitePolynomialHe` | other | **D3** | — | — | — | — | — | HermitePolynomialHeExtend | hermite_polynomial_he_reg_base.h | CalcVoidTmpSizeV2 | no | i32 | 仅 reg-based |
| `Hypot` | other | **D3** | — | — | — | — | — | Hypot | — | CalcVoidTmpSizeV2 | no | f16/f32/bf16 | 仅 reg-based |
| `I0` | other | **D3** | — | — | — | — | — | I0Extend | i0_reg_base.h | CalcVoidTmpSizeV2 | no | f32 | 仅 reg-based |
| `I0e` | other | **D3** | — | — | — | — | — | I0eExtend | i0e_reg_base.h/modified_bessel_i0_reg_base.h/modified_bessel_utils_reg_base.h | CalcVoidTmpSizeV2 | no | f32 | 仅 reg-based |
| `I1e` | other | **D3** | — | — | — | — | — | I1eExtend | i1e_reg_base.h/modified_bessel_i1_reg_base.h/modified_bessel_utils_reg_base.h | CalcVoidTmpSizeV2 | no | f32 | 仅 reg-based |
| `Igamma` | other | **D3** | — | — | — | — | — | IgammaExtend | igamma_reg_base.h/igammac_helper/asymptotic_series_reg_base.h/igammac_helper/continued_fraction_reg_base.h/igammac_helper/series_complement_reg_base.h/igammac_helper/series_reg_base.h/igammac_reg_base.h | CalcIgammaTmpSizeV2 | no | f32 | 仅 reg-based |
| `Igammac` | other | **D3** | — | — | — | — | — | IgammacExtend | igammac_helper/asymptotic_series_reg_base.h/igammac_helper/continued_fraction_reg_base.h/igammac_helper/series_complement_reg_base.h/igammac_helper/series_reg_base.h/igammac_reg_base.h | CalcIgammacTmpSizeV2 | no | f32 | 仅 reg-based |
| `IndexExpr` | elementwise | **D0** | IndexExpr | — | — | no | n/a | IndexExpr | — | — | no | n/a | 两侧一致且不依赖扩展头 |
| `IndirectLoad` | gather | **D3** | — | — | — | — | — | IndirectLoad | datacopy_reg_base.h/indirect_load_simd_policy_reg_base.h/indirect_load_simd_reg_base.h/indirect_load_simt_reg_base.h/indirect_load_sk_reg_base.h/transpose_reg_base.h | custom | no | i32/i64 | 仅 reg-based |
| `IsFinite` | elementwise | **D2** | IsFiniteExtend | isfinite.h | CalcIsFiniteTmpSize | no | f16/f32 | IsFinite | — | — | no | f16/f32/bf16 | API 名 `IsFiniteExtend` vs `IsFinite`；扩展头 isfinite.h vs -；tmp 要 vs 不要 |
| `IsInf` | elementwise | **D2** | IsInfExtend | is_inf.h | CalcIsInfTmpSize | no | f16/f32 | IsInf | — | — | no | f16/f32/bf16 | API 名 `IsInfExtend` vs `IsInf`；扩展头 is_inf.h vs -；tmp 要 vs 不要 |
| `Isnan` | elementwise | **D2** | IsnanExtend | isnan.h | CalcIsnanTmpSize | no | f16/f32 | IsNan | — | — | no | f16/f32/bf16 | API 名 `IsnanExtend` vs `IsNan`；扩展头 isnan.h vs -；tmp 要 vs 不要 |
| `LShift` | other | **D3** | — | — | — | — | — | ShiftLeft | — | — | no | i16/i32/i8/i64 | 仅 reg-based |
| `LaguerrePolynomialL` | other | **D3** | — | — | — | — | — | LaguerrePolynomialLExtend | laguerre_polynomial_l_reg_base.h | — | no | i16/i32/ui8/ui16/i8/i64/ui32/ui64 | 仅 reg-based |
| `Le` | elementwise | **D2** | LE | compare.h/compare_v2.h | CalcLeTmpSize | 2nd | f16/f32/i32/i64 | LE | compare_reg_base.h | GetCompareSizeV2 | 2nd | f16/f32/i16/i32/ui8/ui16/bf16/i8/i64/ui32/ui64/bool | 扩展头 compare.h/compare_v2.h vs compare_reg_base.h |
| `LeakyRelu` | elementwise | **D0** | LeakyRelu | — | — | no | f16/f32 | LeakyRelu | — | — | no | f16/f32 | 两侧一致且不依赖扩展头 |
| `LegendrePolynomialP` | other | **D3** | — | — | — | — | — | LegendrePolynomialPExtend | legendre_polynomial_p_reg_base.h | — | no | i16/i32/ui8/ui16/i8/i64/ui32/ui64 | 仅 reg-based |
| `Lgamma` | other | **D3** | — | — | — | — | — | Lgamma | — | CalcLgammaTmpSizeV2 | no | f16/f32/bf16 | 仅 reg-based |
| `Ln` | elementwise | **D0** | Ln | — | — | no | f16/f32 | Ln | — | — | no | f16/f32/bf16 | 两侧一致且不依赖扩展头 |
| `Load` | datamove | **D2** | Load | datacopy.h | — | no | f16/f32/i16/i32/ui8/ui16/bf16/i8/i64/ui32 | DataCopyPadExtend | datacopy_reg_base.h | — | no | f16/f32/i16/i32/ui8/ui16/bf16/i8/i64/ui32/ui64/bool | API 名 `Load` vs `DataCopyPadExtend`；扩展头 datacopy.h vs datacopy_reg_base.h |
| `Log10` | other | **D3** | — | — | — | — | — | Log10 | — | — | no | f16/f32/bf16 | 仅 reg-based |
| `Log1p` | other | **D3** | — | — | — | — | — | Log1p | log1p_reg_base.h | — | no | f16/f32/bf16 | 仅 reg-based |
| `Log2` | other | **D3** | — | — | — | — | — | Log2 | — | CalcLog2TmpSizeV2 | no | f16/f32/bf16 | 仅 reg-based |
| `LogNdtr` | other | **D3** | — | — | — | — | — | LogNdtrExtend | log_ndtr_reg_base.h | CalcVoidTmpSizeV2 | no | f32 | 仅 reg-based |
| `LogicalAnd` | elementwise | **D2** | LogicalAnd | logical.h | CalcLogicalAndTmpSize | 2nd+swap(2nd) | f16/f32/i16/i32/ui8 | LogicalAndExtend | logical_reg_base.h | — | 2nd+swap(2nd) | f16/f32/i16/i32/ui8/ui16/bf16/i8/i64/ui32/ui64/bool | API 名 `LogicalAnd` vs `LogicalAndExtend`；扩展头 logical.h vs logical_reg_base.h；tmp 要 vs 不要 |
| `LogicalNot` | elementwise | **D2** | LogicalNot | logical_not.h | CalcLogicalNotTmpSize | no | f16/f32/i16/i32/ui8 | LogicalNotExtend | logical_not_reg_base.h | — | no | f16/f32/i16/i32/ui8/ui16/bf16/i8/i64/ui32/ui64/bool | API 名 `LogicalNot` vs `LogicalNotExtend`；扩展头 logical_not.h vs logical_not_reg_base.h；tmp 要 vs 不要 |
| `LogicalOr` | elementwise | **D2** | LogicalOr | logical.h | CalcLogicalOrTmpSize | 2nd+swap(2nd) | f16/f32/i16/i32/ui8 | LogicalOrExtend | logical_reg_base.h | — | no | f16/f32/i16/i32/ui8/ui16/bf16/i8/i64/ui32/ui64/bool | API 名 `LogicalOr` vs `LogicalOrExtend`；扩展头 logical.h vs logical_reg_base.h；tmp 要 vs 不要 |
| `LogicalXor` | other | **D3** | — | — | — | — | — | LogicalXorExtend | logical_xor_reg_base.h | — | no | f16/f32/i16/i32/ui8/ui16/bf16/i8/i64/ui32/ui64/bool | 仅 reg-based |
| `Lt` | elementwise | **D2** | LT | compare.h/compare_v2.h | CalcLtTmpSize | 2nd | f16/f32/i32 | LT | compare_reg_base.h | GetCompareSizeV2 | 2nd | f16/f32/i16/i32/ui8/ui16/bf16/i8/i64/ui32/ui64/bool | 扩展头 compare.h/compare_v2.h vs compare_reg_base.h |
| `MaskedFill` | elementwise | **D2** | Where | duplicate.h/where.h | CalcMaskedFillTmpSize | !is_scalar_list[1] | f16/f32 | WhereExtend | where_reg_base.h | CalcMaskedFillTmpSize | !is_scalar_list[1] | f16/f32/bf16 | API 名 `Where` vs `WhereExtend`；扩展头 duplicate.h/where.h vs where_reg_base.h |
| `MatMul` | cube | **D1** | MatMul | mat_mul_pingpong_basic_cmct.h/mat_mul_pingpong_basic_cmct_dynamic.h/mat_mul_tiling_key.h/mat_mul_tiling_key_dynamic.h/matmul.h/matmul_dynamic.h/matmul_include_headers.h | — | no | f16/f32/bf16 | MatMul | mat_mul_pingpong_basic_cmct.h/mat_mul_pingpong_basic_cmct_dynamic.h/mat_mul_tiling_key.h/mat_mul_tiling_key_dynamic.h/matmul.h/matmul_dynamic.h/matmul_include_headers.h | — | no | f16/f32/bf16 | 两侧共用扩展头 mat_mul_pingpong_basic_cmct.h/mat_mul_pingpong_basic_cmct_dynamic.h/mat_mul_tiling_key.h/mat_mul_tiling_key_dynamic.h/matmul.h/matmul_dynamic.h/matmul_include_headers.h |
| `MatMulBias` | cube | **D1** | MatMul | mat_mul_pingpong_basic_cmct.h/mat_mul_pingpong_basic_cmct_dynamic.h/mat_mul_tiling_key.h/mat_mul_tiling_key_dynamic.h/matmul.h/matmul_dynamic.h/matmul_include_headers.h | — | no | f16/f32/bf16 | MatMul | mat_mul_pingpong_basic_cmct.h/mat_mul_pingpong_basic_cmct_dynamic.h/mat_mul_tiling_key.h/mat_mul_tiling_key_dynamic.h/matmul.h/matmul_dynamic.h/matmul_include_headers.h | — | no | f16/f32/bf16 | 两侧共用扩展头 mat_mul_pingpong_basic_cmct.h/mat_mul_pingpong_basic_cmct_dynamic.h/mat_mul_tiling_key.h/mat_mul_tiling_key_dynamic.h/matmul.h/matmul_dynamic.h/matmul_include_headers.h |
| `MatMulOffset` | cube | **D1** | MatMul | mat_mul_pingpong_basic_cmct.h/mat_mul_pingpong_basic_cmct_dynamic.h/mat_mul_tiling_key.h/mat_mul_tiling_key_dynamic.h/matmul.h/matmul_dynamic.h/matmul_include_headers.h | — | no | f16/f32/bf16 | MatMul | mat_mul_pingpong_basic_cmct.h/mat_mul_pingpong_basic_cmct_dynamic.h/mat_mul_tiling_key.h/mat_mul_tiling_key_dynamic.h/matmul.h/matmul_dynamic.h/matmul_include_headers.h | — | no | f16/f32/bf16 | 两侧共用扩展头 mat_mul_pingpong_basic_cmct.h/mat_mul_pingpong_basic_cmct_dynamic.h/mat_mul_tiling_key.h/mat_mul_tiling_key_dynamic.h/matmul.h/matmul_dynamic.h/matmul_include_headers.h |
| `MatMulOffsetBias` | cube | **D1** | MatMul | mat_mul_pingpong_basic_cmct.h/mat_mul_pingpong_basic_cmct_dynamic.h/mat_mul_tiling_key.h/mat_mul_tiling_key_dynamic.h/matmul.h/matmul_dynamic.h/matmul_include_headers.h | — | no | f16/f32/bf16 | MatMul | mat_mul_pingpong_basic_cmct.h/mat_mul_pingpong_basic_cmct_dynamic.h/mat_mul_tiling_key.h/mat_mul_tiling_key_dynamic.h/matmul.h/matmul_dynamic.h/matmul_include_headers.h | — | no | f16/f32/bf16 | 两侧共用扩展头 mat_mul_pingpong_basic_cmct.h/mat_mul_pingpong_basic_cmct_dynamic.h/mat_mul_tiling_key.h/mat_mul_tiling_key_dynamic.h/matmul.h/matmul_dynamic.h/matmul_include_headers.h |
| `Max` | reduce | **D2** | ReduceMax | reduce.h/reduce_init.h | CalcReduceTmpSize | no | f16/f32 | ReduceMax | reduce_extend_reg_base.h/reduce_init_reg_base.h | CalcReduceTmpSizeV2 | no | f16/f32/i16/i32/ui8/bf16/i8/i64 | 扩展头 reduce.h/reduce_init.h vs reduce_extend_reg_base.h/reduce_init_reg_base.h |
| `Maximum` | elementwise | **D1** | AscendC::Max | scalar_maximum.h | — | 2nd+swap(2nd) | f16/f32/i16/i32 | AscendC::Max | scalar_maximum.h | — | 2nd+swap(2nd) | f16/f32/i16/i32/ui8/ui16/bf16/i8/i64/ui32/ui64 | 两侧共用扩展头 scalar_maximum.h |
| `Mean` | reduce | **D2** | ReduceMean | reduce.h/reduce_init.h | CalcReduceTmpSize | no | f32 | ReduceMean | reduce_extend_reg_base.h/reduce_init_reg_base.h | CalcReduceTmpSizeV2 | no | f32 | 扩展头 reduce.h/reduce_init.h vs reduce_extend_reg_base.h/reduce_init_reg_base.h |
| `Min` | reduce | **D2** | ReduceMin | reduce.h/reduce_init.h | CalcReduceTmpSize | no | f16/f32 | ReduceMin | reduce_extend_reg_base.h/reduce_init_reg_base.h | CalcReduceTmpSizeV2 | no | f16/f32/i16/i32/ui8/bf16/i8/i64 | 扩展头 reduce.h/reduce_init.h vs reduce_extend_reg_base.h/reduce_init_reg_base.h |
| `Minimum` | elementwise | **D1** | AscendC::Min | scalar_minimum.h | — | 2nd+swap(2nd) | f16/f32/i16/i32 | AscendC::Min | scalar_minimum.h | — | 2nd+swap(2nd) | f16/f32/i16/i32/ui8/ui16/bf16/i8/i64/ui32/ui64 | 两侧共用扩展头 scalar_minimum.h |
| `Mod` | other | **D3** | — | — | — | — | — | FmodExtend | fmod_reg_base.h | CalcVoidTmpSizeV2 | no | f16/f32/i16/ui8/bf16/i8 | 仅 reg-based |
| `ModifiedBesselI0` | other | **D3** | — | — | — | — | — | ModifiedBesselI0Extend | modified_bessel_i0_reg_base.h/modified_bessel_utils_reg_base.h | CalcModifiedBesselI0TmpSizeV2 | no | f32 | 仅 reg-based |
| `ModifiedBesselI1` | other | **D3** | — | — | — | — | — | ModifiedBesselI1Extend | modified_bessel_i1_reg_base.h/modified_bessel_utils_reg_base.h | CalcModifiedBesselI1TmpSizeV2 | no | f32 | 仅 reg-based |
| `ModifiedBesselK0` | other | **D3** | — | — | — | — | — | ModifiedBesselK0Extend | modified_bessel_i0_reg_base.h/modified_bessel_k0_reg_base.h/modified_bessel_utils_reg_base.h | CalcModifiedBesselK0TmpSizeV2 | no | f32 | 仅 reg-based |
| `ModifiedBesselK1` | other | **D3** | — | — | — | — | — | ModifiedBesselK1Extend | modified_bessel_i1_reg_base.h/modified_bessel_k1_reg_base.h/modified_bessel_utils_reg_base.h | CalcModifiedBesselK1TmpSizeV2 | no | f32 | 仅 reg-based |
| `Mul` | elementwise | **D2** | Mul | scalar_mul.h | — | 2nd+swap(2nd) | f16/f32/i16/i32 | MulExtend | mul_reg_base.h | — | no | f16/f32/i16/i32/ui8/bf16/i8/i64/ui32/ui64/bool | API 名 `Mul` vs `MulExtend`；扩展头 scalar_mul.h vs mul_reg_base.h |
| `Nddma` | layout | **D3** | — | — | — | — | — | DataCopyNddma | datacopy_nddma_reg_base.h | — | no | f16/f32/i16/i32/ui8/ui16/bf16/i8/i64/ui32/ui64/bool | 仅 reg-based |
| `Ndtr` | other | **D3** | — | — | — | — | — | NdtrExtend | ndtr_reg_base.h | CalcVoidTmpSizeV2 | no | f32 | 仅 reg-based |
| `Ndtri` | other | **D3** | — | — | — | — | — | NdtriExtend | ndtri_reg_base.h | CalcVoidTmpSizeV2 | no | f32 | 仅 reg-based |
| `Ne` | elementwise | **D2** | NE | compare.h/compare_v2.h | CalcNeTmpSize | 2nd+swap(2nd) | f16/f32/i32/i64 | NE | compare_reg_base.h | GetCompareSizeV2 | 2nd+swap(2nd) | f16/f32/i16/i32/ui8/ui16/bf16/i8/i64/ui32/ui64/bool | 扩展头 compare.h/compare_v2.h vs compare_reg_base.h |
| `Neg` | elementwise | **D2** | NegExtend | neg.h | — | no | f16/f32/i16/i32 | NegExtend | neg_reg_base.h | — | no | f16/f32/i16/i32/bf16/i8/i64 | 扩展头 neg.h vs neg_reg_base.h |
| `NextAfter` | other | **D3** | — | — | — | — | — | NextAfterExtend | next_after_reg_base.h | CalcVoidTmpSizeV2 | no | f32 | 仅 reg-based |
| `Nop` | structural | **D0** | Nop | — | — | no | f16/f32/i16/i32/ui8/ui16/i8/ui32/ui64 | Nop | — | — | no | f16/f32/i16/i32/ui8/ui16/i8/ui32/ui64 | 两侧一致且不依赖扩展头 |
| `Output` | structural | **D0** | Output | — | — | no | f16/f32/i16/i32/ui8/ui16/bf16/i8/i64/ui32/ui64 | Output | — | — | no | f16/f32/i16/i32/ui8/ui16/bf16/i8/i64/ui32/ui64/bool | 两侧一致且不依赖扩展头 |
| `Pad` | layout | **D0** | Pad | — | CalcPadTmpSize | no | f16/f32/i16/i32/ui16/ui32 | Pad | — | CalcVoidTmpSizeV2 | no | f16/f32/i16/i32/ui16/ui32 | 两侧一致且不依赖扩展头 |
| `PolyGamma` | other | **D3** | — | — | — | — | — | PolyGammaExtend | polygamma_reg_base.h/trigonometric_function_utils_reg_base.h/zeta_reg_base.h | CalcPolygammaTmpSizeV2 | no | i32 | 仅 reg-based |
| `Pow` | elementwise | **D2** | Pow | pow.h | CalcPowTmpSize | !std::all_of(is_scalar_list.begin(), is_ | f16/f32/i32 | Pow | pow_reg_base.h | CalcPowTmpSizeV2 | any | f16/f32/i16/i32/ui8/ui16/bf16/i8/ui32 | 扩展头 pow.h vs pow_reg_base.h |
| `Prod` | reduce | **D2** | ReduceProd | reduce.h/reduce_init.h | CalcReduceTmpSize | no | f32 | ReduceProd | reduce_extend_reg_base.h/reduce_init_reg_base.h | CalcReduceTmpSizeV2 | no | f32 | 扩展头 reduce.h/reduce_init.h vs reduce_extend_reg_base.h/reduce_init_reg_base.h |
| `RShift` | other | **D3** | — | — | — | — | — | ShiftRight | — | — | no | i16/i32/i8/i64 | 仅 reg-based |
| `Rand` | layout | **D3** | — | — | — | — | — | Rand | random_reg_base.h | — | no | n/a | 仅 reg-based |
| `Randn` | layout | **D3** | — | — | — | — | — | Rand | random_reg_base.h | — | no | n/a | 仅 reg-based |
| `Reciprocal` | elementwise | **D2** | ReciprocalExtend | reciprocal.h | CalcDefaultTmpSize | no | f16/f32 | ? | reciprocal_reg_base.h | custom | no | f16/f32/bf16 | API 名 `ReciprocalExtend` vs `?`；扩展头 reciprocal.h vs reciprocal_reg_base.h |
| `Relu` | elementwise | **D0** | Relu | — | — | no | f16/f32/i32 | Relu | — | — | no | f16/f32/i32/ui8/i64 | 两侧一致且不依赖扩展头 |
| `Remainder` | elementwise | **D2** | RemainderExtend | remainder.h | CalcRemainderTmpSize | no | f32/i32 | RemainderExtend | remainder_reg_base.h | CalcVoidTmpSizeV2 | no | f16/f32/i32/bf16 | 扩展头 remainder.h vs remainder_reg_base.h |
| `RemovePad` | layout | **D1** | RemovePad | removepad.h | — | no | f16/f32/i16/i32/ui16/bf16/ui32 | RemovePad | removepad.h | — | no | f16/f32/i16/i32/ui16/bf16/ui32 | 两侧共用扩展头 removepad.h |
| `Round` | other | **D3** | — | — | — | — | — | RoundExtend | round_reg_base.h | CalcVoidTmpSizeV2 | no | f16/f32/bf16 | 仅 reg-based |
| `RoundToInt` | other | **D3** | — | — | — | — | — | RoundToInt | cast_reg_base.h | CalcVoidTmpSizeV2 | no | f16/f32/bf16 | 仅 reg-based |
| `Rsqrt` | elementwise | **D2** | RsqrtExtend | rsqrt.h | CalcRsqrtTmpSize | no | f16/f32 | ? | rsqrt_reg_base.h | custom | no | f16/f32/bf16 | API 名 `RsqrtExtend` vs `?`；扩展头 rsqrt.h vs rsqrt_reg_base.h |
| `Scalar` | elementwise | **D0** | Scalar | — | — | no | n/a | Scalar | — | — | no | n/a | 两侧一致且不依赖扩展头 |
| `ScalarData` | elementwise | **D0** | Scalar | — | — | no | n/a | Scalar | — | — | no | n/a | 两侧一致且不依赖扩展头 |
| `ScaledModifiedBesselK0` | other | **D3** | — | — | — | — | — | ScaledModifiedBesselK0Extend | modified_bessel_i0_reg_base.h/modified_bessel_k0_reg_base.h/modified_bessel_utils_reg_base.h/scaled_modified_bessel_k0_reg_base.h | CalcScaledModifiedBesselK0TmpSizeV2 | no | f32 | 仅 reg-based |
| `ScaledModifiedBesselK1` | other | **D3** | — | — | — | — | — | ScaledModifiedBesselK1Extend | modified_bessel_i1_reg_base.h/modified_bessel_k1_reg_base.h/modified_bessel_utils_reg_base.h/scaled_modified_bessel_k1_reg_base.h | CalcScaledModifiedBesselK1TmpSizeV2 | no | f32 | 仅 reg-based |
| `Select` | elementwise | **D2** | Where | duplicate.h/where.h | CalcSelectTmpSize | is_scalar_list[0] == false | f16/f32/i16/i32/i64 | WhereExtend | where_reg_base.h | — | !is_scalar_list[0] | f16/f32/i16/i32/i64 | API 名 `Where` vs `WhereExtend`；扩展头 duplicate.h/where.h vs where_reg_base.h；tmp 要 vs 不要 |
| `ShiftedChebyshevPolynomialT` | other | **D3** | — | — | — | — | — | ShiftedChebyshevPolynomialTExtend | shifted_chebyshev_polynomial_t_reg_base.h/shifted_chebyshev_polynomial_utils_reg_base.h | CalcVoidTmpSizeV2 | no | f32 | 仅 reg-based |
| `ShiftedChebyshevPolynomialU` | other | **D3** | — | — | — | — | — | ShiftedChebyshevPolynomialUExtend | shifted_chebyshev_polynomial_u_reg_base.h/shifted_chebyshev_polynomial_utils_reg_base.h | CalcVoidTmpSizeV2 | no | f32 | 仅 reg-based |
| `ShiftedChebyshevPolynomialV` | other | **D3** | — | — | — | — | — | ShiftedChebyshevPolynomialVExtend | shifted_chebyshev_polynomial_utils_reg_base.h/shifted_chebyshev_polynomial_v_reg_base.h | CalcVoidTmpSizeV2 | no | f32 | 仅 reg-based |
| `ShiftedChebyshevPolynomialW` | other | **D3** | — | — | — | — | — | ShiftedChebyshevPolynomialWExtend | shifted_chebyshev_polynomial_utils_reg_base.h/shifted_chebyshev_polynomial_w_reg_base.h | CalcVoidTmpSizeV2 | no | f32 | 仅 reg-based |
| `Sigmoid` | elementwise | **D2** | SigmoidExtend | sigmoid.h | CalcSigmoidTmpSize | no | f16/f32 | Sigmoid | — | CalcVoidTmpSizeV2 | no | f16/f32/bf16 | API 名 `SigmoidExtend` vs `Sigmoid`；扩展头 sigmoid.h vs - |
| `Sign` | elementwise | **D2** | SignExtend | cast.h/sign.h | CalcSignTmpSize | no | f16/f32/i32 | SignExtend | cast.h/sign_reg_base.h | CalcVoidTmpSizeV2 | no | f16/f32/i32/ui8/bf16/i64 | 扩展头 cast.h/sign.h vs cast.h/sign_reg_base.h |
| `SignBit` | other | **D3** | — | — | — | — | — | SignBitExtend | signbit_reg_base.h | CalcVoidTmpSizeV2 | no | f32/i32 | 仅 reg-based |
| `Sin` | other | **D3** | — | — | — | — | — | SinExtend | sin_reg_base.h | CalcSinTmpSizeV2 | no | f16/f32/bf16 | 仅 reg-based |
| `Sinh` | other | **D3** | — | — | — | — | — | SinhExtend | sinh_reg_base.h | CalcSinhTmpSizeV2 | no | f16/f32/bf16 | 仅 reg-based |
| `Softmax` | reduce | **D3** | — | — | — | — | — | SoftmaxARFullLoadExtend | softmax_af_reg_base.h | CalcSoftmaxTmpSizeV2 | no | f16/f32/bf16 | 仅 reg-based |
| `SphericalBesselJ0` | other | **D3** | — | — | — | — | — | SphericalBesselJ0Extend | bessel_j_utils_reg_base.h/spherical_bessel_j0_reg_base.h/trigonometric_function_utils_reg_base.h | CalcVoidTmpSizeV2 | no | f32 | 仅 reg-based |
| `Split` | split | **D2** | — | — | — | no | ? | Split | split_reg_base.h | CalcSplitTmpSizeV2 | no | f16/f32/i16/i32/ui8/ui16/bf16/i8/i64/ui32/ui64/bool/f64/c64/c128/qi8/qi16/qi32/qui8/qui16 | API 名 `-` vs `Split`；扩展头 - vs split_reg_base.h；tmp 不要 vs 要 |
| `Sqrt` | elementwise | **D2** | Sqrt | — | — | no | f16/f32 | ? | sqrt_reg_base.h | — | no | f16/f32/bf16 | API 名 `Sqrt` vs `?`；扩展头 - vs sqrt_reg_base.h |
| `Square` | other | **D3** | — | — | — | — | — | Square | scalar_mul.h | — | no | f16/f32/i16/i32/ui8/ui16/bf16/i64/ui32/ui64 | 仅 reg-based |
| `Store` | datamove | **D2** | Store | datacopy.h | — | no | f16/f32/i16/i32/ui8/ui16/bf16/i8/i64/ui32 | DataCopyPadExtend | datacopy_reg_base.h | — | no | f16/f32/i16/i32/ui8/ui16/bf16/i8/i64/ui32/ui64/bool | API 名 `Store` vs `DataCopyPadExtend`；扩展头 datacopy.h vs datacopy_reg_base.h |
| `Sub` | elementwise | **D2** | Sub | scalar_sub.h | CalcSubTmpSize | any | f16/f32/i16/i32 | SubExtend | sub_reg_base.h | — | no | f16/f32/i16/i32/ui8/ui16/bf16/i8/i64/ui32/ui64 | API 名 `Sub` vs `SubExtend`；扩展头 scalar_sub.h vs sub_reg_base.h；tmp 要 vs 不要 |
| `Sum` | reduce | **D2** | ReduceSum | reduce.h/reduce_init.h | CalcReduceTmpSize | no | f32/i32 | ReduceSum | reduce_extend_reg_base.h/reduce_init_reg_base.h | CalcReduceTmpSizeV2 | no | f16/f32/i16/i32/bf16/i8/i64 | 扩展头 reduce.h/reduce_init.h vs reduce_extend_reg_base.h/reduce_init_reg_base.h |
| `Tan` | other | **D3** | — | — | — | — | — | TanExtend | tan_reg_base.h | CalcTanTmpSizeV2 | no | f16/f32/bf16 | 仅 reg-based |
| `Tanh` | elementwise | **D2** | Tanh | — | CalcTanhTmpSize | no | f16/f32 | TanhExtend | tanh_reg_base.h | CalcVoidTmpSizeV2 | no | f16/f32/bf16 | API 名 `Tanh` vs `TanhExtend`；扩展头 - vs tanh_reg_base.h |
| `Transpose` | layout | **D2** | Transpose | transpose.h/transpose_base_type.h | CalcDefaultTmpSize | no | f16/f32/i16/i32/ui16/ui32 | TransposeExtend | transpose_reg_base.h | CalcTransposeTmpSizeV2 | no | f16/f32/i16/i32/ui16/bf16/i64/ui32/bool | API 名 `Transpose` vs `TransposeExtend`；扩展头 transpose.h/transpose_base_type.h vs transpose_reg_base.h |
| `TrueDiv` | elementwise | **D2** | TrueDivExtend | scalar_div.h/true_div.h | CalcTrueDivTmpSize | any | f16/f32/i32 | DivExtend | div_reg_base.h | — | no | f16/f32/bf16 | API 名 `TrueDivExtend` vs `DivExtend`；扩展头 scalar_div.h/true_div.h vs div_reg_base.h；tmp 要 vs 不要 |
| `Trunc` | other | **D3** | — | — | — | — | — | Trunc | — | CalcTruncTmpSizeV2 | no | f16/f32/bf16 | 仅 reg-based |
| `TruncDiv` | other | **D3** | — | — | — | — | — | TruncDivExtend | trunc_div_reg_base.h | — | no | f16/f32/bf16 | 仅 reg-based |
| `TruncToInt` | other | **D3** | — | — | — | — | — | TruncToInt | cast_reg_base.h | CalcVoidTmpSizeV2 | no | f16/f32/bf16 | 仅 reg-based |
| `Ub2ub` | layout | **D0** | DataCopy | — | — | no | f16/f32/i16/i32/ui8/ui16/i8/ui32/ui64 | DataCopy | — | — | no | f16/f32/i16/i32/ui8/ui16/i8/i64/ui32/ui64 | 两侧一致且不依赖扩展头 |
| `Unsupported` | structural | **D0** | Unsupported | — | — | no | n/a | Unsupported | — | — | no | n/a | 两侧一致且不依赖扩展头 |
| `VectorFunc` | structural | **D2** | — | — | — | no | all | — | utils_reg_base.h | — | no | all | 扩展头 - vs utils_reg_base.h |
| `Where` | elementwise | **D2** | Where | duplicate.h/where.h | CalcWhereTmpSize | is_scalar_list[0] == false | f16/f32/i16/i32/i64 | WhereExtend | where_reg_base.h | — | !is_scalar_list[0] | f16/f32/i16/i32/ui8/ui16/bf16/i8/i64/ui32/ui64 | API 名 `Where` vs `WhereExtend`；扩展头 duplicate.h/where.h vs where_reg_base.h；tmp 要 vs 不要 |
| `Workspace` | structural | **D0** | Workspace | — | — | no | f16/f32/i16/i32/ui8/ui16/bf16/i8/i64/ui32/ui64 | Workspace | — | — | no | f16/f32/i16/i32/ui8/ui16/bf16/i8/i64/ui32/ui64/bool | 两侧一致且不依赖扩展头 |
| `Xor` | other | **D3** | — | — | — | — | — | Xor | — | CalcXorTmpSizeV2 | no | i16/ui8/ui16 | 仅 reg-based |
| `Zeta` | other | **D3** | — | — | — | — | — | ZetaExtend | zeta_reg_base.h | CalcVoidTmpSizeV2 | no | f32 | 仅 reg-based |

## 引用了未注册头名的 op

无。所有 impl 引用的头名都能在 `AscendCApiRegistry` 的三批注册里找到。

## API 名书写异常

`GetApiName()` 的约定是**裸名**（`Add` / `AbsExtend`），由 emitter 决定要不要加命名空间前缀。下列 op 破了这个约定，迁移时 emitter 不能无条件拼 `AscendC::`。

| op | mem API | reg API |
|---|---|---|
| `Maximum` | `AscendC::Max` | `AscendC::Max` |
| `Minimum` | `AscendC::Min` | `AscendC::Min` |
| `VectorFunc` | `-` | `-` |

## 生成命令

```bash
python3 autofuse/mlir/tools/af-mlir-tools/scan_ascir_op_lowering_matrix.py \
    --format markdown --output autofuse/mlir/docs/guides/ascir_op_lowering_matrix.md
```
