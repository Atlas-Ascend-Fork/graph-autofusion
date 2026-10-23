# Schedule

The Schedule module of AutoFuse is the core component connecting computation definition and efficient code generation. Its core capability is to flexibly rearrange and optimize the computation process: without changing the computation semantics, flexibly adjust the computation implementation through computation rearrangement and scheduling primitive calls to obtain better performance. Optimization methods include loop merging, solution space generation, parallel optimization, memory optimization, and multi-template generation.

When a user defines a HintGraph describing Scalar computation logic through AscIR, the Schedule module will perform scheduling optimization on the computation based on hardware characteristics, and generate multiple ImplGraphs expressing splitting and memory relationships, providing a foundation for the Codegen and Auto Tiling modules to support them in generating high-performance Kernel code.

## Loop Merging

Loop merging is an important loop transformation technique. Its core role is to reduce memory access times, reduce control overhead, improve data locality by reconstructing the loop structure, and pave the way for subsequent optimizations, ultimately improving program execution efficiency without changing the computation result.

For example, suppose there are the following two layers of loops:

```text
for i in range(N):
    for j in range(M):
        C[i][j] = A[i][j] + B[i][j]
```

Add is an element-wise operation, so it can be merged into a single linear loop:

```text
for fused in range(N * M):
    i = fused // M
    j = fused % M
    C[i][j] = A[i][j] + B[i][j]
```

The corresponding AscIR expression is as follows:

```python
z0 = graph.create_axis("z0", N)
z1 = graph.create_axis("z1", M)

Load load0("load_0");
load0.x = data0.y;
load0.attr.sched.axis = {z0.id, z1.id};
load0.y.axis = {z0.id, z1.id};
load0.y.repeats = {N, M};
load0.y.strides = {M, 1};

Load load1("load_1");
load1.x = data1.y;
load1.attr.sched.axis = {z0.id, z1.id};
load1.y.axis = {z0.id, z1.id};
load1.y.repeats = {N, M};
load1.y.strides = {M, 1};

Add add("add");
add.x1 = load0.y;
add.x2 = load1.y;
add.attr.sched.axis = {z0.id, z1.id};
add.y.axis = {z0.id, z1.id};
add.y.repeats = {N, M};
add.y.strides = {M, 1};
```

After loop merging:

```python
Load load0("load_0");
load0.x = data0.y;
load0.attr.sched.axis = {z0z1.id};
load0.y.axis = {z0z1.id};
load0.y.repeats = {N * M};
load0.y.strides = {1};

Load load1("load_1");
load1.x = data1.y;
load1.attr.sched.axis = {z0z1.id};
load1.y.axis = {z0z1.id};
load1.y.repeats = {N * M};
load1.y.strides = {1};

Add add("add");
add.x1 = load0.y;
add.x2 = load1.y;
add.attr.sched.axis = {z0z1.id};
add.y.axis = {z0z1.id};
add.y.repeats = {N * M};
add.y.strides = {1};
```

## Generating TilingCase

In automatic fusion technology, tiling solution space generation is a key link in achieving efficient computation scheduling. Its core goal is to provide diverse tiling strategy options for complex computation tasks, so that the subsequent optimizer can screen out the optimal solution from them. In simple terms, the process of generating tiling solution space can be understood as a systematic enumeration of "blocking possibilities" for input data or computation tasks, and each solution space is called a TilingCase.

The design of the splitting method is closely related to the operator implementation characteristics. To achieve systematic splitting strategy enumeration for diverse operators, operators are first abstracted into 9 compute_types based on their implementation characteristics (compute and view operators as shown in the figure below). Operators of the same compute_type have similar computation logic and data access patterns, so they can share a set of tiling splitting strategy frameworks.

<div style="text-align: center;">
<img src="../../figures/schedule_tiling_strategy.png" alt="Tiling splitting strategy" style="width: 50%; max-width: 800px;">
</div>

To materialize this strategy framework, we normalize and group the axes of operators, and uniformly divide all axes into three dimensional sets (xgroup, ygroup, rgroup), with specific definitions as follows:

- **xgroup**: A grouping designed specifically for view operators such as Concat. Taking Concat as an example, it will be divided along the Concat axis, the axes before the Concat axis are divided into xgroup, and the Concat axis and subsequent axes are divided into ygroup.
- **ygroup**: Corresponding to the loop axis grouping of Elementwise, Broadcast and other types of operators.
- **rgroup**: Reduce operations usually have special splitting requirements for the reduce axis, so all reduce axes are placed separately in rgroup.

> **Note**: The core reason for introducing xgroup, ygroup, rgroup is to support the "double splitting" requirement in complex scenarios. For example, in a computation graph containing mixed reduce, ygroup controls the loop splitting of elementwise, and the axes in rgroup control the loop splitting in the reduce direction.

After completing the axis grouping of a single operator, the grouping strategies of all operators in the computation graph need to be merged through preset merging rules (Merge). The merged result will be used as a unified splitting strategy applicable to all operators in the entire graph, providing a foundation for subsequent solution space generation.

Through the above grouping and merging mechanism, two core functions can be achieved:

- Filter out splitting methods applicable to all nodes in the computation graph to form effective TilingCases.
- Verify the fusibility of two graphs by judging whether the tiling groups of different AscGraphs can be successfully merged.

Taking the following case as an example, the principle of generating TilingCase through tiling group merging is introduced:

```python
z0 = graph.create_axis("z0", s0)
z1 = graph.create_axis("z1", s1)

data = ascir.ops.Data('data', graph)
data.y.dtype = ascir.dtypes.float32

# Declare the Load operator
load = ascir.ops.Load('load')
load.attr.sched.axis = [z0, z1]  # Scheduling axis
load.x = data.y
load.y.axis = [z0, z1]  # Tensor output axis
load.y.size = [s0, s1]  # Tensor output size
load.y.strides = [s1, 1]  # Tensor output stride

# Declare the Abs operator
abs = ascir.ops.Abs('abs')
abs.attr.sched.axis = [z0, z1]
abs.x = load.y
abs.y.axis = [z0, z1]
abs.y.size = [s0, s1]
abs.y.strides = [s1, 1]

# Declare the Max operator
max = ascir.ops.Max('max')
max.attr.sched.axis = [z0, z1]
max.x = abs.y
max.y.axis = [z0, z1]
max.y.size = [s0, 1]  # Reduce operation on z1 axis
max.y.strides = [1, 0]
```

abs is an elewise operator, and each axis has no difference in computation. Therefore, as long as the memory is continuous, multiple axes can be merged into one axis for splitting. As shown in the figure below, first block the axis, 15 is divided into 3 blocks, block0 is the purple part, and tiling blocking is performed within block0. At this time, the tiling block does not fill the part allocated by block0, so an additional for loop is needed within the block.

<div style="text-align: center;">
<img src="../../figures/schedule_elewise_split.png" alt="elewise operator splitting" style="width: 45%; max-width: 800px;">
</div>

The splitting of reduce is more complex and requires double splitting in implementation: the row direction is the elewise axis, and the column direction is the reduce axis; first block in the row direction, write loops within the block, and then add another layer of for loops in the column direction.

<div style="text-align: center;">
<img src="../../figures/schedule_reduce_split.png" alt="reduce operator splitting" style="width: 45%; max-width: 800px;">
</div>

Through the merging rule of TilingGroup:

```text
()(z0,z1)()  Merge  ()(z0)(z1)  =>  ()(z0)(z1)
```

The (z0, z1) axes originally belonging to ygroup of the abs operator are split, and the z1 axis is adjusted to rgroup. The core purpose of this adjustment is to make the abs operator maintain a unified splitting strategy with the subsequent reduce type operator.

## Parallel Optimization

### Loop Splitting

The core role of loop splitting is to clarify the division between outer loops suitable for parallelism and inner loops suitable for vectorization by introducing new loop levels.

For each TilingCase, the axes existing in xgroup, ygroup, rgroup are split into ub_out and ub_in:

<div style="text-align: center;">
<img src="../../figures/schedule_loop_split.png" alt="Loop splitting" style="width: 50%; max-width: 800px;">
</div>

For example: {z0, z1, z2}, splitting on z1, splits z1 into z1T and z1t, where z1T is ub_out and z1t is ub_in.

### Vectorization

Vectorization is a key technology that uses hardware SIMD (Single-Instruction Multiple-Data stream processing) units to improve the efficiency of data parallel computing. By converting single-element operations into vector operations, it significantly reduces the number of instruction executions and improves hardware utilization.

For example, for the following loop:

```text
for (int i = 0; i < 256; i++) {
    c[i] = a[i] + b[i];
}
```

Non-vectorized execution requires 256 addition instructions, while vectorized execution only requires one addition instruction.

After selecting an axis as the ub splitting axis in each group, ub_in and its inner axes are used as vectorization axes. At this time, since the grouping is generated according to xyr, the vectorization axes generated in this order have inconsistent axis order with memory layout, which will introduce non-contiguous data movement, so reordering is required according to the output axis order.

<div style="text-align: center;">
<img src="../../figures/schedule_vectorize_reorder.png" alt="Vectorization reordering" style="width: 60%; max-width: 800px;">
</div>

For example, the output tensor axis order is (a, b, c, d), the axis grouping is (a, c),(b, d)(), the vectorization axis group generated in this order is (a_in, c, b_in, d), which needs to be adjusted to (a_in, b_in, c, d).

> **Note**: The Schedule phase sets the same vectorization axis for the entire graph. For some APIs, due to instruction limitations, not all vectorization axes can be vectorized. At this time, the CodeGen phase needs to throw out the axes that cannot be vectorized and process them with for loops.
>
> For example, if the vectorization axes are [z1,z2,z3], it is equivalent to 3 layers of loops. If the instruction supports 3 layers of loops, it can be written as `vector[z1,z2,z3]`; but if the instruction can only support two layers of loops, the CodeGen needs to give the following code:
>
> ```text
> for (i in z1) {
>     vector[z2,z3]
> }
> ```

### Loop Merging, Loop BindCore

Loop merging and loop bindcore are often used together for optimization. The loop splitting stage produces ub_out axes in each group.

1. Merge all non-reduce axes into one axis, and merge all reduce axes into one axis to reduce the number of loop nesting layers.
2. Split the merged loop to get outer and inner layers.
3. Bind the split outer loop to multiple blocks to achieve parallelism, and the inner layer is digested through loops.

<div style="text-align: center;">
<img src="../../figures/schedule_loop_merge_bindcore.png" alt="Loop merging and loop bindcore" style="width: 60%; max-width: 800px;">
</div>

For example: {z0, z1T, z1t, z2}, z0 and z1T will first be merged into z0z1T, then multi-core splitting is performed on z0z1T. Since z0z1T may exceed the number of logical AI Core cores participating in the calculation, an additional layer of loop is needed after core splitting, and z0z1T will be further split into z0z1TB and z0z1Tb. The specific values are calculated by Auto Tiling in the Tiling phase.

## Memory Optimization

Currently, memory reuse is mainly based on node reference relationships. To improve the reuse effect, memory with similar occupied sizes is allocated to the same group as much as possible, and then reused within the group.

The pseudo-code for memory reuse is as follows:

```python
for (node in all nodes) {
    for (output in node.outputs) {
        # Mark the number of dependencies of the tensor
        output->sch.depends = output->anchor->GetPeerInDataNodesSize();
        # try reuse from free queue
    }
    for (input in node.inputs) {
        input->sch.depends--;
        if (input->sch.depends == 0) {
            Enqueue(input->opt.reuse_id); # Mark as freeTensor, can be reused by subsequent nodes
        }
    }
}
```

For some APIs, the output can directly reuse the input. For this type of API, Inplace reuse can be used, that is, the output directly reuses the input memory. As shown in the figure below, 3 blocks of memory are required before Inplace reuse:

<div style="text-align: center;">
<img src="../../figures/schedule_mem_reuse_before.png" alt="Before memory reuse" style="width: 25%; max-width: 800px;">
</div>

After Inplace reuse, only 2 blocks of memory are needed:

<div style="text-align: center;">
<img src="../../figures/schedule_mem_reuse_after.png" alt="After memory reuse" style="width: 25%; max-width: 800px;">
</div>

## Multi-template Generation

For a computation graph, there may be multiple implementation methods. Taking tail-axis concat as an example, you can first combine multiple small packets into a large packet on UB through ub_concat and then move the complete packet out, or you can directly use non-contiguous data movement to complete the rearrangement on GM (Global Memory). The former can significantly improve MTE (Memory Transfer Engine, the data transfer engine of an AI Core) efficiency in small-shape scenarios, thereby providing better performance. However, ub_concat also requires the inner axis to be fully loaded, which makes it unavailable in some scenarios. When the Schedule phase cannot determine which template to choose, it usually generates a general template suitable for any shape and a performance-optimized template for specific scenarios; the Auto Tiling module decides which template to use during the tiling phase.

- UB concat template:

<div style="text-align: center;">
<img src="../../figures/schedule_ub_concat_template.png" alt="UB concat template" style="width: 35%; max-width: 800px;">
</div>

- Graph rewrite template:

<div style="text-align: center;">
<img src="../../figures/schedule_graph_rewrite_template.png" alt="Graph rewrite template" style="width: 35%; max-width: 800px;">
</div>

<a id="template-semantic-propagation"></a>

## Semantic Propagation for New Templates

New templates and changes affecting implementation categories must follow the [Codegen semantic naming requirements](codegen.md#template-semantic-naming). Schedule records actual scheduling outcomes; it must not infer features that only final Codegen API selection can establish.

- Set `ScheduleTask.semantic_name` at the actual generator/strategy selection point and propagate it to `ScheduledResult.semantic_name`. Reuse existing graph tags for required roles and transformations.
- Apply a role only to the affected logical group. A graph-level parent strategy must not overwrite group responsibilities such as `Phase1`, `Phase2`, or `Norm`. Groups may have different names; a graph need not have one uniform name.
- Record a transform only after the graph or data-movement transformation succeeds. Failure must not contaminate the original graph or other candidates. Preserve corresponding tags through copying/derivation and reflect the final outcome after fallback.
- Preserve complete graph/result/group/key/subcase/template ownership. Restore the current logical identity after reuse; neither a local `group_id` nor a physical primary uniquely identifies it. Commit candidate-private selection state with the final tiling data.
- State category or inheritance rationale and information sources in the PR. Reuse existing tests for affected selection, copying, failure, fallback, and multiple-group/reuse boundaries; adapt final API features under the Codegen requirements.

## Related Links

- Back to [AutoFuse Architecture Introduction](../../introduction/architecture.md)
