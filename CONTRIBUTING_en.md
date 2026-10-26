# Contributing Guide

This project welcomes developers to experience and participate in contributions. Before participating in community contributions, see [cann-community](https://gitcode.com/cann/community) to understand the code of conduct, sign the CLA agreement, and understand the contribution process of the source code repository. This repository details the prerequisites for participating in CANN open source project contributions, including but not limited to:
1. How to submit a PR
2. Gitcode workflow
3. Pipeline trigger commands
4. Code review
5. Other notes
For details, see [cann-community](https://gitcode.com/cann/community).

In addition, developers need to pay attention to the following points when preparing local code and submitting PRs:

1. When submitting a PR, carefully fill in the business background, purpose, solution, and other information according to the PR template.
2. If your modification is not a simple bug fix but involves new features, new interfaces, new configuration parameters, or modified code flow, discuss the solution through an Issue first to avoid your code being rejected. If you are unsure whether your modification can be classified as a "simple bug fix," you can also discuss the solution by submitting an Issue.
3. When submitting a PR, ensure that your code conforms to the project code specifications. For details, see Google's [Open Source Code Style Guide](https://google.github.io/styleguide/), including but not limited to:
   - Code formatting
   - Comment specifications
   - Variable naming specifications
   - Function naming specifications
   - Class naming specifications
   - Interface naming specifications
   - Configuration parameter naming specifications
   - Code flow specifications
4. The pipeline runs a code check and rejects code that has not been formatted by clang-format through pre-commit. We recommend installing pre-commit and running `pre-commit install` in the repository root so that each local commit is automatically checked and formatted.
5. When submitting a PR, if there are multiple invalid commits, it is recommended that you perform a rebase operation before submitting the PR to merge multiple commits into one to maintain code simplicity and readability. For details, see [git rebase](https://git-scm.com/docs/git-rebase). At the same time, the commit message must also conform to the project code specifications and clearly describe the intent and content of this change. The format is: <type>: <brief description>. For example:

|Type|Description|Example|
|--|--|--|
|feat|New feature|feat: Add user registration function|
|fix|Bug fix|fix: Fix login session expiration issue|
|docs|Documentation update|docs: Update API usage instructions|
|style|Code format adjustment (does not affect logic)|style: Adjust code indentation|
|refactor|Refactoring (non-feature addition/fix)|refactor: Optimize user service class structure|
|perf|Performance optimization|perf: Reduce database query count|
|test|Test related|test: Add login function unit test|
|chore|Build/toolchain change|chore: Update webpack configuration|
|ci|CI configuration related|ci: Add automated test process|

Developer contribution scenarios mainly include:

- Bug Fix

  If you discover certain bugs in this project and want to fix them, you are welcome to create an Issue for feedback and tracking.

  You can follow the [Submit Issue/Handle Issue Task](https://gitcode.com/cann/community#提交Issue处理Issue任务) guide to create a `Bug-Report|Defect Feedback` type Issue to describe the bug, then enter "/assign" or "/assign @yourself" in the comment box to assign the Issue to yourself for handling.


- Contribute New Features

  If you discover certain missing features in this project and want to add them, you are welcome to create an Issue for feedback and tracking.

  You can follow the [Submit Issue/Handle Issue Task](https://gitcode.com/cann/community#提交Issue处理Issue任务) guide to create a `Requirement|Feature Suggestion` type Issue to describe the new feature and provide your design solution, then enter "/assign" or "/assign @yourself" in the comment box to assign the Issue to yourself for tracking implementation.


- Document Correction

  If you discover certain document description errors in this project, you are welcome to create an Issue for feedback and correction.

  You can follow the [Submit Issue/Handle Issue Task](https://gitcode.com/cann/community#提交Issue处理Issue任务) guide to create a `Documentation|Document Feedback` type Issue to point out the corresponding document problem, then enter "/assign" or "/assign @yourself" in the comment box to assign the Issue to yourself to correct the corresponding document description.

- Help Solve Others' Issues

  If you have appropriate solutions for problems encountered by others in the community, you are welcome to post comments in the Issue to exchange ideas, help others solve problems and pain points, and jointly optimize usability.

  If the corresponding Issue requires code modification, you can enter "/assign" or "/assign @yourself" in the Issue comment box to assign the Issue to yourself to track and assist in solving the problem.

## Compiler and Runtime Library Compatibility

The minimum GCC version for source builds is **7.3.0**; see [build dependencies](docs/en/build.md#421-install-dependencies).
Avoid introducing features unsupported by GCC 7.3.0. The project does not require support for older compilers.
The Autofuse [runtime requirements](docs/en/autofuse/introduction/enable_autofuse_inductor.md#runtime-requirements) separately specify GCC >= 9.5.0.
Check the requirements for the actual component and use case; do not treat the build compiler requirement as the runtime library ABI baseline.

### Development and Review Checks

1. **Check language, standard library, and compiler extension support separately.** Enabling `-std=c++17` does not mean GCC 7.3.0 implements every C++17 feature.
   Check new language features against the [GCC C++ support table](https://gcc.gnu.org/projects/cxx-status.html), and new standard library interfaces against the
   [libstdc++ support table](https://gcc.gnu.org/onlinedocs/libstdc++/manual/status.html).
   For example, the standard `<filesystem>` header was introduced in GCC 8 and cannot be assumed available in GCC 7.3.0.
   Prefer existing compatible implementations in the project. Before using features that require a newer compiler, define the support changes and delivery impact.
2. **Compile with the minimum supported toolchain.** Use GCC/G++ 7.3.0 with its matching headers and standard library to configure and build the relevant targets in a separate build directory.
   Limit all builds to `-j 8` and run the relevant UT/ST. Reconfigure the build directory when changing compilers to avoid reusing an old cache.
   Using `-std=c++17` with a newer compiler or checking `__GNUC__` alone does not demonstrate compatibility with the minimum version.
3. **Inspect optimized ELF dependencies.** Compilers can turn ordinary arithmetic into runtime library calls. For example, signed `__int128` division and remainder operations on the same operands
   can generate `__divmodti4@GCC_7.0.0` in optimized x86-64 builds.
   Build with the actual delivery toolchain and optimization options. For each architecture, compare `GCC_*`, `GLIBCXX_*`, `CXXABI_*`, and `GLIBC_*` against its accepted delivery baseline,
   and verify loading in the target runtime environment. Include third-party static libraries and generated code when inspecting the final shared library.
4. **Explain new symbol version dependencies.** Record their source, whether the target runtime libraries provide them, and whether runtime requirements need to change.
   `GCC_7.0.0` is an ELF symbol version name, not a compiler version. This repository currently provides no separate libgcc 4.8 compatibility guarantee; do not infer one.
   If delivery requirements include compatibility with older runtime libraries, eliminate unnecessary new dependencies and add regression tests for actual compiled artifacts.

Use the following commands to inspect delivered shared libraries and their dependency chains. `gcc --version` does not identify the runtime library versions actually loaded by a process:

```bash
readelf --wide --dyn-syms <delivered-library.so>
readelf --wide --version-info <delivered-library.so>
ldd -v <trusted-delivered-library.so>
```

The PGO timestamp ABI regression test compiles the actual source file with both `-O2` and `-O3` and inspects its ELF dynamic symbols.
It guards against reintroducing the dependency described above and does not replace compatibility verification of the complete delivery package and target runtime environment:

```bash
python -m pytest autofuse/tests/st/python/test_pgo_timestamp_abi.py -q
```
