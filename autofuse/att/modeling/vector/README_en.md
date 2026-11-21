# Vector Performance Models

Follow the categories in `autofuse/codegen/api_call/`: `broadcast/`, `concat/`, `elewise/`, `gather/`, `reduce/`, `transpose/`.
Place `cast` under `elewise/cast/` and add other elementwise API models under `elewise/` as needed.
Archive datacopy load/store models under mte2/mte3 respectively; `utils` contains shared utilities, not model categories.

## Content to Add

- Modeling methods: derive v35 expressions from AscendC API implementation logic and document underlying cost sources.
- Test case design and execution instructions.
- Validation results, error analysis, and applicability.

See the [archive conventions](../README_en.md) for the model file structure.
