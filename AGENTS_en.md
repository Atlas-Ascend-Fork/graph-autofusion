# Graph-autofusion Agent Guide

See [AGENTS.md](AGENTS.md) for the full repository guidance.

## Sequence Diagram Colors

When feature designs, issue analyses, or PR descriptions use sequence diagrams to explain a workflow change, show the original and updated workflows and identify the key steps. Include a legend beside the diagram and use both color and text labels inside it.

- **Light red · Original workflow** shows the previous call order, key branches, and confirmed failure points.
- **Light green · Current implementation** shows the updated call order and identifies the steps that were added or changed.
- **Light blue · Proposed workflow** shows the target workflow or follow-up changes still under design.

In Mermaid `sequenceDiagram`, use `rect rgba(...)` to highlight the relevant steps and `Note over` to label the phase and its key changes. Shared steps can use the default styling for easy comparison.
