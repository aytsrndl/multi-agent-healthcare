# Multi-Agent LLM Framework for Healthcare Question Revision

This repository contains a modular multi-agent LLM framework for revising patient-generated healthcare questions while attempting to preserve the patient's original clinical intent.

The current research focus is on **very low health-literacy questions**. Multiple role-specialized healthcare agents revise the same patient question under different communication topologies. The resulting revised questions can then be evaluated using downstream healthcare LLMs and external fidelity / quality evaluation.

## Research Objective

The main objective is to study whether the **communication structure between role-specialized LLM agents** affects:

- semantic fidelity to the original patient question,
- clarity of the revised question,
- downstream healthcare-answer quality,
- agreement between agents,
- and computational cost.

The framework separates the underlying LLM from the multi-agent communication topology so that the same experiment can be repeated across different model backbones.

Conceptually:

```text
Patient Question
      |
      v
Role-Specialized Clinical Agents
      |
      v
Communication Topology
      |
      v
Candidate Revisions
      |
      v
Consensus / Anonymous Voting
      |
      v
Final Revised Question
