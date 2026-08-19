## Purpose

Gets the SparkVSR and reference-model weights onto persistent Modal storage ahead of time, so that GPU containers start from a warm local filesystem instead of downloading tens of gigabytes on every cold start.

## ADDED Requirements

### Requirement: One-shot weight provisioning

The project SHALL provide a single command that downloads all weights required for inference into persistent storage, and SHALL be safe to re-run.

#### Scenario: First run

- **WHEN** the provisioning command runs against empty storage
- **THEN** it downloads the SparkVSR pipeline weights and the reference model weights and reports the total size written

#### Scenario: Re-run with weights already present

- **WHEN** the provisioning command runs again with weights already in place
- **THEN** it completes without re-downloading unchanged files

#### Scenario: Missing credentials

- **WHEN** the provisioning command runs without the credential needed to fetch the weights
- **THEN** it fails with a message naming the missing credential rather than leaving partial weights in storage

### Requirement: Precomputed empty-prompt embedding

Provisioning SHALL precompute and store the text embedding for the empty prompt, so that inference never loads the text encoder.

#### Scenario: Provisioning completes

- **WHEN** provisioning finishes
- **THEN** the empty-prompt embedding is stored alongside the model weights

#### Scenario: Inference container startup

- **WHEN** a GPU container loads the model for inference
- **THEN** it loads the stored empty-prompt embedding and does not load the text encoder or tokenizer into memory

#### Scenario: Embedding missing

- **WHEN** a GPU container starts and the stored empty-prompt embedding is absent
- **THEN** startup fails with a message directing the operator to run provisioning, rather than silently loading the text encoder

### Requirement: Inference does not download weights

A GPU container SHALL read all weights from persistent storage and SHALL NOT download model weights at inference time.

#### Scenario: Cold start

- **WHEN** a GPU container starts with weights present in persistent storage
- **THEN** it loads them from storage with no network fetch of model files
- **AND** the model load time is recorded in the logs

### Requirement: Weight integrity is verified before serving

The service SHALL detect incomplete or missing weights at container startup rather than mid-job.

#### Scenario: Incomplete weights

- **WHEN** a GPU container starts and the expected weight files are missing or incomplete
- **THEN** startup fails immediately with a message identifying what is missing
- **AND** no job is accepted by that container
