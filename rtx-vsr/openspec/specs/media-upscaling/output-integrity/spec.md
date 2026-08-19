## Purpose

Defines the guarantees the service makes about the pixels it returns, so that a defect in
the underlying super-resolution engine surfaces as a visible failure rather than as a
corrupted image the caller cannot distinguish from a good one.

## Requirements

### Requirement: Corrupted super-resolution output is rejected

The super-resolution engine can return a structurally valid image whose contents are
corrupt — most notably a colour channel collapsed to near-zero, which renders as a strong
colour cast. The service SHALL inspect every upscaled frame and SHALL fail the job when the
result is corrupt, rather than returning it.

Detection MUST cover a colour channel collapsing to near-zero when the corresponding input
channel carried signal, and non-finite pixel values.

#### Scenario: Collapsed colour channel

- **WHEN** the engine returns a frame whose blue channel is near-zero while the source frame
  had blue content
- **THEN** the job fails with a server-side error naming the affected channel
- **AND** no image is offered to the caller for download

#### Scenario: Non-finite pixel values

- **WHEN** the engine returns a frame containing NaN or infinite values
- **THEN** the job fails with a server-side error

#### Scenario: Valid output passes through

- **WHEN** the engine returns a frame whose channels all carry signal consistent with the
  source
- **THEN** the frame is delivered unchanged by the integrity check

### Requirement: Output above the verified super-resolution limit

Super-resolution output is only trusted up to an edge of 15360 pixels. For a larger
requested output the service SHALL still deliver the full requested size, by running
super-resolution at or below the trusted limit and resampling the result up to the
requested dimensions.

#### Scenario: Request above the trusted limit

- **WHEN** a request computes an output of 16384x16384
- **THEN** super-resolution runs at an edge no greater than 15360
- **AND** the super-resolution output passes the integrity check
- **AND** the delivered image is exactly the computed output size

#### Scenario: Request within the trusted limit

- **WHEN** a request computes an output whose edges are all 15360 or smaller
- **THEN** super-resolution produces the final image directly with no resampling step

### Requirement: Frame independence

Each returned frame SHALL contain the pixels produced for that frame. Processing a later
frame MUST NOT alter the contents of a frame already produced.

#### Scenario: Batch of distinct frames

- **WHEN** a video whose frames have visibly different content is upscaled in a single batch
- **THEN** each output frame corresponds to its own input frame
- **AND** no output frame is a duplicate of a neighbouring frame
