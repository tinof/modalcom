## Purpose

Lets a caller submit a video for SparkVSR super-resolution and retrieve the restored result, defining the request contract, job lifecycle, and the quality and fidelity guarantees of the returned file.

## ADDED Requirements

### Requirement: Video submission and asynchronous retrieval

The service SHALL accept a video submission over an authenticated HTTP endpoint, return a job identifier immediately, and serve the finished file from a separate retrieval endpoint once processing completes.

#### Scenario: Successful submission and retrieval

- **WHEN** a caller submits a supported video file with valid parameters
- **THEN** the service returns a job identifier without waiting for processing to finish
- **AND** requests for that job's result return a pending status while processing continues
- **AND** requests for that job's result return the processed video file once processing completes

#### Scenario: Unauthenticated request

- **WHEN** a request arrives without valid credentials
- **THEN** the service rejects it and no GPU work is started

#### Scenario: Processing failure

- **WHEN** processing fails for a submitted job
- **THEN** the result endpoint reports the failure with a diagnostic message rather than returning a partial or corrupt file

### Requirement: Pre-GPU request validation

The service SHALL validate every request parameter before allocating GPU capacity, and SHALL reject invalid requests with an error identifying the offending parameter.

#### Scenario: Unsupported input

- **WHEN** a caller submits a file that is not a decodable video, or requests a reference mode, target resolution, or chunk configuration outside the supported range
- **THEN** the service rejects the request with a message naming the invalid parameter
- **AND** no GPU container is started for that request

#### Scenario: Reference index spacing violated

- **WHEN** a caller supplies explicit reference frame indices spaced 4 frames apart or closer
- **THEN** the service rejects the request, because the model requires reference indices to be more than 4 frames apart

### Requirement: Output resolution control

The service SHALL let the caller choose the output resolution, defaulting to 3840x2160, and SHALL preserve the source aspect ratio.

#### Scenario: Default 4K output

- **WHEN** a caller submits a 1920x1080 video without specifying a target resolution
- **THEN** the output video is 3840x2160

#### Scenario: Same-resolution detail restoration

- **WHEN** a caller requests a target resolution equal to the source resolution
- **THEN** the output video has the same dimensions as the source and has been processed by the model rather than passed through unchanged

#### Scenario: Non-16:9 source

- **WHEN** a caller submits a source whose aspect ratio differs from the requested target dimensions
- **THEN** the output preserves the source aspect ratio rather than stretching the image

### Requirement: Output encoding fidelity

The output file SHALL be encoded so that the restored detail survives compression and playback color is correct.

#### Scenario: Encoder selection

- **WHEN** the service encodes an output video
- **THEN** it uses the highest-quality encoder it has verified to work in the running container, falling back to a software encoder if hardware encoding is unavailable
- **AND** the chosen encoder is recorded in the job logs

#### Scenario: Color signalling

- **WHEN** the output video is produced
- **THEN** its color primaries, transfer characteristics, and matrix coefficients are explicitly tagged as bt709

#### Scenario: Frame rate preservation

- **WHEN** the source has a fractional frame rate such as 23.976 or 29.97
- **THEN** the output frame rate matches the source exactly rather than being rounded

### Requirement: Audio preservation

The service SHALL carry the source audio into the output without re-encoding it, unless the caller opts out.

#### Scenario: Video with audio

- **WHEN** a caller submits a video containing an audio track and does not disable audio
- **THEN** the output contains that audio track, bit-identical to the source, in sync with the video

#### Scenario: Video without audio

- **WHEN** a caller submits a video with no audio track
- **THEN** processing completes normally and the output has no audio track

### Requirement: Long-input handling

The service SHALL process inputs longer than a single GPU invocation can complete by segmenting them, and the joined output SHALL be free of visible seams at segment boundaries.

#### Scenario: Episode-length input

- **WHEN** a caller submits a video too long to process in one GPU invocation
- **THEN** the service segments it, processes the segments, and returns a single joined output file
- **AND** segment boundaries fall on scene cuts so the join introduces no visible discontinuity

### Requirement: Processing observability

The service SHALL report per-stage timing and resource usage for every job.

#### Scenario: Completed job

- **WHEN** a job completes
- **THEN** the logs record wall-clock time for decoding, reference generation, model inference, and encoding, along with peak GPU memory and the number of chunks processed
