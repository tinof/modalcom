## Purpose

Defines how an upscale request is turned into concrete output dimensions, including the
limits the super-resolution engine imposes, so callers get either a predictable image size
or a clear error rather than a silently wrong or unservable result.

## Requirements

### Requirement: Scale-based output sizing

When the caller selects scale-by-multiplier resizing, the service SHALL multiply the input
dimensions by the supplied factor. The factor MUST be greater than or equal to 1.0, because
the super-resolution engine can only enlarge an image.

#### Scenario: Valid upscale factor

- **WHEN** a 800x400 image is submitted with scale-by-multiplier resizing and a factor of 2.0
- **THEN** the service produces an image of 1600x800

#### Scenario: Factor below one is rejected

- **WHEN** a request uses scale-by-multiplier resizing with a factor below 1.0
- **THEN** the service rejects the request with a client error identifying the invalid factor
- **AND** no GPU work is started

### Requirement: Target-dimension output sizing

When the caller selects target-dimension resizing, the service SHALL fit the output inside
the requested width and height while preserving the input aspect ratio by default. The caller
MAY disable aspect-ratio preservation to stretch the image to the exact requested dimensions.

#### Scenario: Aspect ratio preserved by default

- **WHEN** a 800x400 image is submitted with target dimensions 1920x1080 and no explicit
  aspect-ratio preference
- **THEN** the output fits inside 1920x1080 without distortion, giving 1920x960

#### Scenario: Caller opts into stretching

- **WHEN** the same request sets aspect-ratio preservation to false
- **THEN** the output is exactly 1920x1080 and the image is stretched

### Requirement: Rejection of non-upscaling requests

The service SHALL reject any request whose computed output is smaller than the input in
either dimension, because the super-resolution engine cannot downscale.

#### Scenario: Target smaller than source

- **WHEN** a 1920x1080 image is submitted with target dimensions 640x480
- **THEN** the service rejects the request with a client error explaining that only
  upscaling is supported
- **AND** the response reports both the input and the requested output size

### Requirement: Maximum output size limit

The service SHALL reject any request whose computed output edge exceeds 16384 pixels, so an
oversized request fails fast instead of exhausting GPU memory mid-job.

#### Scenario: Request beyond the maximum edge in target-dimension mode

- **WHEN** a target-dimension request computes an output edge larger than 16384 pixels
- **THEN** the service rejects the request with a client error naming the limit
- **AND** no GPU work is started

#### Scenario: Request beyond the maximum edge in scale-by-multiplier mode

- **WHEN** a scale-by-multiplier request computes an output edge larger than 16384 pixels
- **THEN** the worker rejects the request and surfaces a client error naming the limit

#### Scenario: Request at the maximum edge

- **WHEN** a request computes an output edge of exactly 16384 pixels
- **THEN** the service accepts the request and processes it

### Requirement: Output dimension alignment

The service SHALL align computed output dimensions to a multiple of 8, matching the
behaviour of the reference ComfyUI node, and SHALL never align a dimension below 8.

#### Scenario: Unaligned target is aligned

- **WHEN** a request computes an output width of 1001 pixels
- **THEN** the delivered width is the nearest multiple of 8

#### Scenario: Alignment never collapses a dimension

- **WHEN** alignment would produce a dimension smaller than 8
- **THEN** the dimension is set to 8
