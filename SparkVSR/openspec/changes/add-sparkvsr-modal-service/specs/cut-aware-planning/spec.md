## Purpose

Derives a processing plan from an input video that respects scene cuts, so that the model's temporal propagation never carries detail from one shot into the next — the dominant artifact risk when restoring episodic broadcast television.

## ADDED Requirements

### Requirement: Scene cut detection

The system SHALL detect scene cuts in the input video and expose the detected boundaries in the job's plan output.

#### Scenario: Video with cuts

- **WHEN** planning runs on a video containing scene changes
- **THEN** the plan lists the frame index of each detected cut
- **AND** the plan is recorded in the job output so a caller can compare it against the visible scene changes

#### Scenario: Single continuous shot

- **WHEN** planning runs on a video with no detected cuts
- **THEN** the plan contains exactly one shot spanning the whole video

### Requirement: Chunks never straddle a cut

Every temporal chunk in the plan SHALL lie entirely within a single detected shot.

#### Scenario: Cut inside a nominal chunk span

- **WHEN** a scene cut falls partway through what would otherwise be one chunk
- **THEN** the plan splits at the cut, producing two chunks that each stay within one shot

#### Scenario: Temporal blending scope

- **WHEN** adjacent chunks are blended in their overlap region
- **THEN** blending is applied only between chunks belonging to the same shot, and never across a shot boundary

### Requirement: Long shots are windowed with overlap

Shots longer than the maximum chunk length SHALL be covered by overlapping windows, and every frame of the shot SHALL appear in at least one window.

#### Scenario: Shot longer than one chunk

- **WHEN** a shot exceeds the configured chunk length
- **THEN** the plan covers it with overlapping windows whose union is the entire shot
- **AND** consecutive windows within the shot share a non-zero overlap region

### Requirement: Model length constraints are satisfied

Each planned window SHALL satisfy the model's temporal length constraints, and the padding SHALL not appear in the output.

#### Scenario: Short shot

- **WHEN** a detected shot is shorter than the model's minimum frame count
- **THEN** the window is padded up to a valid length
- **AND** the output is cropped back to the shot's exact original frame count

#### Scenario: Frame count not a valid model length

- **WHEN** a window's frame count is not one of the lengths the model accepts
- **THEN** the window is padded to the next valid length and cropped back after inference

### Requirement: Every window has a reference frame

Every planned window SHALL contain at least one reference frame index, and multiple reference indices within a window SHALL be spaced more than 4 frames apart.

#### Scenario: Window reference assignment

- **WHEN** the plan is produced
- **THEN** each window carries at least one reference frame index that falls inside that window

#### Scenario: Long shot with multiple references

- **WHEN** a shot is long enough to warrant more than one reference frame
- **THEN** the assigned reference indices are spaced more than 4 frames apart

### Requirement: Cut-aware planning can be disabled

The caller SHALL be able to disable cut detection and fall back to fixed-length overlapping chunks over the whole video.

#### Scenario: Cut-awareness disabled

- **WHEN** a caller disables cut-aware planning
- **THEN** the plan is a single sequence of fixed-length overlapping chunks covering the video, ignoring scene boundaries
- **AND** the remaining constraints on frame count and reference presence still hold
