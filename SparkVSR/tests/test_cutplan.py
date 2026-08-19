"""Unit tests for SparkVSR cut-aware window planner (synthetic tests, no GPU required)."""

import unittest

from modal_app.pipeline.cutplan import (
    calculate_window_padding,
    plan_cut_aware_windows,
    plan_fixed_windows,
)


class TestCutPlan(unittest.TestCase):
    def test_calculate_window_padding(self):
        """Verify CogVideoX 8n+1 length constraints and minimum 9 frames."""
        # Test short lengths (< 9)
        for raw in range(1, 9):
            target, pad_before, pad_after = calculate_window_padding(raw)
            self.assertEqual(target, 9)
            self.assertEqual(pad_before, 0)
            self.assertEqual(pad_after, 9 - raw)
            self.assertEqual((target - 1) % 8, 0)

        # Test exact 8n+1 lengths
        for n in (1, 2, 6, 10):
            exact_len = 8 * n + 1
            target, pad_before, pad_after = calculate_window_padding(exact_len)
            self.assertEqual(target, exact_len)
            self.assertEqual(pad_after, 0)

        # Test non-8n+1 lengths
        for raw in (10, 25, 48, 50, 100):
            target, pad_before, pad_after = calculate_window_padding(raw)
            self.assertGreaterEqual(target, raw)
            self.assertEqual((target - 1) % 8, 0)
            self.assertEqual(pad_after, target - raw)

    def test_plan_cut_aware_single_shot(self):
        """Verify planner on a single continuous shot."""
        total_frames = 120
        fps = 24.0
        plan = plan_cut_aware_windows(
            shots=[(0, total_frames)],
            total_frames=total_frames,
            fps=fps,
            chunk_size=49,
            overlap=8,
        )

        self.assertTrue(plan["cut_aware"])
        self.assertEqual(plan["total_frames"], total_frames)
        self.assertEqual(len(plan["shots"]), 1)
        self.assertGreater(len(plan["windows"]), 0)

        # Verify every frame is covered
        covered = [False] * total_frames
        for win in plan["windows"]:
            self.assertTrue(0 <= win["start_frame"] < win["end_frame"] <= total_frames)
            self.assertEqual(win["raw_frames"], win["end_frame"] - win["start_frame"])
            self.assertGreaterEqual(win["target_frames"], win["raw_frames"])
            self.assertEqual((win["target_frames"] - 1) % 8, 0)
            self.assertGreaterEqual(len(win["ref_indices"]), 1)

            for idx in win["ref_indices"]:
                self.assertTrue(win["start_frame"] <= idx < win["end_frame"])

            for f in range(win["start_frame"], win["end_frame"]):
                covered[f] = True

        self.assertTrue(all(covered), "All frames must be covered by at least one window")

    def test_plan_cut_aware_multiple_shots(self):
        """Verify no window crosses a scene cut and all shots are covered."""
        shots = [
            (0, 30),     # Short shot (30 frames)
            (30, 150),   # Long shot (120 frames, needs multi-window)
            (150, 156),  # Very short shot (6 frames, < 9 frames minimum)
            (156, 200),  # Medium shot (44 frames)
        ]
        total_frames = 200
        fps = 25.0

        plan = plan_cut_aware_windows(
            shots=shots,
            total_frames=total_frames,
            fps=fps,
            chunk_size=49,
            overlap=8,
        )

        self.assertEqual(len(plan["shots"]), 4)

        # Verify no window crosses a scene cut
        for win in plan["windows"]:
            s_id = win["shot_id"]
            shot_start, shot_end = shots[s_id]
            self.assertGreaterEqual(win["start_frame"], shot_start)
            self.assertLessEqual(win["end_frame"], shot_end)
            self.assertGreaterEqual(win["target_frames"], 9)
            self.assertEqual((win["target_frames"] - 1) % 8, 0)
            self.assertGreaterEqual(len(win["ref_indices"]), 1)

        # Verify coverage across all frames
        covered = [False] * total_frames
        for win in plan["windows"]:
            for f in range(win["start_frame"], win["end_frame"]):
                covered[f] = True
        self.assertTrue(all(covered), "All frames across all shots must be covered")

    def test_plan_fixed_windows(self):
        """Verify fixed-window fallback planner."""
        total_frames = 150
        fps = 30.0
        plan = plan_fixed_windows(
            total_frames=total_frames,
            fps=fps,
            chunk_size=49,
            overlap=8,
        )

        self.assertEqual(plan["total_frames"], total_frames)
        self.assertEqual(len(plan["shots"]), 1)

        covered = [False] * total_frames
        for win in plan["windows"]:
            for f in range(win["start_frame"], win["end_frame"]):
                covered[f] = True
            self.assertGreaterEqual(len(win["ref_indices"]), 1)
        self.assertTrue(all(covered))

    def test_reference_spacing(self):
        """Verify reference indices are spaced > 4 frames apart (>= 5 frames)."""
        shots = [(0, 100), (100, 200)]
        plan = plan_cut_aware_windows(
            shots=shots,
            total_frames=200,
            fps=24.0,
            chunk_size=49,
            overlap=8,
        )

        all_refs = sorted(plan["all_ref_indices"])
        for i in range(len(all_refs) - 1):
            spacing = all_refs[i + 1] - all_refs[i]
            self.assertGreaterEqual(spacing, 5, f"Reference spacing must be > 4 frames, got {spacing}")


    def test_every_window_reference_is_inside_its_window(self):
        """A reference outside its window is silently dropped by the executor.

        Regression test for the short-shot path: when many tiny adjacent shots
        make the >4-frame spacing heuristic unsatisfiable, the planner must still
        return an in-window reference rather than borrowing a neighbour's.
        """
        # Densely packed short shots, all within 4 frames of each other.
        shots = [(i, i + 3) for i in range(0, 60, 3)]
        plan = plan_cut_aware_windows(
            shots=shots,
            total_frames=60,
            fps=25.0,
            chunk_size=49,
            overlap=8,
        )

        self.assertGreater(len(plan["windows"]), 0)
        for win in plan["windows"]:
            self.assertGreaterEqual(
                len(win["ref_indices"]), 1, f"window {win['window_id']} has no reference"
            )
            for idx in win["ref_indices"]:
                self.assertTrue(
                    win["start_frame"] <= idx < win["end_frame"],
                    f"reference {idx} falls outside window "
                    f"[{win['start_frame']}, {win['end_frame']})",
                )

    def test_no_redundant_trailing_window(self):
        """A runt tail must be absorbed, not turned into a near-duplicate window.

        The old planner appended a final window at `shot_end - chunk_size`, which
        for a 137-frame shot overlapped its predecessor by 43 of 49 frames: one
        wasted inference per shot, and an overlap region with three contributors
        whose ramp weights summed above 1.
        """
        for shots, total in (
            ([(300, 437)], 437),
            ([(0, 120), (120, 300), (300, 437)], 437),
            ([(0, 3000)], 3000),
            ([(0, 250)], 250),
        ):
            plan = plan_cut_aware_windows(
                shots=shots, total_frames=total, fps=25.0, chunk_size=49, overlap=8
            )
            windows = plan["windows"]

            depth = [0] * total
            for win in windows:
                for f in range(win["start_frame"], win["end_frame"]):
                    depth[f] += 1
            self.assertLessEqual(
                max(depth), 2, f"no frame may have 3+ contributors (shots={shots})"
            )

            for prev, nxt in zip(windows, windows[1:]):
                if prev["shot_id"] != nxt["shot_id"]:
                    continue
                overlap = prev["end_frame"] - nxt["start_frame"]
                self.assertLessEqual(
                    overlap,
                    8,
                    f"windows overlap by {overlap} frames, expected at most the "
                    f"configured overlap of 8 (shots={shots})",
                )

    def test_padding_crops_back_to_exact_source_length(self):
        """Padded windows must record exactly how much to crop back off."""
        shots = [(0, 30), (30, 156), (156, 162)]
        plan = plan_cut_aware_windows(
            shots=shots, total_frames=162, fps=25.0, chunk_size=49, overlap=8
        )
        for win in plan["windows"]:
            raw = win["end_frame"] - win["start_frame"]
            self.assertEqual(win["raw_frames"], raw)
            self.assertEqual(
                win["target_frames"] - win["pad_before"] - win["pad_after"],
                raw,
                "cropping the recorded padding must recover the exact source length",
            )


if __name__ == "__main__":
    unittest.main()
