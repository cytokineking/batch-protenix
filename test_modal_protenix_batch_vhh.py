import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import modal_protenix_batch as mpb


def _analysis_dict(sequence: str) -> dict:
    return {
        "binder_sequence": sequence,
        "binder_sequence_sha256": hashlib.sha256(sequence.encode("utf-8")).hexdigest(),
        "numbering_scheme": "imgt",
        "chain_class": "vhh",
        "chain_type": "H",
        "fr1": "A",
        "cdr1": "B",
        "fr2": "C",
        "cdr2": "D",
        "fr3": "E",
        "cdr3": "FG",
        "fr4": "H",
        "fr1_length": 1,
        "fr2_length": 1,
        "fr3_length": 1,
        "fr4_length": 1,
        "cdr1_length": 1,
        "cdr2_length": 1,
        "cdr3_length": 2,
        "cdr1_register": "H27",
        "cdr2_register": "H56",
        "cdr3_register": "H105,H106",
        "total_binder_length": len(sequence),
        "framework_hash": "fh",
        "canonical_template_key_json": '{"x":"y"}',
        "canonical_template_key_hash": "hash1234",
        "lengths_only_template_key_json": '{"x":"len"}',
        "lengths_only_template_key_hash": "lenhash1234",
    }


class ModalProtenixBatchVhhHelpersTest(unittest.TestCase):
    def test_render_prepare_recommended_run_flags_uses_resolved_output_dir(self) -> None:
        rendered = mpb._render_prepare_recommended_run_flags(
            pair_csv="./pairs.csv",
            output_dir="/tmp/vhh_prep/run_pipeline_results",
            mmseqs_mode_n="colabfold",
            host_url="https://api.colabfold.com",
            mmseqs_host_policy="strict",
            resolved_db_tag="colabfold_env",
            mmseqs_pairing_strategy="greedy",
            mmseqs_local_db_profile="uniref100_only",
            mmseqs_local_gpu="A100-80GB",
            mmseqs_fallback_n="none",
            mmseqs_local_workers=4,
            mmseqs_local_batch_size=8,
            mmseqs_local_max_seqs=300,
            mmseqs_local_prefilter_mode=1,
            msa_min_submit_interval_s=1.0,
            msa_global_rate_key="msa_global",
        )
        self.assertIn("--output-dir /tmp/vhh_prep/run_pipeline_results", rendered)
        self.assertIn("--msa-min-submit-interval-s 1.000", rendered)
        self.assertIn("--msa-global-rate-key msa_global", rendered)
        self.assertNotIn("--mmseqs-local-workers", rendered)

    def test_lookup_cached_msa_entry_finds_fallback_namespace(self) -> None:
        sequence = "EVQL"
        host_url = "https://api.colabfold.com"
        context_hash = "ctx123"
        context_meta = {
            "mmseqs_mode": "local_gpu",
            "mmseqs_db_tag": "uniref100_v1",
            "mmseqs_pairing_strategy": "greedy",
            "mmseqs_context_hash": context_hash,
            "mmseqs_db_profile": "uniref100_only",
            "mmseqs_manifest_sha256": "manifest123",
            "mmseqs_version": "18.8",
            "mmseqs_build_fingerprint": "build123",
            "mmseqs_local_max_seqs": 300,
            "mmseqs_local_prefilter_mode": 1,
        }
        fallback_key = mpb._build_cache_key(
            sequence=sequence,
            role="binder",
            host_url=host_url,
            msa_mode="colabfold_fallback",
            pairing_strategy="greedy",
            db_tag="uniref100_v1",
            context_hash=context_hash,
        )
        metadata = {
            "sequence_sha256": hashlib.sha256(sequence.encode("utf-8")).hexdigest(),
            "role": "binder",
            "host_url": host_url,
            "msa_mode": "colabfold_fallback",
            "pairing_strategy": "greedy",
            "db_tag": "uniref100_v1",
            "context_hash": context_hash,
            "fallback_from": "local_gpu",
        }
        metadata.update(context_meta)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir)
            with mock.patch.object(mpb, "MSA_CACHE_ROOT", cache_root):
                mpb._write_cache_entry(
                    cache_dir=cache_root / fallback_key,
                    pairing=">query\nEVQL\n",
                    non_pairing=">query\nEVQL\n",
                    metadata=metadata,
                )
                info = mpb._lookup_cached_msa_entry(
                    sequence=sequence,
                    role="binder",
                    host_url=host_url,
                    msa_mode="local_gpu",
                    pairing_strategy="greedy",
                    db_tag="uniref100_v1",
                    context_hash=context_hash,
                    context_meta=context_meta,
                    fallback_mode="colabfold",
                    require_complete_marker=True,
                )

        self.assertIsNotNone(info)
        self.assertEqual(info["status"], "cached")
        self.assertEqual(info["cache_key"], fallback_key)
        self.assertEqual(info["fallback"], "colabfold")

    def test_prepare_vhh_binder_msas_analyze_only_skips_context_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pair_csv = Path(tmpdir) / "pairs.csv"
            pair_csv.write_text(
                "binder_name,binder_sequence,target_name,target_sequence\n"
                "b1,EVQL,target1,AAAA\n",
                encoding="utf-8",
            )
            output_dir = Path(tmpdir) / "out"
            remote_results = {
                "EVQL": {"ok": True, "analysis": _analysis_dict("EVQL"), "stage": "numbering"}
            }
            with mock.patch.object(mpb.analyze_vhh_sequences_remote, "remote", return_value=remote_results):
                with mock.patch.object(
                    mpb,
                    "_resolve_prepare_msa_context",
                    side_effect=AssertionError("context resolution should be skipped"),
                ):
                    mpb.prepare_vhh_binder_msas(
                        pair_csv=str(pair_csv),
                        output_dir=str(output_dir),
                        analyze_only=True,
                        emit_recommended_run_flags=False,
                    )

            manifest = json.loads((output_dir / "vhh_template_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["execution_skip_reason"], "analyze_only")
            self.assertEqual(manifest["resolved_mmseqs_mode"], "")
            self.assertEqual(manifest["template_count"], 1)

    def test_prepare_vhh_binder_msas_strict_numbering_fails_before_context_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pair_csv = Path(tmpdir) / "pairs.csv"
            pair_csv.write_text(
                "binder_name,binder_sequence,target_name,target_sequence\n"
                "b1,EVQL,target1,AAAA\n",
                encoding="utf-8",
            )
            output_dir = Path(tmpdir) / "out"
            remote_results = {
                "EVQL": {"ok": False, "error": "numbering failed", "stage": "numbering"}
            }
            with mock.patch.object(mpb.analyze_vhh_sequences_remote, "remote", return_value=remote_results):
                with mock.patch.object(
                    mpb,
                    "_resolve_prepare_msa_context",
                    side_effect=AssertionError("context resolution should be skipped"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "Rejected 1 binder sequence\\(s\\)"):
                        mpb.prepare_vhh_binder_msas(
                            pair_csv=str(pair_csv),
                            output_dir=str(output_dir),
                            strict_numbering=True,
                            emit_recommended_run_flags=False,
                        )

            manifest = json.loads((output_dir / "vhh_template_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["execution_skip_reason"], "strict_numbering_rejected")
            self.assertEqual(manifest["rejected_binder_count"], 1)
            self.assertTrue((output_dir / "rejected_binders.csv").exists())

    def test_prepare_vhh_binder_msas_default_framework_mode_is_lengths_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pair_csv = Path(tmpdir) / "pairs.csv"
            pair_csv.write_text(
                "binder_name,binder_sequence,target_name,target_sequence\n"
                "b1,EVQL,target1,AAAA\n"
                "b2,QVQL,target2,BBBB\n",
                encoding="utf-8",
            )
            output_dir = Path(tmpdir) / "out"
            seq1 = "EVQL"
            seq2 = "QVQL"
            remote_results = {
                seq1: {
                    "ok": True,
                    "analysis": {
                        **_analysis_dict(seq1),
                        "fr1": "AAAA",
                        "fr2": "BBBB",
                        "fr3": "CCCC",
                        "fr4": "DDDD",
                        "canonical_template_key_json": '{"mode":"exact","fr1":"AAAA"}',
                        "canonical_template_key_hash": "exact1111",
                    },
                    "stage": "numbering",
                },
                seq2: {
                    "ok": True,
                    "analysis": {
                        **_analysis_dict(seq2),
                        "fr1": "WWWW",
                        "fr2": "XXXX",
                        "fr3": "YYYY",
                        "fr4": "ZZZZ",
                        "canonical_template_key_json": '{"mode":"exact","fr1":"WWWW"}',
                        "canonical_template_key_hash": "exact2222",
                    },
                    "stage": "numbering",
                },
            }
            with mock.patch.object(mpb.analyze_vhh_sequences_remote, "remote", return_value=remote_results):
                with mock.patch.object(
                    mpb,
                    "_resolve_prepare_msa_context",
                    side_effect=AssertionError("context resolution should be skipped"),
                ):
                    mpb.prepare_vhh_binder_msas(
                        pair_csv=str(pair_csv),
                        output_dir=str(output_dir),
                        analyze_only=True,
                        emit_recommended_run_flags=False,
                    )

            manifest = json.loads((output_dir / "vhh_template_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["framework_mode"], "lengths_only")
            self.assertEqual(manifest["template_count"], 1)


if __name__ == "__main__":
    unittest.main()
