import unittest
from unittest import mock
import types
import sys

import vhh_msa_templates as vhh


class VhhMsaTemplatesTest(unittest.TestCase):
    def test_rewrite_query_in_non_pairing_a3m_normalizes_query_and_preserves_hits(self) -> None:
        content = ">query_0\nAAAA\n>hit_b\nBBBB\n>hit_a\nCCCC\n"
        rewritten = vhh.rewrite_query_in_non_pairing_a3m(content, "DdDd")
        self.assertEqual(
            rewritten,
            ">query\nDDDD\n>hit_a\nCCCC\n>hit_b\nBBBB\n",
        )

    def test_pairing_from_strategy(self) -> None:
        non_pairing = ">query\nAAAA\n>hit\nBBBB\n"
        self.assertEqual(
            vhh.pairing_from_strategy(non_pairing, "CCCC", "greedy"),
            ">query\nCCCC\n",
        )
        self.assertEqual(
            vhh.pairing_from_strategy(non_pairing, "CCCC", "query_only"),
            ">query\nCCCC\n",
        )
        self.assertEqual(
            vhh.pairing_from_strategy(non_pairing, "CCCC", "copy_non_pairing"),
            non_pairing,
        )

    def test_extract_unique_binders_keeps_first_occurrence(self) -> None:
        rows = [
            {"row_index": 0, "binder_name": "b0", "binder_seq": "AAAA"},
            {"row_index": 1, "binder_name": "b1", "binder_seq": "BBBB"},
            {"row_index": 2, "binder_name": "b2", "binder_seq": "AAAA"},
        ]
        unique = vhh.extract_unique_binders(rows)
        self.assertEqual(
            unique,
            [
                {"binder_name_first": "b0", "binder_sequence": "AAAA", "first_row_index": 0},
                {"binder_name_first": "b1", "binder_sequence": "BBBB", "first_row_index": 1},
            ],
        )

    def test_build_template_groups_uses_first_occurrence_representative(self) -> None:
        analyzed = [
            {
                "binder_name_first": "late",
                "binder_sequence": "SEQ2",
                "binder_sequence_sha256": "sha2",
                "first_row_index": 5,
                "numbering_scheme": "imgt",
                "chain_class": "vhh",
                "fr1": "AAA",
                "fr2": "BBB",
                "fr3": "CCC",
                "fr4": "DDD",
                "fr1_length": 3,
                "fr2_length": 3,
                "fr3_length": 3,
                "fr4_length": 3,
                "cdr1_length": 3,
                "cdr2_length": 3,
                "cdr3_length": 4,
                "cdr1_register": "H27,H28,H29",
                "cdr2_register": "H56,H57,H58",
                "cdr3_register": "H105,H106,H107,H108",
                "framework_hash": "fh",
                "canonical_template_key_json": "same-key",
                "canonical_template_key_hash": "abc12345def6",
                "lengths_only_template_key_json": "same-len-key",
                "lengths_only_template_key_hash": "len12345def6",
            },
            {
                "binder_name_first": "early",
                "binder_sequence": "SEQ1",
                "binder_sequence_sha256": "sha1",
                "first_row_index": 2,
                "numbering_scheme": "imgt",
                "chain_class": "vhh",
                "fr1": "AAA",
                "fr2": "BBB",
                "fr3": "CCC",
                "fr4": "DDD",
                "fr1_length": 3,
                "fr2_length": 3,
                "fr3_length": 3,
                "fr4_length": 3,
                "cdr1_length": 3,
                "cdr2_length": 3,
                "cdr3_length": 4,
                "cdr1_register": "H27,H28,H29",
                "cdr2_register": "H56,H57,H58",
                "cdr3_register": "H105,H106,H107,H108",
                "framework_hash": "fh",
                "canonical_template_key_json": "same-key",
                "canonical_template_key_hash": "abc12345def6",
                "lengths_only_template_key_json": "same-len-key",
                "lengths_only_template_key_hash": "len12345def6",
            },
        ]
        groups = vhh.build_template_groups(analyzed)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["representative"]["binder_name_first"], "early")
        self.assertEqual(groups[0]["members"][0]["binder_sequence"], "SEQ1")
        self.assertTrue(groups[0]["template_id"].startswith("vhh_tpl_0001_"))

    def test_build_template_groups_lengths_only_merges_framework_variants(self) -> None:
        analyzed = [
            {
                "binder_name_first": "a",
                "binder_sequence": "SEQ1",
                "binder_sequence_sha256": "sha1",
                "first_row_index": 0,
                "numbering_scheme": "imgt",
                "chain_class": "vhh",
                "fr1": "AAA",
                "fr2": "BBB",
                "fr3": "CCC",
                "fr4": "DDD",
                "fr1_length": 3,
                "fr2_length": 3,
                "fr3_length": 3,
                "fr4_length": 3,
                "cdr1_length": 3,
                "cdr2_length": 3,
                "cdr3_length": 4,
                "cdr1_register": "H27,H28,H29",
                "cdr2_register": "H56,H57,H58",
                "cdr3_register": "H105,H106,H107,H108",
                "framework_hash": "fh1",
                "canonical_template_key_json": "exact-a",
                "canonical_template_key_hash": "exact-a-hash",
                "lengths_only_template_key_json": "len-same",
                "lengths_only_template_key_hash": "len-same-hash",
            },
            {
                "binder_name_first": "b",
                "binder_sequence": "SEQ2",
                "binder_sequence_sha256": "sha2",
                "first_row_index": 1,
                "numbering_scheme": "imgt",
                "chain_class": "vhh",
                "fr1": "XXX",
                "fr2": "YYY",
                "fr3": "ZZZ",
                "fr4": "WWW",
                "fr1_length": 3,
                "fr2_length": 3,
                "fr3_length": 3,
                "fr4_length": 3,
                "cdr1_length": 3,
                "cdr2_length": 3,
                "cdr3_length": 4,
                "cdr1_register": "H27,H28,H29",
                "cdr2_register": "H56,H57,H58",
                "cdr3_register": "H105,H106,H107,H108",
                "framework_hash": "fh2",
                "canonical_template_key_json": "exact-b",
                "canonical_template_key_hash": "exact-b-hash",
                "lengths_only_template_key_json": "len-same",
                "lengths_only_template_key_hash": "len-same-hash",
            },
        ]
        exact_groups = vhh.build_template_groups(analyzed, framework_mode="exact")
        lengths_groups = vhh.build_template_groups(analyzed, framework_mode="lengths_only")
        self.assertEqual(len(exact_groups), 2)
        self.assertEqual(len(lengths_groups), 1)
        self.assertEqual(lengths_groups[0]["template_grouping_mode"], "lengths_only")

    def test_build_template_groups_trims_materialization_subset_only(self) -> None:
        analyzed = [
            {
                "binder_name_first": "a",
                "binder_sequence": "SEQ1",
                "binder_sequence_sha256": "sha1",
                "first_row_index": 0,
                "numbering_scheme": "imgt",
                "chain_class": "vhh",
                "fr1": "AAA",
                "fr2": "BBB",
                "fr3": "CCC",
                "fr4": "DDD",
                "fr1_length": 3,
                "fr2_length": 3,
                "fr3_length": 3,
                "fr4_length": 3,
                "cdr1_length": 3,
                "cdr2_length": 3,
                "cdr3_length": 4,
                "cdr1_register": "H27,H28,H29",
                "cdr2_register": "H56,H57,H58",
                "cdr3_register": "H105,H106,H107,H108",
                "framework_hash": "fh",
                "canonical_template_key_json": "same-key",
                "canonical_template_key_hash": "abc12345def6",
                "lengths_only_template_key_json": "same-len-key",
                "lengths_only_template_key_hash": "len12345def6",
            },
            {
                "binder_name_first": "b",
                "binder_sequence": "SEQ2",
                "binder_sequence_sha256": "sha2",
                "first_row_index": 1,
                "numbering_scheme": "imgt",
                "chain_class": "vhh",
                "fr1": "AAA",
                "fr2": "BBB",
                "fr3": "CCC",
                "fr4": "DDD",
                "fr1_length": 3,
                "fr2_length": 3,
                "fr3_length": 3,
                "fr4_length": 3,
                "cdr1_length": 3,
                "cdr2_length": 3,
                "cdr3_length": 4,
                "cdr1_register": "H27,H28,H29",
                "cdr2_register": "H56,H57,H58",
                "cdr3_register": "H105,H106,H107,H108",
                "framework_hash": "fh",
                "canonical_template_key_json": "same-key",
                "canonical_template_key_hash": "abc12345def6",
                "lengths_only_template_key_json": "same-len-key",
                "lengths_only_template_key_hash": "len12345def6",
            },
        ]
        groups = vhh.build_template_groups(analyzed, max_members_per_template=1)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["group_size"], 2)
        self.assertEqual(len(groups[0]["members"]), 2)
        self.assertEqual(len(groups[0]["members_for_materialization"]), 1)
        self.assertEqual(groups[0]["members_for_materialization"][0]["binder_sequence"], "SEQ1")

    def test_merge_binder_analyses_combines_remote_results(self) -> None:
        unique = [
            {"binder_name_first": "ok", "binder_sequence": "AAAA", "first_row_index": 0},
            {"binder_name_first": "bad", "binder_sequence": "BBBB", "first_row_index": 1},
        ]
        remote = {
            "AAAA": {
                "ok": True,
                "analysis": {
                    "binder_sequence": "AAAA",
                    "binder_sequence_sha256": "shaA",
                    "numbering_scheme": "imgt",
                    "chain_class": "vhh",
                    "chain_type": "H",
                    "fr1": "A",
                    "cdr1": "A",
                    "fr2": "A",
                    "cdr2": "A",
                    "fr3": "A",
                    "cdr3": "AA",
                    "fr4": "A",
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
                    "total_binder_length": 8,
                    "framework_hash": "fh",
                    "canonical_template_key_json": "{}",
                    "canonical_template_key_hash": "hash",
                    "lengths_only_template_key_json": "{\"x\":\"len\"}",
                    "lengths_only_template_key_hash": "lenhash",
                },
                "stage": "numbering",
            },
            "BBBB": {"ok": False, "error": "numbering failed", "stage": "numbering"},
        }
        merged = vhh.merge_binder_analyses(unique, remote)
        self.assertEqual(len(merged["analyzed"]), 1)
        self.assertEqual(merged["analyzed"][0]["binder_name_first"], "ok")
        self.assertEqual(len(merged["rejected"]), 1)
        self.assertEqual(merged["rejected"][0]["binder_name"], "bad")

    def test_analyze_vhh_sequence_uses_numbering_output(self) -> None:
        segmentation = vhh.VhhSegmentation(
            sequence="EVQL",
            numbering_scheme="imgt",
            chain_class="vhh",
            chain_type="H",
            fr1="E",
            cdr1="V",
            fr2="Q",
            cdr2="L",
            fr3="",
            cdr3="",
            fr4="",
            cdr1_register="H27",
            cdr2_register="H56",
            cdr3_register="",
            fr1_length=1,
            fr2_length=1,
            fr3_length=0,
            fr4_length=0,
            cdr1_length=1,
            cdr2_length=1,
            cdr3_length=0,
            total_binder_length=4,
            framework_hash="fh",
        )
        with mock.patch.object(vhh, "number_vhh_sequence", return_value=segmentation):
            analysis = vhh.analyze_vhh_sequence("EVQL")
        self.assertEqual(analysis["binder_sequence"], "EVQL")
        self.assertEqual(analysis["chain_class"], "vhh")
        self.assertIn("canonical_template_key_json", analysis)
        self.assertIn("lengths_only_template_key_json", analysis)

    def test_number_vhh_sequence_uses_anarcii_backend(self) -> None:
        class FakePosition:
            def __init__(self, label: str) -> None:
                self.label = label

            def format(self, chain_type: bool = True, region: bool = False) -> str:
                return self.label

            def __lt__(self, other: object) -> bool:
                if not isinstance(other, FakePosition):
                    return NotImplemented
                return self.label < other.label

        class FakeChain:
            called_kwargs = None

            def __init__(self, sequence: str, **kwargs) -> None:
                FakeChain.called_kwargs = dict(kwargs)
                self.chain_type = "H"
                self.seq = sequence
                self.tail = ""
                self.regions = {
                    "FR1": {FakePosition("H1"): "A"},
                    "CDR1": {FakePosition("H27"): "B"},
                    "FR2": {FakePosition("H39"): "C"},
                    "CDR2": {FakePosition("H56"): "D"},
                    "FR3": {FakePosition("H66"): "E"},
                    "CDR3": {FakePosition("H105"): "F"},
                    "FR4": {FakePosition("H118"): "G"},
                }

        fake_abnumber = types.SimpleNamespace(Chain=FakeChain)
        with mock.patch.dict(sys.modules, {"abnumber": fake_abnumber}):
            seg = vhh.number_vhh_sequence("ABCDEFG")
        self.assertEqual(seg.sequence, "ABCDEFG")
        self.assertIsNotNone(FakeChain.called_kwargs)
        self.assertTrue(FakeChain.called_kwargs["use_anarcii"])


if __name__ == "__main__":
    unittest.main()
