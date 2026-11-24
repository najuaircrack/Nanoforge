from __future__ import annotations

from nanoforge.data.fim import FIMConfig, transform_fim_text
from nanoforge.data.formats import DatasetRecord
from nanoforge.data.modes import encode_training_record
from nanoforge.data.tokenizer import ByteTokenizer


def test_fim_transform_is_deterministic_and_ordered():
    transformed = transform_fim_text("abcdef", FIMConfig(split_fraction=0.5))
    assert transformed == "<|fim_prefix|>abc<|fim_suffix|>ef<|fim_middle|>d"


def test_fim_mode_uses_transformed_sequence():
    encoded = encode_training_record(
        DatasetRecord("abcdef", "sample.py", {"format": "code"}),
        ByteTokenizer(),
        mode="fim",
        add_bos=False,
        add_eos=False,
    )
    decoded = ByteTokenizer().decode(encoded.ids)
    assert "<|fim_prefix|>" in decoded
    assert "<|fim_suffix|>" in decoded
    assert "<|fim_middle|>" in decoded
