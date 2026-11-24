from __future__ import annotations

import copy
from unittest.mock import patch

import pytest
import torch

from nanoforge.config import ModelConfig, MoEConfig
from nanoforge.data.tokenizer import ByteTokenizer
from nanoforge.generation.engine import GenerationEngine
from nanoforge.generation.sampling import SamplingConfig
from nanoforge.model.attention import CausalSelfAttention
from nanoforge.model.rope import RotaryEmbedding, _scaled_positions
from nanoforge.model.transformer import NanoforgeForCausalLM


@pytest.fixture(autouse=True)
def fixed_cpu_seed():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    yield
    torch.set_num_threads(previous)


def small_config(**changes):
    values = dict(vocab_size=260, max_seq_len=32, d_model=32,
                  n_layers=3, n_heads=4, n_kv_heads=2, use_flash=False)
    values.update(changes)
    return ModelConfig(**values)


@pytest.mark.parametrize("moe", [None, MoEConfig(num_experts=2, top_k=1)])
def test_checkpointing_preserves_loss_auxiliary_and_every_gradient(moe):
    plain = NanoforgeForCausalLM(small_config(moe=moe)).train()
    checked = copy.deepcopy(plain)
    checked.config.gradient_checkpointing = True
    checked.backbone.config.gradient_checkpointing = True
    ids = torch.randint(0, 260, (2, 8))
    labels = torch.randint(0, 260, (2, 8))
    expected = plain(ids, labels=labels)
    actual = checked(ids, labels=labels)
    torch.testing.assert_close(actual.loss, expected.loss)
    if moe is not None:
        assert actual.aux_loss is not None
        torch.testing.assert_close(actual.aux_loss, expected.aux_loss)
    expected.loss.backward()
    actual.loss.backward()
    for (name, left), (_, right) in zip(plain.named_parameters(), checked.named_parameters()):
        if left.grad is not None:
            assert right.grad is not None, name
            torch.testing.assert_close(right.grad, left.grad, msg=name)


@pytest.mark.parametrize("backend", ["manual", "sdpa", "chunked", "sparse", "paged"])
def test_cached_attention_matches_full_across_multiple_evictions(backend):
    from nanoforge.registry import ATTENTION_BACKENDS

    config = small_config(attention_backend=backend, sliding_window=4)
    attention = ATTENTION_BACKENDS.create(backend, config).eval()
    hidden = torch.randn(1, 11, 32)
    with torch.no_grad():
        expected, _ = attention(hidden)
        cache = None
        for position in range(hidden.shape[1]):
            actual, cache = attention(hidden[:, position:position + 1], cache=cache, use_cache=True)
            torch.testing.assert_close(actual, expected[:, position:position + 1], atol=3e-6, rtol=3e-5)
        assert cache.length <= 4
        assert cache.tokens_seen == 11


def test_long_prefill_stays_causal_and_matches_uncached_attention():
    attention = CausalSelfAttention(small_config(sliding_window=4)).eval()
    hidden = torch.randn(1, 9, 32)
    altered = hidden.clone()
    altered[:, -1] += 5
    with torch.no_grad():
        expected, _ = attention(hidden)
        actual, _ = attention(hidden, use_cache=True)
        changed, _ = attention(altered, use_cache=True)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[:, 0], changed[:, 0])


def test_rope_uses_one_bounded_growable_table():
    rope = RotaryEmbedding(64, 256)
    for length in range(1, 257):
        rope.get_cos_sin(length, torch.device("cpu"), torch.float32)
    tables = list(rope._cache.values())
    assert len(tables) == 1
    retained = sum(t.numel() * t.element_size() for pair in tables for t in pair)
    assert retained <= 2 * 256 * 64 * 4


@pytest.mark.parametrize("ffn", ["rwkv_mixing", "mamba_selective_state"])
@pytest.mark.parametrize("block", ["transformer", "parallel_residual"])
def test_sequence_mixer_state_preserves_incremental_model_logits(ffn, block):
    model = NanoforgeForCausalLM(small_config(ffn_type=ffn, block_type=block)).eval()
    ids = torch.randint(0, 260, (2, 11))
    with torch.no_grad():
        expected = model(ids).logits
        cache = None
        for position in range(ids.shape[1]):
            actual = model(ids[:, position:position + 1], caches=cache, use_cache=True)
            cache = actual.caches
            torch.testing.assert_close(actual.logits, expected[:, position:position + 1],
                                       atol=3e-6, rtol=3e-5)
        assert cache[0].ffn_state.shape[1] <= 2


def tiny_engine(**changes):
    return GenerationEngine(NanoforgeForCausalLM(small_config(n_layers=1, **changes)),
                            ByteTokenizer(), device="cpu")


def test_unicode_stream_matches_whole_token_decode():
    engine = tiny_engine()
    expected = "\u00e9\u4f60\U0001f642 hello"
    with patch.object(engine, "generate_ids", return_value=iter(engine.tokenizer.encode(expected))):
        assert engine.complete("x", stop_tokens=[]) == expected


def test_exact_prefix_cache_preserves_predictions_and_has_no_duplicate_prefill():
    engine = tiny_engine()
    ids = engine.tokenizer.encode("abc", add_bos=True)
    sampling = SamplingConfig(mode="deterministic", stop_on_repetition=False)
    expected = list(engine.generate_ids(ids, 4, sampling))
    engine.cache_prefix("abc")
    with patch.object(engine.model, "forward", wraps=engine.model.forward) as forward:
        actual = list(engine.generate_ids(ids, 4, sampling))
    assert actual == expected
    assert forward.call_count == 3
    assert all(call.args[0].shape[1] == 1 for call in forward.call_args_list)


def test_long_prompt_penalties_use_generated_history():
    engine = tiny_engine(max_seq_len=4)
    observed = []

    def sample(logits, history, sampling, miro):
        observed.append(history.tolist())
        return torch.tensor([[69]], device=logits.device)

    with patch("nanoforge.generation.engine.sample_next", side_effect=sample):
        list(engine.generate_ids(list(range(4, 14)), 3, SamplingConfig(stop_on_repetition=False)))
    assert observed == [[], [69], [69, 69]]


def test_manual_backend_does_not_call_sdpa_or_external_flash():
    attention = CausalSelfAttention(small_config(attention_backend="manual", use_flash=True)).eval()
    with patch("nanoforge.model.attention.F.scaled_dot_product_attention") as sdpa:
        with patch("nanoforge.model.attention._flash_attn_func") as flash:
            attention(torch.randn(1, 4, 32))
    sdpa.assert_not_called()
    flash.assert_not_called()


@pytest.mark.parametrize("caches", [[], [None] * 4])
def test_invalid_cache_count_rejected(caches):
    model = NanoforgeForCausalLM(small_config())
    with pytest.raises(ValueError, match="cache"):
        model(torch.tensor([[4, 5]]), caches=caches, use_cache=True)


@pytest.mark.parametrize("factor", [1, 2])
def test_unknown_rope_scaling_rejected_even_without_effect(factor):
    with pytest.raises(ValueError, match="scaling"):
        _scaled_positions(3, torch.device("cpu"), {"type": "ntk", "factor": factor})


@pytest.mark.parametrize("position", ["none", "alibi"])
def test_non_rotary_attention_accepts_odd_head_dimension(position):
    attention = CausalSelfAttention(small_config(d_model=3, n_heads=1, n_kv_heads=1,
                                                  position_embedding=position))
    actual, _ = attention(torch.randn(1, 4, 3))
    assert actual.shape == (1, 4, 3)


def test_paged_cache_uses_blocks_and_matches_dense_attention():
    from nanoforge.model.attention import PagedCausalSelfAttention, PagedKVCache

    config = small_config(attention_backend="paged")
    dense = CausalSelfAttention(config).eval()
    paged = PagedCausalSelfAttention(config).eval()
    paged.load_state_dict(dense.state_dict())
    hidden = torch.randn(1, 17, 32)
    with torch.no_grad():
        expected, _ = dense(hidden)
        actual, cache = paged(hidden, use_cache=True)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    assert isinstance(cache, PagedKVCache)
    assert sum(key.shape[2] for key, _ in cache.iter_blocks()) == 17


def test_invalid_generation_inputs_fail_before_model_execution():
    engine = tiny_engine()
    with pytest.raises(ValueError):
        list(engine.generate_ids([], 3, SamplingConfig()))
    with pytest.raises(ValueError):
        list(engine.generate_ids([4], -1, SamplingConfig()))
    with pytest.raises(ValueError):
        engine.beam_search("x", num_beams=0)


@pytest.mark.parametrize("text", ['{}', '{"ok":true,"n":[1,-2.3e+4,null]}',
                                   '"\\u1234"', '"\u00e9\u4f60\U0001f642"'])
def test_context_free_json_grammar_accepts_valid_json_with_unicode(text):
    from nanoforge.generation.grammar import json_byte_grammar

    tokenizer = ByteTokenizer()
    grammar = json_byte_grammar(tokenizer)
    ids = tokenizer.encode(text)
    for position, token in enumerate(ids):
        assert token in grammar.allowed_tokens(ids[:position])
    assert grammar.is_accepting(ids)
    assert tokenizer.eos_id in grammar.allowed_tokens(ids)


@pytest.mark.parametrize("text", ['{"x":}', '[1,]', '{"x":01}', '"\\q"', '{"x":true false}'])
def test_json_grammar_rejects_invalid_syntax(text):
    from nanoforge.generation.grammar import json_byte_grammar

    grammar = json_byte_grammar(ByteTokenizer())
    assert not grammar.is_accepting(ByteTokenizer().encode(text))


def test_grammar_enforces_actual_generation_and_fails_on_incomplete_budget():
    from nanoforge.generation.grammar import TokenGrammar

    engine = tiny_engine()
    expected = '{"ok":true}'
    grammar = TokenGrammar.choices(engine.tokenizer, [expected], eos_id=engine.tokenizer.eos_id)
    sampling = SamplingConfig(mode="deterministic", stop_on_repetition=False)
    assert engine.complete("x", 20, sampling, stop_tokens=[], constraint=grammar) == expected
    with pytest.raises(ValueError, match="accepting"):
        engine.complete("x", 3, sampling, stop_tokens=[], constraint=grammar)


def test_context_free_recursive_and_nullable_rules_and_parser_budget():
    from nanoforge.generation.grammar import TokenGrammar

    grammar = TokenGrammar({"s": [(), (4, "s", 5, "s")]}, "s", eos_id=2)
    assert grammar.is_accepting([])
    assert grammar.is_accepting([4, 4, 5, 5])
    assert not grammar.is_accepting([4, 5, 5])
    assert grammar.allowed_tokens([4]) == {4, 5}
    with pytest.raises(ValueError, match="budget"):
        TokenGrammar({"s": [(), (4, "s", 5)]}, "s", max_states=2).allowed_tokens([])


@pytest.mark.parametrize("different_draft", [False, True])
def test_greedy_speculation_matches_target_with_acceptance_and_rejection(different_draft):
    target = tiny_engine(max_seq_len=16)
    draft = GenerationEngine(copy.deepcopy(target.model), ByteTokenizer(), device="cpu")
    if different_draft:
        draft.model.head.proj.weight = torch.nn.Parameter(target.model.head.proj.weight.roll(17, 0).clone())
    ids = target.tokenizer.encode("abcdef", add_bos=True)
    sampling = SamplingConfig(mode="deterministic", stop_on_repetition=False)
    expected = list(target.generate_ids(ids, 25, sampling, eos_id=target.tokenizer.eos_id))
    actual = list(target.speculative_generate_ids(ids, draft, 25, draft_tokens=4))
    assert actual == expected
    assert target.speculative_stats["verification_calls"] > 0
    if different_draft:
        assert target.speculative_stats["accepted"] < target.speculative_stats["drafted"]
    else:
        assert target.speculative_stats["accepted"] == target.speculative_stats["drafted"]
        assert target.speculative_stats["target_forward_calls"] < 26


def test_contrastive_search_scores_hidden_state_candidates():
    engine = tiny_engine()
    sampling = SamplingConfig(sampler="contrastive", contrastive_top_k=3,
                              contrastive_alpha=.6, stop_on_repetition=False)
    with patch.object(engine.model, "forward", wraps=engine.model.forward) as forward:
        tokens = list(engine.generate_ids(engine.tokenizer.encode("abc", add_bos=True), 3, sampling))
    assert len(tokens) == 3
    assert forward.call_count == 12  # one state forward + three candidate evaluations per token
    assert all(call.kwargs.get("return_hidden_states") for call in forward.call_args_list)


def test_continuous_batch_forward_matches_independent_generations_and_admits_next_request():
    from nanoforge.generation.continuous import ContinuousBatcher

    engine = tiny_engine()
    sampling = SamplingConfig(mode="deterministic", stop_on_repetition=False)
    ids = [engine.tokenizer.encode(text, add_bos=True) for text in ("abc", "def", "ghi")]
    expected = {str(i): list(engine.generate_ids(prompt, 3, sampling,
                                               eos_id=engine.tokenizer.eos_id))
                for i, prompt in enumerate(ids)}
    batcher = ContinuousBatcher(engine, max_active_requests=2, max_batch_tokens=16)
    for i, prompt in enumerate(ids):
        batcher.submit(prompt, 3, sampling, request_id=str(i))
    actual = {str(i): [] for i in range(3)}
    events = list(batcher.run())
    for event in events:
        assert event.error is None
        if event.token_id is not None:
            actual[event.request_id].append(event.token_id)
    assert actual == expected
    assert batcher.stats["maximum_batch_size"] == 2
    assert batcher.stats["admitted_requests"] == 3
    assert batcher.stats["model_forward_calls"] == 6
    assert batcher.reserved_tokens == 0
    assert batcher.cache_bytes == 0


def test_continuous_batch_cancellation_and_resource_limits():
    from nanoforge.generation.continuous import ContinuousBatcher

    engine = tiny_engine()
    batcher = ContinuousBatcher(engine, max_active_requests=1, max_batch_tokens=6,
                                max_pending_requests=2, max_generated_tokens=4)
    first = batcher.submit([4], 3, request_id="first")
    batcher.submit([5], 3, request_id="second")
    with pytest.raises(ValueError, match="limit"):
        batcher.submit([6], 3)
    assert batcher.cancel(first)
    events = batcher.step()
    assert events[0].error == "cancelled"
    assert list(batcher.active) == ["second"]
    with pytest.raises(ValueError, match="budget"):
        batcher.submit([4], 5)


def test_mirostat_versions_run_independent_entropy_control():
    from nanoforge.generation.sampling import MirostatState, sample_next

    logits = torch.arange(12, dtype=torch.float32).neg().reshape(1, 1, 12)
    for version in (1, 2):
        config = SamplingConfig(mirostat=True, mirostat_version=version, top_k=None, top_p=None,
                                mirostat_tau=2., mirostat_eta=.1)
        state = MirostatState(config.mirostat_tau)
        for _ in range(30):
            token = sample_next(logits, torch.tensor([], dtype=torch.long), config, state)
            assert 0 <= token.item() < 12
        assert state.mu != 4.


def test_no_valid_sampling_logits_raise_even_in_greedy_mode():
    from nanoforge.generation.sampling import sample_next

    with pytest.raises(ValueError, match="continuation"):
        sample_next(torch.full((1, 1, 3), -torch.inf), torch.tensor([], dtype=torch.long),
                    SamplingConfig(deterministic=True))


def test_hybrid_queries_use_real_global_and_local_masks():
    from nanoforge.model.attention import HybridLocalGlobalCausalAttention

    config = small_config(sliding_window=2)
    config.hybrid_global_stride = 4
    attention = HybridLocalGlobalCausalAttention(config)
    mask = attention.mask(0, 6, 0, 6, torch.device("cpu"), torch.float32, None)
    assert torch.isfinite(mask[4, :5]).all()
    assert not torch.isfinite(mask[5, :4]).any()
    assert torch.isfinite(mask[5, 4:6]).all()


def test_paged_cache_has_multiple_bounded_pages_and_clone_isolation():
    from nanoforge.model.attention import PagedKVCache

    cache = PagedKVCache(max_length=9, page_size=3)
    cache.append(torch.randn(1, 2, 8, 4), torch.randn(1, 2, 8, 4))
    clone = cache.clone_detached()
    cache.append(torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 4))
    assert cache.tokens_seen == 13 and cache.length == 9
    assert clone.tokens_seen == 8 and clone.length == 8
    assert all(key.shape[2] <= 3 for key, _ in cache.iter_blocks())


def test_moe_capacity_has_visible_training_overflow_and_no_inference_drop():
    from nanoforge.model.moe import MoEFeedForward

    moe = MoEFeedForward(8, 16, MoEConfig(num_experts=2, top_k=1, capacity_factor=.5))
    with torch.no_grad():
        moe.router.weight.zero_()
    moe(torch.randn(1, 16, 8))
    assert moe.routing_stats["capacity_enforced"]
    assert moe.routing_stats["dropped_assignments"] > 0
    moe.eval()
    moe(torch.randn(1, 16, 8))
    assert moe.routing_stats["dropped_assignments"] == 0


def test_observed_generation_resource_report_has_real_token_counts():
    from nanoforge.benchmark import benchmark_generation

    engine = tiny_engine()
    report = benchmark_generation(engine, "abc", 5, repeats=2, warmup_steps=0)
    assert report["generated_tokens"] == sum(run["generated_tokens"] for run in report["runs"])
    assert report["tokens_per_sec"] > 0
    assert report["peak_rss_scope"] == "process_lifetime"
    assert report["quality_status"] == "not_evaluated"
    assert report["actual_attention_backends"] == ["manual"]


def test_forward_benchmark_validates_context_and_restores_threads(tmp_path):
    from nanoforge.benchmark import benchmark_forward

    config = tmp_path / "bench.yaml"
    config.write_text("model:\n  vocab_size: 260\n  d_model: 32\n  n_layers: 1\n"
                      "  n_heads: 4\n  max_seq_len: 16\n"
                      "data:\n  tokenizer_type: byte\n  seq_len: 8\n", encoding="utf-8")
    threads = torch.get_num_threads()
    report = benchmark_forward(config, steps=2, warmup_steps=1, device="cpu", cpu_threads=1)
    assert report["steps"] == 2 and report["tokens_per_sec"] > 0
    assert report["params"] == sum(p.numel() for p in NanoforgeForCausalLM(
        small_config(n_layers=1, max_seq_len=16, n_kv_heads=4)).parameters())
    assert torch.get_num_threads() == threads
    with pytest.raises(ValueError):
        benchmark_forward(config, steps=0)


def test_stochastic_speculation_matches_target_distribution_with_bad_draft():
    from nanoforge.model.attention import KVCache
    from nanoforge.model.heads import ModelOutput

    class ConstantModel(torch.nn.Module):
        def __init__(self, probabilities):
            super().__init__()
            self.config = small_config(n_layers=1)
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            logits = torch.full((260,), -torch.inf)
            logits[69:72] = torch.tensor(probabilities).log()
            self.register_buffer("constant", logits)

        def forward(self, ids, caches=None, use_cache=False):
            batch, length = ids.shape
            cache = caches[0] if caches else KVCache(max_length=32)
            cache.append(torch.zeros(batch, 1, length, 2), torch.zeros(batch, 1, length, 2))
            return ModelOutput(logits=self.constant.reshape(1, 1, -1).expand(batch, length, -1),
                               caches=[cache])

    target = GenerationEngine(ConstantModel([.1, .3, .6]), ByteTokenizer(), "cpu")
    draft = GenerationEngine(ConstantModel([.7, .2, .1]), ByteTokenizer(), "cpu")
    config = SamplingConfig(temperature=1., top_k=None, top_p=None, stop_on_repetition=False)
    counts = [0, 0, 0]
    for _ in range(1600):
        token = list(target.stochastic_speculative_generate_ids([4], draft, 1,
                                                               draft_tokens=1, sampling=config))[0]
        counts[token - 69] += 1
    observed = torch.tensor(counts) / sum(counts)
    torch.testing.assert_close(observed, torch.tensor([.1, .3, .6]), atol=.035, rtol=0)


def test_tokenizer_identity_rejects_incompatible_special_ids():
    from nanoforge.generation.speculative import tokenizer_identity

    first = ByteTokenizer()
    second = ByteTokenizer()
    second.eos_id = 12
    assert tokenizer_identity(first) != tokenizer_identity(second)


def test_dynamic_rotary_cached_attention_rerotates_retained_keys():
    attention = CausalSelfAttention(small_config(
        rope_scaling={"type": "dynamic", "factor": 2.,
                      "original_max_position_embeddings": 4})).eval()
    hidden = torch.randn(1, 9, 32)
    cache = None
    with torch.no_grad():
        for position in range(hidden.shape[1]):
            expected, _ = attention(hidden[:, :position + 1])
            actual, cache = attention(hidden[:, position:position + 1], cache=cache, use_cache=True)
            torch.testing.assert_close(actual, expected[:, -1:], atol=3e-6, rtol=3e-5)


def test_stream_cooperative_interrupt_and_request_local_usage():
    engine = tiny_engine()
    usage = {}
    calls = 0

    def interrupted():
        nonlocal calls
        calls += 1
        return calls > 2

    with patch("nanoforge.generation.engine.sample_next", return_value=torch.tensor([[69]])):
        text = "".join(engine.stream("abc", max_new_tokens=20,
                                    sampling=SamplingConfig(mode="deterministic", stop_on_repetition=False),
                                    stop_tokens=[], interrupt=interrupted, usage=usage))
    assert usage["generated_tokens"] == 2
    assert usage["output_tokens"] == 2
    assert usage["counting"] == "exact_generated_ids"
    assert engine.count_tokens(text) == 2
    assert engine.count_tokens("\u00e9") == 2


def test_generation_deadline_and_invalid_wall_budget():
    engine = tiny_engine()
    with pytest.raises(TimeoutError, match="wall-clock"):
        list(engine.generate_ids([4], 3, SamplingConfig(), max_wall_seconds=1e-12))
    with pytest.raises(ValueError, match="wall-clock"):
        list(engine.generate_ids([4], 3, SamplingConfig(), max_wall_seconds=0))


def test_prefix_cache_budgets_and_lru_eviction():
    engine = GenerationEngine(NanoforgeForCausalLM(small_config(n_layers=1)), ByteTokenizer(),
                              "cpu", prefix_cache_max_entries=1, prefix_cache_max_bytes=100000)
    engine.cache_prefix("abc")
    first = next(iter(engine.prefix_cache))
    engine.cache_prefix("def")
    assert len(engine.prefix_cache) == 1 and first not in engine.prefix_cache
    assert 0 < engine.prefix_cache_bytes <= 100000
    engine.clear_prefix_cache()
    assert engine.prefix_cache_bytes == 0


def test_generator_registry_wrappers_select_real_search_contracts():
    from nanoforge.generation.sampling import beam_generate, contrastive_generate, top_k_top_p_generate

    engine = tiny_engine()
    prompt = engine.tokenizer.encode("abc", add_bos=True)
    for generate in (beam_generate, contrastive_generate, top_k_top_p_generate):
        tokens = list(generate(engine, prompt, 2, SamplingConfig(mode="deterministic",
                               stop_on_repetition=False)))
        assert len(tokens) == 2


@pytest.mark.parametrize("window,expected_keys", [(None, 5), (2, 2)])
def test_attention_entropy_is_opt_in_and_measured_on_a_bounded_query(window, expected_keys):
    import math

    attention = CausalSelfAttention(small_config(sliding_window=window)).eval()
    with torch.no_grad():
        attention.q_proj.weight.zero_()
        attention.k_proj.weight.zero_()
        attention(torch.randn(1, 5, 32))
        assert attention.last_attention_diagnostics is None
        attention.collect_diagnostics = True
        attention(torch.randn(1, 5, 32))
    diagnostic = attention.last_attention_diagnostics
    assert diagnostic["scope"] == "reference_last_query_pre_dropout"
    assert diagnostic["query_count"] == 1
    assert diagnostic["entropy_nats_mean"] == pytest.approx(math.log(expected_keys), abs=1e-6)


def test_beam_complete_honors_context_free_grammar_and_default_stops():
    from nanoforge.generation.grammar import TokenGrammar

    engine = tiny_engine()
    grammar = TokenGrammar.choices(engine.tokenizer, ['{"ok":true}'], eos_id=engine.tokenizer.eos_id)
    assert engine.complete("x", 20, SamplingConfig(sampler="beam", num_beams=2),
                           constraint=grammar) == '{"ok":true}'


@pytest.mark.parametrize("values", [{"mirostat_tau": float("inf")}, {"temperature": True},
                                    {"deterministic": "false"}, {"top_k": 1.5}])
def test_sampling_configuration_rejects_wrong_types_and_nonfinite_values(values):
    with pytest.raises(ValueError):
        SamplingConfig(**values)


def test_unigram_blocking_and_invalid_history_are_enforced():
    from nanoforge.generation.sampling import sample_next

    logits = torch.tensor([[[1., 2., 3.]]])
    history = torch.tensor([2], dtype=torch.long)
    assert sample_next(logits, history, SamplingConfig(deterministic=True,
                                                       no_repeat_ngram_size=1)).item() == 1
    with pytest.raises(ValueError, match="history"):
        sample_next(logits, torch.tensor([-1]), SamplingConfig(deterministic=True))


@pytest.mark.parametrize("backend", ["sdpa", "chunked", "sparse", "hybrid_local_global", "paged"])
def test_traced_logits_preserve_dynamic_sequence_lengths_across_tile_boundaries(backend):
    import warnings

    class LogitsOnly(torch.nn.Module):
        def __init__(self):
            super().__init__()
            config = small_config(n_layers=1, attention_backend=backend)
            config.attention_chunk_size = 4
            self.model = NanoforgeForCausalLM(config).eval()

        def forward(self, ids):
            return self.model(ids).logits

    wrapper = LogitsOnly().eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", torch.jit.TracerWarning)
        traced = torch.jit.trace(wrapper, torch.tensor([[4, 5, 6]]), check_trace=False)
    for length in (2, 5, 9):
        ids = torch.randint(4, 260, (1, length))
        torch.testing.assert_close(traced(ids), wrapper(ids), atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("scaling", [None, {"type": "linear", "factor": 2},
                                    {"type": "dynamic", "factor": 2, "original_max_position_embeddings": 4},
                                    {"type": "yarn", "factor": 2, "original_max_position_embeddings": 4}])
def test_rotary_range_equals_full_positions_and_long_cached_decode_stays_bounded(scaling):
    rope = RotaryEmbedding(8, 32, scaling=scaling)
    full_cos, full_sin = rope.get_cos_sin(100, torch.device("cpu"), torch.float32)
    cos, sin = rope.get_cos_sin_range(95, 100, torch.device("cpu"), torch.float32)
    torch.testing.assert_close(cos, full_cos[95:])
    torch.testing.assert_close(sin, full_sin[95:])
    attention = CausalSelfAttention(small_config(sliding_window=4, rope_scaling=scaling)).eval()
    cache = None
    with torch.no_grad():
        for _ in range(100):
            _, cache = attention(torch.randn(1, 1, 32), cache=cache, use_cache=True)
    assert cache.tokens_seen == 100
    assert cache.length == 4
    assert max(cos.shape[0] for cos, _ in attention.rope._cache.values()) <= 32







