from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

Symbol = int | str
Item = tuple[str, tuple[Symbol, ...], int, int]


@dataclass(frozen=True)
class TokenGrammar:
    """Bounded Earley parser for context-free grammars over integer token IDs.

    Production symbols are nonterminal names (str) or token IDs (int). Epsilon
    productions are empty tuples. EOS is allowed only when the start rule is
    complete; it is not part of the language. This is a reference parser, not
    a tokenizer-independent JSON-schema compiler or an optimized grammar kernel.
    """

    rules: Mapping[str, Sequence[Sequence[Symbol]]]
    start: str
    eos_id: int | None = None
    max_states: int = 100_000
    max_input_tokens: int = 4096

    def __post_init__(self):
        if self.start not in self.rules or self.max_states < 1 or self.max_input_tokens < 1:
            raise ValueError("Invalid token grammar start or parser budget.")
        immutable = {name: tuple(tuple(rhs) for rhs in alternatives)
                     for name, alternatives in self.rules.items()}
        if "__start__" in immutable:
            raise ValueError("Reserved grammar symbol __start__.")
        for alternatives in immutable.values():
            for rhs in alternatives:
                for symbol in rhs:
                    if isinstance(symbol, bool) or not isinstance(symbol, (int, str)):
                        raise ValueError("Grammar symbols must be token integers or rule names.")
                    if isinstance(symbol, str) and symbol not in immutable:
                        raise ValueError(f"Unknown grammar rule: {symbol}")
                    if isinstance(symbol, int) and symbol < 0:
                        raise ValueError("Grammar token IDs must be nonnegative.")
        object.__setattr__(self, "rules", immutable)

    def _chart(self, tokens):
        if len(tokens) > self.max_input_tokens:
            raise ValueError("Grammar input token budget exceeded.")
        chart: list[set[Item]] = [set() for _ in range(len(tokens) + 1)]
        chart[0].add(("__start__", (self.start,), 0, 0))
        count = 1
        for position in range(len(chart)):
            agenda = list(chart[position])
            cursor = 0
            while cursor < len(agenda):
                lhs, rhs, dot, origin = agenda[cursor]
                cursor += 1
                additions = []
                if dot == len(rhs):
                    for plhs, prhs, pdot, porigin in tuple(chart[origin]):
                        if pdot < len(prhs) and prhs[pdot] == lhs:
                            additions.append((plhs, prhs, pdot + 1, porigin))
                elif isinstance(rhs[dot], str):
                    name = rhs[dot]
                    additions.extend((name, alternative, 0, position)
                                     for alternative in self.rules[name])
                    # A nullable rule may already have completed at this position.
                    if any(clhs == name and cdot == len(crhs) and corigin == position
                           for clhs, crhs, cdot, corigin in chart[position]):
                        additions.append((lhs, rhs, dot + 1, origin))
                elif position < len(tokens) and rhs[dot] == tokens[position]:
                    item = (lhs, rhs, dot + 1, origin)
                    if item not in chart[position + 1]:
                        chart[position + 1].add(item)
                        count += 1
                for item in additions:
                    if item not in chart[position]:
                        chart[position].add(item)
                        agenda.append(item)
                        count += 1
                if count > self.max_states:
                    raise ValueError("Grammar parser state budget exceeded.")
            if position < len(tokens) and not chart[position + 1]:
                return None
        return chart[-1]

    def allowed_tokens(self, generated):
        last = self._chart(generated)
        if last is None:
            return set()
        allowed = {rhs[dot] for _, rhs, dot, _ in last
                   if dot < len(rhs) and isinstance(rhs[dot], int)}
        if self.eos_id is not None and ("__start__", (self.start,), 1, 0) in last:
            allowed.add(self.eos_id)
        return allowed

    def is_accepting(self, generated):
        last = self._chart(generated)
        return last is not None and ("__start__", (self.start,), 1, 0) in last

    @classmethod
    def choices(cls, tokenizer, texts, *, eos_id=None, **budgets):
        if not texts:
            raise ValueError("Grammar choices must not be empty.")
        rules = {"choice": [tokenizer.encode(text, add_bos=False, add_eos=False) for text in texts]}
        return cls(rules, "choice", eos_id, **budgets)


def json_byte_grammar(tokenizer, **budgets) -> TokenGrammar:
    """Strict JSON syntax over a byte tokenizer, including valid UTF-8 strings.

    This validates JSON syntax. Application schemas and tool policy are separate
    checks; callers must not describe it as schema validation.
    """
    offset = getattr(tokenizer, "offset", None)
    if offset is None or tokenizer.encode("{}", add_bos=False, add_eos=False) != [123 + offset, 125 + offset]:
        raise ValueError("JSON byte grammar requires an unmerged byte tokenizer.")

    def terminal(text):
        return tuple(byte + offset for byte in text.encode("ascii"))

    rules = {}
    rules["json"] = [("ws", "value", "ws")]
    rules["ws"] = [(), (32 + offset, "ws"), (9 + offset, "ws"),
                   (10 + offset, "ws"), (13 + offset, "ws")]
    rules["value"] = [("object",), ("array",), ("string",), ("number",),
                      terminal("true"), terminal("false"), terminal("null")]
    rules["object"] = [(123 + offset, "ws", 125 + offset),
                       (123 + offset, "ws", "members", "ws", 125 + offset)]
    rules["members"] = [("member",), ("member", "ws", 44 + offset, "ws", "members")]
    rules["member"] = [("string", "ws", 58 + offset, "ws", "value")]
    rules["array"] = [(91 + offset, "ws", 93 + offset),
                      (91 + offset, "ws", "values", "ws", 93 + offset)]
    rules["values"] = [("value",), ("value", "ws", 44 + offset, "ws", "values")]
    rules["string"] = [(34 + offset, "chars", 34 + offset)]
    rules["chars"] = [(), ("char", "chars")]
    rules["char"] = [(i + offset,) for i in range(32, 127) if i not in (34, 92)]
    rules["char"] += [(92 + offset, "escape"), ("utf8",)]
    rules["escape"] = [(ord(char) + offset,) for char in '"\\/bfnrt']
    rules["escape"] += [(117 + offset, "hex", "hex", "hex", "hex")]
    rules["hex"] = [(ord(char) + offset,) for char in "0123456789abcdefABCDEF"]
    rules["digit"] = [(ord(char) + offset,) for char in "0123456789"]
    rules["nonzero"] = [(ord(char) + offset,) for char in "123456789"]
    rules["digits"] = [("digit",), ("digit", "digits")]
    rules["integer"] = [(48 + offset,), ("nonzero",), ("nonzero", "digits")]
    rules["sign"] = [(), (45 + offset,)]
    rules["fraction"] = [(), (46 + offset, "digits")]
    rules["exponent"] = [(), ("e", "exp_sign", "digits")]
    rules["e"] = [(69 + offset,), (101 + offset,)]
    rules["exp_sign"] = [(), (43 + offset,), (45 + offset,)]
    rules["number"] = [("sign", "integer", "fraction", "exponent")]
    rules["continuation"] = [(i + offset,) for i in range(0x80, 0xC0)]
    rules["cont_a0"] = [(i + offset,) for i in range(0xA0, 0xC0)]
    rules["cont_9f"] = [(i + offset,) for i in range(0x80, 0xA0)]
    rules["cont_90"] = [(i + offset,) for i in range(0x90, 0xC0)]
    rules["cont_8f"] = [(i + offset,) for i in range(0x80, 0x90)]
    rules["utf8"] = [(i + offset, "continuation") for i in range(0xC2, 0xE0)]
    rules["utf8"] += [(0xE0 + offset, "cont_a0", "continuation"),
                       (0xED + offset, "cont_9f", "continuation")]
    rules["utf8"] += [(i + offset, "continuation", "continuation")
                      for i in range(0xE1, 0xF0) if i != 0xED]
    rules["utf8"] += [(0xF0 + offset, "cont_90", "continuation", "continuation"),
                       (0xF4 + offset, "cont_8f", "continuation", "continuation")]
    rules["utf8"] += [(i + offset, "continuation", "continuation", "continuation")
                      for i in range(0xF1, 0xF4)]
    return TokenGrammar(rules, "json", tokenizer.eos_id, **budgets)
