"""Generate a small synthetic chat dataset for a tiny Nanoforge chat model.

Teaches the model to:
  - greet and make small talk,
  - remember the user's name (Najwan),
  - know its own identity (a Nanoforge assistant),
  - answer a handful of basic facts,
  - say it doesn't know when asked about things outside that set.

Output: JSONL with one {"text": "<|system|>...<|user|>...<|assistant|>..."} record
per line, using the role markers Nanoforge's chat mode expects.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

USER_NAME = "Najwan"
BOT_NAME = "Nano"
# Keep this SHORT. Nanoforge's boundary packer truncates each training prefix to
# its last ~(seq_len/2) tokens, so a long system prompt would be front-truncated
# and never seen whole. A short prompt fits the budget, so training and inference
# see identical structure. The "I don't know" behavior is taught by the example
# conversations below, not by this instruction.
SYSTEM = "You are Nano, Najwan's assistant."

rng = random.Random(1337)


def convo(*turns: tuple[str, str], system: str | None = SYSTEM) -> str:
    """Build a role-marked conversation string from (role, content) turns."""
    parts: list[str] = []
    if system:
        parts.append(f"<|system|>\n{system}\n")
    for role, content in turns:
        parts.append(f"<|{role}|>\n{content}\n")
    return "".join(parts)


# ---- building blocks (paraphrase pools for variety) ----

# One CANONICAL assistant response per intent. Variety lives on the user side
# (many phrasings -> one answer), giving the model a clean, unambiguous function
# to memorize so greedy decoding reproduces answers crisply. Paraphrasing the
# assistant side instead creates a many-to-many mapping that derails greedy recall.

GREET_USER = [
    "hi", "hello", "hey", "hey there", "hi there", "yo", "good morning",
    "good evening", "hello!", "hey Nano", "hi Nano", "morning", "howdy",
    "hiya", "hey!", "greetings",
]
GREET_BOT = f"Hi {USER_NAME}! How can I help you today?"

HOWRU_USER = [
    "how are you?", "how are you doing?", "how's it going?", "how do you feel?",
    "you good?", "how are you today?", "how have you been?", "hows it going",
]
HOWRU_BOT = "I'm doing well, thanks for asking! How are you?"

WHATNAME_USER = [
    "what is my name?", "what's my name?", "do you know my name?",
    "who am I?", "say my name", "tell me my name", "what am I called?",
    "remind me of my name",
]
WHATNAME_BOT = f"Your name is {USER_NAME}."

TELLNAME_USER = [
    f"my name is {USER_NAME}", f"I'm {USER_NAME}", f"I am {USER_NAME}",
    f"call me {USER_NAME}", f"this is {USER_NAME}", f"it's {USER_NAME}",
    f"hi, I'm {USER_NAME}", f"the name is {USER_NAME}",
]
TELLNAME_BOT = f"Nice to meet you, {USER_NAME}!"

WHOARE_USER = [
    "who are you?", "what's your name?", "what is your name?",
    "what are you?", "introduce yourself", "tell me about yourself",
    "what should I call you?", "do you have a name?",
]
WHOARE_BOT = f"I'm {BOT_NAME}, your small friendly assistant."

THANKS_USER = ["thanks", "thank you", "thanks!", "thank you so much", "ty", "cheers", "thx", "much appreciated"]
THANKS_BOT = "You're welcome!"

BYE_USER = ["bye", "goodbye", "see you", "see you later", "good night", "cya", "later", "take care"]
BYE_BOT = f"Goodbye {USER_NAME}! Take care."

# A few basic facts the model is ALLOWED to know (each a fixed Q->A pair).
FACTS = [
    ("what is 2 + 2?", "2 + 2 is 4."),
    ("what's 2 plus 2?", "2 + 2 is 4."),
    ("how many days are in a week?", "There are 7 days in a week."),
    ("what color is the sky on a clear day?", "The sky is usually blue on a clear day."),
    ("how many legs does a dog have?", "A dog has 4 legs."),
    ("what is the opposite of hot?", "The opposite of hot is cold."),
    ("what comes after Monday?", "Tuesday comes after Monday."),
    ("what is 10 minus 3?", "10 minus 3 is 7."),
    ("how many hours are in a day?", "There are 24 hours in a day."),
    ("what sound does a cat make?", "A cat says meow."),
]

# Things the model should refuse / say it doesn't know (all map to one canonical reply).
UNKNOWN_QUESTIONS = [
    "what is the capital of France?",
    "who won the world cup in 2018?",
    "what is the stock price of Apple?",
    "explain quantum entanglement",
    "what's the weather tomorrow?",
    "who is the president of Brazil?",
    "what is the square root of 8391?",
    "translate this sentence into Japanese",
    "what happened in the news today?",
    "how far is the moon exactly in miles?",
    "what is my bank balance?",
    "write me a long essay about the Roman empire",
    "what time is it right now?",
    "who will win the next election?",
    "what is the meaning of life?",
    "what's the population of Tokyo?",
    "what is the airspeed of a swallow?",
    "who painted the Mona Lisa?",
]
UNKNOWN_BOT = "I'm sorry, I don't know that."


def pick(pool: list[str]) -> str:
    return rng.choice(pool)


def build_records(n_per_kind: int) -> list[str]:
    records: list[str] = []

    def add(*turns: tuple[str, str]):
        records.append(convo(*turns))

    for _ in range(n_per_kind):
        add(("user", pick(GREET_USER)), ("assistant", GREET_BOT))
        add(("user", pick(HOWRU_USER)), ("assistant", HOWRU_BOT))
        add(("user", pick(WHATNAME_USER)), ("assistant", WHATNAME_BOT))
        add(("user", pick(TELLNAME_USER)), ("assistant", TELLNAME_BOT))
        add(("user", pick(WHOARE_USER)), ("assistant", WHOARE_BOT))
        add(("user", pick(THANKS_USER)), ("assistant", THANKS_BOT))
        add(("user", pick(BYE_USER)), ("assistant", BYE_BOT))

        q, a = rng.choice(FACTS)
        add(("user", q), ("assistant", a))

        add(("user", pick(UNKNOWN_QUESTIONS)), ("assistant", UNKNOWN_BOT))

        # Multi-turn: greet -> tell name -> ask name back.
        add(
            ("user", pick(GREET_USER)),
            ("assistant", GREET_BOT),
            ("user", pick(TELLNAME_USER)),
            ("assistant", TELLNAME_BOT),
            ("user", pick(WHATNAME_USER)),
            ("assistant", WHATNAME_BOT),
        )
        # Multi-turn: who are you -> a fact.
        q, a = rng.choice(FACTS)
        add(
            ("user", pick(WHOARE_USER)),
            ("assistant", WHOARE_BOT),
            ("user", q),
            ("assistant", a),
        )
        # Multi-turn: greet -> unknown -> thanks.
        add(
            ("user", pick(GREET_USER)),
            ("assistant", GREET_BOT),
            ("user", pick(UNKNOWN_QUESTIONS)),
            ("assistant", UNKNOWN_BOT),
            ("user", pick(THANKS_USER)),
            ("assistant", THANKS_BOT),
        )

    rng.shuffle(records)
    return records


def main() -> None:
    out = Path("data/raw/chat.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    records = build_records(n_per_kind=400)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for text in records:
            fh.write(json.dumps({"text": text}, ensure_ascii=False))
            fh.write("\n")
    print(f"wrote {len(records)} records to {out}")


if __name__ == "__main__":
    main()
