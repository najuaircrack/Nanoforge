
from pathlib import Path
import pandas as pd

INPUT_DIR = Path("data/raw")
OUT_FILE = Path("data/chat/chat_data.txt")

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

SPECIAL_TOKENS = {
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "system": "<|system|>"
}

with OUT_FILE.open("w", encoding="utf-8") as out:

    for parquet_file in INPUT_DIR.glob("*.parquet"):

        print("Loading:", parquet_file)

        df = pd.read_parquet(parquet_file)

        if "messages" not in df.columns:
            print("Skipping no messages:", parquet_file)
            continue

        for msgs in df["messages"]:

            try:
                conversation = []

                for msg in msgs:

                    role = msg.get("role", "").strip()
                    content = msg.get("content", "").strip()

                    if not content:
                        continue

                    token = SPECIAL_TOKENS.get(
                        role,
                        "<|user|>"
                    )

                    conversation.append(
                        f"{token}\n{content}\n"
                    )

                if conversation:

                    text = "\n".join(conversation)

                    text += "\n<|endoftext|>\n\n"

                    out.write(text)

            except Exception as e:
                print("Bad row:", e)

print("DONE")

