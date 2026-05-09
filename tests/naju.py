text = "Hello my name is Najwan"

ids = tokenizer.encode(text)
print(ids)

decoded = tokenizer.decode(ids)
print(decoded)