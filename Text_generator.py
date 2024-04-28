import random

# Prepare the text dataset
with open('text_data.txt', 'r') as f:
    text = f.read()

# Build the Markov chain model
n = 2  # Use 2-grams for this example
grams = {}
for i in range(len(text) - n):
    key = text[i:i+n]
    if key not in grams:
        grams[key] = []
    grams[key].append(text[i+n])

# Implement a function to generate text
def generate_text(grams, n, start_text):
    text = start_text
    for i in range(100):  # Generate 100 words for this example
        key = text[-n:]
        if key in grams:
            text += random.choice(grams[key])
        else:
            break
    return text

# Generate the text
start_text = 'The quick brown '
generated_text = generate_text(grams, n, start_text)
print(generated_text)