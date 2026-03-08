#!/usr/bin/env python3
"""
Extract sentences from test_english.tsv using the exact same parsing mechanism
as samples_generator.py
"""

import pandas as pd

# Constants (same as samples_generator.py)
LABEL2ID = {"O": 0, "B-IDIOM": 1, "I-IDIOM": 2}

def read_bio_tsv(file_path: str) -> pd.DataFrame:
    """
    Reads a BIO-formatted TSV file and returns a Pandas DataFrame.
    Each sentence is treated as a separate sample with a unique ID.

    This is the EXACT same function from samples_generator.py
    """
    data = []
    tokens = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:  # Empty line indicates a new sentence
                if tokens:
                    data.append(([w for w, _ in tokens], [t for _, t in tokens]))
                    tokens = []
                continue

            parts = line.split("\t")
            if len(parts) == 2:
                word, tag = parts
                tokens.append((word, tag))

    # Add the last sentence if not already added
    if tokens:
        data.append(([w for w, _ in tokens], [t for _, t in tokens]))

    df = pd.DataFrame(data, columns=["tokens", "tags"])

    # Add a column with tag ids
    df["tag_ids"] = df["tags"].apply(lambda x: [LABEL2ID[t] for t in x])

    # Add a column with a list of the MWEs in each sentence
    def extract_idioms(row):
        tags = row["tags"]
        tokens = row["tokens"]
        idioms = []

        while "B-IDIOM" in tags:
            start = tags.index("B-IDIOM")
            # Read until the next "O" tag
            end = start + 1
            while end < len(tags) and tags[end] != "O":
                end += 1
            idioms.append(" ".join(tokens[start:end]).strip())
            tags = tags[end:]
            tokens = tokens[end:]

        return idioms

    df["true_idioms"] = df.apply(extract_idioms, axis=1)

    # Add a column with the full sentence
    # NOTE: This joins tokens with NO SEPARATOR - this is the actual implementation!
    df["sentence"] = df["tokens"].apply(lambda x: "".join(x))

    # Make this column the first one
    df = df[["sentence", "tokens", "tags", "tag_ids", "true_idioms"]]

    return df


if __name__ == "__main__":
    # Read the TSV file using the exact same parsing mechanism
    print("Reading test_english.tsv using the EXACT parsing mechanism from samples_generator.py...")
    df = read_bio_tsv("test_english.tsv")

    print(f"\nFound {len(df)} sentences")
    print("\nFirst 5 sentences after parsing:")
    print("=" * 80)

    for idx, row in df.head(10).iterrows():
        print(f"\nSentence {idx + 1}:")
        print(f"  Parsed sentence: {row['sentence']}")
        print(f"  Tokens: {row['tokens']}")
        print(f"  Tags: {row['tags']}")
        print(f"  True idioms: {row['true_idioms']}")

    # Save just the sentences to a text file
    output_file = "parsed_sentences.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, sentence in enumerate(df['sentence'], 1):
            f.write(f"{idx}. {sentence}\n")

    print(f"\n\nSentences saved to: {output_file}")

    # Also save the full dataframe to CSV
    csv_output = "parsed_sentences_full.csv"
    df.to_csv(csv_output, index=False)
    print(f"Full data saved to: {csv_output}")
