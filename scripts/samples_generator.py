#!/usr/bin/env python3
"""
Confusing Context Variant Generator

This script generates confusing context variants from test_english.tsv data
and returns them in a structured DataFrame format with proper BIO tagging.

Combines:
- Variant generation logic from confusing_context_pipeline.py
- Data structure and BIO processing from display_test_data_simple.py
"""

import pandas as pd
import re
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.google import Gemini
from datetime import datetime
from dotenv import load_dotenv
import os

from system_prompts import SYSTEM_PROMPT_V2, SYSTEM_PROMPT_V3_ITALIAN, SYSTEM_PROMPT_V3_SPANISH, SYSTEM_PROMPT_V3_GERMAN

# Load environment variables
load_dotenv()

# Constants
LABEL2ID = {"O": 0, "B-IDIOM": 1, "I-IDIOM": 2}
LABELS = list(LABEL2ID.keys())

# Gemini 2.5 Pro
model_name = "gemini-2.5-pro"
model = Gemini(id=model_name, api_key=os.getenv("GEMINI_API_KEY"))

class ConfusingVariants(BaseModel):
    """Structured output model for confusing context variants"""
    sentence: str = Field(..., description="The original input sentence")
    bio_tag: Optional[str] = Field(None, description="The BIO tag if present, None if no idiom")
    variants: List[str] = Field(..., description="Confusing context variants of the sentence")

def read_bio_tsv(file_path: str) -> pd.DataFrame:
    """
    Reads a BIO-formatted TSV file and returns a Pandas DataFrame.
    Each sentence is treated as a separate sample with a unique ID.
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
    df["sentence"] = df["tokens"].apply(lambda x: "".join(x))

    # Make this column the first one
    df = df[["sentence", "tokens", "tags", "tag_ids", "true_idioms"]]

    return df

def create_confusing_context_agent(num_variants: int = 3, language: str = "english"):
    """Create an Agno agent for generating confusing context variants

    Args:
        num_variants: Number of variants to generate per sentence
        language: Language to use ("english", "italian", "spanish", or "german")
    """

    # Select the appropriate system prompt based on language
    if language.lower() == "italian":
        system_prompt = SYSTEM_PROMPT_V3_ITALIAN.format(num_variants=num_variants)
    elif language.lower() == "spanish":
        system_prompt = SYSTEM_PROMPT_V3_SPANISH.format(num_variants=num_variants)
    elif language.lower() == "german":
        system_prompt = SYSTEM_PROMPT_V3_GERMAN.format(num_variants=num_variants)
    else:
        system_prompt = SYSTEM_PROMPT_V2.format(num_variants=num_variants)

    agent = Agent(
        model=model,
        output_schema=ConfusingVariants,
        instructions=system_prompt,
        markdown=False
    )

    return agent

def tokenize_and_tag_variant(variant_sentence: str, original_idioms: List[str]) -> Dict[str, Any]:
    """
    Tokenizes a variant sentence and generates proper BIO tagging based on original idioms.
    
    Args:
        variant_sentence: The generated variant sentence
        original_idioms: List of idioms from the original sentence
    
    Returns:
        Dictionary with tokens, tags, tag_ids, and preserved idioms
    """
    # Basic tokenization (split on whitespace and separate punctuation)
    # This mimics the tokenization pattern from the original TSV
    tokens = []
    
    # Split on whitespace first
    words = variant_sentence.split()
    
    for word in words:
        # Separate punctuation from words
        parts = re.findall(r'\w+|[^\w\s]', word, re.UNICODE)
        tokens.extend(parts)
    
    # Initialize all tags as "O"
    tags = ["O"] * len(tokens)
    tag_ids = [LABEL2ID["O"]] * len(tokens)
    
    # Find and tag idioms in the variant sentence
    preserved_idioms = []
    
    for idiom in original_idioms:
        # Clean idiom for matching (remove extra spaces)
        idiom_clean = " ".join(idiom.split())
        
        # Try to find the idiom in the variant sentence
        variant_clean = " ".join(tokens)
        
        # Look for exact matches first
        if idiom_clean.lower() in variant_clean.lower():
            # Find the position of the idiom
            idiom_tokens = idiom_clean.split()
            
            # Search for the idiom sequence in tokens
            for i in range(len(tokens) - len(idiom_tokens) + 1):
                # Check if the sequence matches (case-insensitive)
                token_sequence = [t.lower() for t in tokens[i:i+len(idiom_tokens)]]
                idiom_sequence = [t.lower() for t in idiom_tokens]
                
                if token_sequence == idiom_sequence:
                    # Tag this sequence as an idiom
                    for j, token_idx in enumerate(range(i, i + len(idiom_tokens))):
                        if j == 0:
                            tags[token_idx] = "B-IDIOM"
                            tag_ids[token_idx] = LABEL2ID["B-IDIOM"]
                        else:
                            tags[token_idx] = "I-IDIOM"
                            tag_ids[token_idx] = LABEL2ID["I-IDIOM"]
                    
                    preserved_idioms.append(idiom_clean)
                    break  # Only tag the first occurrence
    
    return {
        "tokens": tokens,
        "tags": tags,
        "tag_ids": tag_ids,
        "true_idioms": preserved_idioms
    }

def create_variant_rows(original_row: pd.Series, variants: List[str], num_variants: int) -> List[Dict[str, Any]]:
    """
    Creates multiple DataFrame rows from an original sentence and its variants.
    
    Args:
        original_row: Original row from the DataFrame
        variants: List of generated variant sentences
        num_variants: Number of variants to create
    
    Returns:
        List of dictionaries representing new DataFrame rows
    """
    variant_rows = []
    
    sentence = original_row["sentence"]
    original_idioms = original_row["true_idioms"]
    
    # Ensure we have the requested number of variants
    variants_to_use = variants[:num_variants] if len(variants) >= num_variants else variants
    
    # If we don't have enough variants, pad with the last variant or original sentence
    while len(variants_to_use) < num_variants:
        if variants_to_use:
            variants_to_use.append(variants_to_use[-1])  # Repeat last variant
        else:
            variants_to_use.append(sentence)  # Use original as fallback
    
    for variant_num, variant_sentence in enumerate(variants_to_use, 1):
        # Tokenize and tag the variant
        variant_data = tokenize_and_tag_variant(variant_sentence, original_idioms)
        
        # Create the row
        variant_row = {
            "sentence": sentence,
            "variant_number": variant_num,
            "variant_sentence": variant_sentence,
            "tokens": variant_data["tokens"],
            "tags": variant_data["tags"],
            "tag_ids": variant_data["tag_ids"],
            "true_idioms": variant_data["true_idioms"]
        }
        
        variant_rows.append(variant_row)
    
    return variant_rows

def generate_variants_dataframe(tsv_file_path: str, num_variants: int = 3, max_sentences: Optional[int] = None, language: str = "english", offset: int = 0) -> pd.DataFrame:
    """
    Main function that generates confusing context variants from a TSV file.

    Args:
        tsv_file_path: Path to the BIO-formatted TSV file
        num_variants: Number of variants to generate per sentence (default: 3)
        max_sentences: Maximum number of sentences to process (None for all)
        language: Language to use ("english", "italian", "spanish", or "german")
        offset: Number of sentences to skip from the beginning (default: 0)

    Returns:
        DataFrame with variants and proper BIO tagging
    """
    print(f"Loading data from {tsv_file_path}...")

    # Load the original data
    original_df = read_bio_tsv(tsv_file_path)

    # Apply offset first
    total_sentences = len(original_df)
    if offset > 0:
        if offset >= total_sentences:
            raise ValueError(f"Offset {offset} is >= total sentences {total_sentences}")
        original_df = original_df.iloc[offset:].reset_index(drop=True)
        print(f"Skipping first {offset} sentences...")

    # Limit sentences if requested
    if max_sentences:
        original_df = original_df.head(max_sentences)
        print(f"Processing sentences {offset + 1} to {offset + len(original_df)} (total: {len(original_df)} sentences)...")
    else:
        print(f"Processing sentences {offset + 1} to {total_sentences} (total: {len(original_df)} sentences)...")

    # Create the variant generation agent
    print(f"Initializing variant generation agent for {language}...")
    variant_agent = create_confusing_context_agent(num_variants, language)

    # Process each sentence
    all_variant_rows = []

    for idx, row in original_df.iterrows():
        sentence = row["sentence"]
        true_idioms = row["true_idioms"]

        # Calculate actual sentence number (offset + 1-based index)
        actual_sentence_num = offset + idx + 1
        print(f"Processing sentence {actual_sentence_num} ({idx + 1}/{len(original_df)}): {sentence[:50]}...")

        # Create the prompt for variant generation
        bio_tag = ", ".join(true_idioms) if true_idioms else None
        prompt = f"""
        Original sentence: "{sentence}"
        BIO tag: {bio_tag if bio_tag else "None (no idiom)"}

        Generate exactly {num_variants} confusing context variants of this sentence. If there are idioms, preserve those exact idiom phrases while creating ambiguous contexts around them.
        """

        try:
            # Generate variants
            response = variant_agent.run(prompt)
            variants = response.content.variants

            print(f"  ✓ Generated {len(variants)} variants")

            # Create variant rows
            variant_rows = create_variant_rows(row, variants, num_variants)
            all_variant_rows.extend(variant_rows)

        except Exception as e:
            print(f"  ✗ Error generating variants for sentence {actual_sentence_num}: {e}")

            # Create fallback rows with original sentence
            fallback_variants = [sentence] * num_variants
            variant_rows = create_variant_rows(row, fallback_variants, num_variants)
            all_variant_rows.extend(variant_rows)

    # Create the final DataFrame
    print("\nCreating final DataFrame...")
    result_df = pd.DataFrame(all_variant_rows)

    # Reorder columns for better readability
    column_order = [
        "sentence",
        "variant_number",
        "variant_sentence",
        "tokens",
        "tags",
        "tag_ids",
        "true_idioms"
    ]
    result_df = result_df[column_order]

    print(f"Generated {len(result_df)} variant rows from {len(original_df)} original sentences")

    return result_df

def save_variants_to_file(df: pd.DataFrame, output_path: str, format_type: str = "json") -> None:
    """
    Save the variants DataFrame to a file.
    
    Args:
        df: DataFrame with variants
        output_path: Path to save the file
        format_type: File format ('csv', 'json', 'pickle')
    """
    if format_type.lower() == "csv":
        df.to_csv(output_path, index=False)
    elif format_type.lower() == "json":
        df.to_json(output_path, orient="records", indent=2)
    elif format_type.lower() == "pickle":
        df.to_pickle(output_path)
    else:
        raise ValueError(f"Unsupported format: {format_type}")
    
    print(f"Variants saved to {output_path}")

def display_sample_variants(df: pd.DataFrame, num_samples: int = 3) -> None:
    """
    Display sample variants for inspection.
    
    Args:
        df: DataFrame with variants
        num_samples: Number of sample sentences to display
    """
    print("\nSample Generated Variants:")
    print("=" * 60)
    
    # Get unique original sentences for sampling
    unique_originals = df["sentence"].unique()[:num_samples]
    
    for i, original in enumerate(unique_originals, 1):
        print(f"\nOriginal Sentence {i}:")
        print(f"Text: {original}")
        
        # Get all variants for this original sentence
        sentence_variants = df[df["sentence"] == original]
        
        # Show the first row to get original idioms
        first_row = sentence_variants.iloc[0]
        print(f"Original Idioms: {first_row['true_idioms']}")
        
        for _, variant_row in sentence_variants.iterrows():
            print(f"\nVariant {variant_row['variant_number']}:")
            print(f"  Text: {variant_row['variant_sentence']}")
            print(f"  Preserved Idioms: {variant_row['true_idioms']}")
            print(f"  Tokens: {variant_row['tokens'][:10]}{'...' if len(variant_row['tokens']) > 10 else ''}")
            print(f"  BIO Tags: {variant_row['tags'][:10]}{'...' if len(variant_row['tags']) > 10 else ''}")
        
        print("-" * 60)

def main():
    """Main execution function with configuration options"""
    
    # Configuration
    TSV_FILE_PATH = "./test_english.tsv"
    OUTPUT_PATH = f"./variants_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    NUM_VARIANTS = 3
    MAX_SENTENCES = 5  # Set to None to process all sentences, or a number for testing
    OUTPUT_FORMAT = "json"  # Options: 'csv', 'json', 'pickle'
    
    print("🔄 Confusing Context Variant Generator")
    print("=" * 50)
    print(f"Input file: {TSV_FILE_PATH}")
    print(f"Number of variants per sentence: {NUM_VARIANTS}")
    print(f"Max sentences to process: {MAX_SENTENCES if MAX_SENTENCES else 'All'}")
    print(f"Output format: {OUTPUT_FORMAT}")
    print("=" * 50)
    
    # Check if input file exists
    if not os.path.exists(TSV_FILE_PATH):
        print(f"❌ Error: Input file not found at {TSV_FILE_PATH}")
        print("Please ensure the test_english.tsv file is in the current directory.")
        return
    
    # Check environment variables
    if not os.getenv("GITHUB_TOKEN_MODEL"):
        print("❌ Error: GITHUB_TOKEN_MODEL environment variable not set")
        print("Please set your OpenAI API key in the environment variables.")
        return
    
    try:
        # Generate variants
        print("\n🚀 Starting variant generation...")
        variants_df = generate_variants_dataframe(
            tsv_file_path=TSV_FILE_PATH,
            num_variants=NUM_VARIANTS,
            max_sentences=MAX_SENTENCES
        )
        
        # Display statistics
        print(f"\n📊 Generation Statistics:")
        print(f"  Total variant rows generated: {len(variants_df)}")
        print(f"  Original sentences processed: {len(variants_df) // NUM_VARIANTS}")
        print(f"  Variants per sentence: {NUM_VARIANTS}")
        
        # Count sentences with idioms
        idiom_counts = variants_df['true_idioms'].apply(len)
        variants_with_idioms = len(idiom_counts[idiom_counts > 0])
        variants_without_idioms = len(idiom_counts[idiom_counts == 0])
        
        print(f"  Variants with preserved idioms: {variants_with_idioms}")
        print(f"  Variants without idioms: {variants_without_idioms}")
        
        # Save results
        print(f"\n💾 Saving results...")
        output_filename = f"{OUTPUT_PATH}.{OUTPUT_FORMAT}"
        save_variants_to_file(variants_df, output_filename, OUTPUT_FORMAT)
        
        # Display sample results
        display_sample_variants(variants_df, num_samples=2)
        
        print(f"\n✅ Process completed successfully!")
        print(f"Results saved to: {output_filename}")
        print(f"You can now use this DataFrame for further analysis or testing.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
    except Exception as e:
        print(f"\n❌ An error occurred during processing:")
        print(f"Error: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        print("\nPlease check your configuration and try again.")

if __name__ == "__main__":
    main()