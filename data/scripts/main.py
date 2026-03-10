#!/usr/bin/env python3
"""
Main script for Confusing Context Variant Generator

This script runs the variants generator with user-configurable options.
"""

import os
import sys
import random
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────
_HERE           = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH   = os.path.join(_HERE, "..", "raw_id10m_data", "english", "id10m_english.csv")
GENERATIONS_DIR = os.path.join(_HERE, "generations")
VALID_FORMATS   = ("json", "csv", "pkl")   # pkl = pickle


def check_environment() -> bool:
    """Check if required environment variables are set"""
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Missing required environment variable: GEMINI_API_KEY")
        print("Please set it in your .env file.")
        return False
    return True


def check_input_file(path: str) -> bool:
    """Check if the input CSV file exists"""
    if not os.path.exists(path):
        print(f"❌ Input file not found: {path}")
        return False
    print(f"✅ Input file found: {path}")
    return True


def get_user_input():
    """Get user configuration for the variant generation"""
    print("\n📋 Configuration Options:")
    print("=" * 40)

    # Language selection
    while True:
        language = input("Select language (english/german, default: english): ").strip().lower()
        if not language:
            language = "english"
        if language in ["english", "german"]:
            break
        print("Please choose from: english, german")

    # Figurative filter
    while True:
        fig_input = input("Filter by usage? (figurative/literal/all, default: all): ").strip().lower()
        if not fig_input or fig_input == "all":
            figurative_filter = None
            break
        elif fig_input == "figurative":
            figurative_filter = True
            break
        elif fig_input == "literal":
            figurative_filter = False
            break
        print("Please choose from: figurative, literal, all")

    # Number of variants per sentence
    while True:
        try:
            num_variants = input("Number of variants per sentence (default: 3): ").strip()
            num_variants = int(num_variants) if num_variants else 3
            if num_variants >= 1:
                break
            print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")

    # Offset
    while True:
        try:
            offset_input = input("Offset - sentences to skip from start (default: 0): ").strip()
            offset = int(offset_input) if offset_input else 0
            if offset >= 0:
                break
            print("Please enter a non-negative number.")
        except ValueError:
            print("Please enter a valid number.")

    # Max sentences
    while True:
        try:
            max_input = input("Number of sentences to process (default: all): ").strip()
            if not max_input or max_input.lower() == "all":
                max_sentences = None
                break
            max_sentences = int(max_input)
            if max_sentences >= 1:
                break
            print("Please enter a positive number or 'all'.")
        except ValueError:
            print("Please enter a valid number or 'all'.")

    # Random selection
    use_random = False
    if max_sentences:
        while True:
            random_choice = input("Take random sentences? (y/n, default: n): ").strip().lower()
            if random_choice in ('', 'n', 'no'):
                break
            elif random_choice in ('y', 'yes'):
                use_random = True
                break
            print("Please enter 'y' or 'n'.")

    # Output format
    while True:
        fmt_input = input(f"Output format ({'/'.join(VALID_FORMATS)}, default: json): ").strip().lower()
        if not fmt_input:
            output_format = "json"
            break
        if fmt_input in VALID_FORMATS:
            output_format = fmt_input
            break
        print(f"Please choose from: {', '.join(VALID_FORMATS)}")

    return language, figurative_filter, num_variants, offset, max_sentences, use_random, output_format


def run_generator(language: str, figurative_filter: Optional[bool], num_variants: int,
                  offset: int, max_sentences: Optional[int], use_random: bool, output_format: str) -> bool:
    """Run the variant generator with the specified configuration"""
    try:
        from variants_generator import generate_variants_dataframe, save_variants_to_file, display_sample_variants, model_name
        import pandas as pd
        from datetime import datetime
    except ImportError as e:
        print(f"❌ Error importing variants_generator: {e}")
        return False

    data_path                = CSV_FILE_PATH
    temp_csv                 = None
    display_figurative_filter = figurative_filter  # preserve for logging before it may be cleared

    # Random sentence selection: pre-sample the df and write a temp CSV
    if use_random and max_sentences:
        try:
            df_all = pd.read_csv(data_path)
            if figurative_filter is not None:
                label  = "figurative" if figurative_filter else "literal"
                df_all = df_all[df_all["is_figurative"] == figurative_filter].reset_index(drop=True)
                print(f"Filtered to {label} sentences only: {len(df_all)} sentences remaining")
            available = df_all.iloc[offset:]
            if max_sentences >= len(available):
                print(f"Requested {max_sentences} sentences but only {len(available)} available — using all.")
                max_sentences = None
                use_random    = False
            else:
                chosen    = available.sample(n=max_sentences).sort_index()
                temp_csv  = os.path.join(_HERE, "_temp_random.csv")
                chosen.to_csv(temp_csv, index=False)
                data_path        = temp_csv
                offset           = 0
                figurative_filter = None  # already applied
                max_sentences    = None
                print(f"✅ Randomly selected {len(chosen)} sentences")
        except Exception as e:
            print(f"❌ Error during random selection: {e}")
            return False

    try:
        print(f"\n🚀 Starting variant generation...")
        print(f"  Language         : {language}")
        print(f"  Figurative filter: {'figurative' if display_figurative_filter is True else 'literal' if display_figurative_filter is False else 'all'}")
        print(f"  Variants/sentence: {num_variants}")
        print(f"  Offset           : {offset}")
        print(f"  Max sentences    : {max_sentences or 'all'}")
        print(f"  Selection        : {'random' if use_random else 'sequential'}")
        print(f"  Output format    : {output_format}")
        print("-" * 50)

        variants_df = generate_variants_dataframe(
            data_path=data_path,
            num_variants=num_variants,
            max_sentences=max_sentences,
            language=language,
            offset=offset,
            figurative_filter=figurative_filter,
        )

        # Build output filename
        os.makedirs(GENERATIONS_DIR, exist_ok=True)
        timestamp       = datetime.now().strftime('%Y%m%d_%H%M%S')
        sentences_count = len(variants_df) // num_variants
        fig_label       = "fig" if figurative_filter is True else "lit" if figurative_filter is False else "all"
        safe_model      = model_name.replace("-", "_").replace(".", "_")
        output_filename = os.path.join(
            GENERATIONS_DIR,
            f"{language}_{safe_model}_{fig_label}_{sentences_count}s_{num_variants}v_{timestamp}.{output_format}"
        )

        print(f"\n💾 Saving results to {output_filename}...")
        save_variants_to_file(variants_df, output_filename, output_format)
        display_sample_variants(variants_df, num_samples=2)
        print(f"\n✅ Done! Results saved to: {output_filename}")
        return True

    except Exception as e:
        print(f"\n❌ Error during variant generation: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if temp_csv and os.path.exists(temp_csv):
            os.remove(temp_csv)


def main():
    print("🎯 Confusing Context Variant Generator")
    print("=" * 60)

    if not check_environment():
        sys.exit(1)

    if not check_input_file(CSV_FILE_PATH):
        sys.exit(1)

    language, figurative_filter, num_variants, offset, max_sentences, use_random, output_format = get_user_input()

    success = run_generator(language, figurative_filter, num_variants, offset, max_sentences, use_random, output_format)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
