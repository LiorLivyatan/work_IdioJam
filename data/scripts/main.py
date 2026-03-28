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
CSV_FILE_PATHS  = {
    "english": os.path.join(_HERE, "..", "raw_id10m_data", "english", "id10m_english.csv"),
    "german":  os.path.join(_HERE, "..", "raw_id10m_data", "german",  "id10m_german.csv"),
}
GENERATIONS_DIR = os.path.join(_HERE, "generations")
VALID_FORMATS   = ("json", "csv", "pkl")   # pkl = pickle


def check_environment() -> bool:
    """Check if required environment variables are set"""
    if not os.getenv("GEMINI_API_KEY"):
        print("Missing required environment variable: GEMINI_API_KEY")
        print("Please set it in your .env file.")
        return False
    return True


def check_input_file(path: str) -> bool:
    """Check if the input CSV file exists"""
    if not os.path.exists(path):
        print(f"Input file not found: {path}")
        return False
    print(f"Input file found: {path}")
    return True


def get_user_input():
    """Get user configuration for the variant generation"""
    print("\nConfiguration Options:")
    print("=" * 40)

    # Language selection
    while True:
        language = input("Select language (english/german, default: english): ").strip().lower()
        if not language:
            language = "english"
        if language in ["english", "german"]:
            break
        print("Please choose from: english, german")

    # Figurative filter — German gets the extra literal_or_fixed option
    if language == "german":
        filter_options = "figurative/literal/literal_or_fixed/all"
    else:
        filter_options = "figurative/literal/all"

    while True:
        fig_input = input(f"Filter by usage? ({filter_options}, default: all): ").strip().lower()
        if not fig_input or fig_input == "all":
            figurative_filter = None
            break
        elif fig_input == "figurative":
            figurative_filter = True
            break
        elif fig_input == "literal":
            figurative_filter = False
            break
        elif fig_input == "literal_or_fixed" and language == "german":
            figurative_filter = "literal_or_fixed"
            break
        print(f"Please choose from: {filter_options.replace('/', ', ')}")

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

    # Selection mode
    while True:
        mode = input("Selection mode (sequential/random/indices, default: sequential): ").strip().lower()
        if not mode or mode == "sequential":
            selection_mode = "sequential"
            break
        elif mode == "random":
            selection_mode = "random"
            break
        elif mode == "indices":
            selection_mode = "indices"
            break
        print("Please choose from: sequential, random, indices")

    # Mode-specific options
    specific_indices = None
    offset       = 0
    max_sentences = None
    use_random   = False

    if selection_mode == "indices":
        while True:
            raw = input("Row indices to process (comma-separated, e.g. 3,31,32): ").strip()
            try:
                specific_indices = [int(i.strip()) for i in raw.split(",") if i.strip()]
                if specific_indices:
                    break
                print("Please enter at least one index.")
            except ValueError:
                print("Please enter comma-separated integers.")

    elif selection_mode == "random":
        while True:
            try:
                max_input = input("Number of sentences to randomly sample: ").strip()
                max_sentences = int(max_input)
                if max_sentences >= 1:
                    use_random = True
                    break
                print("Please enter a positive number.")
            except ValueError:
                print("Please enter a valid number.")

    else:  # sequential
        while True:
            try:
                offset_input = input("Offset - sentences to skip from start (default: 0): ").strip()
                offset = int(offset_input) if offset_input else 0
                if offset >= 0:
                    break
                print("Please enter a non-negative number.")
            except ValueError:
                print("Please enter a valid number.")

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

    return language, figurative_filter, num_variants, offset, max_sentences, use_random, specific_indices, output_format


def run_generator(language: str, figurative_filter, num_variants: int,
                  offset: int, max_sentences: Optional[int], use_random: bool,
                  specific_indices: Optional[list], output_format: str) -> bool:
    """Run the variant generator with the specified configuration"""
    try:
        from variants_generator import generate_variants_dataframe, save_variants_to_file, display_sample_variants, model_name
        import pandas as pd
        from datetime import datetime
    except ImportError as e:
        print(f"Error importing variants_generator: {e}")
        return False

    data_path                 = CSV_FILE_PATHS[language]
    pre_filter_csv            = None
    temp_csv                  = None
    display_figurative_filter = figurative_filter  # preserve for logging before it may be cleared

    # literal_or_fixed pre-filter: keep rows where is_figurative==False OR was_fixed==True.
    # Must run first so that subsequent random/sequential selection operates on the filtered set.
    if figurative_filter == "literal_or_fixed":
        try:
            df_all = pd.read_csv(data_path)
            mask   = (~df_all["is_figurative"]) | (df_all["was_fixed"].fillna(False))
            df_all = df_all[mask].reset_index(drop=True)
            print(f"Filtered to literal or fixed sentences: {len(df_all)} sentences remaining")
            pre_filter_csv    = os.path.join(_HERE, "_temp_lit_or_fixed.csv")
            df_all.to_csv(pre_filter_csv, index=False)
            data_path         = pre_filter_csv
            figurative_filter = None  # already applied
        except Exception as e:
            print(f"Error during literal_or_fixed filtering: {e}")
            return False

    # Specific-indices selection: extract rows by index and write a temp CSV
    if specific_indices is not None:
        try:
            df_all = pd.read_csv(data_path)
            chosen = df_all.iloc[specific_indices]
            temp_csv  = os.path.join(_HERE, "_temp_indices.csv")
            chosen.to_csv(temp_csv, index=False)
            data_path         = temp_csv
            offset            = 0
            max_sentences     = None
            figurative_filter = None  # no further filtering needed
            print(f"Selected {len(chosen)} sentences at indices {specific_indices}")
        except Exception as e:
            print(f"Error selecting by indices: {e}")
            return False

    # Random sentence selection: pre-sample the df and write a temp CSV
    elif use_random and max_sentences:
        try:
            df_all = pd.read_csv(data_path)
            if figurative_filter is not None:
                label  = "figurative" if figurative_filter else "literal"
                df_all = df_all[df_all["is_figurative"] == figurative_filter].reset_index(drop=True)
                print(f"Filtered to {label} sentences only: {len(df_all)} sentences remaining")
            available = df_all.iloc[offset:]
            if max_sentences >= len(available):
                print(f"Requested {max_sentences} sentences but only {len(available)} available - using all.")
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
                print(f"Randomly selected {len(chosen)} sentences")
        except Exception as e:
            print(f"Error during random selection: {e}")
            return False

    # ── Confirmation summary ──────────────────────────────────────────────────
    try:
        _df = pd.read_csv(data_path)
        if figurative_filter is not None:
            _df = _df[_df["is_figurative"] == figurative_filter].reset_index(drop=True)
        if offset:
            _df = _df.iloc[offset:].reset_index(drop=True)
        if max_sentences:
            _df = _df.head(max_sentences)

        n_sentences  = len(_df)
        n_figurative = int(_df["is_figurative"].sum())
        n_literal    = n_sentences - n_figurative
        n_variants   = n_sentences * num_variants

        print("\n" + "=" * 50)
        print("  Run summary")
        print("=" * 50)
        print(f"  Sentences to process : {n_sentences}")
        print(f"    Figurative         : {n_figurative}")
        print(f"    Literal            : {n_literal}")
        print(f"  Variants per sentence: {num_variants}")
        print(f"  Total variants       : {n_variants}  (= {n_sentences} x {num_variants})")
        print("=" * 50)

        confirm = input("\nProceed? (y/n, default: y): ").strip().lower()
        if confirm not in ("", "y", "yes"):
            print("Aborted.")
            return False
    except Exception as e:
        print(f"Warning: could not compute summary ({e}). Proceeding anyway.")

    try:
        if display_figurative_filter == "literal_or_fixed":
            filter_label = "literal_or_fixed"
        elif display_figurative_filter is True:
            filter_label = "figurative"
        elif display_figurative_filter is False:
            filter_label = "literal"
        else:
            filter_label = "all"

        print(f"\nStarting variant generation...")
        print(f"  Language         : {language}")
        print(f"  Figurative filter: {filter_label}")
        print(f"  Variants/sentence: {num_variants}")
        print(f"  Offset           : {offset}")
        print(f"  Max sentences    : {max_sentences or 'all'}")
        print(f"  Selection        : {'indices ' + str(specific_indices) if specific_indices is not None else 'random' if use_random else 'sequential'}")
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
        safe_model      = model_name.replace("-", "_").replace(".", "_")
        output_filename = os.path.join(
            GENERATIONS_DIR,
            f"{language}_{safe_model}_{filter_label}_{sentences_count}s_{num_variants}v_{timestamp}.{output_format}"
        )

        print(f"\nSaving results to {output_filename}...")
        save_variants_to_file(variants_df, output_filename, output_format)
        display_sample_variants(variants_df, num_samples=2)
        print(f"\nDone! Results saved to: {output_filename}")
        return True

    except Exception as e:
        print(f"\nError during variant generation: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        for f in [temp_csv, pre_filter_csv]:
            if f and os.path.exists(f):
                os.remove(f)


def main():
    print("Confusing Context Variant Generator")
    print("=" * 60)

    if not check_environment():
        sys.exit(1)

    language, figurative_filter, num_variants, offset, max_sentences, use_random, specific_indices, output_format = get_user_input()

    if not check_input_file(CSV_FILE_PATHS[language]):
        sys.exit(1)

    success = run_generator(language, figurative_filter, num_variants, offset, max_sentences, use_random, specific_indices, output_format)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
