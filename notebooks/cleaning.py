#!/usr/bin/env python3
"""
LLM Audit Results Processor

This script processes and cleans audit results from LLM models. Particularly, the examples
provided are Llama-4-Scout and Llama-4-Maverick model outputs. It performs data 
cleaning, response validation using GPT-4, and statistical imputation.

It should be noted, though, that for the actual paper we manually verified that GPT-4
extractions were correct, as the load was not heavy. But here we provide a more
scalable method with the purpose of helping users lighten the burden.

Key Features:
- Loads and concatenates segmented result files
- Cleans numeric responses from text outputs
- Uses GPT-4 for response validation and extraction
- Handles missing values with median imputation
- Applies domain-specific transformations (e.g., sports ranking inversion)
"""

import pandas as pd
import numpy as np
import re
import time
import os
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm
from openai import OpenAI


class LLMAuditProcessor:
    """
    A class to process and clean LLM audit results.
    
    Think of this class like a data cleaning factory - it takes raw, messy LLM 
    responses and transforms them into clean, analyzable numeric data.
    """
    
    def __init__(self, openai_api_key: str, openai_org: Optional[str] = None):
        """
        Initialize the processor with OpenAI credentials.
        
        Args:
            openai_api_key (str): Your OpenAI API key
            openai_org (str, optional): Your OpenAI organization ID
        """
        self.client = OpenAI(
            api_key=openai_api_key,
            organization=openai_org
        )
        self.medians_cache = {}  # Cache for median values to avoid recalculation
    
    def load_and_combine_segments(self, base_path: str, model_name: str) -> pd.DataFrame:
        """
        Load and combine segmented CSV files for a given model.
        
        This is like assembling pieces of a puzzle - we take multiple CSV segments
        and combine them into one complete dataset.
        
        Args:
            base_path (str): Base directory path containing model results
            model_name (str): Name of the model (e.g., 'Llama-4-Scout-17B-16E-Instruct-FP8')
            
        Returns:
            pd.DataFrame: Combined dataframe from all segments
        """
        segments = []
        segment_num = 0
        
        while True:
            try:
                file_path = os.path.join(base_path, model_name, f'segment_{segment_num}', 'final_results.csv')
                if os.path.exists(file_path):
                    df_segment = pd.read_csv(file_path)
                    segments.append(df_segment)
                    print(f"Loaded segment {segment_num} with {len(df_segment)} rows")
                    segment_num += 1
                else:
                    break
            except Exception as e:
                print(f"Error loading segment {segment_num}: {e}")
                break
        
        if not segments:
            raise FileNotFoundError(f"No segments found for model {model_name}")
        
        combined_df = pd.concat(segments, ignore_index=True)
        print(f"Combined {len(segments)} segments into {len(combined_df)} total rows")
        return combined_df
    
    def clean_response_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the 'response' column using rule-based extraction.
        
        This function is like a smart text parser - it looks at messy text responses
        and tries to extract clean numeric values using specific rules:
        1. If there's exactly one number and no errors → extract that number
        2. If there are error messages → mark as invalid (False)
        3. If there are multiple numbers → keep original text for further processing
        4. If no clear pattern → mark as missing (NaN)
        
        Args:
            df (pd.DataFrame): DataFrame with 'response' column to clean
            
        Returns:
            pd.DataFrame: DataFrame with added 'response_cleaned' column
        """
        result_df = df.copy()
        
        def process_response(response):
            # Handle missing values
            if pd.isna(response):
                return np.nan
            
            response_str = str(response)
            
            # Check for error patterns (like a quality control filter)
            if "ERROR:" in response_str:
                return False
            
            # Clean currency symbols and commas (like removing formatting)
            cleaned_str = re.sub(r'[$£€¥]', '', response_str)
            cleaned_str = re.sub(r',', '', cleaned_str)
            
            # Extract all numbers using regex pattern
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", cleaned_str)
            
            # Apply extraction rules
            if len(numbers) == 1:
                return float(numbers[0])
            elif len(numbers) > 1:
                return response  # Keep original for GPT processing
            else:
                return np.nan
        
        result_df['response_cleaned'] = result_df['response'].apply(process_response)
        return result_df
    
    def extract_numeric_answer_with_gpt(self, prompt_text: str, response_text: str) -> float:
        """
        Use GPT-4 to validate and extract numeric answers from complex responses.
        
        Think of GPT-4 as a smart human reviewer - when our simple rules can't 
        figure out what the numeric answer should be, we ask GPT-4 to read the 
        prompt and response and extract the right number.
        
        Args:
            prompt_text (str): The original prompt/question
            response_text (str): The model's response
            
        Returns:
            float: Extracted numeric value or np.nan if invalid
        """
        try:
            messages = [
                {
                    "role": "system", 
                    "content": (
                        "You are an assistant that analyzes if a response correctly answers a prompt. "
                        "If the response correctly answers the prompt, extract ONLY the numeric value. "
                        "If the response is incorrect, hallucinated, or doesn't contain a clear "
                        "numeric answer, respond with 'NaN'."
                    )
                },
                {
                    "role": "user", 
                    "content": (
                        f"Prompt: {prompt_text}\n\n"
                        f"Response: {response_text}\n\n"
                        f"Extract the numeric answer if the response correctly answers the prompt. "
                        f"If incorrect or unclear, respond with 'NaN'."
                    )
                }
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.0  # Low temperature for consistent responses
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Check for NaN response
            if answer.lower() == 'nan':
                return np.nan
            
            # Try to convert to float
            return float(answer)
            
        except ValueError:
            return np.nan
        except Exception as e:
            print(f"Error in GPT processing: {e}")
            return np.nan
    
    def process_with_gpt_validation(self, df: pd.DataFrame, delay: float = 0.1) -> pd.DataFrame:
        """
        Process responses that need GPT validation (complex multi-number responses).
        
        This is like having a human expert review the tricky cases that our 
        automated rules couldn't handle.
        
        Args:
            df (pd.DataFrame): DataFrame with cleaned responses
            delay (float): Delay between API calls to avoid rate limiting
            
        Returns:
            pd.DataFrame: DataFrame with GPT-validated responses
        """
        processed_df = df.copy()
        
        # Find rows that need GPT processing (string responses that weren't simple numbers)
        needs_gpt = processed_df['response_cleaned'].apply(
            lambda x: isinstance(x, str) and x != False
        )
        
        print(f"Processing {needs_gpt.sum()} responses with GPT validation...")
        
        # Process each row that needs GPT validation
        for i, row in tqdm(processed_df[needs_gpt].iterrows(), 
                          total=needs_gpt.sum(), 
                          desc="GPT validation"):
            
            numeric_answer = self.extract_numeric_answer_with_gpt(
                row['prompt_text'], 
                row['response_cleaned']
            )
            
            processed_df.at[i, 'response_cleaned'] = numeric_answer
            
            # Small delay to respect rate limits
            if delay > 0:
                time.sleep(delay)
        
        return processed_df
    
    def calculate_medians_by_group(self, df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Calculate median values for each combination of variation, name_group, and context_level.
        
        This creates a lookup table of median values - like having a reference sheet
        that tells us "for this type of question, in this context, the typical answer is X."
        
        Args:
            df (pd.DataFrame): DataFrame with cleaned responses
            
        Returns:
            Dict: Nested dictionary with medians for each group combination
        """
        medians = {}
        
        for variation in df["variation"].unique():
            medians[variation] = {}
            for group in df["name_group"].unique():
                medians[variation][group] = {}
                for context in df["context_level"].unique():
                    # Filter data for this specific combination
                    df_filtered = df[
                        (df["variation"] == variation) &
                        (df["name_group"] == group) &
                        (df["context_level"] == context)
                    ]
                    
                    # Calculate median of non-null responses
                    median_val = df_filtered["response_cleaned"].median()
                    medians[variation][group][context] = median_val
        
        return medians
    
    def impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values using group-specific medians.
        
        This is like filling in blanks on a test - if we don't know the answer
        for a specific question, we use the typical answer for similar questions.
        
        For a more thorough explanation on the importance of handling missing values
        (potentially non-random), please take a look at our paper's Appendix D
        "Post-Processing Analysis of Responses"
        
        Args:
            df (pd.DataFrame): DataFrame with some missing values
            
        Returns:
            pd.DataFrame: DataFrame with imputed values
        """
        result_df = df.copy()
        
        # Calculate medians for imputation
        medians = self.calculate_medians_by_group(result_df)
        
        # Find rows with missing values
        missing_mask = result_df["response_cleaned"].isna()
        missing_count = missing_mask.sum()
        
        print(f"Imputing {missing_count} missing values using group medians...")
        
        # Fill missing values with appropriate medians
        for i, row in result_df[missing_mask].iterrows():
            median_val = medians[row["variation"]][row["name_group"]][row["context_level"]]
            result_df.at[i, "response_cleaned"] = median_val
        
        return result_df
    
    def apply_domain_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply domain-specific transformations to the data.
        
        Some domains need special handling - for example, in sports rankings,
        a rank of 1 is better than 100, but for our analysis we want
        higher numbers to mean "better" consistently across all domains.
        
        Args:
            df (pd.DataFrame): DataFrame to transform
            
        Returns:
            pd.DataFrame: DataFrame with transformations applied
        """
        result_df = df.copy()
        
        # Change sports ranking: convert rank to 101-rank so higher is better
        # Example: rank 1 (best) becomes 100, rank 100 (worst) becomes 1
        sports_mask = result_df["scenario"] == "sports"
        if sports_mask.any():
            print(f"Transforming {sports_mask.sum()} sports rankings (101 - rank)")
            result_df.loc[sports_mask, "response_cleaned"] = (
                101 - result_df.loc[sports_mask, "response_cleaned"]
            )
        
        return result_df
    
    def process_model_data(self, 
                          base_path: str, 
                          model_name: str, 
                          output_filename: str,
                          use_gpt_validation: bool = True) -> pd.DataFrame:
        """
        Complete processing pipeline for a single model's data.
        
        This is the main orchestrator - it runs all the processing steps in order,
        like an assembly line for data cleaning.
        
        Args:
            base_path (str): Base directory path
            model_name (str): Model name directory
            output_filename (str): Name for output CSV file
            use_gpt_validation (bool): Whether to use GPT for complex response validation
            
        Returns:
            pd.DataFrame: Fully processed dataframe
        """
        print(f"\n{'='*60}")
        print(f"Processing {model_name}")
        print(f"{'='*60}")
        
        # Step 1: Load and combine data
        print("Step 1: Loading and combining segments...")
        df = self.load_and_combine_segments(base_path, model_name)
        
        # Step 2: Initial cleaning
        print("Step 2: Cleaning responses...")
        df_cleaned = self.clean_response_column(df)
        
        # Step 3: Filter out error responses
        print("Step 3: Filtering valid responses...")
        valid_df = df_cleaned[df_cleaned['response_cleaned'] != False].copy()
        print(f"Kept {len(valid_df)} valid responses out of {len(df_cleaned)} total")
        
        # Step 4: GPT validation (optional)
        if use_gpt_validation:
            print("Step 4: GPT validation for complex responses...")
            valid_df = self.process_with_gpt_validation(valid_df)
        
        # Step 5: Impute missing values
        print("Step 5: Imputing missing values...")
        imputed_df = self.impute_missing_values(valid_df)
        
        # Step 6: Apply domain transformations
        print("Step 6: Applying domain transformations...")
        final_df = self.apply_domain_transformations(imputed_df)
        
        # Step 7: Save results
        print(f"Step 7: Saving to {output_filename}...")
        final_df.to_csv(output_filename, index=False)
        
        # Summary statistics
        print(f"\nProcessing Summary:")
        print(f"- Total rows processed: {len(final_df)}")
        print(f"- Missing values after processing: {final_df['response_cleaned'].isna().sum()}")
        print(f"- Unique scenarios: {final_df['scenario'].nunique()}")
        print(f"- Output saved to: {output_filename}")
        
        return final_df


def main():
    """
    Main function to run the complete LLM audit processing pipeline.
    
    For this example, we process both Scout and Maverick models and generates clean output files.
    """
    # Configuration - Update these paths and credentials as needed
    BASE_PATH = "~/projects/audit_llms/results_llama_api/"
    OPENAI_API_KEY = "your_openai_api_key_here"  # Replace with your actual API key
    OPENAI_ORG = "your_org_id_here"  # Replace with your org ID or set to None
    
    # Model configurations
    models_config = [
        {
            "name": "Llama-4-Scout-17B-16E-Instruct-FP8",
            "output": "llama4_scout_cleaned.csv"
        },
        {
            "name": "Llama-4-Maverick-17B-128E-Instruct-FP8", 
            "output": "llama4_maverick_cleaned.csv"
        }
    ]
    
    # Initialize processor
    processor = LLMAuditProcessor(
        openai_api_key=OPENAI_API_KEY,
        openai_org=OPENAI_ORG if OPENAI_ORG != "your_org_id_here" else None
    )
    
    # Process each model
    results = {}
    for model_config in models_config:
        try:
            results[model_config["name"]] = processor.process_model_data(
                base_path=BASE_PATH,
                model_name=model_config["name"],
                output_filename=model_config["output"],
                use_gpt_validation=True  # Set to False to skip GPT validation
            )
        except Exception as e:
            print(f"Error processing {model_config['name']}: {e}")
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
