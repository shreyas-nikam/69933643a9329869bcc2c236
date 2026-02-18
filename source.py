import os
import re
import json
import pandas as pd
import tiktoken
import nltk
from openai import OpenAI
# Although imported in original, not used, but kept for fidelity.
from difflib import SequenceMatcher

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# --- Constants for LLM Prompts ---
PROMPT_NAIVE = """Summarize this earnings call transcript:

{transcript}"""

SYSTEM_PROMPT = """You are a senior equity research analyst at a top-tier investment bank. You produce concise, accurate earnings call summaries for portfolio managers.

RULES:
1.  Use ONLY information explicitly stated in the transcript.
2.  Do NOT infer, speculate, or add information from your training data.
3.  Quote specific numbers exactly as stated (revenue, EPS, margins).
4.  If guidance is mentioned, distinguish between RAISED, LOWERED, MAINTAINED, and INTRODUCED guidance.
5.  Flag any qualitative hedging language (e.g., "cautiously optimistic", "headwinds") that may signal uncertainty.
6.  If a question was deflected or answered vaguely, note this."""

TASK_PROMPT = """Summarize the following earnings call transcript into a structured analyst brief with these sections:

1.  HEADLINE: One sentence capturing the most material takeaway.
2.  QUARTERLY PERFORMANCE: Revenue, EPS, margins, segment performance. Include YoY and QoQ comparisons if mentioned.
3.  GUIDANCE & OUTLOOK: Forward guidance (raised/lowered/maintained), management commentary on future quarters, macro outlook.
4.  KEY Q&A EXCHANGES: The 3-5 most important analyst questions and management responses. Focus on questions about guidance, margins, capital allocation, and competitive dynamics.
5.  RISKS & CONCERNS: Any risks, headwinds, or cautionary language.
6.  TONE ASSESSMENT: Overall management tone (confident/cautious/defensive) with supporting evidence from the transcript.

TRANSCRIPT:
{transcript}"""

FEW_SHOT_EXAMPLE = """Here is an example of a good earnings summary:

HEADLINE: ABC Corp beats Q3 estimates on cloud strength; raises FY guidance by 2%.

QUARTERLY PERFORMANCE:
Revenue: $10.5B (+8% YoY), beating consensus of $10.2B
EPS: $1.05 vs $0.95 expected
Cloud segment: +15% YoY, driving majority of upside
Operating margin: 28.5%, up 120bps YoY
[...]
"""

EXTRACTION_PROMPT = """Extract the following financial metrics from the earnings call transcript. Return ONLY a JSON object.
If a metric is not mentioned, use null.
Quote numbers exactly as stated. Include units and comparisons.

Required fields:
{{
    "company": "string",
    "quarter": "string (e.g., Q4 FY2024)",
    "revenue": {{"value": "string", "yoy_change": "string or null", "vs_consensus": "string or null"}},
    "eps": {{"value": "string", "vs_consensus": "string or null"}},
    "operating_margin": "string or null",
    "guidance_revenue_fy": {{"value": "string", "change": "raised|lowered|maintained|null"}},
    "key_segment_performance": [
        {{"segment": "string", "revenue": "string", "growth": "string"}}
    ],
    "capital_return": {{"buyback": "string or null", "dividend": "string or null"}},
    "headcount_change": "string or null",
    "management_tone": "confident|cautious|defensive|mixed"
}}

TRANSCRIPT:
{transcript}"""

# --- Utility Functions ---


def get_estimated_api_cost(prompt_tokens: int, completion_tokens: int,
                           input_cost_per_million: float = 2.50,
                           output_cost_per_million: float = 10.00) -> float:
    """Calculates the estimated cost for an LLM API call."""
    cost = (prompt_tokens * input_cost_per_million / 1_000_000) + \
           (completion_tokens * output_cost_per_million / 1_000_000)
    return cost


def setup_transcript_files(base_path: str = 'transcripts') -> None:
    """
    Ensures the 'transcripts' directory exists and writes example transcript content to files.
    """
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    aapl_transcript_content = """OPERATOR: Good day, and welcome to the Apple Q4 2024 Earnings Conference Call.
TIM COOK: Thank you. Good afternoon, everyone, and thanks for joining us. We finished a record-breaking fiscal 2024 with a strong September quarter. We delivered September quarter revenue of $90.1 billion, up 8% year-over-year. This growth was driven by continued strong performance in our Services segment, which grew 15% to $22.3 billion, and resilience in iPhone sales. We saw robust demand for our new iPhone 16 models. Our diluted earnings per share for the quarter were $1.40, compared to $1.35 in the prior year. This was above analyst consensus of $1.38. Our operating margin was 38.5%, an increase of 100 basis points YoY. We are seeing some macroeconomic headwinds in certain geographies, particularly in Europe, which we are closely monitoring. We expect December quarter revenue to be between $117 billion and $122 billion. Our guidance for fiscal year 2025 revenue is maintained at $390 billion to $400 billion. We remain confident in our long-term strategy.
LUCA MAESTRI: Thanks, Tim. Our balance sheet remains incredibly strong. We returned over $29 billion to shareholders through dividends and share repurchases in the September quarter.
ANALYST 1: Tim, can you elaborate on the European headwinds?
TIM COOK: Yes, we are seeing some softness, but believe our product pipeline remains strong.
ANALYST 2: What about the Q1 2025 EPS guidance?
LUCA MAESTRI: We typically do not provide specific quarterly EPS guidance, but we are confident in our full-year outlook.
"""
    with open(os.path.join(base_path, 'AAPL_Q4_2024.txt'), 'w', encoding='utf-8') as f:
        f.write(aapl_transcript_content)

    jpm_transcript_content = """OPERATOR: Welcome to the JPMorgan Chase & Co. Fourth Quarter 2024 Earnings Call.
JAMIE DIMON: Good morning, everyone. We concluded 2024 with record earnings, reflecting the strength of our diversified business model. For the fourth quarter of 2024, net revenue was $40.5 billion, up 10% from the prior year, exceeding expectations of $39.8 billion. Diluted earnings per share were $3.98, compared to $3.57 in Q4 2023, well above consensus estimates of $3.75. Our net interest income (NII) grew by 18%. We saw continued strength in our consumer and community banking segment. We are actively managing credit risk given the evolving economic environment. We are raising our guidance for full-year 2025 NII to approximately $90 billion, from our previous guidance of $88 billion. Our expenses were $22.5 billion, higher than expected due to investments in technology. We are confident in our capital position and continued growth trajectory.
ANALYST 1: Jamie, could you comment on the expense growth?
JAMIE DIMON: We are making strategic investments that we believe will drive long-term value.
"""
    with open(os.path.join(base_path, 'JPM_Q4_2024.txt'), 'w', encoding='utf-8') as f:
        f.write(jpm_transcript_content)

    tsla_transcript_content = """OPERATOR: Good afternoon, and welcome to the Tesla, Inc. Q4 2024 Earnings Q&A Webcast.
ELON MUSK: Thanks. Q4 2024 was a pivotal quarter for Tesla. We achieved revenue of $27.0 billion, a 20% increase year-over-year, which was in line with analyst expectations. Our diluted earnings per share came in at $0.95, slightly below the $1.00 consensus estimate. Automotive gross margin was 17.5%, down from 20% a year ago, primarily due to price reductions. We expect to see continued pressure on margins in the near term as we scale production of Cybertruck. Our energy storage business saw significant growth, with deployments up 45%. We are lowering our guidance for vehicle deliveries in 2025 due to global supply chain challenges and economic uncertainty. We now project 2025 deliveries to be around 1.8 million units, a decrease from our prior forecast of 2.0 million units. We are accelerating our AI and robotics initiatives.
ANALYST 1: Elon, about the margin pressure, how will that impact profitability?
ELON MUSK: We are focused on cost efficiency and innovation.
"""
    with open(os.path.join(base_path, 'TSLA_Q4_2024.txt'), 'w', encoding='utf-8') as f:
        f.write(tsla_transcript_content)
    print(f"Transcript files created/ensured in '{base_path}' directory.")


def load_transcripts_data(transcript_names: list[str], base_path: str = 'transcripts') -> dict[str, str]:
    """
    Loads transcript content from specified files and calculates token counts and estimated costs.

    Args:
        transcript_names (list[str]): A list of transcript file names (without extension).
        base_path (str): The directory where transcript files are located.

    Returns:
        dict[str, str]: A dictionary where keys are transcript names and values are their content.
    """
    transcripts = {}
    # Assuming a common model for token counting
    enc = tiktoken.encoding_for_model("gpt-4o")
    input_cost_per_million = 2.50

    print("\n--- Loading Transcripts ---")
    for company in transcript_names:
        filepath = os.path.join(base_path, f'{company}.txt')
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()

            tokens = enc.encode(text)

            print(f"Transcript: {filepath}")
            print(f"  Words: {len(text.split()):,}")
            print(f"  Tokens: {len(tokens):,}")

            estimated_input_cost = (
                len(tokens) * input_cost_per_million) / 1_000_000
            print(
                f"  Estimated input cost (input @ ${input_cost_per_million:.2f}/1M): ${estimated_input_cost:.4f}")

            transcripts[company] = text
        except FileNotFoundError:
            print(f" {company}: File not found at {filepath}")
            print("  Please ensure transcript files are in the 'transcripts' directory.")

    print("Environment setup complete. Transcripts loaded.")
    return transcripts

# --- LLM Interaction Functions ---


def explain_financial_concept(client: OpenAI, concept_query: str, model: str = 'gpt-4o') -> str:
    """
    Asks the LLM to explain a financial concept.

    Args:
        client (OpenAI): The OpenAI client instance.
        concept_query (str): The financial concept to explain.
        model (str): The LLM model to use (default: 'gpt-4o').

    Returns:
        str: The LLM's explanation.
    """
    messages = [
        {"role": "system", "content": "You are a highly knowledgeable financial analyst. Explain complex financial concepts clearly and concisely, focusing on real-world relevance."},
        {"role": "user", "content": f"Explain the significance of {concept_query} for a senior investment professional."}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )
    return response.choices[0].message.content


def summarize_naive(client: OpenAI, transcript: str, model: str = 'gpt-4o') -> str:
    """
    Summarizes a transcript using a naive prompting strategy.

    Args:
        client (OpenAI): The OpenAI client instance.
        transcript (str): The earnings call transcript text.
        model (str): The LLM model to use (default: 'gpt-4o').

    Returns:
        str: The LLM's naive summary.
    """
    messages = [
        {"role": "user", "content": PROMPT_NAIVE.format(transcript=transcript)}]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=500
    )
    return response.choices[0].message.content


def summarize_with_strategy(client: OpenAI, transcript: str, strategy: str = 'structured', model: str = 'gpt-4o') -> tuple[str, dict]:
    """
    Summarizes a transcript using selected prompting strategy (naive, structured, few_shot).
    Also returns token usage.

    Args:
        client (OpenAI): The OpenAI client instance.
        transcript (str): The earnings call transcript text.
        strategy (str): The prompting strategy ('naive', 'structured', 'few_shot').
        model (str): The LLM model to use (default: 'gpt-4o').

    Returns:
        tuple[str, dict]: The LLM's summary and token usage statistics.
    """
    messages = []
    if strategy == 'naive':
        messages = [
            {"role": "user", "content": PROMPT_NAIVE.format(transcript=transcript)}]
    elif strategy == 'structured':
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": TASK_PROMPT.format(
                transcript=transcript)}
        ]
    elif strategy == 'few_shot':
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": FEW_SHOT_EXAMPLE +
                "\nNow summarize the following transcript in the same format:\n" + TASK_PROMPT.format(transcript=transcript)}
        ]
    else:
        raise ValueError(
            "Invalid strategy. Choose 'naive', 'structured', or 'few_shot'.")

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=2000
    )

    # Debugging: print token usage for this call
    print(response.usage.to_dict())
    return response.choices[0].message.content, response.usage.to_dict()


def chunk_transcript(transcript: str, max_chunk_tokens: int = 8000, model: str = 'gpt-4o') -> list[tuple[str, str]]:
    """
    Split transcript into manageable chunks by section, then by sentence if needed.

    Args:
        transcript (str): The full earnings call transcript.
        max_chunk_tokens (int): Maximum tokens allowed per chunk.
        model (str): The LLM model encoding to use for token counting.

    Returns:
        list[tuple[str, str]]: A list of (label, chunk_text) tuples.
    """
    enc = tiktoken.encoding_for_model(model)
    chunks = []

    def split_if_needed(text: str, label: str) -> list[tuple[str, str]]:
        tokens = enc.encode(text)
        if len(tokens) <= max_chunk_tokens:
            return [(label, text)]

        sentences = nltk.sent_tokenize(text)
        if not sentences:  # Handle empty text
            return []

        # Try to split by paragraph/double newline first
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            mid_para = len(paragraphs) // 2
            part1 = '\n\n'.join(paragraphs[:mid_para])
            part2 = '\n\n'.join(paragraphs[mid_para:])
            # Ensure parts are not empty after split
            if enc.encode(part1) and enc.encode(part2):
                return split_if_needed(part1, f'{label}_part1') + split_if_needed(part2, f'{label}_part2')

        # Fallback to splitting by sentence if paragraphs don't work or are too large
        mid = len(sentences) // 2
        part1 = " ".join(sentences[:mid])
        part2 = " ".join(sentences[mid:])

        if not part1.strip() or not part2.strip():  # Avoid infinite recursion with single large sentence
            # If a single sentence exceeds max_chunk_tokens, we can't split further
            print(
                f"Warning: A single sentence/paragraph in '{label}' exceeds max_chunk_tokens ({max_chunk_tokens}). This chunk will be oversized.")
            return [(label, text)]

        return split_if_needed(part1, f'{label}_part1') + split_if_needed(part2, f'{label}_part2')

    qa_markers = ['Question-and-Answer', 'Q&A Session', 'Operator:']
    qa_start_idx = len(transcript)
    for marker in qa_markers:
        idx = transcript.find(marker)
        if 0 < idx < qa_start_idx:
            qa_start_idx = idx

    prepared_remarks = transcript[:qa_start_idx].strip()
    qa_session = transcript[qa_start_idx:].strip()

    if prepared_remarks:
        chunks.extend(split_if_needed(prepared_remarks, 'Prepared_Remarks'))
    if qa_session:
        chunks.extend(split_if_needed(qa_session, 'QA_Session'))

    print("\n--- Transcript Chunks and Token Counts ---")
    for label, chunk in chunks:
        n_tok = len(enc.encode(chunk))
        print(f" {label}: {n_tok:,} tokens")

    return chunks


def hierarchical_summarize(client: OpenAI, transcript: str, model: str = 'gpt-4o') -> str:
    """
    Summarize a long transcript by section, then merge into a final summary.

    Args:
        client (OpenAI): The OpenAI client instance.
        transcript (str): The full earnings call transcript.
        model (str): The LLM model to use (default: 'gpt-4o').

    Returns:
        str: The final merged summary.
    """
    chunks = chunk_transcript(transcript)
    section_summaries = []

    for label, chunk in chunks:
        print(f"\nSummarizing: {label}...")
        summary, _ = summarize_with_strategy(
            client, chunk, strategy='structured', model=model)
        section_summaries.append((label, summary))

    merge_prompt = f"""You are given summaries of different sections of an earnings call. Merge them into a single coherent analyst brief following the standard format (Headline, Performance, Guidance, Q&A, Risks, Tone). Remove redundancies and ensure consistency.

Section summaries:
{chr(10).join([f'--- {label} ---{chr(10)}{s}' for label, s in section_summaries])}"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": merge_prompt}
        ],
        temperature=0.2,
        max_tokens=2000
    )
    return response.choices[0].message.content


def extract_metrics(client: OpenAI, transcript: str, model: str = 'gpt-4o') -> dict:
    """
    Extracts structured financial metrics from a transcript as a JSON object.

    Args:
        client (OpenAI): The OpenAI client instance.
        transcript (str): The earnings call transcript text.
        model (str): The LLM model to use (default: 'gpt-4o').

    Returns:
        dict: A dictionary of extracted financial metrics.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a financial data extractor. Return only valid JSON."},
            {"role": "user", "content": EXTRACTION_PROMPT.format(
                transcript=transcript)}
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
        max_tokens=1500
    )
    metrics = json.loads(response.choices[0].message.content)
    return metrics


def hallucination_audit(summary: str, transcript: str) -> tuple[dict, float]:
    """
    Audits a summary for potential hallucinations.
    Checks: numbers, proper nouns (implicitly by checking presence), and forward-looking statements.

    Args:
        summary (str): The LLM-generated summary.
        transcript (str): The original earnings call transcript.

    Returns:
        tuple[dict, float]: A dictionary of audit results ('verified', 'flagged', 'missing')
                           and the calculated hallucination rate.
    """
    audit_results = {'verified': [], 'flagged': [], 'missing': []}

    # Normalize transcript for easier matching
    clean_transcript = transcript.replace(',', '').replace('$', '').lower()

    # Find numbers and numerical phrases in summary
    summary_numbers = re.findall(
        r'\$?[\d,]+\.?\d*(?:[BMK%]| billion| million| trillion)?', summary, re.IGNORECASE)

    for num in summary_numbers:
        clean_num = num.replace(',', '').replace(
            '$', '').replace(' percent', '%').lower()

        if num.lower() in transcript.lower():
            audit_results['verified'].append(
                f"VERIFIED: '{num}' found directly in transcript")
        elif clean_num in clean_transcript:
            audit_results['verified'].append(
                f"VERIFIED (format diff): '{num}' found as '{clean_num}' in transcript")
        else:
            audit_results['flagged'].append(
                f"FLAGGED: '{num}' NOT found in transcript")

    # Check for guidance terms that might be fabricated
    guidance_terms = ['raised', 'lowered', 'maintained',
                      'introduced', 'increased', 'decreased', 'revised']
    summary_lower = summary.lower()
    transcript_lower = transcript.lower()

    for term in guidance_terms:
        if term in summary_lower:
            # Simple check: if a guidance term appears in summary but not at all in transcript, it's suspicious
            # This is a crude check and might flag valid inferences
            if term not in transcript_lower:
                audit_results['flagged'].append(
                    f"FLAGGED: Summary uses guidance term '{term}' but it's absent from transcript (may be inferred).")

    # Check for presence of key sections/topics for completeness
    key_topics = ['revenue', 'earnings', 'margin',
                  'guidance', 'outlook', 'q&a', 'risks', 'tone']
    for topic in key_topics:
        if topic not in summary_lower:
            audit_results['missing'].append(
                f"MISSING: '{topic}' not explicitly mentioned or clearly addressed in summary (check completeness).")

    total_verifiable_items = len(
        audit_results['verified']) + len(audit_results['flagged'])
    hallucination_rate = (
        len(audit_results['flagged']) / max(1, total_verifiable_items)) * 100

    print("\n--- HALLUCINATION AUDIT REPORT ---")
    print("=" * 50)
    print(f"Total numbers/claims checked: {total_verifiable_items}")
    print(f"Numbers/Claims verified: {len(audit_results['verified'])}")
    print(f"Numbers/Claims flagged: {len(audit_results['flagged'])}")
    print(f"Topics potentially missing: {len(audit_results['missing'])}")
    print(f"\nHallucination rate: {hallucination_rate:.2f}%")

    if audit_results['flagged']:
        print("\nFLAGGED ITEMS (require manual verification):")
        for item in audit_results['flagged']:
            print(f" - {item}")

    if audit_results['missing']:
        print("\nMISSING TOPICS (check for completeness):")
        for item in audit_results['missing']:
            print(f" - {item}")

    return audit_results, hallucination_rate

# --- Analysis and Comparison Functions ---


def run_prompt_strategy_comparison(client: OpenAI, transcript_to_use: str, company_name: str, key_facts_template: dict[str, list[str]], model: str = 'gpt-4o') -> pd.DataFrame:
    """
    Runs an experiment comparing different prompting strategies for summarization.

    Args:
        client (OpenAI): The OpenAI client instance.
        transcript_to_use (str): The transcript text for the comparison.
        company_name (str): The name of the company for which the transcript belongs.
        key_facts_template (dict): A dictionary mapping company names to a list of key facts.
        model (str): The LLM model to use (default: 'gpt-4o').

    Returns:
        pd.DataFrame: A DataFrame containing comparison results for each strategy.
    """
    strategies = ['naive', 'structured', 'few_shot']
    comparison = {}
    enc = tiktoken.encoding_for_model(model)

    key_facts_transcript = key_facts_template.get(company_name, [])

    for strategy in strategies:
        print(f"\n--- Running summary with {strategy} strategy ---")
        summary, usage = summarize_with_strategy(
            client, transcript_to_use, strategy=strategy, model=model)
        audit, h_rate = hallucination_audit(summary, transcript_to_use)

        N_transcript_tokens = len(enc.encode(transcript_to_use))
        N_summary_tokens = len(enc.encode(summary))
        cr = 1 - (N_summary_tokens /
                  N_transcript_tokens) if N_transcript_tokens > 0 else 0

        facts_in_summary_count = 0
        for fact in key_facts_transcript:
            if fact.lower() in summary.lower():
                facts_in_summary_count += 1
        recall_facts = facts_in_summary_count / \
            len(key_facts_transcript) if key_facts_transcript else 0

        comparison[strategy] = {
            'word_count': len(summary.split()),
            'section_coverage': sum(1 for s in ('HEADLINE', 'QUARTERLY PERFORMANCE', 'GUIDANCE', 'Q&A', 'RISKS', 'TONE') if s.lower() in summary.lower()),
            'hallucination_rate': h_rate,
            'numbers_verified': len(audit['verified']),
            'numbers_flagged': len(audit['flagged']),
            'cost': get_estimated_api_cost(usage['prompt_tokens'], usage['completion_tokens']),
            'compression_ratio': cr,
            'information_retention_recall': recall_facts
        }

    comp_df = pd.DataFrame(comparison).T
    return comp_df


def run_temperature_calibration_experiment(client: OpenAI, transcript: str, temperatures: list[float], model: str = 'gpt-4o') -> pd.Series:
    """
    Runs an experiment to evaluate summary consistency across different temperatures.

    Args:
        client (OpenAI): The OpenAI client instance.
        transcript (str): The transcript text to summarize.
        temperatures (list[float]): A list of temperatures to test.
        model (str): The LLM model to use (default: 'gpt-4o').

    Returns:
        pd.Series: A Series containing Jaccard similarity scores for each temperature.
    """
    print("\n--- Temperature Calibration Experiment (Consistency) ---")
    print("=" * 60)
    temperature_consistency_results = {}

    for temp in temperatures:
        # Override temperature in summarize_with_strategy
        # This requires a slight modification to summarize_with_strategy or a direct call here.
        # For simplicity, we'll make direct calls for this experiment.
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": TASK_PROMPT.format(
                transcript=transcript)}
        ]

        response1 = client.chat.completions.create(
            model=model, messages=messages, temperature=temp, max_tokens=2000
        )
        run1 = response1.choices[0].message.content

        response2 = client.chat.completions.create(
            model=model, messages=messages, temperature=temp, max_tokens=2000
        )
        run2 = response2.choices[0].message.content

        words1 = set(run1.lower().split())
        words2 = set(run2.lower().split())
        jaccard_similarity = len(words1.intersection(
            words2)) / len(words1.union(words2))

        temperature_consistency_results[temp] = jaccard_similarity
        print(
            f"Temp={temp}: Jaccard similarity between runs = {jaccard_similarity:.3f}")

    return pd.Series(temperature_consistency_results)


def calculate_roi_brief(comparison_df: pd.DataFrame,
                        total_transcripts_processed: int,
                        annual_companies: int = 30,
                        annual_quarters: int = 4,
                        analyst_read_time_min: int = 45,
                        ai_review_time_min: int = 5,
                        analyst_focused_review_time_min: int = 10,
                        analyst_hourly_cost: int = 200) -> dict:
    """
    Calculates and presents the ROI and ethical brief for LLM adoption.

    Args:
        comparison_df (pd.DataFrame): DataFrame containing prompt strategy comparison results.
        total_transcripts_processed (int): Number of transcripts used in the demo.
        annual_companies (int): Number of companies tracked annually.
        annual_quarters (int): Number of quarters per year for each company.
        analyst_read_time_min (int): Time analyst spends reading a transcript manually (minutes).
        ai_review_time_min (int): Time analyst spends reviewing AI summary (minutes).
        analyst_focused_review_time_min (int): Time analyst spends focusing on key points after AI review (minutes).
        analyst_hourly_cost (int): Estimated hourly cost of an analyst.

    Returns:
        dict: A dictionary containing all calculated ROI metrics and ethical brief points.
    """
    results = {}

    structured_metrics = comparison_df.loc['structured']
    total_demo_cost = comparison_df['cost'].sum()

    results['operational_efficiency'] = {
        'total_demo_transcripts': total_transcripts_processed,
        'total_demo_cost': total_demo_cost,
        'avg_cost_per_structured_summary': structured_metrics['cost']
    }

    estimated_cost_per_transcript = structured_metrics['cost']
    projected_annual_llm_cost = annual_companies * \
        annual_quarters * estimated_cost_per_transcript
    results['operational_efficiency']['projected_annual_llm_cost'] = projected_annual_llm_cost

    total_analyst_time_saved_per_call_min = analyst_read_time_min - \
        (ai_review_time_min + analyst_focused_review_time_min)
    total_analyst_time_saved_per_call_hours = total_analyst_time_saved_per_call_min / 60

    total_calls_per_year = annual_companies * annual_quarters
    projected_annual_hours_saved = total_calls_per_year * \
        total_analyst_time_saved_per_call_hours
    projected_annual_analyst_value_saved = projected_annual_hours_saved * analyst_hourly_cost

    results['projected_annual_roi'] = {
        'analyst_time_saved_per_call_min': total_analyst_time_saved_per_call_min,
        'projected_annual_hours_saved': projected_annual_hours_saved,
        'projected_annual_analyst_value_saved': projected_annual_analyst_value_saved
    }

    roi = ((projected_annual_analyst_value_saved - projected_annual_llm_cost) /
           projected_annual_llm_cost) * 100 if projected_annual_llm_cost else 0
    results['projected_annual_roi']['roi_percentage'] = roi

    results['ethical_considerations'] = {
        'llm_role_shift': "LLM summarization shifts the analyst role from 'reader' to 'reviewer'.",
        'audit_necessity': "The Hallucination Audit is NOT optional; it's a mandatory compliance safeguard.",
        'structured_hallucination_rate': structured_metrics['hallucination_rate'],
        'compliance_risk': "Unverified AI-generated figures or mischaracterized guidance risks violating CFA Standard V(A): Diligence and Reasonable Basis.",
        'documentation_importance': "Importance of documenting the audit process for transparency and accountability."
    }

    return results


def print_roi_brief(roi_results: dict) -> None:
    """Prints the formatted ROI and ethical brief."""
    print("\n--- LLM ADOPTION: ROI AND ETHICAL BRIEF ---")
    print("=" * 60)

    op_eff = roi_results['operational_efficiency']
    print("\n**1. Operational Efficiency & Cost Savings**")
    print(
        f"  - Total demonstration transcripts processed: {op_eff['total_demo_transcripts']}")
    print(
        f"  - Total estimated API cost for these {op_eff['total_demo_transcripts']} transcripts: ${op_eff['total_demo_cost']:.4f}")
    print(
        f"  - Average cost per structured summary: ${op_eff['avg_cost_per_structured_summary']:.4f}")
    print(
        f"\n  - Projected Annual LLM Cost (for 30 companies, 4 quarters): ${op_eff['projected_annual_llm_cost']:.2f}")

    proj_roi = roi_results['projected_annual_roi']
    print(f"\n**2. Projected Annual ROI**")
    print(
        f"  - Estimated analyst time saved per call: {proj_roi['analyst_time_saved_per_call_min']:.1f} minutes")
    print(
        f"  - Projected annual analyst hours saved (for 30 companies): {proj_roi['projected_annual_hours_saved']:.1f} hours")
    print(
        f"  - Projected annual value of analyst time saved: ${proj_roi['projected_annual_analyst_value_saved']:,.2f}")
    print(f"  - Projected Annual ROI: {proj_roi['roi_percentage']:,.0f}%")

    eth_cons = roi_results['ethical_considerations']
    print("\n**3. Ethical Considerations & Compliance (CFA Standard V(A))**")
    print(f"  - {eth_cons['llm_role_shift']}")
    print(f"  - {eth_cons['audit_necessity']}")
    print(
        f"  - Example: Structured prompt hallucination rate was {eth_cons['structured_hallucination_rate']:.2f}%. Any flagged item requires human verification.")
    print(f"  - {eth_cons['compliance_risk']}")
    print(f"  - {eth_cons['documentation_importance']}")

# --- Main Workflow Function ---


def run_earnings_analysis_workflow(api_key: str | None = None) -> None:
    """
    Main function to run the entire earnings call analysis workflow.

    Args:
        api_key (str | None): Your OpenAI API key. If None, it will try to
                               read from OPENAI_API_KEY environment variable.
    """
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key or api_key == "YOUR_OPENAI_KEY_HERE":
        print("Error: OpenAI API key not provided or set. Please set the OPENAI_API_KEY environment variable or pass it to the function.")
        return

    client = OpenAI(api_key=api_key)

    transcript_files_to_setup = ['AAPL_Q4_2024', 'JPM_Q4_2024', 'TSLA_Q4_2024']
    setup_transcript_files()
    transcripts_data = load_transcripts_data(transcript_files_to_setup)

    if not transcripts_data:
        print("No transcripts loaded. Exiting.")
        return

    # 1. Explain Financial Concept
    print("\n--- LLM Explanation of Yield Curve Inversion ---")
    yield_curve_explanation = explain_financial_concept(
        client, "the significance of the yield curve inversion in 2023")
    print(yield_curve_explanation)

    # 2. Naive Summary Example
    aapl_transcript = transcripts_data.get('AAPL_Q4_2024')
    if aapl_transcript:
        print("\n--- Naive Summary of AAPL Q4 2024 Earnings ---")
        aapl_summary_naive = summarize_naive(client, aapl_transcript)
        print(aapl_summary_naive)

        # 3. Structured Summary Example
        print("\n--- Structured Summary of AAPL Q4 2024 Earnings ---")
        aapl_summary_structured, usage_structured = summarize_with_strategy(
            client, aapl_transcript, strategy='structured')
        print(aapl_summary_structured)
        print(f"\nStrategy: structured")
        print(f" Input tokens: {usage_structured['prompt_tokens']:,}")
        print(f" Output tokens: {usage_structured['completion_tokens']:,}")
        cost_structured = get_estimated_api_cost(
            usage_structured['prompt_tokens'], usage_structured['completion_tokens'])
        print(f" Cost: ${cost_structured:.4f}")

        # 4. Hallucination Audit for Structured Summary
        _, _ = hallucination_audit(aapl_summary_structured, aapl_transcript)

    # 5. Hierarchical Summarization Example (for JPM, assuming it's longer or just for demo)
    jpm_transcript = transcripts_data.get('JPM_Q4_2024')
    if jpm_transcript:
        print("\n--- Final Hierarchical Summary of JPM Q4 2024 Earnings ---")
        jpm_final_summary = hierarchical_summarize(client, jpm_transcript)
        print(jpm_final_summary)

    # 6. Metric Extraction Example
    if aapl_transcript:
        aapl_metrics = extract_metrics(client, aapl_transcript)
        print("\n--- Extracted Financial Metrics for AAPL Q4 2024 (JSON) ---")
        print(json.dumps(aapl_metrics, indent=2))

    # 7. Prompt Strategy Comparison
    if aapl_transcript:
        key_facts_for_recall = {
            'AAPL_Q4_2024': [
                "revenue of $90.1 billion", "up 8% year-over-year", "Services segment which grew 15%", "$22.3 billion",
                "diluted earnings per share for the quarter were $1.40", "above analyst consensus of $1.38",
                "operating margin was 38.5%", "increase of 100 basis points YoY",
                "macroeconomic headwinds in certain geographies, particularly in Europe",
                "December quarter revenue to be between $117 billion and $122 billion",
                "guidance for fiscal year 2025 revenue is maintained at $390 billion to $400 billion",
                "returned over $29 billion to shareholders"
            ]
        }
        comp_df = run_prompt_strategy_comparison(
            client, aapl_transcript, 'AAPL_Q4_2024', key_facts_for_recall)
        print("\n--- PROMPT STRATEGY COMPARISON ---")
        print("=" * 60)
        print(comp_df.to_string())
    else:
        comp_df = None
        print("Skipping prompt strategy comparison due to missing AAPL transcript.")

    # 8. Temperature Calibration Experiment
    if aapl_transcript:
        temperatures = [0.0, 0.2, 0.5, 0.8, 1.0]
        temperature_consistency_results = run_temperature_calibration_experiment(
            client, aapl_transcript, temperatures)
        print("\n--- Temperature Consistency Results ---")
        print(temperature_consistency_results.to_string())

    # 9. ROI and Ethical Brief
    if comp_df is not None:
        # Total transcripts used in the demo run, not just comparison
        total_transcripts_processed_demo = len(transcripts_data)
        roi_report = calculate_roi_brief(
            comp_df, total_transcripts_processed_demo)
        print_roi_brief(roi_report)
    else:
        print("\nSkipping ROI and Ethical Brief due to missing comparison data.")


# --- Entry Point for Script Execution ---
if __name__ == "__main__":
    # In a real app.py, you would get the API key from environment variables
    # or a secure configuration management system.
    # For this script, we'll try to get it from env, or prompt if not found.
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key or openai_api_key == "YOUR_OPENAI_KEY_HERE":
        print("OPENAI_API_KEY environment variable not set or placeholder used.")
        print("Please set your OpenAI API key as an environment variable or manually paste it below.")
        # As an alternative for local testing, you might uncomment this line and paste your key:
        # openai_api_key = "PASTE_YOUR_KEY_HERE"
        openai_api_key = input(
            "Enter your OpenAI API Key (or set OPENAI_API_KEY env var): ")
        if not openai_api_key:
            print("API Key not provided. Exiting.")
            exit()

    run_earnings_analysis_workflow(api_key=openai_api_key)
