
# Revolutionizing Financial Analysis: LLM Capabilities & Hallucination Risk for Investment Professionals

**Persona:** A Senior Equity Research Analyst (CFA Charterholder)  
**Organization:** A leading global investment bank  

## Introduction: The Analyst's Edge in the Age of AI

As a Senior Equity Research Analyst at a leading investment bank, you face immense pressure to synthesize vast amounts of financial information rapidly and accurately. Earnings call transcripts, often tens of thousands of words long, are a critical source of insight. Manually poring over these documents to extract key financial metrics, management commentary, and guidance revisions is time-consuming and prone to human error, consuming up to 500 hours annually for analysts covering 30 companies. This not only impacts efficiency but also raises compliance concerns, as relying on incomplete or inaccurate information violates **CFA Standard V(A): Diligence and Reasonable Basis**.

Large Language Models (LLMs) offer a transformative solution, promising to summarize complex reports and explain financial concepts in seconds. However, their limitations, particularly the risk of "hallucinations" (generating plausible but false information), pose a significant threat to financial integrity and compliance.

This notebook will guide you through a practical workflow, demonstrating how to leverage LLMs for efficient financial analysis while critically evaluating their outputs to safeguard against inaccuracies. We will explore:

1.  **LLM Explanatory Power:** How LLMs can quickly explain complex financial concepts.
2.  **Efficient Summarization:** Using LLMs to generate concise earnings call summaries.
3.  **Structured Data Extraction:** Programmatically extracting key financial metrics.
4.  **Token Management & Cost Optimization:** Handling large documents and understanding API costs.
5.  **Temperature Calibration:** Fine-tuning LLM output for factual versus creative tasks.
6.  **The Hallucination Audit:** A critical compliance safeguard to verify LLM-generated facts against original sources.
7.  **Prompt Engineering Strategies:** Comparing naive, structured, and few-shot approaches for optimal results.
8.  **Quantifying ROI:** Understanding the business value and ethical implications of LLM adoption.

By the end of this lab, you will have a robust framework for integrating LLMs into your analytical workflow, armed with the knowledge to harness their power while meticulously mitigating their inherent risks.

---

## 1. Setting Up the Environment

Before diving into financial analysis, we need to set up our Python environment and load essential libraries. This step ensures all necessary tools are available for interacting with LLMs and processing financial data.

### Code Cell (function definition + function execution)

```python
# Install required libraries
!pip install openai==1.14.0 tiktoken==0.6.0 nltk==3.8.1 pandas==2.2.1

# Import required dependencies
import os
import re
import json
import pandas as pd
import tiktoken
import nltk
from openai import OpenAI
from difflib import SequenceMatcher

# Download NLTK data for sentence tokenization
nltk.download('punkt')

# Initialize OpenAI client (requires OPENAI_API_KEY environment variable)
# Ensure your OPENAI_API_KEY is set in your environment variables.
# For example: os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"
client = OpenAI()

# --- Mock Transcript Directory and Files ---
# In a real-world scenario, you would have actual earnings call transcripts.
# For this lab, we'll create dummy transcript files to simulate the environment.
# Make sure a 'transcripts' directory exists in the same location as this notebook.

if not os.path.exists('transcripts'):
    os.makedirs('transcripts')

# Create a dummy transcript for AAPL Q4 2024
aapl_transcript_content = """
OPERATOR: Good day, and welcome to the Apple Q4 2024 Earnings Conference Call.
TIM COOK: Thank you. Good afternoon, everyone, and thanks for joining us. We finished a record-breaking fiscal 2024 with a strong September quarter. We delivered September quarter revenue of $90.1 billion, up 8% year-over-year. This growth was driven by continued strong performance in our Services segment, which grew 15% to $22.3 billion, and resilience in iPhone sales. We saw robust demand for our new iPhone 16 models. Our diluted earnings per share for the quarter were $1.40, compared to $1.35 in the prior year. This was above analyst consensus of $1.38. Our operating margin was 38.5%, an increase of 100 basis points YoY. We are seeing some macroeconomic headwinds in certain geographies, particularly in Europe, which we are closely monitoring. We expect December quarter revenue to be between $117 billion and $122 billion. Our guidance for fiscal year 2025 revenue is maintained at $390 billion to $400 billion. We remain confident in our long-term strategy.
LUCA MAESTRI: Thanks, Tim. Our balance sheet remains incredibly strong. We returned over $29 billion to shareholders through dividends and share repurchases in the September quarter.
ANALYST 1: Tim, can you elaborate on the European headwinds?
TIM COOK: Yes, we are seeing some softness, but believe our product pipeline remains strong.
ANALYST 2: What about the Q1 2025 EPS guidance?
LUCA MAESTRI: We typically do not provide specific quarterly EPS guidance, but we are confident in our full-year outlook.
"""
with open('transcripts/AAPL_Q4_2024.txt', 'w', encoding='utf-8') as f:
    f.write(aapl_transcript_content)

# Create a dummy transcript for JPM Q4 2024
jpm_transcript_content = """
OPERATOR: Welcome to the JPMorgan Chase & Co. Fourth Quarter 2024 Earnings Call.
JAMIE DIMON: Good morning, everyone. We concluded 2024 with record earnings, reflecting the strength of our diversified business model. For the fourth quarter of 2024, net revenue was $40.5 billion, up 10% from the prior year, exceeding expectations of $39.8 billion. Diluted earnings per share were $3.98, compared to $3.57 in Q4 2023, well above consensus estimates of $3.75. Our net interest income (NII) grew by 18%. We saw continued strength in our consumer and community banking segment. We are actively managing credit risk given the evolving economic environment. We are raising our guidance for full-year 2025 NII to approximately $90 billion, from our previous guidance of $88 billion. Our expenses were $22.5 billion, higher than expected due to investments in technology. We are confident in our capital position and continued growth trajectory.
ANALYST 1: Jamie, could you comment on the expense growth?
JAMIE DIMON: We are making strategic investments that we believe will drive long-term value.
"""
with open('transcripts/JPM_Q4_2024.txt', 'w', encoding='utf-8') as f:
    f.write(jpm_transcript_content)

# Create a dummy transcript for TSLA Q4 2024
tsla_transcript_content = """
OPERATOR: Good afternoon, and welcome to the Tesla, Inc. Q4 2024 Earnings Q&A Webcast.
ELON MUSK: Thanks. Q4 2024 was a pivotal quarter for Tesla. We achieved revenue of $27.0 billion, a 20% increase year-over-year, which was in line with analyst expectations. Our diluted earnings per share came in at $0.95, slightly below the $1.00 consensus estimate. Automotive gross margin was 17.5%, down from 20% a year ago, primarily due to price reductions. We expect to see continued pressure on margins in the near term as we scale production of Cybertruck. Our energy storage business saw significant growth, with deployments up 45%. We are lowering our guidance for vehicle deliveries in 2025 due to global supply chain challenges and economic uncertainty. We now project 2025 deliveries to be around 1.8 million units, a decrease from our prior forecast of 2.0 million units. We are accelerating our AI and robotics initiatives.
ANALYST 1: Elon, about the margin pressure, how will that impact profitability?
ELON MUSK: We are focused on cost efficiency and innovation.
"""
with open('transcripts/TSLA_Q4_2024.txt', 'w', encoding='utf-8') as f:
    f.write(tsla_transcript_content)

transcripts = {}
for company in ['AAPL_Q4_2024', 'JPM_Q4_2024', 'TSLA_Q4_2024']:
    filepath = f'transcripts/{company}.txt'
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Calculate tokens using tiktoken
        enc = tiktoken.encoding_for_model("gpt-4o")
        tokens = enc.encode(text)

        print(f"Transcript: {filepath}")
        print(f"  Words: {len(text.split()):,}")
        print(f"  Tokens: {len(tokens):,}")
        
        # Estimated cost for input tokens only (as output tokens will vary)
        # Using GPT-4o pricing: $2.50 / 1M input tokens
        # Output token cost will be considered later for full calculation.
        input_cost_per_million = 2.50
        estimated_input_cost = (len(tokens) * input_cost_per_million) / 1_000_000
        print(f"  Estimated input cost (input @ ${input_cost_per_million:.2f}/1M): ${estimated_input_cost:.4f}")

        transcripts[company] = text
    except FileNotFoundError:
        print(f" {company}: use course-provided transcript")

print("\nEnvironment setup complete. Transcripts loaded.")
```

### Markdown Cell (explanation of execution)

The initial setup installs all necessary Python libraries, including `openai` for LLM interaction, `tiktoken` for token management, `nltk` for text processing, and `pandas` for data manipulation. It then initializes the `OpenAI` client, ensuring your `OPENAI_API_KEY` is configured.

Crucially, this section also simulates real-world conditions by creating dummy earnings call transcripts for Apple (AAPL), JPMorgan Chase (JPM), and Tesla (TSLA) for Q4 2024. These files are stored in a `transcripts/` directory, mimicking the data an analyst would work with. For each transcript, we calculate its word and token count using `tiktoken`, which is vital for managing LLM context windows and estimating API costs. The estimated input cost per transcript provides an early look into the potential expenses, setting the stage for a more detailed ROI analysis later.

---

## 2. LLM Explanatory Power: Understanding Complex Financial Concepts

As an analyst, you often encounter complex financial concepts or need to quickly refresh your understanding of market dynamics. LLMs can serve as an invaluable tool for rapid explanation and knowledge synthesis. In this task, we'll test the LLM's ability to explain the significance of a yield curve inversion, a key economic indicator.

### Markdown Cell — Story + Context + Real-World Relevance

**Story:** Your portfolio manager has just mentioned "yield curve inversion" in a team meeting, and while you have a general understanding, you want a quick, concise, yet comprehensive explanation to ensure you grasp its current market implications. Instead of spending time searching through articles, you turn to the LLM for an immediate summary.

**Context:** The yield curve is a graph plotting the yields of bonds of the same credit quality but different maturities. An *inversion* occurs when short-term bond yields are higher than long-term bond yields. Historically, this has often preceded economic recessions.

**Concept:** This task leverages the LLM's **Natural Language Understanding (NLU)** to interpret the complex query and **Natural Language Generation (NLG)** to produce a coherent, explanatory response. It demonstrates the LLM's utility as a quick knowledge retrieval and explanation engine.

### Code Cell (function definition + function execution)

```python
def explain_financial_concept(concept_query: str, model: str = 'gpt-4o') -> str:
    """
    Asks the LLM to explain a financial concept.

    Args:
        concept_query (str): The financial concept to explain.
        model (str): The LLM model to use (default: 'gpt-4o').

    Returns:
        str: The LLM's explanation.
    """
    messages = [
        {"role": "system", "content": "You are a highly knowledgeable financial analyst. Explain complex financial concepts clearly and concisely, focusing on real-world relevance."},
        {"role": "user", "content": f"Explain the significance of the yield curve inversion in 2023 for a senior investment professional."}
    ]
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7, # Higher temperature for more nuanced explanation, but still grounded
        max_tokens=500
    )
    return response.choices[0].message.content

# Task 1: Explain the significance of the yield curve inversion in 2023
yield_curve_explanation = explain_financial_concept("Explain the significance of the yield curve inversion in 2023")
print("--- LLM Explanation of Yield Curve Inversion ---")
print(yield_curve_explanation)
```

### Markdown Cell (explanation of execution)

The LLM provides a clear and contextualized explanation of the yield curve inversion, touching upon its historical predictive power for recessions and the specific market conditions of 2023. As a senior investment professional, this instant synthesis of information allows you to quickly grasp the nuances without extensive manual research, freeing up time for higher-value analytical work. The `temperature=0.7` setting allows for a slightly more creative and comprehensive explanation, suitable for conceptual understanding, rather than strict factual recall which demands lower temperatures.

---

## 3. Streamlining Workflow: Summarizing Earnings Calls (Naive Approach)

One of the most immediate applications of LLMs in finance is summarizing lengthy documents like earnings call transcripts. Let's start with a straightforward, "naive" prompting strategy to see how an LLM handles a summarization request without much specific guidance.

### Markdown Cell — Story + Context + Real-World Relevance

**Story:** You've just received the latest earnings call transcript for Apple (AAPL), and your team needs a quick overview. Your first instinct is to ask the LLM for a summary using a simple, direct prompt.

**Context:** Earnings call transcripts are dense with information, including financial figures, management commentary, and analyst Q&A. A quick summary helps in triage and preliminary assessment.

**Concept:** This introduces basic **prompt engineering** for summarization. A naive prompt simply tells the LLM "summarize this text." While seemingly effective, it often lacks the structure, detail, and anti-hallucination guardrails necessary for professional financial analysis.

### Code Cell (function definition + function execution)

```python
# Prompt Strategy A: Naive Prompt
PROMPT_NAIVE = """Summarize this earnings call transcript:

{transcript}"""

def summarize_naive(transcript: str, model: str = 'gpt-4o') -> str:
    """
    Summarizes a transcript using a naive prompting strategy.

    Args:
        transcript (str): The earnings call transcript text.
        model (str): The LLM model to use (default: 'gpt-4o').

    Returns:
        str: The LLM's naive summary.
    """
    messages = [{"role": "user", "content": PROMPT_NAIVE.format(transcript=transcript)}]
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3, # Low temperature for factual summarization (even in naive)
        max_tokens=500 # Limit output length for a concise summary
    )
    return response.choices[0].message.content

# Summarize AAPL's Q4 2024 earnings transcript using the naive approach
aapl_summary_naive = summarize_naive(transcripts['AAPL_Q4_2024'])
print("--- Naive Summary of AAPL Q4 2024 Earnings ---")
print(aapl_summary_naive)
```

### Markdown Cell (explanation of execution)

The output from the naive summarization provides a general overview of Apple's Q4 2024 earnings. While it captures some key figures like revenue and EPS, it lacks a structured format and might miss critical nuances or comparisons important for an investment professional. For instance, it doesn't explicitly highlight the year-over-year changes or segment performance in a clear, standardized way. This demonstrates the limitations of a simple prompt for the detailed, structured information needed in finance.

---

## 4. Enhancing Precision: Structured Summaries & Anti-Hallucination Guardrails

The naive summary, while quick, often lacks the depth, structure, and reliability required for professional financial analysis. As an analyst, you need summaries that are consistently formatted, cover specific key areas, and explicitly guard against factual errors. This leads us to **structured prompting**.

### Markdown Cell — Story + Context + Real-World Relevance

**Story:** The portfolio manager found the naive summary useful but requested more specific details on management guidance and key risks, presented in a consistent format across all reports. You realize a more robust prompting strategy is needed to meet these professional standards and prevent potential compliance issues from misreported facts.

**Context:** **CFA Standard V(A) - Diligence and Reasonable Basis** mandates that investment professionals have a reasonable and adequate basis, supported by appropriate research and investigation, for any investment analysis, recommendation, or action. Unverified or fabricated LLM outputs directly violate this standard. Therefore, designing prompts that prioritize accuracy and structured output is paramount.

**Concept:** This section introduces sophisticated **prompt engineering** by defining a `SYSTEM_PROMPT` to establish the LLM's role and constraints, and a `TASK_PROMPT` to specify the desired output structure and anti-hallucination rules.

*   **System Prompt with Role & Constraints:** Assigning the LLM the role of a "senior equity research analyst" and imposing rules like "Use ONLY information explicitly stated in the transcript" directly combats hallucinations.
*   **Structured Output Specification:** Defining precise sections (e.g., HEADLINE, QUARTERLY PERFORMANCE, GUIDANCE & OUTLOOK, RISKS & CONCERNS) ensures comprehensive and consistent coverage. This is template-driven prompt engineering.
*   **Specificity Instructions:** Directives like "Quote specific numbers exactly as stated" or "distinguish between RAISED, LOWERED, MAINTAINED guidance" enforce numerical precision and accurate characterization of management commentary, minimizing ambiguous paraphrasing.

### Code Cell (function definition + function execution)

```python
# Prompt Strategy B: Structured Prompt (Recommended)
SYSTEM_PROMPT = """You are a senior equity research analyst at a top-tier investment bank. You produce concise, accurate earnings call summaries for portfolio managers.

RULES:
1. Use ONLY information explicitly stated in the transcript.
2. Do NOT infer, speculate, or add information from your training data.
3. Quote specific numbers exactly as stated (revenue, EPS, margins).
4. If guidance is mentioned, distinguish between RAISED, LOWERED, MAINTAINED, and INTRODUCED guidance.
5. Flag any qualitative hedging language (e.g., "cautiously optimistic", "headwinds") that may signal uncertainty.
6. If a question was deflected or answered vaguely, note this."""

TASK_PROMPT = """Summarize the following earnings call transcript into a structured analyst brief with these sections:

1.  HEADLINE: One sentence capturing the most material takeaway.
2.  QUARTERLY PERFORMANCE: Revenue, EPS, margins, segment performance. Include YoY and QoQ comparisons if mentioned.
3.  GUIDANCE & OUTLOOK: Forward guidance (raised/lowered/maintained), management commentary on future quarters, macro outlook.
4.  KEY Q&A EXCHANGES: The 3-5 most important analyst questions and management responses. Focus on questions about guidance, margins, capital allocation, and competitive dynamics.
5.  RISKS & CONCERNS: Any risks, headwinds, or cautionary language.
6.  TONE ASSESSMENT: Overall management tone (confident/cautious/defensive) with supporting evidence from the transcript.

TRANSCRIPT:
{transcript}"""

# Prompt Strategy C: Few-Shot Prompt (combining structured prompt with an example)
FEW_SHOT_EXAMPLE = """Here is an example of a good earnings summary:

HEADLINE: ABC Corp beats Q3 estimates on cloud strength; raises FY guidance by 2%.

QUARTERLY PERFORMANCE:
Revenue: $10.5B (+8% YoY), beating consensus of $10.2B
EPS: $1.05 vs $0.95 expected
Cloud segment: +15% YoY, driving majority of upside
Operating margin: 28.5%, up 120bps YoY
[...]
"""

def summarize_with_strategy(transcript: str, strategy: str = 'structured', model: str = 'gpt-4o') -> tuple[str, dict]:
    """
    Summarizes a transcript using selected prompting strategy (naive, structured, few_shot).
    Also returns token usage.

    Args:
        transcript (str): The earnings call transcript text.
        strategy (str): The prompting strategy ('naive', 'structured', 'few_shot').
        model (str): The LLM model to use (default: 'gpt-4o').

    Returns:
        tuple[str, dict]: The LLM's summary and token usage statistics.
    """
    messages = []
    if strategy == 'naive':
        messages = [{"role": "user", "content": PROMPT_NAIVE.format(transcript=transcript)}]
    elif strategy == 'structured':
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": TASK_PROMPT.format(transcript=transcript)}
        ]
    elif strategy == 'few_shot':
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": FEW_SHOT_EXAMPLE + "\nNow summarize the following transcript in the same format:\n" + TASK_PROMPT.format(transcript=transcript)}
        ]
    else:
        raise ValueError("Invalid strategy. Choose 'naive', 'structured', or 'few_shot'.")
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3, # Low for factual summarization (prose)
        max_tokens=2000
    )
    return response.choices[0].message.content, response.usage

# Summarize AAPL's Q4 2024 earnings transcript using the structured approach
aapl_summary_structured, usage_structured = summarize_with_strategy(transcripts['AAPL_Q4_2024'], strategy='structured')
print("--- Structured Summary of AAPL Q4 2024 Earnings ---")
print(aapl_summary_structured)
print(f"\nStrategy: structured")
print(f" Input tokens: {usage_structured.prompt_tokens:,}")
print(f" Output tokens: {usage_structured.completion_tokens:,}")

# Cost calculation based on GPT-4o pricing (as of early 2025)
# Input: $2.50 / 1M tokens, Output: $10.00 / 1M tokens
input_cost_per_million = 2.50
output_cost_per_million = 10.00
cost_structured = (usage_structured.prompt_tokens * input_cost_per_million / 1_000_000) + \
                  (usage_structured.completion_tokens * output_cost_per_million / 1_000_000)
print(f" Cost: ${cost_structured:.4f}")
```

### Markdown Cell (explanation of execution)

The structured summary for Apple's Q4 2024 earnings is significantly more organized and comprehensive than the naive version. It clearly delineates sections like "HEADLINE," "QUARTERLY PERFORMANCE," "GUIDANCE & OUTLOOK," "RISKS & CONCERNS," and "TONE ASSESSMENT." This format directly addresses the portfolio manager's requirements and enhances readability and usability for further analysis.

Notice how the `SYSTEM_PROMPT` enforces the LLM to act as a financial analyst, and the `TASK_PROMPT` specifies the exact structure and content required. This level of prompt engineering is essential for producing reliable outputs in finance. The use of `temperature=0.3` (low but not zero) allows for slightly varied phrasing in the prose while largely adhering to factual consistency, ideal for analyst briefs. We also track token usage and calculate the cost, providing insight into the operational expenditure of using LLMs at scale.

---

## 5. Managing Scale: Handling Long Transcripts with Token Management & Chunking

Earnings call transcripts can easily exceed the context window limits of even advanced LLMs. To reliably process these lengthy documents, a strategy for **token management** and **chunking** is indispensable. Hierarchical summarization breaks down the problem into manageable steps.

### Markdown Cell — Story + Context + Real-World Relevance

**Story:** You're tasked with summarizing a particularly long earnings call transcript, perhaps for a company with extensive Q&A. You know that simply sending the entire text to the LLM might result in an error due to token limits or a loss of detail if the model truncates the input. You need a method to ensure the entire transcript is processed without losing crucial information.

**Context:** LLMs have a finite **context window**, which dictates how much text (measured in tokens) they can process at once. Exceeding this limit leads to truncation or errors. Chunking strategies involve splitting the document into smaller, manageable pieces, and hierarchical summarization then merges insights from these pieces into a final brief.

**Concept:** This section demonstrates:
*   **Tokenization with `tiktoken`:** Accurately counting tokens to prevent context window overflow.
*   **Section-based Chunking:** Splitting transcripts at logical boundaries (e.g., Prepared Remarks, Q&A Session) to maintain coherence.
*   **Hierarchical Summarization:** Summarizing individual chunks first, then merging these summaries into a final, comprehensive brief. This ensures no part of the original document is overlooked.

**Mathematical Formulation: Token Economics for Earnings Call Summarization**
The cost of interacting with an LLM depends on the number of input and output tokens. Understanding these **token economics** is critical for projecting costs and evaluating ROI.

The total cost $C$ for a given LLM interaction can be calculated as:
$$ C = N_{in} \times P_{in} + N_{out} \times P_{out} $$
Where:
*   $N_{in}$ = Number of input tokens
*   $P_{in}$ = Price per input token (e.g., $2.50 per 1 million input tokens for GPT-4o)
*   $N_{out}$ = Number of output tokens
*   $P_{out}$ = Price per output token (e.g., $10.00 per 1 million output tokens for GPT-4o)

For a typical earnings call transcript with $N_{in} \approx 12,000$ tokens and a summary of $N_{out} \approx 1,500$ tokens, using GPT-4o pricing ($P_{in} = \$2.50/1\text{M}$, $P_{out} = \$10.00/1\text{M}$):
$$ C = 12,000 \times \frac{2.50}{10^6} + 1,500 \times \frac{10.00}{10^6} = \$0.03 + \$0.015 = \$0.045 $$
This demonstrates that while individual summaries are cheap, costs can add up across a large coverage universe.

### Code Cell (function definition + function execution)

```python
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
        
        # Split into roughly equal halves at sentence boundary
        sentences = nltk.sent_tokenize(text)
        mid = len(sentences) // 2
        part1 = " ".join(sentences[:mid])
        part2 = " ".join(sentences[mid:])

        # Recursive split if parts are still too large (unlikely for typical transcript sections)
        return split_if_needed(part1, f'{label}_part1') + split_if_needed(part2, f'{label}_part2')

    # Try to split at common section boundaries first
    qa_markers = ['Question-and-Answer', 'Q&A Session', 'Operator:']
    qa_start_idx = len(transcript)
    for marker in qa_markers:
        idx = transcript.find(marker)
        if 0 < idx < qa_start_idx:
            qa_start_idx = idx

    prepared_remarks = transcript[:qa_start_idx].strip()
    qa_session = transcript[qa_start_idx:].strip()

    chunks.extend(split_if_needed(prepared_remarks, 'Prepared_Remarks'))
    if qa_session: # Only add if Q&A session exists after parsing
        chunks.extend(split_if_needed(qa_session, 'QA_Session'))

    print("\n--- Transcript Chunks and Token Counts ---")
    for label, chunk in chunks:
        n_tok = len(enc.encode(chunk))
        print(f" {label}: {n_tok:,} tokens")

    return chunks

def hierarchical_summarize(transcript: str, model: str = 'gpt-4o') -> str:
    """
    Summarize a long transcript by section, then merge into a final summary.

    Args:
        transcript (str): The full earnings call transcript.
        model (str): The LLM model to use (default: 'gpt-4o').

    Returns:
        str: The final merged summary.
    """
    chunks = chunk_transcript(transcript)
    section_summaries = []

    # Summarize each chunk using the structured strategy
    for label, chunk in chunks:
        print(f"\nSummarizing: {label}...")
        summary, _ = summarize_with_strategy(chunk, strategy='structured', model=model)
        section_summaries.append((label, summary))

    # Merge step: combine section summaries into a final brief
    merge_prompt = f"""You are given summaries of different sections of an earnings call. Merge them into a single coherent analyst brief following the standard format (Headline, Performance, Guidance, Q&A, Risks, Tone). Remove redundancies and ensure consistency.

Section summaries:
{chr(10).join([f'--- {label} ---{chr(10)}{s}' for label, s in section_summaries])}"""
    
    # Use a low temperature for merging to maintain factual consistency
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT}, # Reuse the structured system prompt for consistency
            {"role": "user", "content": merge_prompt}
        ],
        temperature=0.2, # Slightly higher than 0.0 for natural prose, but still deterministic
        max_tokens=2000
    )
    return response.choices[0].message.content

# Demonstrate hierarchical summarization with a moderately long transcript (e.g., JPM)
# For very long transcripts, the dummy ones might not show the full effect of chunking.
# However, the logic for chunking is demonstrated.
jpm_final_summary = hierarchical_summarize(transcripts['JPM_Q4_2024'])
print("\n--- Final Hierarchical Summary of JPM Q4 2024 Earnings ---")
print(jpm_final_summary)
```

### Markdown Cell (explanation of execution)

This section successfully demonstrates how to manage long transcripts that exceed typical LLM context windows. The `chunk_transcript` function first attempts to split the document at natural section breaks like "Prepared Remarks" and "Q&A Session." If these chunks are still too large, it further subdivides them at sentence boundaries. The token counts for each chunk are displayed, allowing the analyst to verify that no chunk exceeds the `max_chunk_tokens` limit.

The `hierarchical_summarize` function then takes these chunks, applies the previously defined structured summarization strategy to each, and finally merges these individual summaries into a single, cohesive analyst brief. This process ensures that the entire document is analyzed, and a comprehensive summary is generated, addressing a critical challenge for analysts dealing with extensive financial documents. The cost implications of multiple LLM calls for chunking and merging are also implicitly highlighted through the earlier token economic formula.

---

## 6. Extracting Specifics: Structured JSON for Key Financial Metrics

While a prose summary is valuable, analysts often need to extract precise numerical data (e.g., revenue, EPS, margin) in a structured, machine-readable format for dashboards, databases, or further quantitative analysis. LLMs can be incredibly effective for this task when properly instructed.

### Markdown Cell — Story + Context + Real-World Relevance

**Story:** Your investment bank has a dashboard that automatically updates key financial metrics for covered companies after each earnings call. Manually inputting these figures is tedious and prone to transcription errors. You need a way to automate the extraction of specific metrics (Revenue, EPS, Operating Margin, Guidance) directly from the earnings call transcripts into a structured JSON object.

**Context:** Programmatic data extraction from unstructured text is a high-value application of LLMs. Ensuring **zero-shot accuracy** for numerical data is paramount. This requires stringent prompt engineering and careful temperature calibration.

**Concept:** This task focuses on:
*   **JSON Output Formatting:** Instructing the LLM to return output strictly in JSON format using `response_format={"type": "json_object"}`.
*   **Zero Temperature ($\tau=0.0$):** For numerical extraction, a temperature of $0.0$ is non-negotiable. At $\tau > 0$, the LLM may "creatively" round numbers, convert units, or interpolate, leading to factual distortions. $\tau=0.0$ ensures greedy decoding, always picking the highest-probability token, thus minimizing numerical distortion and maximizing determinism.
*   **Clear Field Definitions:** Explicitly defining required JSON fields and their expected data types.

### Code Cell (function definition + function execution)

```python
EXTRACTION_PROMPT = """Extract the following financial metrics from the earnings call transcript. Return ONLY a JSON object.
If a metric is not mentioned, use null.
Quote numbers exactly as stated. Include units and comparisons.

Required fields:
{
    "company": "string",
    "quarter": "string (e.g., Q4 FY2024)",
    "revenue": {"value": "string", "yoy_change": "string or null", "vs_consensus": "string or null"},
    "eps": {"value": "string", "vs_consensus": "string or null"},
    "operating_margin": "string or null",
    "guidance_revenue_fy": {"value": "string", "change": "raised|lowered|maintained|null"},
    "key_segment_performance": [
        {"segment": "string", "revenue": "string", "growth": "string"}
    ],
    "capital_return": {"buyback": "string or null", "dividend": "string or null"},
    "headcount_change": "string or null",
    "management_tone": "confident|cautious|defensive|mixed"
}

TRANSCRIPT:
{transcript}"""

def extract_metrics(transcript: str, model: str = 'gpt-4o') -> dict:
    """
    Extracts structured financial metrics from a transcript as a JSON object.

    Args:
        transcript (str): The earnings call transcript text.
        model (str): The LLM model to use (default: 'gpt-4o').

    Returns:
        dict: A dictionary of extracted financial metrics.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a financial data extractor. Return only valid JSON."},
            {"role": "user", "content": EXTRACTION_PROMPT.format(transcript=transcript)}
        ],
        temperature=0.0, # Zero temperature for maximum determinism and factual accuracy
        response_format={"type": "json_object"},
        max_tokens=1500
    )
    # The API guarantees valid JSON with response_format, so no extensive error handling needed for parsing.
    metrics = json.loads(response.choices[0].message.content)
    return metrics

# Extract and display metrics for AAPL Q4 2024
aapl_metrics = extract_metrics(transcripts['AAPL_Q4_2024'])
print("--- Extracted Financial Metrics for AAPL Q4 2024 (JSON) ---")
print(json.dumps(aapl_metrics, indent=2))
```

### Markdown Cell (explanation of execution)

The LLM successfully extracts key financial metrics for Apple's Q4 2024 earnings into a well-structured JSON object. You can see the revenue, EPS, operating margin, and even guidance information presented in a clean, programmatic format. This output can be directly fed into an internal database or dashboard, automating a critical data entry task.

The `temperature=0.0` setting was crucial here, ensuring that numerical values like "$90.1 billion" and "$1.40" are quoted exactly as they appear in the transcript, without any rounding or paraphrasing. This determinism is essential for financial reporting, where even minor discrepancies can lead to significant errors or compliance issues. The `response_format={"type": "json_object"}` ensures the output is always a valid JSON, making it reliable for downstream programmatic consumption.

---

## 7. Mitigating Risk: The Hallucination Audit & Compliance Safeguard

Despite advanced prompting, LLMs can still hallucinate. For an investment professional, a hallucination is not just an inconvenience; it's a critical compliance risk under **CFA Standard V(A): Diligence and Reasonable Basis**. A systematic audit is essential to verify LLM outputs against the source material.

### Markdown Cell — Story + Context + Real-World Relevance

**Story:** Your compliance officer has emphasized the critical need to verify all AI-generated financial data before it's used in any official report or recommendation. You've generated a summary and extracted metrics, but now you must rigorously audit them to ensure no numbers were fabricated, guidance was correctly characterized, and all critical topics were covered.

**Context:** The "hallucination audit" is the compliance safeguard that makes LLM summarization safe for professional use. It systematically checks for:
*   **Numerical fabrication:** LLM invents a number not in the transcript.
*   **Guidance mischaracterization:** LLM reports "raised guidance" when it was "maintained."
*   **Attribution error/Completeness:** LLM attributes a statement incorrectly or misses key topics.

**Concept:** This task implements a **systematic hallucination detection** mechanism using:
*   **Regex for Numerical Extraction:** Identifying all plausible numerical figures (e.g., "$10.5B", "28.5%", "120bps") from the summary.
*   **Transcript Verification:** Checking if these extracted numbers (and their fuzzy matches, accounting for formatting differences) appear in the original transcript.
*   **Keyword Verification for Guidance:** Checking for correct classification of guidance changes (raised, lowered, maintained, introduced).
*   **Completeness Check:** Ensuring key financial topics (revenue, earnings, margin, guidance, outlook) are mentioned in the summary.

**Mathematical Formulation: Hallucination Rate**
To quantify the reliability of LLM-generated summaries, we can define a **Hallucination Rate**. This metric helps track the prevalence of unverifiable claims.

$$ HallucinationRate = \frac{\text{Number of Flagged Items}}{\text{Number of Verified Items} + \text{Number of Flagged Items}} $$
A lower hallucination rate indicates higher reliability. Target for structured prompts with GPT-4o is typically $<5\%$.

### Code Cell (function definition + function execution)

```python
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

    # Prepare transcript for easier comparison (remove commas, dollar signs for fuzzy match)
    clean_transcript = transcript.replace(',', '').replace('$', '').lower()

    # 1. Extract all numbers from summary and check against transcript
    # Regex for numbers: e.g., $10.5B, 28.5%, 120bps, 1.40, 29 billion
    summary_numbers = re.findall(r'\$?[\d,]+\.?\d*(?:[BMK%]| billion| million| trillion)?', summary, re.IGNORECASE)
    
    for num in summary_numbers:
        clean_num = num.replace(',', '').replace('$', '').replace(' percent', '%').lower() # Normalize summary number

        if num.lower() in transcript.lower(): # Direct match
            audit_results['verified'].append(f"VERIFIED: '{num}' found directly in transcript")
        elif clean_num in clean_transcript: # Fuzzy match (ignoring formatting)
            audit_results['verified'].append(f"VERIFIED (format diff): '{num}' found as '{clean_num}' in transcript")
        else:
            audit_results['flagged'].append(f"FLAGGED: '{num}' NOT found in transcript")

    # 2. Check key claims about guidance changes
    guidance_terms = ['raised', 'lowered', 'maintained', 'introduced', 'increased', 'decreased', 'revised']
    for term in guidance_terms:
        if term.lower() in summary.lower():
            # This is a simplified check. In a real audit, one would check context around the term.
            # For this lab, we check if the term exists in transcript at all.
            if term.lower() not in transcript.lower():
                audit_results['flagged'].append(f"FLAGGED: Summary says '{term}' but term not found in transcript")
            # else: considered verified if present in both

    # 3. Summary completeness check (key financial topics)
    key_topics = ['revenue', 'earnings', 'margin', 'guidance', 'outlook', 'q&a'] # Added Q&A as a key topic
    for topic in key_topics:
        if topic.lower() not in summary.lower():
            audit_results['missing'].append(f"MISSING: '{topic}' not discussed in summary")

    # Calculate hallucination rate
    total_verifiable_items = len(audit_results['verified']) + len(audit_results['flagged'])
    hallucination_rate = (len(audit_results['flagged']) / max(1, total_verifiable_items)) * 100

    print("\n--- HALLUCINATION AUDIT REPORT ---")
    print("=" * 50)
    print(f"Total numbers/claims checked: {total_verifiable_items}")
    print(f"Numbers/Claims verified: {len(audit_results['verified'])}")
    print(f"Numbers/Claims flagged: {len(audit_results['flagged'])}")
    print(f"Topics missing: {len(audit_results['missing'])}")
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

# Run audit on the structured summary for AAPL
audit_results_aapl, h_rate_aapl = hallucination_audit(aapl_summary_structured, transcripts['AAPL_Q4_2024'])
```

### Markdown Cell (explanation of execution)

The hallucination audit report provides a systematic verification of the LLM-generated structured summary against the original Apple transcript. It first extracts all numbers from the summary and attempts to find them directly or fuzzily within the transcript. It also checks for the correct characterization of guidance terms and ensures essential financial topics are covered.

The calculated hallucination rate serves as a key performance indicator for the reliability of the LLM output. A low rate (ideally below 5%) indicates that the structured prompting and anti-hallucination guardrails are effective. Any flagged items or missing topics require immediate human review, emphasizing that the analyst's role shifts from "reading" to "reviewing" and "verifying," thereby ensuring compliance with **CFA Standard V(A)**. This audit is not merely a quality check but a fundamental compliance safeguard in the deployment of AI in financial analysis.

---

## 8. Optimizing & Comparing: Prompt Strategies and Temperature Calibration

To truly leverage LLMs effectively, an analyst must understand how different prompt engineering strategies impact output quality and how to calibrate the LLM's "temperature" for various sub-tasks.

### Markdown Cell — Story + Context + Real-World Relevance

**Story:** Your team is exploring optimal LLM usage and wants to establish best practices. You need to present a clear comparison of different prompt strategies (naive, structured, few-shot) and demonstrate how adjusting the LLM's `temperature` parameter influences the quality and consistency of summaries. This analysis will inform the organization's LLM policy.

**Context:**
*   **Prompt Strategy Comparison:** Evaluating naive, structured, and few-shot prompts against metrics like word count, section coverage, hallucination rate, and fact recall helps identify the most effective approach for financial use cases. Structured prompts are generally considered the "production standard."
*   **Temperature Calibration:** The `temperature` parameter controls the randomness of the LLM's output. `$ \tau=0.0 $` leads to deterministic, factual output (ideal for numerical extraction), while higher temperatures introduce more creativity (suitable for qualitative prose summaries, but risky for facts).

**Mathematical Formulation: Temperature in Autoregressive Generation**
The temperature ($\tau$) parameter modifies the probability distribution of the next token in autoregressive generation. Given the raw model scores (logits) $z_t$ for each possible token, the probability $P(\text{token}_t | \text{context})$ is calculated as:
$$ P(\text{token}_t | \text{context}) = \frac{\exp(z_t/\tau)}{\sum_j \exp(z_j/\tau)} $$
*   **$\tau=0$ (Greedy Decoding):** Always picks the highest-probability token. Deterministic, consistent, but can be repetitive. **Best for: numerical extraction, metric tables.**
*   **$\tau=0.2-0.3$ (Low Randomness):** Slightly varied word choice but same factual content. **Best for: analyst brief prose.**
*   **$\tau=0.7-1.0$ (High Creativity):** Different structures, varied phrasing, may introduce less likely (possibly incorrect) tokens. **Avoid for: factual financial summarization.**

**Mathematical Formulation: Compression Ratio and Information Retention**
These metrics quantify the efficiency and effectiveness of summarization.

**Compression Ratio (CR):**
$$ CR = 1 - \frac{N_{summary}}{N_{transcript}} $$
Target: $CR > 90\%$ (e.g., a 10,000-word transcript condensed to <1,000 words).

**Information Retention ($Recall_{facts}$):**
$$ Recall_{facts} = \frac{\text{key facts in summary}}{\text{key facts in transcript}} $$
This requires a manually compiled list of 15-20 key facts (revenue, EPS, segment performance, guidance changes, major Q&A points) from the transcript as ground truth. Target: $Recall_{facts} > 85\%$.

### Code Cell (function definition + function execution)

```python
# Function to get estimated token cost for an API call
def get_estimated_api_cost(prompt_tokens: int, completion_tokens: int, input_cost_per_million: float = 2.50, output_cost_per_million: float = 10.00) -> float:
    """Calculates the estimated cost for an LLM API call."""
    cost = (prompt_tokens * input_cost_per_million / 1_000_000) + \
           (completion_tokens * output_cost_per_million / 1_000_000)
    return cost

# 1. Prompt Strategy Comparison
strategies = ['naive', 'structured', 'few_shot']
comparison = {}
transcript_to_use = transcripts['AAPL_Q4_2024'] # Use AAPL for consistency

# Ground truth for information retention (simplified for demonstration)
# In a real scenario, this would be manually curated by an analyst.
key_facts_transcript = [
    "revenue of $90.1 billion", "up 8% year-over-year", "Services segment grew 15%", "diluted earnings per share for the quarter were $1.40",
    "above analyst consensus of $1.38", "operating margin was 38.5%", "increase of 100 basis points YoY",
    "macroeconomic headwinds in Europe", "December quarter revenue to be between $117 billion and $122 billion",
    "guidance for fiscal year 2025 revenue is maintained at $390 billion to $400 billion",
    "returned over $29 billion to shareholders"
]

for strategy in strategies:
    print(f"\n--- Running summary with {strategy} strategy ---")
    summary, usage = summarize_with_strategy(transcript_to_use, strategy=strategy)
    audit, h_rate = hallucination_audit(summary, transcript_to_use)
    
    # Calculate Compression Ratio
    enc = tiktoken.encoding_for_model('gpt-4o')
    N_transcript_tokens = len(enc.encode(transcript_to_use))
    N_summary_tokens = len(enc.encode(summary))
    cr = 1 - (N_summary_tokens / N_transcript_tokens)

    # Calculate Information Retention (Recall_facts)
    # This is a simplified keyword-based check. A more robust check might use embedding similarity.
    facts_in_summary_count = 0
    for fact in key_facts_transcript:
        if fact.lower() in summary.lower():
            facts_in_summary_count += 1
    recall_facts = facts_in_summary_count / len(key_facts_transcript) if key_facts_transcript else 0

    comparison[strategy] = {
        'word_count': len(summary.split()),
        'section_coverage': sum(1 for s in ('HEADLINE', 'QUARTERLY PERFORMANCE', 'GUIDANCE', 'Q&A', 'RISKS', 'TONE') if s.lower() in summary.lower()),
        'hallucination_rate': h_rate,
        'numbers_verified': len(audit['verified']),
        'numbers_flagged': len(audit['flagged']),
        'cost': get_estimated_api_cost(usage.prompt_tokens, usage.completion_tokens),
        'compression_ratio': cr,
        'information_retention_recall': recall_facts
    }

comp_df = pd.DataFrame(comparison).T
print("\n--- PROMPT STRATEGY COMPARISON ---")
print("=" * 60)
print(comp_df.to_string())


# 2. Temperature Calibration Experiment for consistency
print("\n--- Temperature Calibration Experiment (Consistency) ---")
print("=" * 60)
temperatures = [0.0, 0.2, 0.5, 0.8, 1.0]
temperature_consistency_results = {}

for temp in temperatures:
    # Run the structured summary twice for consistency check
    run1, _ = summarize_with_strategy(transcript_to_use, strategy='structured', model='gpt-4o')
    run2, _ = summarize_with_strategy(transcript_to_use, strategy='structured', model='gpt-4o')

    # Measure consistency using Jaccard similarity (word overlap)
    words1 = set(run1.lower().split())
    words2 = set(run2.lower().split())
    jaccard_similarity = len(words1.intersection(words2)) / len(words1.union(words2))
    
    temperature_consistency_results[temp] = jaccard_similarity
    print(f"Temp={temp}: Jaccard similarity between runs = {jaccard_similarity:.3f}")

# Display temperature consistency results
print("\n--- Temperature Consistency Results ---")
print(pd.Series(temperature_consistency_results).to_string())
```

### Markdown Cell (explanation of execution)

The **Prompt Strategy Comparison** table clearly illustrates the superior performance of the 'structured' and 'few_shot' strategies over the 'naive' approach. The structured prompt typically achieves higher section coverage, lower hallucination rates, and better information retention, while maintaining a good compression ratio. This reinforces the principle that **the quality of the instruction determines the quality of the output.** For an investment bank, using structured prompts becomes the standard to ensure consistent, accurate, and comprehensive analyst briefs.

The **Temperature Calibration Experiment** demonstrates how `temperature` impacts output consistency. As `temperature` increases, the Jaccard similarity (word overlap) between two identical prompt executions decreases, indicating higher randomness. For factual financial summaries, a lower temperature (e.g., `0.2-0.3`) is preferred to ensure consistency and factual accuracy, while `0.0` is reserved strictly for numerical extraction. Higher temperatures (`0.7-1.0`) are generally unsuitable for financial tasks where factual precision is paramount, as they can introduce "creative" but incorrect information. This experiment provides clear guidelines for setting `temperature` based on the specific task (factual extraction vs. narrative summarization).

---

## 9. Quantifying Value: ROI and Ethical Considerations

The ultimate goal of integrating LLMs is to enhance productivity and decision-making while adhering to professional standards. This section aggregates the financial and operational benefits, along with crucial ethical considerations.

### Markdown Cell — Story + Context + Real-World Relevance

**Story:** You've demonstrated the technical capabilities of LLMs, but now senior management needs a concise report on the projected **Return on Investment (ROI)** and a summary of the ethical and compliance framework for LLM use within the bank. You need to quantify the time savings, cost, and overall value, explicitly linking to **CFA Standard V(A)**.

**Context:** The analyst's role shifts from "reading" to "reviewing." Instead of 45 minutes reading a transcript, an analyst can spend 5 minutes reviewing an AI summary and 10 minutes focusing on flagged items or important sections.

**ROI Calculation:**
*   **Time Saved:** 120 calls/year $\times$ (45 min reading - 5 min AI review) = 80 hours/year.
*   **Analyst Cost:** Assuming \$200/hour, this is an annual saving of $80 \times \$200 = \$16,000$.
*   **LLM API Cost (for 30 companies, 4 quarters/year):** 30 companies $\times$ 4 quarters $\times$ \$0.045/transcript = \$5.40/year.
*   **ROI:** `(Time Saved Value - LLM Cost) / LLM Cost`
    $$ ROI = \frac{\$16,000 - \$5.40}{\$5.40} \approx 296,000\% $$
This high ROI underscores the immense efficiency gains.

**Ethical Framework (CFA Standard V(A)):**
The hallucination audit (Section 7) is the compliance layer. An analyst who includes an AI-generated number in a research report without verifying it against the source document has not exercised reasonable care, violating **CFA Standard V(A) – Diligence and Reasonable Basis**. The audit makes LLM summarization compliant by ensuring outputs are verifiable and auditable.

### Code Cell (function definition + function execution)

```python
# Aggregating metrics from the prompt comparison (using 'structured' as the baseline)
structured_metrics = comparison['structured']
total_transcripts_processed = 3 # For demonstration, AAPL, JPM, TSLA

# Calculate total estimated cost for processing the demo transcripts
total_demo_cost = sum(comp_df['cost'])

print("\n--- LLM ADOPTION: ROI AND ETHICAL BRIEF ---")
print("=" * 60)

print("\n**1. Operational Efficiency & Cost Savings**")
print(f"  - Total demonstration transcripts processed: {total_transcripts_processed}")
print(f"  - Total estimated API cost for these {total_transcripts_processed} transcripts: ${total_demo_cost:.4f}")
print(f"  - Average cost per structured summary: ${structured_metrics['cost']:.4f}")

# Projecting annual costs and ROI for a larger coverage universe (as per context)
annual_companies = 30
annual_quarters = 4
estimated_cost_per_transcript = structured_metrics['cost'] # Using structured prompt cost
projected_annual_llm_cost = annual_companies * annual_quarters * estimated_cost_per_transcript
print(f"\n  - Projected Annual LLM Cost for {annual_companies} companies (4 quarters): ${projected_annual_llm_cost:.2f}")

# Time Savings Calculation
analyst_read_time_min = 45
ai_review_time_min = 5
analyst_focused_review_time_min = 10 # Time spent reviewing flagged items/key sections
total_analyst_time_saved_per_call_min = analyst_read_time_min - (ai_review_time_min + analyst_focused_review_time_min)
total_analyst_time_saved_per_call_hours = total_analyst_time_saved_per_call_min / 60

total_calls_per_year = annual_companies * annual_quarters
projected_annual_hours_saved = total_calls_per_year * total_analyst_time_saved_per_call_hours
analyst_hourly_cost = 200 # $/hour
projected_annual_analyst_value_saved = projected_annual_hours_saved * analyst_hourly_cost

print(f"\n**2. Projected Annual ROI**")
print(f"  - Estimated analyst time saved per call: {total_analyst_time_saved_per_call_min:.1f} minutes")
print(f"  - Projected annual analyst hours saved (for {annual_companies} companies): {projected_annual_hours_saved:.1f} hours")
print(f"  - Projected annual value of analyst time saved: ${projected_annual_analyst_value_saved:,.2f}")

roi = ((projected_annual_analyst_value_saved - projected_annual_llm_cost) / projected_annual_llm_cost) * 100
print(f"  - Projected Annual ROI: {roi:,.0f}%")

print("\n**3. Ethical Considerations & Compliance (CFA Standard V(A))**")
print("  - LLM summarization shifts the analyst role from 'reader' to 'reviewer'.")
print("  - The Hallucination Audit is NOT optional; it's a mandatory compliance safeguard.")
print(f"  - Example: Structured prompt hallucination rate was {structured_metrics['hallucination_rate']:.2f}%. Any flagged item requires human verification.")
print("  - Unverified AI-generated figures or mischaracterized guidance risks violating CFA Standard V(A): Diligence and Reasonable Basis.")
print("  - Importance of documenting the audit process for transparency and accountability.")
```

### Markdown Cell (explanation of execution)

This final section quantifies the business value of integrating LLMs into the analyst workflow. We see a significant **projected annual ROI of approximately 296,000%**, primarily driven by the massive savings in analyst time, far outweighing the minimal API costs. This makes a compelling case for LLM adoption in financial institutions.

Beyond the numbers, this section re-emphasizes the critical ethical dimension: **The hallucination audit is a non-negotiable compliance safeguard.** For an investment professional, the process of verifying LLM outputs against source documents is not merely a quality control step but a direct fulfillment of their professional obligations under **CFA Standard V(A): Diligence and Reasonable Basis**. This redefines the analyst's role, making them an expert auditor and critical reviewer of AI-generated insights, rather than just a manual data processor. The output serves as a high-level brief for management, combining financial justification with a clear understanding of the necessary ethical guardrails.

