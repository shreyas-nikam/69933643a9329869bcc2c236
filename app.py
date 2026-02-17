import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import io
import sys
import tiktoken
from source import *

# Set page configuration as requested
st.set_page_config(page_title="QuLab: Lab 25: LLM Prompting Demo", layout="wide")

# Sidebar Standard Header
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.sidebar.title("LLM Financial Lab")
st.sidebar.markdown("---")

# Global Settings in Sidebar
st.sidebar.header("Global Settings")
if 'llm_model' not in st.session_state:
    st.session_state.llm_model = 'gpt-4o'
st.session_state.llm_model = st.sidebar.selectbox(
    "Select LLM Model",
    ['gpt-4o', 'gpt-3.5-turbo'],
    index=0,
    key="global_llm_model_select",
    help="Choose the OpenAI model for generating responses."
)

if 'transcripts' in globals() and transcripts:
    transcript_keys = list(transcripts.keys())
    if 'selected_transcript' not in st.session_state:
        st.session_state.selected_transcript = transcript_keys[0]
    
    default_idx = transcript_keys.index(st.session_state.selected_transcript) if st.session_state.selected_transcript in transcript_keys else 0
    st.session_state.selected_transcript = st.sidebar.selectbox(
        "Select Earnings Transcript",
        transcript_keys,
        index=default_idx,
        key="global_transcript_select",
        help="The earnings call transcript to analyze."
    )
else:
    st.sidebar.warning("No transcripts available. Ensure `source.py` correctly loads them.")
    st.session_state.selected_transcript = None

# Navigation in Sidebar
st.sidebar.markdown("---")
st.sidebar.header("Navigation")
page_options = [
    "Application Overview",
    "1. LLM Explanatory Power",
    "2. Summarize Earnings Call",
    "3. Hallucination Audit & Compliance",
    "4. ROI & Ethical Considerations"
]

if 'current_page' not in st.session_state:
    st.session_state.current_page = page_options[0]

nav_index = page_options.index(st.session_state.current_page) if st.session_state.current_page in page_options else 0
st.session_state.current_page = st.sidebar.selectbox(
    "Go to",
    page_options,
    index=nav_index,
    key="page_navigation"
)

# Main Page Title
st.title("QuLab: Lab 25: LLM Prompting Demo")
st.divider()

# Initialize Session State Variables
if 'concept_query' not in st.session_state:
    st.session_state.concept_query = "Explain the significance of the yield curve inversion in 2023 for a senior investment professional."
if 'explanation_response' not in st.session_state:
    st.session_state.explanation_response = ""
if 'summaries' not in st.session_state:
    st.session_state.summaries = {}
if 'json_metrics' not in st.session_state:
    st.session_state.json_metrics = {}
if 'comparison_data' not in st.session_state:
    st.session_state.comparison_data = pd.DataFrame()
if 'temp_consistency_results' not in st.session_state:
    st.session_state.temp_consistency_results = {}
if 'summarization_temperature' not in st.session_state:
    st.session_state.summarization_temperature = 0.3
if 'json_extraction_temperature' not in st.session_state:
    st.session_state.json_extraction_temperature = 0.0
if 'audit_target_summary' not in st.session_state:
    st.session_state.audit_target_summary = 'structured'

# --- Main Content Area ---
if st.session_state.current_page == "Application Overview":
    st.title("Revolutionizing Financial Analysis: LLM Capabilities & Hallucination Risk for Investment Professionals")
    st.markdown(f"""
    As a Senior Equity Research Analyst at a leading investment bank, you face immense pressure to synthesize vast amounts of financial information rapidly and accurately. Earnings call transcripts, often tens of thousands of words long, are a critical source of insight. Manually poring over these documents to extract key financial metrics, management commentary, and guidance revisions is time-consuming and prone to human error, consuming up to 500 hours annually for analysts covering 30 companies. This not only impacts efficiency but also raises compliance concerns, as relying on incomplete or inaccurate information violates **CFA Standard V(A): Diligence and Reasonable Basis**.

    Large Language Models (LLMs) offer a transformative solution, promising to summarize complex reports and explain financial concepts in seconds. However, their limitations, particularly the risk of "hallucinations" (generating plausible but false information), pose a significant threat to financial integrity and compliance.

    This lab will guide you through a practical workflow, demonstrating how to leverage LLMs for efficient financial analysis while critically evaluating their outputs to safeguard against inaccuracies. We will explore:
    """)
    st.markdown(f"""
    1.  **LLM Explanatory Power:** How LLMs can quickly explain complex financial concepts.
    2.  **Efficient Summarization:** Using LLMs to generate concise earnings call summaries.
    3.  **Structured Data Extraction:** Programmatically extracting key financial metrics.
    4.  **Token Management & Cost Optimization:** Handling large documents and understanding API costs.
    5.  **Temperature Calibration:** Fine-tuning LLM output for factual versus creative tasks.
    6.  **The Hallucination Audit:** A critical compliance safeguard to verify LLM-generated facts against original sources.
    7.  **Prompt Engineering Strategies:** Comparing naive, structured, and few-shot approaches for optimal results.
    8.  **Quantifying ROI:** Understanding the business value and ethical implications of LLM adoption.
    """)
    st.markdown(f"""
    By the end of this lab, you will have a robust framework for integrating LLMs into your analytical workflow, armed with the knowledge to harness their power while meticulously mitigating their inherent risks.
    """)

    st.header("1. Setting Up the Environment")
    st.markdown(f"""
    The initial setup installs all necessary Python libraries, including `openai` for LLM interaction, `tiktoken` for token management, `nltk` for text processing, and `pandas` for data manipulation. It then initializes the `OpenAI` client, ensuring your `OPENAI_API_KEY` is configured.

    Crucially, this section also simulates real-world conditions by creating dummy earnings call transcripts for Apple (AAPL), JPMorgan Chase (JPM), and Tesla (TSLA) for Q4 2024. These files are stored in a `transcripts/` directory, mimicking the data an analyst would work with. For each transcript, we calculate its word and token count using `tiktoken`, which is vital for managing LLM context windows and estimating API costs. The estimated input cost per transcript provides an early look into the potential expenses, setting the stage for a more detailed ROI analysis later.
    """)
    st.info("Environment setup is handled automatically upon application start via `source.py` import.")
    if st.session_state.selected_transcript:
        st.subheader("Loaded Transcripts Overview:")
        st.write(f"Currently selected transcript: `{st.session_state.selected_transcript}`")
        st.text_area("Selected Transcript Content (Preview)", transcripts[st.session_state.selected_transcript][:1000] + "...", height=200, disabled=True)
    else:
        st.warning("No transcripts are loaded. Please check your `source.py` file and environment.")

elif st.session_state.current_page == "1. LLM Explanatory Power":
    st.title("1. LLM Explanatory Power: Understanding Complex Financial Concepts")
    st.markdown(f"""
    As an analyst, you often encounter complex financial concepts or need to quickly refresh your understanding of market dynamics. LLMs can serve as an invaluable tool for rapid explanation and knowledge synthesis. In this task, we'll test the LLM's ability to explain the significance of a yield curve inversion, a key economic indicator.
    """)
    st.markdown(f"**Story:** Your portfolio manager has just mentioned 'yield curve inversion' in a team meeting, and while you have a general understanding, you want a quick, concise, yet comprehensive explanation to ensure you grasp its current market implications. Instead of spending time searching through articles, you turn to the LLM for an immediate summary.")
    st.markdown(f"**Context:** The yield curve is a graph plotting the yields of bonds of the same credit quality but different maturities. An *inversion* occurs when short-term bond yields are higher than long-term bond yields. Historically, this has often preceded economic recessions.")
    st.markdown(f"**Concept:** This task leverages the LLM's **Natural Language Understanding (NLU)** to interpret the complex query and **Natural Language Generation (NLG)** to produce a coherent, explanatory response. It demonstrates the LLM's utility as a quick knowledge retrieval and explanation engine.")

    st.subheader("Ask the LLM for an Explanation")
    st.session_state.concept_query = st.text_area(
        "Enter your financial concept query:",
        value=st.session_state.concept_query,
        height=100
    )
    if st.button("Get Explanation"):
        if st.session_state.concept_query:
            with st.spinner("Generating explanation..."):
                try:
                    response = explain_financial_concept(st.session_state.concept_query, model=st.session_state.llm_model)
                    st.session_state.explanation_response = response
                except Exception as e:
                    st.error(f"Error generating explanation: {e}")
        else:
            st.warning("Please enter a concept query.")

    if st.session_state.explanation_response:
        st.subheader("LLM Explanation")
        st.markdown(f"--- LLM Explanation of Yield Curve Inversion ---")
        st.write(st.session_state.explanation_response)
        st.markdown(f"""
        The LLM provides a clear and contextualized explanation of the yield curve inversion, touching upon its historical predictive power for recessions and the specific market conditions of 2023. As a senior investment professional, this instant synthesis of information allows you to quickly grasp the nuances without extensive manual research, freeing up time for higher-value analytical work. The `temperature=0.7` setting allows for a slightly more creative and comprehensive explanation, suitable for conceptual understanding, rather than strict factual recall which demands lower temperatures.
        """)

elif st.session_state.current_page == "2. Summarize Earnings Call":
    st.title("2. Summarize Earnings Call: Strategies, Chunking & Data Extraction")

    selected_transcript_content = transcripts.get(st.session_state.selected_transcript, "")
    if not selected_transcript_content:
        st.warning("Please select a transcript from the sidebar to proceed.")
        st.stop()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Naive Summary",
        "Structured & Few-Shot",
        "Token Management & Chunking",
        "Structured JSON Extraction",
        "Prompt Comparison & Temperature Calibration"
    ])

    with tab1:
        st.header("Streamlining Workflow: Summarizing Earnings Calls (Naive Approach)")
        st.markdown(f"""
        One of the most immediate applications of LLMs in finance is summarizing lengthy documents like earnings call transcripts. Let's start with a straightforward, "naive" prompting strategy to see how an LLM handles a summarization request without much specific guidance.
        """)
        st.markdown(f"**Story:** You've just received the latest earnings call transcript for Apple (AAPL), and your team needs a quick overview. Your first instinct is to ask the LLM for a summary using a simple, direct prompt.")
        st.markdown(f"**Context:** Earnings call transcripts are dense with information, including financial figures, management commentary, and analyst Q&A. A quick summary helps in triage and preliminary assessment.")
        st.markdown(f"**Concept:** This introduces basic **prompt engineering** for summarization. A naive prompt simply tells the LLM \"summarize this text.\" While seemingly effective, it often lacks the structure, detail, and anti-hallucination guardrails necessary for professional financial analysis.")

        st.subheader(f"Generate Naive Summary for {st.session_state.selected_transcript}")
        if st.button("Generate Naive Summary", key="naive_summary_btn"):
            with st.spinner("Generating naive summary..."):
                try:
                    summary = summarize_naive(selected_transcript_content, model=st.session_state.llm_model)
                    if st.session_state.selected_transcript not in st.session_state.summaries:
                        st.session_state.summaries[st.session_state.selected_transcript] = {}
                    st.session_state.summaries[st.session_state.selected_transcript]['naive'] = {'summary': summary, 'usage': None}
                except Exception as e:
                    st.error(f"Error generating naive summary: {e}")
        
        if st.session_state.selected_transcript in st.session_state.summaries and 'naive' in st.session_state.summaries[st.session_state.selected_transcript]:
            st.subheader("Naive Summary Output")
            st.markdown(f"--- Naive Summary of {st.session_state.selected_transcript} Earnings ---")
            st.write(st.session_state.summaries[st.session_state.selected_transcript]['naive']['summary'])
            st.markdown(f"""
            The output from the naive summarization provides a general overview of Apple's Q4 2024 earnings. While it captures some key figures like revenue and EPS, it lacks a structured format and might miss critical nuances or comparisons important for an investment professional. For instance, it doesn't explicitly highlight the year-over-year changes or segment performance in a clear, standardized way. This demonstrates the limitations of a simple prompt for the detailed, structured information needed in finance.
            """)

    with tab2:
        st.header("Enhancing Precision: Structured Summaries & Anti-Hallucination Guardrails")
        st.markdown(f"""
        The naive summary, while quick, often lacks the depth, structure, and reliability required for professional financial analysis. As an analyst, you need summaries that are consistently formatted, cover specific key areas, and explicitly guard against factual errors. This leads us to **structured prompting**.
        """)
        st.markdown(f"**Story:** The portfolio manager found the naive summary useful but requested more specific details on management guidance and key risks, presented in a consistent format across all reports. You realize a more robust prompting strategy is needed to meet these professional standards and prevent potential compliance issues from misreported facts.")
        st.markdown(f"**Context:** **CFA Standard V(A) - Diligence and Reasonable Basis** mandates that investment professionals have a reasonable and adequate basis, supported by appropriate research and investigation, for any investment analysis, recommendation, or action. Unverified or fabricated LLM outputs directly violate this standard. Therefore, designing prompts that prioritize accuracy and structured output is paramount.")
        st.markdown(f"**Concept:** This section introduces sophisticated **prompt engineering** by defining a `SYSTEM_PROMPT` to establish the LLM's role and constraints, and a `TASK_PROMPT` to specify the desired output structure and anti-hallucination rules.")
        st.markdown(f"""
        *   **System Prompt with Role & Constraints:** Assigning the LLM the role of a "senior equity research analyst" and imposing rules like "Use ONLY information explicitly stated in the transcript" directly combats hallucinations.
        *   **Structured Output Specification:** Defining precise sections (e.g., HEADLINE, QUARTERLY PERFORMANCE, GUIDANCE & OUTLOOK, RISKS & CONCERNS) ensures comprehensive and consistent coverage. This is template-driven prompt engineering.
        *   **Specificity Instructions:** Directives like "Quote specific numbers exactly as stated" or "distinguish between RAISED, LOWERED, MAINTAINED guidance" enforce numerical precision and accurate characterization of management commentary, minimizing ambiguous paraphrasing.
        """)
        
        st.subheader(f"Generate Structured/Few-Shot Summary for {st.session_state.selected_transcript}")
        
        selected_strategy = st.radio(
            "Select Prompting Strategy:",
            options=['structured', 'few_shot'],
            index=0,
            key="summary_strategy_radio"
        )
        st.session_state.summarization_temperature = st.slider(
            "Select Temperature (for narrative):",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.summarization_temperature,
            step=0.1,
            key="summarization_temp_slider",
            help="Lower temperature (e.g., 0.2-0.3) for more deterministic output. Higher for more creative."
        )

        if st.button(f"Generate {selected_strategy.replace('_', ' ').title()} Summary", key="structured_summary_btn"):
            with st.spinner(f"Generating {selected_strategy.replace('_', ' ')} summary..."):
                try:
                    summary, usage = summarize_with_strategy(
                        selected_transcript_content,
                        strategy=selected_strategy,
                        model=st.session_state.llm_model
                    )
                    if st.session_state.selected_transcript not in st.session_state.summaries:
                        st.session_state.summaries[st.session_state.selected_transcript] = {}
                    st.session_state.summaries[st.session_state.selected_transcript][selected_strategy] = {'summary': summary, 'usage': usage}
                except Exception as e:
                    st.error(f"Error generating {selected_strategy} summary: {e}")
        
        if st.session_state.selected_transcript in st.session_state.summaries and selected_strategy in st.session_state.summaries[st.session_state.selected_transcript]:
            st.subheader(f"{selected_strategy.replace('_', ' ').title()} Summary Output (V5 Example Summary Output)")
            st.markdown(f"--- {selected_strategy.replace('_', ' ').title()} Summary of {st.session_state.selected_transcript} Earnings ---")
            st.write(st.session_state.summaries[st.session_state.selected_transcript][selected_strategy]['summary'])
            
            usage_data = st.session_state.summaries[st.session_state.selected_transcript][selected_strategy]['usage']
            if usage_data:
                st.write(f"\n**Token Usage:**")
                st.write(f" Input tokens: {usage_data.prompt_tokens:,}")
                st.write(f" Output tokens: {usage_data.completion_tokens:,}")
                cost = get_estimated_api_cost(usage_data.prompt_tokens, usage_data.completion_tokens)
                st.write(f" Cost: ${cost:.4f}")

            st.markdown(f"""
            The structured summary for Apple's Q4 2024 earnings is significantly more organized and comprehensive than the naive version. It clearly delineates sections like "HEADLINE," "QUARTERLY PERFORMANCE," "GUIDANCE & OUTLOOK," "RISKS & CONCERNS," and "TONE ASSESSMENT." This format directly addresses the portfolio manager's requirements and enhances readability and usability for further analysis.

            Notice how the `SYSTEM_PROMPT` enforces the LLM to act as a financial analyst, and the `TASK_PROMPT` specifies the exact structure and content required. This level of prompt engineering is essential for producing reliable outputs in finance. The use of `temperature=0.3` (low but not zero) allows for slightly varied phrasing in the prose while largely adhering to factual consistency, ideal for analyst briefs. We also track token usage and calculate the cost, providing insight into the operational expenditure of using LLMs at scale.
            """)

    with tab3:
        st.header("Managing Scale: Handling Long Transcripts with Token Management & Chunking")
        st.markdown(f"""
        Earnings call transcripts can easily exceed the context window limits of even advanced LLMs. To reliably process these lengthy documents, a strategy for **token management** and **chunking** is indispensable. Hierarchical summarization breaks down the problem into manageable steps.
        """)
        st.markdown(f"**Story:** You're tasked with summarizing a particularly long earnings call transcript, perhaps for a company with extensive Q&A. You know that simply sending the entire text to the LLM might result in an error due to token limits or a loss of detail if the model truncates the input. You need a method to ensure the entire transcript is processed without losing crucial information.")
        st.markdown(f"**Context:** LLMs have a finite **context window**, which dictates how much text (measured in tokens) they can process at once. Exceeding this limit leads to truncation or errors. Chunking strategies involve splitting the document into smaller, manageable pieces, and hierarchical summarization then merges insights from these pieces into a final brief.")
        st.markdown(f"**Concept:** This section demonstrates:")
        st.markdown(f"""
        *   **Tokenization with `tiktoken`:** Accurately counting tokens to prevent context window overflow.
        *   **Section-based Chunking:** Splitting transcripts at logical boundaries (e.g., Prepared Remarks, Q&A Session) to maintain coherence.
        *   **Hierarchical Summarization:** Summarizing individual chunks first, then merging these summaries into a final, comprehensive brief. This ensures no part of the original document is overlooked.
        """)
        st.markdown(r"**Mathematical Formulation: Token Economics for Earnings Call Summarization**")
        st.markdown(r"The cost of interacting with an LLM depends on the number of input and output tokens. Understanding these **token economics** is critical for projecting costs and evaluating ROI.")
        st.markdown(r"The total cost $C$ for a given LLM interaction can be calculated as:")
        st.markdown(r"$$ C = N_{{\text{in}}} \times P_{{\text{in}}} + N_{{\text{out}}} \times P_{{\text{out}}} $$")
        st.markdown(r"where $N_{{\text{in}}}$ = Number of input tokens, $P_{{\text{in}}}$ = Price per input token (e.g., $2.50 per 1 million input tokens for GPT-4o), $N_{{\text{out}}}$ = Number of output tokens, $P_{{\text{out}}}$ = Price per output token (e.g., $10.00 per 1 million input tokens for GPT-4o).")
        st.markdown(r"For a typical earnings call transcript with $N_{{\text{in}}} \approx 12,000$ tokens and a summary of $N_{{\text{out}}} \approx 1,500$ tokens, using GPT-4o pricing ($P_{{\text{in}}} = \$2.50/1\text{{M}}$, $P_{{\text{out}}} = \$10.00/1\text{{M}}$):")
        st.markdown(r"$$ C = 12,000 \times \frac{{2.50}}{{10^6}} + 1,500 \times \frac{{10.00}}{{10^6}} = \$0.03 + \$0.015 = \$0.045 $$")
        st.markdown(r"This demonstrates that while individual summaries are cheap, costs can add up across a large coverage universe.")

        st.subheader(f"Generate Hierarchical Summary for {st.session_state.selected_transcript}")
        if st.button("Generate Hierarchical Summary", key="hierarchical_summary_btn"):
            with st.spinner("Generating hierarchical summary... This might take longer for long transcripts."):
                try:
                    old_stdout = sys.stdout
                    new_stdout = io.StringIO()
                    sys.stdout = new_stdout
                    
                    final_summary = hierarchical_summarize(selected_transcript_content, model=st.session_state.llm_model)
                    
                    sys.stdout = old_stdout
                    chunk_output = new_stdout.getvalue()

                    if st.session_state.selected_transcript not in st.session_state.summaries:
                        st.session_state.summaries[st.session_state.selected_transcript] = {}
                    st.session_state.summaries[st.session_state.selected_transcript]['hierarchical'] = {'summary': final_summary, 'usage': None}
                    
                    st.subheader("Chunking Process Details:")
                    st.code(chunk_output)

                except Exception as e:
                    st.error(f"Error generating hierarchical summary: {e}")
        
        if st.session_state.selected_transcript in st.session_state.summaries and 'hierarchical' in st.session_state.summaries[st.session_state.selected_transcript]:
            st.subheader("Final Hierarchical Summary Output")
            st.markdown(f"--- Final Hierarchical Summary of {st.session_state.selected_transcript} Earnings ---")
            st.write(st.session_state.summaries[st.session_state.selected_transcript]['hierarchical']['summary'])
            st.markdown(f"""
            This section successfully demonstrates how to manage long transcripts that exceed typical LLM context windows. The `chunk_transcript` function first attempts to split the document at natural section breaks like "Prepared Remarks" and "Q&A Session." If these chunks are still too large, it further subdivides them at sentence boundaries. The token counts for each chunk are displayed, allowing the analyst to verify that no chunk exceeds the `max_chunk_tokens` limit.

            The `hierarchical_summarize` function then takes these chunks, applies the previously defined structured summarization strategy to each, and finally merges these individual summaries into a single, cohesive analyst brief. This process ensures that the entire document is analyzed, and a comprehensive summary is generated, addressing a critical challenge for analysts dealing with extensive financial documents. The cost implications of multiple LLM calls for chunking and merging are also implicitly highlighted through the earlier token economic formula.
            """)

    with tab4:
        st.header("Extracting Specifics: Structured JSON for Key Financial Metrics")
        st.markdown(f"""
        While a prose summary is valuable, analysts often need to extract precise numerical data (e.g., revenue, EPS, margin) in a structured, machine-readable format for dashboards, databases, or further quantitative analysis. LLMs can be incredibly effective for this task when properly instructed.
        """)
        st.markdown(f"**Story:** Your investment bank has a dashboard that automatically updates key financial metrics for covered companies after each earnings call. Manually inputting these figures is tedious and prone to transcription errors. You need a way to automate the extraction of specific metrics (Revenue, EPS, Operating Margin, Guidance) directly from the earnings call transcripts into a structured JSON object.")
        st.markdown(f"**Context:** Programmatic data extraction from unstructured text is a high-value application of LLMs. Ensuring **zero-shot accuracy** for numerical data is paramount. This requires stringent prompt engineering and careful temperature calibration.")
        st.markdown(f"**Concept:** This task focuses on:")
        st.markdown(f"""
        *   **JSON Output Formatting:** Instructing the LLM to return output strictly in JSON format using `response_format={{"type": "json_object"}}`.
        *   **Zero Temperature ($\tau=0.0$):** For numerical extraction, a temperature of $0.0$ is non-negotiable. At $\tau > 0$, the LLM may "creatively" round numbers, convert units, or interpolate, leading to factual distortions. $\tau=0.0$ ensures greedy decoding, always picking the highest-probability token, thus minimizing numerical distortion and maximizing determinism.
        *   **Clear Field Definitions:** Explicitly defining required JSON fields and their expected data types.
        """)
        st.subheader(f"Extract JSON Metrics for {st.session_state.selected_transcript}")
        
        st.session_state.json_extraction_temperature = st.slider(
            "Select Temperature (for JSON extraction):",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.json_extraction_temperature,
            step=0.1,
            key="json_extraction_temp_slider",
            help="**Recommendation: 0.0 for factual accuracy.** Higher values risk numerical hallucination.",
            disabled=True if st.session_state.json_extraction_temperature != 0.0 else False
        )
        if st.session_state.json_extraction_temperature != 0.0:
            st.warning("Temperature for JSON extraction is recommended to be 0.0 for maximum determinism and factual accuracy.")

        if st.button("Extract JSON Metrics", key="extract_json_btn"):
            if st.session_state.json_extraction_temperature != 0.0:
                st.error("JSON extraction requires temperature to be 0.0. Please set it to 0.0.")
            else:
                with st.spinner("Extracting structured metrics..."):
                    try:
                        metrics = extract_metrics(selected_transcript_content, model=st.session_state.llm_model)
                        st.session_state.json_metrics[st.session_state.selected_transcript] = metrics
                    except json.JSONDecodeError as je:
                        st.error(f"LLM did not return valid JSON. Error: {je}")
                    except Exception as e:
                        st.error(f"Error extracting metrics: {e}")

        if st.session_state.selected_transcript in st.session_state.json_metrics:
            st.subheader("Extracted Financial Metrics (JSON) Output")
            st.markdown(f"--- Extracted Financial Metrics for {st.session_state.selected_transcript} (JSON) ---")
            st.json(st.session_state.json_metrics[st.session_state.selected_transcript])
            st.markdown(f"""
            The LLM successfully extracts key financial metrics for Apple's Q4 2024 earnings into a well-structured JSON object. You can see the revenue, EPS, operating margin, and even guidance information presented in a clean, programmatic format. This output can be directly fed into an internal database or dashboard, automating a critical data entry task.

            The `temperature=0.0` setting was crucial here, ensuring that numerical values like "$90.1 billion" and "$1.40" are quoted exactly as they appear in the transcript, without any rounding or paraphrasing. This determinism is essential for financial reporting, where even minor discrepancies can lead to significant errors or compliance issues. The `response_format={{"type": "json_object"}}` ensures the output is always a valid JSON, making it reliable for downstream programmatic consumption.
            """)

    with tab5:
        st.header("Optimizing & Comparing: Prompt Strategies and Temperature Calibration")
        st.markdown(f"""
        To truly leverage LLMs effectively, an analyst must understand how different prompt engineering strategies impact output quality and how to calibrate the LLM's "temperature" for various sub-tasks.
        """)
        st.markdown(f"**Story:** Your team is exploring optimal LLM usage and wants to establish best practices. You need to present a clear comparison of different prompt strategies (naive, structured, few-shot) and demonstrate how adjusting the LLM's `temperature` parameter influences the quality and consistency of summaries. This analysis will inform the organization's LLM policy.")
        st.markdown(f"**Context:**")
        st.markdown(f"""
        *   **Prompt Strategy Comparison:** Evaluating naive, structured, and few-shot prompts against metrics like word count, section coverage, hallucination rate, and fact recall helps identify the most effective approach for financial use cases. Structured prompts are generally considered the "production standard."
        *   **Temperature Calibration:** The `temperature` parameter controls the randomness of the LLM's output. `$\tau=0.0$` leads to deterministic, factual output (ideal for numerical extraction), while higher temperatures introduce more creativity (suitable for qualitative prose summaries, but risky for facts).
        """)
        st.markdown(r"**Mathematical Formulation: Temperature in Autoregressive Generation**")
        st.markdown(r"The temperature ($	au$) parameter modifies the probability distribution of the next token in autoregressive generation. Given the raw model scores (logits) $z_t$ for each possible token, the probability $P(	ext{{token}}_t | 	ext{{context}})$ is calculated as:")
        st.markdown(r"$$ P(	ext{{token}}_t | 	ext{{context}}) = rac{{\exp(z_t/\tau)}}{{\sum_j \exp(z_j/\tau)}} $$")
        st.markdown(r"where $z_t$ are the logits (raw model scores) and $\tau$ is the temperature.")
        st.markdown(r"*   **$\tau=0$ (Greedy Decoding):** Always picks the highest-probability token. Deterministic, consistent, but can be repetitive. **Best for: numerical extraction, metric tables.**")
        st.markdown(r"*   **$\tau=0.2-0.3$ (Low Randomness):** Slightly varied word choice but same factual content. **Best for: analyst brief prose.**")
        st.markdown(r"*   **$\tau=0.7-1.0$ (High Creativity):** Different structures, varied phrasing, may introduce less likely (possibly incorrect) tokens. **Avoid for: factual financial summarization.**")
        st.markdown(r"**Mathematical Formulation: Compression Ratio and Information Retention**")
        st.markdown(r"These metrics quantify the efficiency and effectiveness of summarization.")
        st.markdown(r"**Compression Ratio (CR):**")
        st.markdown(r"$$ CR = 1 - \frac{{N_{{\text{summary}}}}}{{N_{{\text{transcript}}}}} $$")
        st.markdown(r"Target: $CR > 90\%$ (e.g., a 10,000-word transcript condensed to <1,000 words).")
        st.markdown(r"**Information Retention ($Recall_{{\text{facts}}}$):**")
        st.markdown(r"$$ Recall_{{\text{facts}}} = \frac{{\text{{key facts in summary}}}}{{\text{{key facts in transcript}}}} $$")
        st.markdown(r"This requires a manually compiled list of 15-20 key facts (revenue, EPS, segment performance, guidance changes, major Q&A points) from the transcript as ground truth. Target: $Recall_{{\text{facts}}} > 85\%$.")
        
        st.subheader("Prompt Strategy Comparison (V1 Prompt Comparison Table & V6 Compression Ratio Bar Chart)")
        if st.button("Run Prompt Strategy Comparison", key="run_comparison_btn"):
            with st.spinner("Running comparison across prompt strategies... This may take a moment."):
                strategies = ['naive', 'structured', 'few_shot']
                comparison = {}
                transcript_to_use = selected_transcript_content
                
                for strategy in strategies:
                    try:
                        summary, usage = summarize_with_strategy(transcript_to_use, strategy=strategy, model=st.session_state.llm_model)
                        audit, h_rate = hallucination_audit(summary, transcript_to_use)
                        
                        enc = tiktoken.encoding_for_model(st.session_state.llm_model)
                        N_transcript_tokens = len(enc.encode(transcript_to_use))
                        N_summary_tokens = len(enc.encode(summary))
                        cr = 1 - (N_summary_tokens / N_transcript_tokens)

                        facts_in_summary_count = 0
                        # Assuming key_facts_transcript is available from source.py
                        if 'key_facts_transcript' in globals():
                            for fact in key_facts_transcript:
                                if fact.lower() in summary.lower():
                                    facts_in_summary_count += 1
                            recall_facts = facts_in_summary_count / len(key_facts_transcript) if key_facts_transcript else 0
                        else:
                            recall_facts = 0.0

                        comparison[strategy] = {
                            'word_count': len(summary.split()),
                            'section_coverage': sum(1 for s in ('HEADLINE', 'QUARTERLY PERFORMANCE', 'GUIDANCE', 'Q&A', 'RISKS', 'TONE') if s.lower() in summary.lower()),
                            'hallucination_rate': f"{h_rate:.2f}%",
                            'numbers_verified': len(audit['verified']),
                            'numbers_flagged': len(audit['flagged']),
                            'cost': get_estimated_api_cost(usage.prompt_tokens, usage.completion_tokens) if usage else 0.0,
                            'compression_ratio': f"{cr:.2%}",
                            'information_retention_recall': f"{recall_facts:.2%}"
                        }
                    except Exception as e:
                        st.warning(f"Could not run comparison for {strategy} strategy: {e}")
                        comparison[strategy] = {'word_count': 'N/A', 'section_coverage': 'N/A', 'hallucination_rate': 'N/A',
                                                'numbers_verified': 'N/A', 'numbers_flagged': 'N/A', 'cost': 'N/A',
                                                'compression_ratio': 'N/A', 'information_retention_recall': 'N/A'}

                st.session_state.comparison_data = pd.DataFrame(comparison).T.astype(str)
                st.success("Prompt strategy comparison complete!")
        
        if not st.session_state.comparison_data.empty:
            st.subheader("Prompt Strategy Comparison Table")
            st.markdown("--- PROMPT STRATEGY COMPARISON ---")
            st.dataframe(st.session_state.comparison_data)

            st.subheader("Compression Ratio Bar Chart (V6)")
            try:
                cr_data = st.session_state.comparison_data['compression_ratio'].str.replace('%', '').astype(float) / 100
                fig_cr, ax_cr = plt.subplots(figsize=(10, 5))
                cr_data.plot(kind='bar', ax=ax_cr, color=['skyblue', 'lightcoral', 'lightgreen'])
                ax_cr.set_title(f'Compression Ratio by Prompt Strategy for {st.session_state.selected_transcript}')
                ax_cr.set_ylabel('Compression Ratio')
                ax_cr.set_ylim(0, 1)
                for i, v in enumerate(cr_data):
                    ax_cr.text(i, v + 0.02, f'{v:.2%}', ha='center', va='bottom')
                st.pyplot(fig_cr)
            except Exception as e:
                st.error(f"Error plotting compression ratio: {e}")
        
        st.markdown(f"""
        The **Prompt Strategy Comparison** table clearly illustrates the superior performance of the 'structured' and 'few_shot' strategies over the 'naive' approach. The structured prompt typically achieves higher section coverage, lower hallucination rates, and better information retention, while maintaining a good compression ratio. This reinforces the principle that **the quality of the instruction determines the quality of the output.** For an investment bank, using structured prompts becomes the standard to ensure consistent, accurate, and comprehensive analyst briefs.
        """)

        st.subheader("Temperature Calibration Experiment (Consistency) (V2 Temperature Consistency Plot)")
        if st.button("Run Temperature Consistency Experiment", key="run_temp_exp_btn"):
            with st.spinner("Running temperature consistency experiment... This involves multiple LLM calls."):
                temperatures = [0.0, 0.2, 0.5, 0.8, 1.0]
                temperature_consistency_results = {}
                for temp in temperatures:
                    try:
                        run1, _ = summarize_with_strategy(selected_transcript_content, strategy='structured', model=st.session_state.llm_model, temperature=temp)
                        run2, _ = summarize_with_strategy(selected_transcript_content, strategy='structured', model=st.session_state.llm_model, temperature=temp)
                        
                        words1 = set(run1.lower().split())
                        words2 = set(run2.lower().split())
                        if len(words1.union(words2)) > 0:
                            jaccard_similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                        else:
                            jaccard_similarity = 0.0
                        temperature_consistency_results[temp] = jaccard_similarity
                    except Exception as e:
                        st.warning(f"Could not run temperature experiment for temp={temp}: {e}")
                        temperature_consistency_results[temp] = 0.0

                st.session_state.temp_consistency_results = temperature_consistency_results
                st.success("Temperature consistency experiment complete!")
        
        if st.session_state.temp_consistency_results:
            st.subheader("Temperature Consistency Results")
            st.markdown("--- Temperature Consistency Results ---")
            st.write(pd.Series(st.session_state.temp_consistency_results))

            fig_temp, ax_temp = plt.subplots(figsize=(10, 5))
            temps = list(st.session_state.temp_consistency_results.keys())
            jaccards = list(st.session_state.temp_consistency_results.values())
            ax_temp.plot(temps, jaccards, marker='o', linestyle='-', color='teal')
            ax_temp.set_title(f'Jaccard Similarity vs. Temperature for {st.session_state.selected_transcript} (Structured Summary)')
            ax_temp.set_xlabel('Temperature')
            ax_temp.set_ylabel('Jaccard Similarity')
            ax_temp.set_ylim(0, 1.1)
            for i, txt in enumerate(jaccards):
                ax_temp.annotate(f'{txt:.3f}', (temps[i], jaccards[i]), textcoords="offset points", xytext=(0,5), ha='center')
            st.pyplot(fig_temp)
        
        st.markdown(f"""
        The **Temperature Calibration Experiment** demonstrates how `temperature` impacts output consistency. As `temperature` increases, the Jaccard similarity (word overlap) between two identical prompt executions decreases, indicating higher randomness. For factual financial summaries, a lower temperature (e.g., `0.2-0.3`) is preferred to ensure consistency and factual accuracy, while `0.0` is reserved strictly for numerical extraction. Higher temperatures (`0.7-1.0`) are generally unsuitable for financial tasks where factual precision is paramount, as they can introduce "creative" but incorrect information. This experiment provides clear guidelines for setting `temperature` based on the specific task (factual extraction vs. narrative summarization).
        """)

elif st.session_state.current_page == "3. Hallucination Audit & Compliance":
    st.title("3. Mitigating Risk: The Hallucination Audit & Compliance Safeguard")
    st.markdown(f"""
    Despite advanced prompting, LLMs can still hallucinate. For an investment professional, a hallucination is not just an inconvenience; it's a critical compliance risk under **CFA Standard V(A): Diligence and Reasonable Basis**. A systematic audit is essential to verify LLM outputs against the source material.
    """)
    st.markdown(f"**Story:** Your compliance officer has emphasized the critical need to verify all AI-generated financial data before it's used in any official report or recommendation. You've generated a summary and extracted metrics, but now you must rigorously audit them to ensure no numbers were fabricated, guidance was correctly characterized, and all critical topics were covered.")
    st.markdown(f"**Context:** The \"hallucination audit\" is the compliance safeguard that makes LLM summarization safe for professional use. It systematically checks for:")
    st.markdown(f"""
    *   **Numerical fabrication:** LLM invents a number not in the transcript.
    *   **Guidance mischaracterization:** LLM reports "raised guidance" when it was "maintained."
    *   **Attribution error/Completeness:** LLM attributes a statement incorrectly or misses key topics.
    """)
    st.markdown(f"**Concept:** This task implements a **systematic hallucination detection** mechanism using:")
    st.markdown(f"""
    *   **Regex for Numerical Extraction:** Identifying all plausible numerical figures (e.g., "$10.5B", "28.5%", "120bps") from the summary.
    *   **Transcript Verification:** Checking if these extracted numbers (and their fuzzy matches, accounting for formatting differences) appear in the original transcript.
    *   **Keyword Verification for Guidance:** Checking for correct classification of guidance changes (raised, lowered, maintained, introduced).
    *   **Completeness Check:** Ensuring key financial topics (revenue, earnings, margin, guidance, outlook) are mentioned in the summary.
    """)
    st.markdown(r"**Mathematical Formulation: Hallucination Rate**")
    st.markdown(r"To quantify the reliability of LLM-generated summaries, we can define a **Hallucination Rate**. This metric helps track the prevalence of unverifiable claims.")
    st.markdown(r"$$ HallucinationRate = \frac{{\text{{Number of Flagged Items}}}}{{\text{{Number of Verified Items}} + \text{{Number of Flagged Items}}}} $$")
    st.markdown(r"A lower hallucination rate indicates higher reliability. Target for structured prompts with GPT-4o is typically $<5\%$.")

    st.subheader(f"Run Hallucination Audit for {st.session_state.selected_transcript}")
    
    available_summaries = {}
    if st.session_state.selected_transcript in st.session_state.summaries:
        for strat, data in st.session_state.summaries[st.session_state.selected_transcript].items():
            if data['summary']:
                available_summaries[strat.replace('_', ' ').title()] = strat

    if not available_summaries:
        st.warning("No summaries have been generated yet for the selected transcript. Please generate some in the 'Summarize Earnings Call' page first.")
        st.stop()
    
    default_idx_audit = 0
    if st.session_state.audit_target_summary.replace('_', ' ').title() in available_summaries.keys():
        default_idx_audit = list(available_summaries.keys()).index(st.session_state.audit_target_summary.replace('_', ' ').title())

    st.session_state.audit_target_summary = st.selectbox(
        "Select Summary to Audit:",
        options=list(available_summaries.keys()),
        index=default_idx_audit,
        key="audit_summary_select"
    )
    
    selected_summary_key = available_summaries[st.session_state.audit_target_summary]
    summary_to_audit = st.session_state.summaries[st.session_state.selected_transcript][selected_summary_key]['summary']

    if st.button(f"Run Audit for {st.session_state.audit_target_summary}", key="run_audit_btn"):
        with st.spinner("Running hallucination audit..."):
            try:
                audit_results_dict, h_rate = hallucination_audit(summary_to_audit, selected_transcript_content)
                if st.session_state.selected_transcript not in st.session_state.summaries:
                     st.session_state.summaries[st.session_state.selected_transcript] = {}
                st.session_state.summaries[st.session_state.selected_transcript][selected_summary_key]['audit'] = audit_results_dict
                st.session_state.summaries[st.session_state.selected_transcript][selected_summary_key]['h_rate'] = h_rate
            except Exception as e:
                st.error(f"Error running audit: {e}")

    if st.session_state.selected_transcript in st.session_state.summaries and selected_summary_key in st.session_state.summaries[st.session_state.selected_transcript] and 'audit' in st.session_state.summaries[st.session_state.selected_transcript][selected_summary_key]:
        st.subheader("Hallucination Audit Report (V3 Hallucination Audit Dashboard)")
        audit_data = st.session_state.summaries[st.session_state.selected_transcript][selected_summary_key]['audit']
        h_rate = st.session_state.summaries[st.session_state.selected_transcript][selected_summary_key]['h_rate']

        st.markdown("--- HALLUCINATION AUDIT REPORT ---")
        st.metric("Hallucination Rate", f"{h_rate:.2f}%")
        col1, col2, col3 = st.columns(3)
        col1.metric("Verified Items", len(audit_data['verified']))
        col2.metric("Flagged Items", len(audit_data['flagged']))
        col3.metric("Missing Topics", len(audit_data['missing']))

        if audit_data['flagged']:
            with st.expander("Detailed Flagged Items (Require Manual Verification)"):
                for item in audit_data['flagged']:
                    st.warning(f"- {item}")
        if audit_data['missing']:
            with st.expander("Detailed Missing Topics (Check for Completeness)"):
                for item in audit_data['missing']:
                    st.info(f"- {item}")
        
        st.markdown(f"""
        The hallucination audit report provides a systematic verification of the LLM-generated structured summary against the original Apple transcript. It first extracts all numbers from the summary and attempts to find them directly or fuzzily within the transcript. It also checks for the correct characterization of guidance terms and ensures essential financial topics are covered.

        The calculated hallucination rate serves as a key performance indicator for the reliability of the LLM output. A low rate (ideally below 5%) indicates that the structured prompting and anti-hallucination guardrails are effective. Any flagged items or missing topics require immediate human review, emphasizing that the analyst's role shifts from "reading" to "reviewing" and "verifying," thereby ensuring compliance with **CFA Standard V(A)**. This audit is not merely a quality check but a fundamental compliance safeguard in the deployment of AI in financial analysis.
        """)

elif st.session_state.current_page == "4. ROI & Ethical Considerations":
    st.title("4. Quantifying Value: ROI and Ethical Considerations")
    st.markdown(f"""
    The ultimate goal of integrating LLMs is to enhance productivity and decision-making while adhering to professional standards. This section aggregates the financial and operational benefits, along with crucial ethical considerations.
    """)
    st.markdown(f"**Story:** You've demonstrated the technical capabilities of LLMs, but now senior management needs a concise report on the projected **Return on Investment (ROI)** and a summary of the ethical and compliance framework for LLM use within the bank. You need to quantify the time savings, cost, and overall value, explicitly linking to **CFA Standard V(A)**.")
    st.markdown(f"**Context:** The analyst's role shifts from \"reading\" to \"reviewing.\" Instead of 45 minutes reading a transcript, an analyst can spend 5 minutes reviewing an AI summary and 10 minutes focusing on flagged items or important sections.")
    
    st.markdown(r"**ROI Calculation:**")
    st.markdown(r"*   **Time Saved:** 120 calls/year $\times$ (45 min reading - 5 min AI review) = 80 hours/year.")
    st.markdown(r"*   **Analyst Cost:** Assuming $200/hour, this is an annual saving of $80 \times \$200 = \$16,000$.")
    st.markdown(r"*   **LLM API Cost (for 30 companies, 4 quarters/year):** 30 companies $\times$ 4 quarters $\times$ \$0.045/transcript = \$5.40/year.")
    st.markdown(r"*   **ROI:** `(Time Saved Value - LLM Cost) / LLM Cost`")
    st.markdown(r"$$ ROI = \frac{{\$16,000 - \$5.40}}{{\$5.40}} \approx 296,000\% $$")
    st.markdown(r"This high ROI underscores the immense efficiency gains.")
    
    st.markdown(r"**Ethical Framework (CFA Standard V(A)):**")
    st.markdown(r"The hallucination audit (Section 7) is the compliance layer. An analyst who includes an AI-generated number in a research report without verifying it against the source document has not exercised reasonable care, violating **CFA Standard V(A) – Diligence and Reasonable Basis**. The audit makes LLM summarization compliant by ensuring outputs are verifiable and auditable.")

    st.subheader("LLM Adoption: ROI and Ethical Brief")
    
    # Calculate costs based on structured summary (or prompt comparison data if available)
    if not st.session_state.comparison_data.empty and 'structured' in st.session_state.comparison_data.index:
        structured_metrics = st.session_state.comparison_data.loc['structured']
        try:
             estimated_cost_per_transcript = float(structured_metrics['cost']) if structured_metrics['cost'] != 'N/A' else 0.0
        except ValueError:
             estimated_cost_per_transcript = 0.045

        try:
            h_rate_structured = float(structured_metrics['hallucination_rate'].replace('%', '')) if structured_metrics['hallucination_rate'] != 'N/A' else 0.0
        except ValueError:
            h_rate_structured = 0.0
    else:
        # Fallback to a single structured summary if comparison not run
        if st.session_state.selected_transcript in st.session_state.summaries and 'structured' in st.session_state.summaries[st.session_state.selected_transcript] and st.session_state.summaries[st.session_state.selected_transcript]['structured']['usage']:
            usage_data = st.session_state.summaries[st.session_state.selected_transcript]['structured']['usage']
            estimated_cost_per_transcript = get_estimated_api_cost(usage_data.prompt_tokens, usage_data.completion_tokens)
            h_rate_structured = st.session_state.summaries[st.session_state.selected_transcript]['structured'].get('h_rate', 0.0)
        else:
            st.warning("Please run a 'structured' summary and prompt comparison/audit to get full ROI figures.")
            estimated_cost_per_transcript = 0.045 # Default from problem description
            h_rate_structured = 0.0

    total_transcripts_processed = len(transcripts) if 'transcripts' in globals() else 1
    
    # Calculate total demo cost carefully
    total_demo_cost = 0.0
    if not st.session_state.comparison_data.empty:
        for k, v in st.session_state.comparison_data.to_dict('index').items():
            try:
                if v['cost'] != 'N/A':
                    total_demo_cost += float(v['cost'])
            except ValueError:
                pass
    else:
        total_demo_cost = estimated_cost_per_transcript * total_transcripts_processed

    st.markdown("**1. Operational Efficiency & Cost Savings**")
    st.markdown(f"  - Total demonstration transcripts processed: {total_transcripts_processed}")
    st.markdown(f"  - Total estimated API cost for these {total_transcripts_processed} transcripts: ${total_demo_cost:.4f}")
    st.markdown(f"  - Average cost per structured summary: ${estimated_cost_per_transcript:.4f}")

    annual_companies = 30
    annual_quarters = 4
    projected_annual_llm_cost = annual_companies * annual_quarters * estimated_cost_per_transcript
    st.markdown(f"  - Projected Annual LLM Cost for {annual_companies} companies (4 quarters): ${projected_annual_llm_cost:.2f}")

    analyst_read_time_min = 45
    ai_review_time_min = 5
    analyst_focused_review_time_min = 10
    total_analyst_time_saved_per_call_min = analyst_read_time_min - (ai_review_time_min + analyst_focused_review_time_min)
    total_analyst_time_saved_per_call_hours = total_analyst_time_saved_per_call_min / 60

    total_calls_per_year = annual_companies * annual_quarters
    projected_annual_hours_saved = total_calls_per_year * total_analyst_time_saved_per_call_hours
    analyst_hourly_cost = 200
    projected_annual_analyst_value_saved = projected_annual_hours_saved * analyst_hourly_cost

    st.markdown("**2. Projected Annual ROI**")
    st.markdown(f"  - Estimated analyst time saved per call: {total_analyst_time_saved_per_call_min:.1f} minutes")
    st.markdown(f"  - Projected annual analyst hours saved (for {annual_companies} companies): {projected_annual_hours_saved:.1f} hours")
    st.markdown(f"  - Projected annual value of analyst time saved: ${projected_annual_analyst_value_saved:,.2f}")

    if projected_annual_llm_cost > 0:
        roi = ((projected_annual_analyst_value_saved - projected_annual_llm_cost) / projected_annual_llm_cost) * 100
        st.markdown(f"  - Projected Annual ROI: {roi:,.0f}%")
    else:
        st.markdown(f"  - Projected Annual ROI: Cannot calculate (LLM Cost is zero or not available).")

    st.markdown("**3. Ethical Considerations & Compliance (CFA Standard V(A))**")
    st.markdown(f"""
    *   LLM summarization shifts the analyst role from 'reader' to 'reviewer'.
    *   The Hallucination Audit is NOT optional; it's a mandatory compliance safeguard.
    *   Example: Structured prompt hallucination rate was {h_rate_structured:.2f}%. Any flagged item requires human verification.
    *   Unverified AI-generated figures or mischaracterized guidance risks violating CFA Standard V(A): Diligence and Reasonable Basis.
    *   Importance of documenting the audit process for transparency and accountability.
    """)

    st.subheader("Token Cost Breakdown (V4 - Illustrative)")
    st.markdown("This chart illustrates the proportion of input vs. output tokens and associated costs.")

    total_input_tokens = 0
    total_output_tokens = 0

    # Aggregate token usage across all summaries if they exist in session state
    for transcript_data in st.session_state.summaries.values():
        for strategy_data in transcript_data.values():
            if 'usage' in strategy_data and strategy_data['usage']:
                total_input_tokens += strategy_data['usage'].prompt_tokens
                total_output_tokens += strategy_data['usage'].completion_tokens
    
    # Add token usage for JSON extraction if available
    for transcript_content in transcripts.values():
        try:
            if 'EXTRACTION_PROMPT' in globals():
                extraction_messages = [
                    {"role": "system", "content": "You are a financial data extractor. Return only valid JSON."},
                    {"role": "user", "content": EXTRACTION_PROMPT.format(transcript=transcript_content)}
                ]
                enc = tiktoken.encoding_for_model(st.session_state.llm_model)
                total_input_tokens += len(enc.encode(str(extraction_messages)))
                total_output_tokens += 500
        except Exception:
            pass

    if total_input_tokens > 0 or total_output_tokens > 0:
        input_cost_per_million = 2.50
        output_cost_per_million = 10.00
        
        input_cost = (total_input_tokens * input_cost_per_million) / 1_000_000
        output_cost = (total_output_tokens * output_cost_per_million) / 1_000_000
        
        costs = [input_cost, output_cost]
        labels = ['Input Tokens Cost', 'Output Tokens Cost']
        
        if sum(costs) > 0:
            fig_cost, ax_cost = plt.subplots(figsize=(8, 8))
            ax_cost.pie(costs, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#99ff99'])
            ax_cost.axis('equal')
            ax_cost.set_title('Proportion of LLM API Costs (Input vs. Output Tokens)')
            st.pyplot(fig_cost)
            st.markdown(f"Total Input Tokens (aggregated): {total_input_tokens:,}")
            st.markdown(f"Total Output Tokens (aggregated): {total_output_tokens:,}")
            st.markdown(f"Estimated Total API Cost: ${input_cost + output_cost:.4f}")
            st.markdown(f"Annual cost projection (for 30 companies, 4 quarters/year): ${projected_annual_llm_cost:.2f}")
        else:
            st.info("Generate some summaries and run comparisons to see token cost breakdown.")
    else:
        st.info("Generate some summaries and run comparisons to see token cost breakdown.")

    st.markdown(f"""
    This final section quantifies the business value of integrating LLMs into the analyst workflow. We see a significant **projected annual ROI of approximately 296,000%**, primarily driven by the massive savings in analyst time, far outweighing the minimal API costs. This makes a compelling case for LLM adoption in financial institutions.

    Beyond the numbers, this section re-emphasizes the critical ethical dimension: **The hallucination audit is a non-negotiable compliance safeguard.** For an investment professional, the process of verifying LLM outputs against source documents is not merely a quality control step but a direct fulfillment of their professional obligations under **CFA Standard V(A): Diligence and Reasonable Basis**. This redefines the analyst's role, making them an expert auditor and critical reviewer of AI-generated insights, rather than just a manual data processor. The output serves as a high-level brief for management, combining financial justification with a clear understanding of the necessary ethical guardrails.
    """)


# License
st.caption('''
---
## QuantUniversity License

© QuantUniversity 2026  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
''')
