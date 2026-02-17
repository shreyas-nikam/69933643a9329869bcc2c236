# QuLab: Lab 25: LLM Prompting Demo - Revolutionizing Financial Analysis

## Project Title

**QuLab: Lab 25: LLM Prompting Demo: Revolutionizing Financial Analysis: LLM Capabilities & Hallucination Risk for Investment Professionals**

## Description

As a Senior Equity Research Analyst, synthesizing vast amounts of financial information rapidly and accurately is paramount. Earnings call transcripts, often tens of thousands of words long, are a critical source, but manually extracting insights is time-consuming and prone to human error, consuming hundreds of hours annually. This not only impacts efficiency but also raises compliance concerns, potentially violating **CFA Standard V(A): Diligence and Reasonable Basis**.

This Streamlit application, "QuLab: Lab 25," provides a hands-on laboratory experience demonstrating how Large Language Models (LLMs) can transform financial analysis workflows. It explores the power of LLMs for tasks like summarizing complex reports and explaining financial concepts, while critically addressing their limitations, particularly the risk of "hallucinations" (generating plausible but false information).

The lab guides users through a practical workflow for leveraging LLMs efficiently, coupled with robust methodologies for evaluating and safeguarding against inaccuracies. By the end, users will have a robust framework for integrating LLMs into their analytical workflow, armed with the knowledge to harness their power while meticulously mitigating inherent risks.

## Features

This application offers a comprehensive suite of features designed to explore LLM capabilities for financial analysis and risk mitigation:

1.  **Application Overview**: A detailed introduction to the lab's objectives, challenges addressed, and the value proposition of LLMs in finance.
2.  **LLM Explanatory Power**:
    *   Demonstrates how LLMs can quickly explain complex financial concepts (e.g., yield curve inversion) with contextual understanding.
    *   Explores the use of `temperature` for generating comprehensive conceptual explanations.
3.  **Summarize Earnings Call**: A multi-faceted section demonstrating various summarization techniques:
    *   **Naive Summary**: A baseline approach to quickly summarize transcripts.
    *   **Structured & Few-Shot Summary**: Advanced prompt engineering techniques using system prompts, structured output formats, and anti-hallucination guardrails to produce professional, compliant analyst briefs.
    *   **Token Management & Chunking**: Strategies for handling lengthy documents that exceed LLM context windows using hierarchical summarization and tokenization (`tiktoken`). Includes a mathematical formulation for token economics.
    *   **Structured JSON Extraction**: Programmatically extracting precise numerical financial metrics (Revenue, EPS, Margin, Guidance) into a machine-readable JSON format, emphasizing `temperature=0.0` for factual accuracy.
    *   **Prompt Comparison & Temperature Calibration**: Comparative analysis of different prompting strategies (naive, structured, few-shot) based on metrics like compression ratio and information retention. Visualizes the impact of `temperature` on output consistency using Jaccard similarity.
4.  **Hallucination Audit & Compliance**:
    *   Implements a critical compliance safeguard: a systematic audit to verify LLM-generated facts (numbers, guidance) against the original source transcript.
    *   Calculates a "Hallucination Rate" and highlights flagged items requiring human verification, directly addressing **CFA Standard V(A): Diligence and Reasonable Basis**.
5.  **ROI & Ethical Considerations**:
    *   Quantifies the potential Return on Investment (ROI) from adopting LLMs for financial analysis, projecting significant time and cost savings.
    *   Discusses the ethical framework, emphasizing the analyst's shift from "reader" to "reviewer" and the non-negotiable role of the hallucination audit in ensuring compliance and ethical AI usage.
    *   Illustrates token cost breakdown for transparency.

## Getting Started

Follow these instructions to set up and run the application on your local machine.

### Prerequisites

Before you begin, ensure you have the following installed:

*   **Python 3.8+**
*   An **OpenAI API Key**: You will need an API key from OpenAI to interact with their LLM models (e.g., GPT-4o, GPT-3.5-turbo).

### Installation

1.  **Clone the repository** (if this project is hosted in a repository):
    ```bash
    git clone <repository_url>
    cd <project_directory>
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required Python packages**:
    Create a `requirements.txt` file in your project root with the following content:
    ```
    streamlit
    pandas
    matplotlib
    openai
    tiktoken
    nltk # Potentially needed for advanced text processing in source.py
    ```
    Then, install them:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If `nltk` is not strictly used in `source.py` for chunking or other text processing, you can omit it from `requirements.txt`.*

4.  **Set your OpenAI API Key**:
    The application expects your OpenAI API key to be available as an environment variable named `OPENAI_API_KEY`.
    *   **Linux/macOS:**
        ```bash
        export OPENAI_API_KEY="your_openai_api_key_here"
        ```
    *   **Windows (Command Prompt):**
        ```bash
        set OPENAI_API_KEY="your_openai_api_key_here"
        ```
    *   **Windows (PowerShell):**
        ```bash
        $env:OPENAI_API_KEY="your_openai_api_key_here"
        ```
    Alternatively, you can place it directly in your Streamlit secrets file (`.streamlit/secrets.toml`):
    ```toml
    OPENAI_API_KEY="your_openai_api_key_here"
    ```

5.  **Prepare dummy transcript data**:
    The application expects earnings call transcripts. The provided `app.py` code imports `transcripts` from `source.py`. Ensure your `source.py` file correctly simulates or loads these transcripts. Typically, you'd have a `transcripts/` directory with `.txt` files (e.g., `AAPL_Q4_2024.txt`, `JPM_Q4_2024.txt`, `TSLA_Q4_2024.txt`). The `source.py` should contain logic to load these and related ground truth `key_facts_transcript`.

## Usage

To run the Streamlit application:

1.  Navigate to your project directory in the terminal.
2.  Ensure your virtual environment is activated and `OPENAI_API_KEY` is set.
3.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
    This will open the application in your default web browser.

### Basic Interaction

*   **Sidebar Navigation**: Use the sidebar on the left to navigate between different sections of the lab: "Application Overview", "1. LLM Explanatory Power", "2. Summarize Earnings Call", "3. Hallucination Audit & Compliance", and "4. ROI & Ethical Considerations".
*   **Global Settings**: In the sidebar, you can select the LLM model (`gpt-4o` or `gpt-3.5-turbo`) and the earnings transcript you wish to analyze.
*   **Interactive Demos**: Each page presents interactive elements (text areas, buttons, sliders) to demonstrate LLM functionalities. Follow the instructions and stories provided on each page.
*   **Temperature Sliders**: Experiment with the `Temperature` slider on summarization and JSON extraction pages to understand its impact on LLM output. Remember that `0.0` is recommended for factual extraction.

## Project Structure

The project is organized as follows:

```
.
├── app.py                      # Main Streamlit application file
├── source.py                   # Contains LLM interaction logic, data loading, and utility functions
├── transcripts/                # Directory for dummy earnings call transcript files
│   ├── AAPL_Q4_2024.txt
│   ├── JPM_Q4_2024.txt
│   └── TSLA_Q4_2024.txt
├── .streamlit/                 # Streamlit configuration (optional, for secrets)
│   └── secrets.toml            # Store API keys and other secrets
└── requirements.txt            # List of Python dependencies
```

### `source.py` Details

The `source.py` file is critical and likely contains the implementations for:
*   `transcripts`: A dictionary or similar structure holding the loaded transcript texts.
*   `key_facts_transcript`: Ground truth facts for audit.
*   `explain_financial_concept()`: Function to query LLM for explanations.
*   `summarize_naive()`: Basic summarization function.
*   `summarize_with_strategy()`: Handles 'structured' and 'few_shot' summarization.
*   `hierarchical_summarize()`: Implements chunking and hierarchical summarization.
*   `extract_metrics()`: Extracts structured JSON data.
*   `hallucination_audit()`: Performs the compliance audit.
*   `get_estimated_api_cost()`: Calculates API costs.
*   `chunk_transcript()`: Utility for splitting long texts.
*   `llm_chat_completion()`: A wrapper for OpenAI API calls.

## Technology Stack

*   **Framework**: [Streamlit](https://streamlit.io/) for creating interactive web applications.
*   **Language**: Python 3.8+
*   **LLM Provider**: OpenAI
*   **Libraries**:
    *   `openai`: Python client for OpenAI API interaction.
    *   `tiktoken`: For tokenization and managing LLM context windows.
    *   `pandas`: For data manipulation and tabular displays.
    *   `matplotlib`: For data visualization (e.g., compression ratio, temperature consistency plots, token cost breakdown).
    *   `json`: (Built-in) for JSON handling.
    *   `io`, `sys`: (Built-in) for managing output streams, especially for chunking process display.
    *   `nltk`: (Potentially) for advanced text processing like sentence tokenization, if implemented in `source.py`.

## Contributing

This project is primarily a lab demonstration. However, if you have suggestions for improvements, feature enhancements, or bug fixes, feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (if you create one).

*(If no specific license file is provided, you might state: "This project is developed as part of QuantUniversity's QuLab series for educational purposes and is not intended for commercial use without explicit permission.")*

## Contact

For any questions, suggestions, or feedback, please reach out to:

*   **QuantUniversity**
*   **Website**: [www.quantuniversity.com](https://www.quantuniversity.com/)
*   **Email**: info@quantuniversity.com
