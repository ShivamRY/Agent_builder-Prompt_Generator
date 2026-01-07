# Agent_builder-Prompt_Generator
Version: v1.0.0
Author: Shivam Yeshwantrao

============================================================
OVERVIEW

Agent Builder – Prompt Generator is a deterministic, dataset-driven system for generating compliant AI calling prompts for regulated use cases such as financial services and collections.

The system is designed to prevent hallucination by enforcing strict dataset selection rules, controlled prompt assembly, and structure-preserving translation. All intelligence and decision-making are handled in the backend, while the UI collects only structured inputs.

============================================================
KEY FEATURES

- Dataset-grounded prompt generation (single-row selection only)
- Zero hallucination by design
- Strict compliance and scope guardrails
- Structured Streamlit UI (no raw metadata input)
- Deterministic placeholder resolution
- Multilingual support with structure preservation
- Mandatory FACT BLOCK for factual correctness
- Optional prompt evaluation and refinement pipeline


============================================================
SYSTEM ARCHITECTURE

User / Admin
    |
    v
Streamlit UI (Structured Inputs Only)
    |
    v
Prompt Generator Backend
    |
    v
Dataset Selection Agent (Azure OpenAI – Controlled)
    |
    v
Approved Prompt Dataset (dataset.xlsx)
    |
    v
Base Prompt Assembler
    |
    v
Final Prompt Output (Downloadable / Executable)


============================================================
FRONTEND (UI)

The Streamlit UI collects only structured inputs.

Company Details:
- Company Name
- Product Category
- Customer Care Number
- Support Email
- Website (optional)
- Collection Stage (Pre-Due / Post-Due)

Agent Details:
- Agent Name
- Agent Description (persona and tone guidance)

Language Selection:
- English
- Hindi
- Marathi

The UI does not allow free-form instructions or raw metadata.


============================================================
BACKEND PIPELINE

1. Parse structured UI input into key-value metadata
2. Validate inputs
3. Select exactly ONE matching row from the approved dataset
4. Assemble the base prompt using dataset content only
5. Extract translatable script sections
6. Translate spoken dialogue only (structure preserved)
7. Merge translated sections safely
8. Resolve placeholders deterministically
9. Prepend mandatory FACT BLOCK
10. Output final prompt

If no dataset row matches, the system fails safely with an error.


============================================================
DATASET ENFORCEMENT

- Exactly one dataset row must be selected
- No row merging allowed
- No paraphrasing of dataset instructions
- No additional content generation
- Dataset is the single source of truth

Failure condition:
ERROR: No exact matching prompt profile found in dataset


============================================================
HALLUCINATION AND COMPLIANCE GUARDRAILS

The system enforces:
- No creative rewriting
- No policy or legal invention
- No topic drift during calls
- Controlled deflection for unrelated queries
- Mandatory use of provided facts for direct questions


============================================================
TRANSLATION LOGIC

Only the following sections are translated:
- Stepwise Script Flow
- Closing and Support
- Exceptions and Handling

Translation rules:
- Spoken dialogue only
- Headings and structure unchanged
- Placeholders preserved
- Temperature set to 0 for determinism


============================================================
FACT BLOCK

Each generated prompt includes a mandatory FACTS FOR THIS CALL block containing:
- Agent Name
- Company Name
- Product Category
- Customer Care Number
- Support Email

The agent must use these facts verbatim when answering direct questions.


============================================================
EVALUATION AND REFINEMENT (OPTIONAL)

The system includes:
- A 35-criteria prompt evaluation engine
- Scoring, critique, and refinement suggestions
- A refinement chain that improves prompts while preserving intent


============================================================
ENVIRONMENT SETUP

Required environment variables:

AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini

API keys must not be committed to version control.


============================================================
RUNNING THE APPLICATION

1. Install dependencies:
   pip install -r requirements.txt

2. Start the UI:
   streamlit run app.py


============================================================
DESIGN GUARANTEES

- Deterministic outputs
- Dataset-grounded prompts
- Zero hallucination by design
- Compliance-first architecture
- Safe multilingual support


============================================================
INTENDED USE CASES

- Loan repayment reminders
- Collections calls
- Financial service notifications
- Regulated outbound calling workflows


============================================================
