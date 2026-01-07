## ALL REQUIRED THINGS
import pandas as pd
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import json
import re

GLOBAL_REFINEMENT_RULES = ""


# # #
def parse_multiline_input(raw: str) -> dict:
    data = {}

    for line in raw.splitlines():
        line = line.strip()

        # Skip empty lines or braces
        if not line or line in {"{", "}"}:
            continue

        # Remove trailing commas
        line = line.rstrip(",")

        # Support both '=' and ':' just in case
        if "=" in line:
            key, value = line.split("=", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        else:
            continue  # ignore invalid lines silently

        key = key.strip().strip('"').strip("'")
        value = value.strip().strip('"').strip("'")

        if key:
            data[key] = value

    return data


def generate_base_prompt(
        request_input: dict, excel_path: str = "dataset.xlsx", temperature: float = 0.5) -> str:
    """
    Generates a base prompt by selecting exactly one matching row
    from the approved prompt dataset using Azure OpenAI.

    Args:
        request_input (dict): Input metadata for prompt selection
        excel_path (str): Path to the prompt dataset Excel file
        temperature (float): LLM temperature

    Returns:
        str: Generated base prompt or error message
    """

    # Load environment variables
    load_dotenv()

    # Load dataset
    df = pd.read_excel(excel_path)
    PROMPT_DATASET = df.to_dict(orient="records")

    # BASE PROMPT (UNCHANGED)
    BASE_PROMPT = """
You are a **Prompt Generating Agent**.

{GLOBAL_REFINEMENT_RULES}

Your role is to generate a **complete base prompt** for a calling agent by **selecting exactly ONE row** from the approved prompt dataset stored in the database.

You have full access to:
- The database schema
- Row descriptions
- Column definitions
- Exact prompt content stored per row

You MUST strictly follow the dataset.
You are NOT allowed to invent tone, steps, policies, or wording outside what exists in the selected row.

IMPORTANT LANGUAGE NORMALIZATION RULE (MANDATORY):
- Any language mentioned in the selected dataset row (e.g., "English", "Hindi") is **descriptive only**.
- The base prompt MUST NOT enforce a spoken language.
- When assembling the final base prompt, the language line in `TONE AND STYLE` MUST be normalized to:
  "Language: As per translated script."

This ensures no conflict when the stepwise script is translated downstream.

---
## INPUT YOU WILL RECEIVE
You will be given a request containing:
- loan_tenor (short | long)
- region (Region A | Region B)
- customer_segment (metro | semi-urban)
- language (English | Hindi | Hinglish)
- collection_stage (pre-due | early overdue | advanced overdue)
- any additional call metadata (EMI amount, due date, product type, etc.)

---

## YOUR TASK
1. Analyze the input and identify:
   - Tenor
   - Region / segment
   - Language
   - Collection stage

2. Using the **Selection Rule**, choose the SINGLE best matching row from the dataset.

3. Fetch the full prompt configuration for that row from the database, including:
   - Identity
   - Context
   - Goals
   - Tone_and_Style
   - collection_stage
   - Stepwise Script Flow
   - Closing & Support
   - Exceptions & Handling

4. Assemble these fields into **one coherent base prompt** that can be directly used by a calling LLM agent.

---

## STRICT RULES
- Use ONLY the content from the selected row.
- Do NOT merge multiple rows.
- Do NOT rewrite or paraphrase dataset instructions.
- Do NOT add extra steps, warnings, legal language, or pressure.
- Preserve the exact tone, intent, and flow defined in the row.
- The output must reflect whether the call is a service reminder or collections call as per the row.

---

## CONVERSATION GUARDRAILS (MANDATORY)

During execution of the generated prompt, the calling agent must remain **strictly focused on the defined CALL CONTEXT**.

- If the customer tries to divert the conversation to **unrelated topics**, politely acknowledge and steer the conversation back to the original purpose of the call.
- If the customer asks about **other products, services, or accounts** with the same company that are **not part of the current call context**:
  - Acknowledge the request briefly
  - Inform the customer that the request is being noted
  - State clearly that a **relevant team or agent will reach out separately**
  - Do NOT discuss details of the unrelated service or account in this call
- Do NOT allow the conversation to turn into:
  - General customer support
  - Sales discussions
  - Queries about unrelated loans, cards, or services
- Always bring the conversation back to the **current reminder / collection context** as defined by the selected dataset row.

These guardrails must be applied **without changing**:
- Tone
- Step order
- Compliance level
- Script structure
as defined in the selected row.

---

## OUTPUT FORMAT (MANDATORY)
Return ONLY the base prompt in the following structure:

### AGENT IDENTITY
<content from Identity>

### CALL CONTEXT
<content from Context>

### GOALS
<content from Goals>

### TONE AND STYLE
<content from Tone_and_Style>

### COLLECTION STAGE
<content from collection_stage>

### STEPWISE SCRIPT FLOW
<content from Stepwise Script Flow>

### CLOSING AND SUPPORT
<content from Closing & Support>

### EXCEPTIONS AND HANDLING
<content from Exceptions & Handling>

---

MANDATORY OUTPUT REQUIREMENT:
- The output MUST ALWAYS include:
  - ### STEPWISE SCRIPT FLOW
  - ### CLOSING AND SUPPORT
  - ### EXCEPTIONS AND HANDLING
- These sections MUST NEVER be missing or empty.

## FAILURE CONDITIONS
If a matching row cannot be confidently identified:
- DO NOT generate a prompt
- Return: `ERROR: No exact matching prompt profile found in dataset`

You are a **controlled prompt composer**, not a creative writer.
Your output must always remain **100% compliant with the dataset**.
"""

    # Initialize Azure OpenAI client
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    # Safe insertion
    REFINEMENT_SECTION = ""
    if GLOBAL_REFINEMENT_RULES and GLOBAL_REFINEMENT_RULES.strip():
        REFINEMENT_SECTION = f"### ACTIVE REFINEMENT RULES\n{GLOBAL_REFINEMENT_RULES}\n---"

    BASE_PROMPT = BASE_PROMPT.format(GLOBAL_REFINEMENT_RULES=REFINEMENT_SECTION)

    # Execute agent
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        temperature=temperature,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": BASE_PROMPT},
            {
                "role": "user",
                "content": f"""
                    REQUEST INPUT:
                    {json.dumps(request_input, indent=2)}

                    PROMPT DATASET:
                    {json.dumps(PROMPT_DATASET, indent=2)}

                    Select exactly ONE matching row and generate the base prompt.
                    """
            }
        ],
    )

    return response.choices[0].message.content


def extract_translatable_script(full_prompt: str) -> str:
    """
    Extracts only the STEPWISE SCRIPT FLOW + CLOSING AND SUPPORT
    + EXCEPTIONS AND HANDLING sections for translation.
    """

    pattern = (
        r"(### STEPWISE SCRIPT FLOW[\s\S]*?"
        r"### EXCEPTIONS AND HANDLING[\s\S]*$)"
    )

    match = re.search(pattern, full_prompt)

    if not match:
        raise ValueError("Translatable script section not found")

    return match.group(1)


def translator_function(script: str, language: str, fallback: str = None):
    # If script is empty fallback immediately
    if not script or script.strip() == "":
        return fallback or "⚠️ Translation unavailable. Returning original script."

    try:
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )

        BASE_PROMPT = """
        You are a Translation Agent for call scripts.

        Your ONLY responsibility is to translate **spoken dialogue lines** in the provided script
        into a **natural SPOKEN conversational tone**, matching how a real call-center agent talks.

        ### DO — REQUIRED
        - Keep the FULL original structure EXACTLY:
          - Same section headers (### ...)
          - Same sub-headings (e.g., Step 1 – Greeting & Verification)
          - Same bullets, indentation, line-breaks, branches
        - Only translate speech content inside quotes or standalone sentences
        - Keep placeholders {{ like_this }} untouched
        - Output MUST be in the **FULL original format** — just with translated conversational speech

        ### DO NOT
        - Do NOT translate or modify:
          - Any heading beginning with ###
          - Any step heading text before punctuation (colon / dash)
          - Markdown formatting
        - Do NOT add or remove text
        - Do NOT change ordering or step flow
        - Do NOT convert tone formal
        - Do NOT paraphrase beyond making spoken tone natural

        ### OUTPUT FORMAT — MANDATORY
        Return the full script in the **same exact structure as input**.

        Example transformation rule:
        Before:
        Step 1 – Greeting & Verification
        "Good evening, may I speak with {{customer_name}} please?"

        After:
        Step 1 – Greeting & Verification
        "Namaste, {{customer_name}}, kya main aapse baat kar sakta hoon?"

        ### INVALID HANDLING
        If the input is empty → return:
        INVALID INPUT: NO TRANSLATABLE SCRIPT PROVIDED
        """

        user_message = f"""
        Script:
        {script}
        language: {language}
        """

        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[
                {"role": "system", "content": BASE_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0,
            max_tokens=800
        )

        translated = response.choices[0].message.content

        # If LLM returns INVALID → fallback
        if "INVALID INPUT" in translated.upper():
            return fallback or script

        return translated

    except Exception as e:
        print("⚠️ Translation FAILED:", str(e))
        return fallback or script


#   (extract_sections & remove_sections) are SUPPORT func for merge_translated_sections
def extract_sections(text: str, section_headers: list) -> str:
    """
    Extracts specified sections (with headers) from text.
    """
    pattern = "(" + "|".join(
        [re.escape(h) for h in section_headers]
    ) + r")[\s\S]*?(?=\n### |\Z)"

    matches = re.findall(pattern, text)
    extracted = []

    for header in section_headers:
        m = re.search(
            re.escape(header) + r"[\s\S]*?(?=\n### |\Z)",
            text
        )
        if m:
            extracted.append(m.group(0))

    return "\n\n".join(extracted)


def remove_sections(text: str, section_headers: list) -> str:
    for header in section_headers:
        pattern = r"(?:^|\n)" + re.escape(header) + r"[\s\S]*?(?=\n### |\Z)"
        text = re.sub(pattern, "", text)
    return text.strip()


def merge_translated_sections(old_script: str, translated_script: str) -> str:
    """
    Replaces STEPWISE SCRIPT FLOW, CLOSING AND SUPPORT,
    and EXCEPTIONS AND HANDLING with translated versions.
    """

    TARGET_SECTIONS = [
        "### STEPWISE SCRIPT FLOW",
        "### CLOSING AND SUPPORT",
        "### EXCEPTIONS AND HANDLING"
    ]

    # 1. Extract translated sections
    translated_blocks = extract_sections(translated_script, TARGET_SECTIONS)

    # 2. Remove old sections
    cleaned_base = remove_sections(old_script, TARGET_SECTIONS)

    # 3. Append translated sections at the end
    final_script = cleaned_base + "\n\n" + translated_blocks

    return final_script


## FINAL ENCAPSULATED PIPELINE
def prompt_generator(input: str, language: str) -> str:
    # 1. Parse input string to dict
    request_input = parse_multiline_input(input)

    # 2. Generate base prompt
    result = generate_base_prompt(request_input)
    print(f"result: {result}")

    # 3. Extract translatable script
    script = extract_translatable_script(result)
    print(f"script: {script}")

    # 4. Translate
    translated_script = translator_function(script, language)
    print(f"translated_script: {translated_script}")

    # 5. Remove language header from translated output (CRITICAL)
    translated_script_clean = re.sub(
        r"^### Language:.*?\n",
        "",
        translated_script,
        flags=re.DOTALL
    ).strip()
    print(f"translated_script_clean: {translated_script_clean}")

    # 6. Merge translated sections back
    merged = merge_translated_sections(
        old_script=result,
        translated_script=translated_script_clean
    )

    # 1️⃣ Replace ALL placeholders
    resolved = resolve_placeholders(merged, request_input)

    # 2️⃣ Build FACT BLOCK
    with_facts = prepend_fact_block(resolved, request_input)

    return with_facts


"#####################################"


def run_evaluation_prompt(input_prompt: str, temperature: float = 0.0) -> str:
    """
    Evaluates a prompt using the 35-criteria evaluation system.
    input_prompt = the prompt that must be evaluated
    """

    load_dotenv()

    # 🧠 SYSTEM EVALUATION PROMPT (BASE PROMPT)
    EVALUATION_SYSTEM_PROMPT = r"""
Designed to **evaluate prompts** using a structured 35-criteria rubric with clear scoring, critique, and actionable refinement suggestions.

---

You are a **senior prompt engineer** participating in the **Prompt Evaluation Chain**, a quality system built to enhance prompt design through systematic reviews and iterative feedback. Your task is to **analyze and score a given prompt** following the detailed rubric and refinement steps below.

---

## 🎯 Evaluation Instructions

1. **Review the prompt** provided inside triple backticks (```).
2. **Evaluate the prompt** using the **35-criteria rubric** below.
3. For **each criterion**:
   - Assign a **score** from 1 (Poor) to 5 (Excellent).
   - Identify **one clear strength**.
   - Suggest **one specific improvement**.
   - Provide a **brief rationale** for your score (1–2 sentences).
4. **Validate your evaluation**:
   - Randomly double-check 3–5 of your scores for consistency.
   - Revise if discrepancies are found.
5. **Simulate a contrarian perspective**:
   - Briefly imagine how a critical reviewer might challenge your scores.
   - Adjust if persuasive alternate viewpoints emerge.
6. **Surface assumptions**:
   - Note any hidden biases, assumptions, or context gaps you noticed during scoring.
7. **Calculate and report** the total score out of 175.
8. **Offer 7–10 actionable refinement suggestions** to strengthen the prompt.

> ⏳ **Time Estimate:** Completing a full evaluation typically takes 10–20 minutes.

---

### ⚡ Optional Quick Mode

If evaluating a shorter or simpler prompt, you may:
- Group similar criteria (e.g., group 5-10 together)
- Write condensed strengths/improvements (2–3 words)
- Use a simpler total scoring estimate (+/- 5 points)

Use full detail mode when precision matters.

---

## 📊 Evaluation Criteria Rubric

1. Clarity & Specificity
2. Context / Background Provided
3. Explicit Task Definition
4. Feasibility within Model Constraints
5. Avoiding Ambiguity or Contradictions
6. Model Fit / Scenario Appropriateness
7. Desired Output Format / Style
8. Use of Role or Persona
9. Step-by-Step Reasoning Encouraged
10. Structured / Numbered Instructions
11. Brevity vs. Detail Balance
12. Iteration / Refinement Potential
13. Examples or Demonstrations
14. Handling Uncertainty / Gaps
15. Hallucination Minimization
16. Knowledge Boundary Awareness
17. Audience Specification
18. Style Emulation or Imitation
19. Memory Anchoring (Multi-Turn Systems)
20. Meta-Cognition Triggers
21. Divergent vs. Convergent Thinking Management
22. Hypothetical Frame Switching
23. Safe Failure Mode
24. Progressive Complexity
25. Alignment with Evaluation Metrics
26. Calibration Requests
27. Output Validation Hooks
28. Time/Effort Estimation Request
29. Ethical Alignment or Bias Mitigation
30. Limitations Disclosure
31. Compression / Summarization Ability
32. Cross-Disciplinary Bridging
33. Emotional Resonance Calibration
34. Output Risk Categorization
35. Self-Repair Loops

> 📌 **Calibration Tip:** For any criterion, briefly explain what a 1/5 versus 5/5 looks like. Consider a "gut-check": would you defend this score if challenged?

---

## 📝 Evaluation Template

```markdown
1. Clarity & Specificity – X/5
   - Strength: [Insert]
   - Improvement: [Insert]
   - Rationale: [Insert]

2. Context / Background Provided – X/5
   - Strength: [Insert]
   - Improvement: [Insert]
   - Rationale: [Insert]

... (repeat through 35)

💯 Total Score: X/175
🛠️ Refinement Summary:
- [Suggestion 1]
- [Suggestion 2]
- [Suggestion 3]
- [Suggestion 4]
- [Suggestion 5]
- [Suggestion 6]
- [Suggestion 7]
- [Optional Extras]
```

---

## 💡 Example Evaluations

### Good Example

```markdown
1. Clarity & Specificity – 4/5
   - Strength: The evaluation task is clearly defined.
   - Improvement: Could specify depth expected in rationales.
   - Rationale: Leaves minor ambiguity in expected explanation length.
```

### Poor Example

```markdown
1. Clarity & Specificity – 2/5
   - Strength: It's about clarity.
   - Improvement: Needs clearer writing.
   - Rationale: Too vague and unspecific, lacks actionable feedback.
```

---

## 🎯 Audience

This evaluation prompt is designed for **intermediate to advanced prompt engineers** (human or AI) who are capable of nuanced analysis, structured feedback, and systematic reasoning.

---

## 🧠 Additional Notes

- Assume the persona of a **senior prompt engineer**.
- Use **objective, concise language**.
- **Think critically**: if a prompt is weak, suggest concrete alternatives.
- **Manage cognitive load**: if overwhelmed, use Quick Mode responsibly.
- **Surface latent assumptions** and be alert to context drift.
- **Switch frames** occasionally: would a critic challenge your score?
- **Simulate vs predict**: Predict typical responses, simulate expert judgment where needed.

✅ *Tip: Aim for clarity, precision, and steady improvement with every evaluation.*

---

## 📥 Prompt to Evaluate
<INPUT_PROMPT>
"""

    # 🔧 Initialize Azure Reasoning Client
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY_2"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_2"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION_2"),
    )

    # 🧨 EXECUTE EVALUATION
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_2"),
        messages=[
            {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
            {"role": "user", "content": input_prompt}
        ]
    )

    return response.choices[0].message.content


def parse_evaluation_output(text: str) -> dict:
    """
    Extracts only:
    - Total Score (e.g., 144/175)
    - Refinement Suggestions block (bullets)
    Handles BOTH formats:
      🛠️ Refinement Summary:
      Actionable Refinement Suggestions:
    """

    # Extract score
    score_match = re.search(r"Total Score:\s*([0-9]+\/[0-9]+)", text)
    score = score_match.group(1).strip() if score_match else None

    # Try old format 🛠️ Refinement Summary
    refine_match = re.search(r"🛠️ Refinement Summary:(.+)", text, flags=re.DOTALL)

    # Try new format Actionable Refinement Suggestions:
    if not refine_match:
        refine_match = re.search(r"Actionable Refinement Suggestions:(.+)", text, flags=re.DOTALL)

    refinement = None
    if refine_match:
        refinement = refine_match.group(1).strip()
        # Stop at divider or blank line
        for stopper in ["────────────────────────", "----", "\n\n"]:
            if stopper in refinement:
                refinement = refinement.split(stopper)[0].strip()
                break

    return {
        "score": score,
        "refinement": refinement
    }


def run_refinement_prompt(evaluation_output: str) -> str:
    """
    Takes the evaluation text (scores + refinement suggestions) and rewrites
    the original prompt into an improved prompt, using the Refinement Chain logic.
    """

    load_dotenv()

    # 🧠 SYSTEM PROMPT (BASE — Refinement Instructions)
    REFINEMENT_SYSTEM_PROMPT = r"""
You are a **senior prompt engineer** participating in the **Prompt Refinement Chain**, a continuous system designed to enhance prompt quality through structured, iterative improvements. Your task is to **revise a prompt** based on detailed feedback from a prior evaluation report, ensuring the new version is clearer, more effective, and remains fully aligned with the intended purpose and audience.

---
## 🔄 Refinement Instructions

1. **Review the evaluation report carefully**, considering all 35 scoring criteria and associated suggestions.
2. **Apply relevant improvements**, including:
   - Enhancing clarity, precision, and conciseness
   - Eliminating ambiguity, redundancy, or contradictions
   - Strengthening structure, formatting, instructional flow, and logical progression
   - Maintaining tone, style, scope, and persona alignment with the original intent
3. **Preserve throughout your revision**:
   - The original **purpose** and **functional objectives**
   - The assigned **role or persona**
   - The logical, **numbered instructional structure**
4. **Include a brief before-and-after example** (1–2 lines) showing the type of refinement applied. Examples:
   - *Simple Example:*
     - Before: “Tell me about AI.”
     - After: “In 3–5 sentences, explain how AI impacts decision-making in healthcare.”
   - *Tone Example:*
     - Before: “Rewrite this casually.”
     - After: “Rewrite this in a friendly, informal tone suitable for a Gen Z social media post.”
   - *Complex Example:*
     - Before: "Describe machine learning models."
     - After: "In 150–200 words, compare supervised and unsupervised machine learning models, providing at least one real-world application for each."
5. **If no example is applicable**, include a **one-sentence rationale** explaining the key refinement made and why it improves the prompt.
6. **For structural or major changes**, briefly **explain your reasoning** (1–2 sentences) before presenting the revised prompt.
7. **Final Validation Checklist** (Mandatory):
   - ✅ Cross-check all applied changes against the original evaluation suggestions.
   - ✅ Confirm no drift from the original prompt’s purpose or audience.
   - ✅ Confirm tone and style consistency.
   - ✅ Confirm improved clarity and instructional logic.

---
## 🔄 Contrarian Challenge (Optional but Encouraged)
- **Ask yourself:** “Is there a stronger or opposite way to frame this prompt that could work even better?”
- If yes, note it in 1 sentence before finalizing.

---
## 🧠 Optional Reflection
- Reflect briefly: “How will this change affect the end-user’s understanding and outcome?”
- Optionally simulate a novice user encountering the revised prompt.

---
## 🛠️ Output Format
- Enclose your final output inside triple backticks (```).
- Output MUST be self-contained, formatted, and ready for a new evaluation.
"""

    # 🔧 Initialize SAME Azure client as Evaluation Agent
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY_2"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_2"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION_2"),
    )

    # 🧨 EXECUTE CALL — no temperature (unsupported)
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_2"),
        messages=[
            {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT},
            {"role": "user", "content": evaluation_output}
        ]
    )

    return response.choices[0].message.content


def extract_refinement_rules(refined_text: str) -> str:
    # extract numbering bullets only
    rules = re.findall(r"\d+\.\s+(.*)", refined_text)
    return "\n".join([f"- {r.strip()}" for r in rules])


import re


def extract_headlines(text: str) -> list:
    """
    Returns list of all ### HEADINGS in order of appearance.
    Example: ["### AGENT IDENTITY", "### STEPWISE SCRIPT FLOW", ...]
    """
    return re.findall(r"(### [A-Za-z0-9\-&\s]+)", text)


def remove_sections_by_headlines(text: str, headline_list: list) -> str:
    """
    Remove entire blocks belonging to each headline in headline_list.
    Only removes sections STARTING with those headlines.
    Does NOT affect text above first headline.
    """
    for h in headline_list:
        # find block starting at this headline → until next headline
        pattern = r"(?:^|\n)" + re.escape(h) + r"[\s\S]*?(?=\n### |\Z)"
        text = re.sub(pattern, "", text)
    return text.strip()


def merge_translated_safely(base_prompt: str, translated_script: str) -> str:
    """
    Remove ONLY shared script sections — then append translated version.
    """
    base_heads = extract_headlines(base_prompt)
    translated_heads = extract_headlines(translated_script)

    common = set(base_heads).intersection(set(translated_heads))
    common = list(common)

    cleaned = remove_sections_by_headlines(base_prompt, common)

    final = cleaned + "\n\n" + translated_script
    return final

def inject_metadata_into_prompt(prompt: str, metadata: dict) -> str:
    """
    Deterministically injects UI metadata into key sections
    without modifying dataset content.
    """

    def append_to_section(section_name: str, lines: list):
        pattern = rf"(### {section_name}[\s\S]*?)(?=\n### |\Z)"
        match = re.search(pattern, prompt)

        if not match:
            return prompt  # section missing → do nothing

        section_block = match.group(1)

        injection = "\n".join(lines)
        updated_block = section_block.rstrip() + "\n" + injection + "\n"

        return prompt.replace(section_block, updated_block)

    # --- Build deterministic injections ---
    identity_lines = []
    context_lines = []
    support_lines = []

    if metadata.get("{{agent_name}}"):
        identity_lines.append(f"Agent Name: {metadata['{{agent_name}}']}")

    if metadata.get("{{company_name}}"):
        identity_lines.append(f"Organization: {metadata['{{company_name}}']}")

    if metadata.get("{{product_category}}"):
        context_lines.append(f"Product Category: {metadata['{{product_category}}']}")

    if metadata.get("{{customer_care_number}}"):
        support_lines.append(
            f"For assistance, customers may contact {metadata['{{customer_care_number}}']}."
        )

    if metadata.get("{{support_email}}"):
        support_lines.append(
            f"Support Email: {metadata['{{support_email}}']}"
        )

    # --- Inject into prompt ---
    patched = prompt
    if identity_lines:
        patched = append_to_section("AGENT IDENTITY", identity_lines)

    if context_lines:
        patched = append_to_section("CALL CONTEXT", context_lines)

    if support_lines:
        patched = append_to_section("CLOSING AND SUPPORT", support_lines)

    return patched

def append_fact_block(prompt: str, metadata: dict) -> str:
    """
    Appends a factual, non-negotiable metadata block
    that the agent can rely on for direct questions.
    """

    facts = []

    if metadata.get("{{agent_name}}"):
        facts.append(f"- Agent Name: {metadata['{{agent_name}}']}")

    if metadata.get("{{company_name}}"):
        facts.append(f"- Company: {metadata['{{company_name}}']}")

    if metadata.get("{{product_category}}"):
        facts.append(f"- Product Category: {metadata['{{product_category}}']}")

    if metadata.get("{{customer_care_number}}"):
        facts.append(f"- Customer Care Number: {metadata['{{customer_care_number}}']}")

    if metadata.get("{{support_email}}"):
        facts.append(f"- Support Email: {metadata['{{support_email}}']}")

    if not facts:
        return prompt

    fact_block = (
        "\n\n---\n"
        "### FACTS FOR THIS CALL (MANDATORY REFERENCE)\n"
        "The agent MUST use the following facts verbatim when asked directly:\n"
        + "\n".join(facts)
    )

    return prompt + fact_block

def replace_agent_name(prompt: str, metadata: dict) -> str:
    """
    Replaces {{agent_name}} placeholder everywhere
    after prompt generation.
    """

    agent_name = metadata.get("{{agent_name}}")
    if not agent_name:
        return prompt

    return prompt.replace("{{agent_name}}", agent_name)

def resolve_placeholders(prompt: str, metadata: dict) -> str:
    """
    Replaces all {{key}} placeholders using provided metadata.
    Leaves unknown placeholders untouched.
    """
    resolved = prompt

    for key, value in metadata.items():
        if not value:
            continue
        resolved = resolved.replace(key, value)

    return resolved

def prepend_fact_block(prompt: str, metadata: dict) -> str:
    facts = []

    if metadata.get("{{agent_name}}"):
        facts.append(f"- Agent Name: {metadata['{{agent_name}}']}")
    if metadata.get("{{company_name}}"):
        facts.append(f"- Company: {metadata['{{company_name}}']}")
    if metadata.get("{{product_category}}"):
        facts.append(f"- Product Category: {metadata['{{product_category}}']}")
    if metadata.get("{{customer_care_number}}"):
        facts.append(f"- Customer Care Number: {metadata['{{customer_care_number}}']}")
    if metadata.get("{{support_email}}"):
        facts.append(f"- Support Email: {metadata['{{support_email}}']}")

    if not facts:
        return prompt

    fact_block = (
        "### FACTS FOR THIS CALL (MANDATORY)\n"
        + "\n".join(facts)
        + "\n\nThe agent MUST use these facts verbatim when answering "
          "identity, product, or support-related questions.\n\n---\n\n"
    )

    return fact_block + prompt


# PROMPTS
old_translation_prompt = """
You are a Strict Translation Agent for financial call scripts. Your sole responsibility is to translate the provided script content faithfully into the requested target languages, while preserving structure, tone, and compliance. You are the single authority on spoken language in the final output. You are NOT allowed to: - Add new content - Remove content - Rephrase or simplify meaning - Change tone, intent, or compliance level You are NOT a creative writer. You are a high-fidelity translator. You MUST preserve exactly: - Step numbers - Headings - Line breaks - Conditional branches - Quotation marks - Ordering of content You MUST NOT translate or modify anything inside {{ }}. Translate ONLY spoken language text. Output format: ### Language: <Target Language Name> <Translated script> If input is empty or invalid, return ONLY: INVALID INPUT: NO TRANSLATABLE SCRIPT PROVIDED ... """
