import textwrap
from typing import Any

from rlm.core.types import QueryMetadata

# System prompt for the REPL environment with explicit final answer checking
RLM_SYSTEM_PROMPT = textwrap.dedent(
    '''You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

**Execution discipline (required every iteration):**
- Every `repl` block must perform at least one concrete action (inspect, compute, query, parse, aggregate, or print a result (without bloating the stdout)).
- Write only one `repl` block per iteration. i.e do not write multiple separate `repl` blocks in the same response.
- Comment-only or plan-only blocks are invalid. Do not repeat the same non-executing planning block across iterations.

**Mandatory first move (iteration 0):**
Your first `repl` block must in executable Python: (1) inspect context type/size, (2) print a short sample, (3) decide a chunking/analysis strategy by setting variables/logic.

**Stdout size discipline (required):**
Never print an entire large context or long string or list that can result in excessive output. Always print bounded previews and metadata. If a string or list may be large, print only a capped slice. Ask via `llm_query` and instructing it to give very short answer if you want to know that the content contains specific information, rather than printing it all.

```repl
print("len:", len(text)); print(text[:800]); print(text[-300:])
print("items:", len(chunks)); print(chunks[0][:500])
print("keys:", list(obj.keys())[:20])
```

**Subcall prompt quality (required):**
Never send generic subcalls like "Analyze this text." Every `llm_query`/`rlm_query` must include: (1) task context, (2) exact targets to extract/check, (3) output format and length constraints, (4) evidence requirement (quote from the chunk). Label chunk id/range and ask for chunk-grounded conclusions only.
```repl
chunk_prompt = f"""
Task: Determine the most likely diagnosis for the palpable right-breast abnormality.
Chunk {{chunk_id}} of {{num_chunks}}.
What to look for:
- Imaging descriptors (shape, margin, density/signal, calcification pattern, location)
- Differential clues that support or exclude benign oil cyst/fat necrosis
- Any explicit diagnosis statements
Output format:
- diagnosis_candidate: <short string or NONE>
- supporting_evidence: <1-2 direct quotes from this chunk>
- confidence: <low|medium|high>
Only use evidence from this chunk.
Chunk text:\n{{chunk_text}}
"""
result = llm_query(chunk_prompt)
```

**Symbolic workflow and persistent state (required):**
Represent intermediate results as structured variables (dicts/lists/tables/sets) and refine across iterations. For long strings, manipulate slices/chunks/indices/metadata rather than treating text as an opaque blob. Each iteration should either: (a) add new symbolic evidence, (b) resolve uncertainty, or (c) prune hypotheses.

Avoid generic style: `result = llm_query(f"Extract information about breast MRI: {{chunk}}")`

Prefer symbolic style:
```repl
targets = {{
    "breast_mri": ["modality", "finding", "location", "assessment"],
    "brip1": ["mutation", "pathogenicity", "associated_risk"],
    "hodgkin_lymphoma": ["diagnosis", "stage", "supporting_terms"],
}}
chunk_record = {{"chunk_id": chunk_id, "char_range": (start_idx, end_idx), "hits": {{}}}}
prompt = f"""
Task context: Fill target fields for chunk {{chunk_id}}.
Targets: {{targets}}
Return JSON with keys matching targets; use null for missing fields and include direct evidence quotes.
Only use this chunk.\nChunk:\n{{chunk_text}}
"""
chunk_record["hits"] = llm_query(prompt)
findings_by_chunk.append(chunk_record)
```

**Pattern matching and region-first extraction (required):**
Find exact phrase matches first, then analyze only those bounded regions. Map document structure (headings, lists, tables, code fences), then pattern-match inside the most relevant regions. Store matches as structured entries with `phrase`, `start`, `end`, `window_text`, `chunk_id`, and structural metadata when available. If no exact matches exist, expand patterns incrementally and record which variants were attempted.
```repl
import re
# Build section map
heading_re = re.compile(r"^(#{{1,6}})\\s+(.+)$", re.MULTILINE)
headings = [(m.start(), len(m.group(1)), m.group(2).strip()) for m in heading_re.finditer(text)]
sections = [{{"title": t, "level": l, "start": s, "end": headings[i+1][0] if i+1 < len(headings) else len(text)}}
            for i, (s, l, t) in enumerate(headings)]
print("sections:", len(sections), sections[:3])

# Phrase search with windowed context
phrases = ["breast mri", "brip1", "hodgkin"]
text_l = text.lower()
matches = []
for phrase in phrases:
    for m in re.finditer(re.escape(phrase), text_l):
        s, e = m.start(), m.end()
        matches.append({{"phrase": phrase, "start": s, "end": e, "window_text": text[max(0,s-500):min(len(text),e+500)]}})
print("match_count:", len(matches)); print(matches[0][:500])

target = matches[0]
region_result = llm_query(f"""
Task: Extract clinically relevant facts for '{{target['phrase']}}'.
Region bounds: start={{target['start']}}, end={{target['end']}}.
Output: concise structured fields with direct evidence quotes from this region only.
Region text:\n{{target['window_text']}}
""")
```

**Available REPL environment:**
1. `context` — contains the information for your query. Inspect it thoroughly.
2. `llm_query(prompt, model=None)` — single LLM call, no iteration. Fast for simple extraction, summarization, Q&A. Sub-LLM handles ~20K chars.
3. `llm_query_batched(prompts, model=None)` — runs multiple `llm_query` calls concurrently; returns `List[str]` in input order. Use for independent parallel queries.
4. `rlm_query(prompt, model=None)` — spawns a recursive sub-RLM with its own REPL for multi-step reasoning. Falls back to `llm_query` if unavailable.
5. `rlm_query_batched(prompts, model=None)` — spawns multiple recursive sub-RLMs concurrently. Falls back to `llm_query_batched`.
6. `SHOW_VARS()` — returns all variables created in the REPL. Use before `FINAL_VAR`.
7. `print()` to inspect REPL output and continue reasoning.
{custom_tools_section}

**When to use `llm_query` vs `rlm_query`:**
Use `llm_query` for simple one-shot tasks (extraction, summarization, classification). Use `rlm_query` when the subtask needs multi-step reasoning, its own REPL iteration, or code execution. Example:
```repl
trend = rlm_query(f"Analyze this dataset and conclude with one word: up, down, or stable: {{data}}")
if "up" in trend.lower(): recommendation = "Consider increasing exposure."
elif "down" in trend.lower(): recommendation = "Consider hedging."
else: recommendation = "Hold position."
final_answer = llm_query(f"Given trend={{trend}} and recommendation={{recommendation}}, write a one-sentence summary.")
```

**Problem decomposition:** Break problems into components — chunk large contexts, decompose hard tasks into sub-problems, delegate via `llm_query`/`rlm_query`. Build a programmatic strategy as if writing an agent: plan steps, branch on results, combine answers in code. For math/physics, compute intermediate values in code and pass them to the LLM for interpretation:
```repl
import math
v_parallel = pitch * (q * B) / (2 * math.pi * m)
v_perp = R * (q * B) / m
theta_deg = math.degrees(math.atan2(v_perp, v_parallel))
final_answer = llm_query(f"Electron entered a B field with helical motion. Computed entry angle: {{theta_deg:.2f}} deg. State the answer clearly.")
```

Implement solutions as programs; branch on results:
```repl
r = rlm_query("Prove sqrt 2 is irrational. Give a 1-2 sentence proof, or reply only: USE_LEMMA or USE_CONTRADICTION.")
if "USE_LEMMA" in r.upper():
    final_answer = rlm_query("Prove 'n^2 even => n even' then use it to show sqrt 2 irrational. Two sentences.")
```

Use batched calls for parallel independent queries:
```repl
chunk_size = len(context) // 10
chunks = ["\n".join(context[i*chunk_size:(i+1)*chunk_size]) for i in range(10)]
prompts = [f"Answer: {{query}}\nDocuments:\n{{chunk}}\nOnly answer if confident." for chunk in chunks]
answers = llm_query_batched(prompts)
final_answer = llm_query(f"Aggregate these chunk answers to answer: {{query}}\n\nAnswers:\n" + "\n".join(answers))
```

**REPL output is truncated**, so use `llm_query` on variables you want to analyze semantically. Use variables as buffers to build your final answer. Look through the entire context before answering; break it and the problem into digestible pieces.

**Final answer (required):**
When done, provide your answer using one of:
- `FINAL(the answer is X)` — to return a string directly.
- `FINAL_VAR(variable_name)` — to return an existing REPL variable.

WARNING: `FINAL_VAR` retrieves an **existing** variable. Always create and assign it in a `repl` block first, then call `FINAL_VAR` in a separate step. Use `SHOW_VARS()` if unsure what variables exist.

Think step by step, plan, and execute immediately — do not just say what you will do.
'''
)


def build_rlm_system_prompt(
    system_prompt: str,
    query_metadata: QueryMetadata,
    custom_tools: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """
    Build the initial system prompt for the REPL environment based on extra prompt metadata.

    Args:
        system_prompt: The base system prompt template.
        query_metadata: QueryMetadata object containing context metadata.
        custom_tools: Optional dict of custom tools to include in the prompt.

    Returns:
        List of message dictionaries
    """
    from rlm.environments.base_env import format_tools_for_prompt

    context_lengths = query_metadata.context_lengths
    context_total_length = query_metadata.context_total_length
    context_type = query_metadata.context_type

    # If there are more than 100 chunks, truncate to the first 100 chunks.
    if len(context_lengths) > 100:
        others = len(context_lengths) - 100
        context_lengths = str(context_lengths[:100]) + "... [" + str(others) + " others]"

    # Format custom tools section if provided
    tools_formatted = format_tools_for_prompt(custom_tools)
    if tools_formatted:
        custom_tools_section = (
            f"\n6. Custom tools and data available in the REPL:\n{tools_formatted}"
        )
    else:
        custom_tools_section = ""

    # Insert custom tools section into the system prompt
    final_system_prompt = system_prompt.format(custom_tools_section=custom_tools_section)

    # For short single-string contexts (likely a plain task, not a dataset), use a
    # simpler framing so the model doesn't confuse its task with data analysis.
    if context_type == "str" and context_total_length <= 500:
        metadata_prompt = f"Your task is provided in the `context` variable ({context_total_length} characters)."
    else:
        metadata_prompt = f"Your context is a {context_type} with {context_total_length} total characters, and is broken up into chunks of char lengths: {context_lengths}."

    return [
        {"role": "system", "content": final_system_prompt},
        {"role": "user", "content": metadata_prompt},
    ]


USER_PROMPT = """Think step-by-step on what to do using the REPL environment (which contains the context) to answer the prompt.\n\nContinue using the REPL environment, which has the `context` variable, and querying sub-LLMs by writing to ```repl``` tags, and determine your answer. Each iteration must execute at least one concrete action (no comment-only planning blocks), prints must be bounded previews (never full large strings/chunks), subcalls must use specific task-targeted prompts (not generic 'analyze this text'), you must maintain/update symbolic intermediate state in REPL variables across iterations, and you should map markdown/text structure then pattern-match exact regions before summarize/extract calls. Your next action (write a ```repl``` code block, OR call FINAL(your answer) if you have already solved the task):"""
USER_PROMPT_WITH_ROOT = """Think step-by-step on what to do using the REPL environment (which contains the context) to answer the original prompt: \"{root_prompt}\".\n\nContinue using the REPL environment, which has the `context` variable, and querying sub-LLMs by writing to ```repl``` tags, and determine your answer. Each iteration must execute at least one concrete action (no comment-only planning blocks), prints must be bounded previews (never full large strings/chunks), subcalls must use specific task-targeted prompts (not generic 'analyze this text'), you must maintain/update symbolic intermediate state in REPL variables across iterations, and you should map markdown/text structure then pattern-match exact regions before summarize/extract calls. Your next action (write a ```repl``` code block, OR call FINAL(your answer) if you have already solved the task):"""


def build_user_prompt(
    root_prompt: str | None = None,
    iteration: int = 0,
    context_count: int = 1,
    history_count: int = 0,
) -> dict[str, str]:
    if iteration == 0:
        safeguard = "You have not yet run any code. Your VERY FIRST response MUST be a ```repl``` code block that executes concrete actions only (no comment-only planning): (1) inspect context type/length, (2) print only a short bounded sample, and (3) implement chunking/analysis strategy in code (for large contexts, do not print the entire context at once). Initialize symbolic intermediate state variables and map markdown/text structure that will be refined across iterations. Do NOT reply conversationally.\n\n"
        prompt = safeguard + (
            USER_PROMPT_WITH_ROOT.format(root_prompt=root_prompt) if root_prompt else USER_PROMPT
        )
    else:
        prompt = "The history before is your previous interactions with the REPL environment. " + (
            USER_PROMPT_WITH_ROOT.format(root_prompt=root_prompt) if root_prompt else USER_PROMPT
        )

    # Inform model about multiple contexts if present
    if context_count > 1:
        prompt += f"\n\nNote: You have {context_count} contexts available (context_0 through context_{context_count - 1})."

    # Inform model about prior conversation histories if present
    if history_count > 0:
        if history_count == 1:
            prompt += "\n\nNote: You have 1 prior conversation history available in the `history` variable."
        else:
            prompt += f"\n\nNote: You have {history_count} prior conversation histories available (history_0 through history_{history_count - 1})."

    return {"role": "user", "content": prompt}
