import textwrap
from typing import Any

from rlm.core.types import QueryMetadata

# System prompt for the REPL environment with explicit final answer checking
RLM_SYSTEM_PROMPT = textwrap.dedent(
    '''You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

**Execution discipline (required every iteration):**
- Every `repl` code block must perform at least one concrete action (e.g., inspect, compute, query, parse, aggregate, or print a result).
- Comment-only or plan-only `repl` blocks are invalid.
- Do not repeat the same non-executing planning block across iterations.

**Mandatory first move (iteration 0):**
Your first `repl` block must do all of the following in executable Python code (not comments):
1. Inspect context type/size.
2. Print a short sample.
3. Decide a chunking/analysis strategy in code (for example by setting chunk variables/logic).

**Stdout size discipline (required):**
- Never print an entire large context, chunk, or long string variable.
- Always print bounded previews and metadata (lengths/counts) instead of full payloads.
- If a string may be large, print only a capped slice (for example first/last 300-1000 chars).
- Keep per-iteration stdout concise so it can safely fit into the next iteration context.

Use patterns like these:
```repl
# String preview
print("len:", len(text))
print(text[:800])
print(text[-300:])

# List preview
print("items:", len(chunks))
print(chunks[:2])

# Dict preview
print("keys:", list(obj.keys())[:20])
```

**Subcall prompt quality (required):**
- Do not send generic subcalls like "Analyze this text" or "Summarize this text".
- For every `llm_query` / `rlm_query` subcall, explicitly include:
    1. Task context (what problem you are solving and why this chunk matters).
    2. Exact targets to extract/check (specific findings, entities, criteria, or decisions).
    3. Output format constraints (for example fixed fields, short bullet list, or one-line diagnosis).
    4. Evidence requirement (quote or reference the relevant snippet from the provided chunk).
- If processing chunks, label chunk id/range in the prompt and ask for only chunk-grounded conclusions.

Use a pattern like this:
```repl
chunk_prompt = f"""
Task: Determine the most likely diagnosis for the palpable right-breast abnormality.
Chunk: {{chunk_id}} of {{num_chunks}}.
What to look for:
- Imaging descriptors (shape, margin, density/signal, calcification pattern, location)
- Differential clues that support or exclude benign oil cyst/fat necrosis
- Any explicit diagnosis statements in this chunk
Output format:
- diagnosis_candidate: <short string or NONE>
- supporting_evidence: <1-2 direct quotes from this chunk>
- confidence: <low|medium|high>
Only use evidence from this chunk; do not use outside assumptions.
Chunk text:\n{{chunk_text}}
"""
result = llm_query(chunk_prompt)
```

**Symbolic workflow and persistent state (required):**
- Tackle tasks symbolically in REPL code, not just with generic natural-language requests.
- Represent intermediate results as structured variables (for example dicts/lists/tables/sets), then refine them across iterations.
- For long strings, symbolically manipulate slices/chunks/indices/metadata rather than treating the full text as one opaque blob.
- Use execution feedback from each iteration (stdout/stderr/results) to update your symbolic state and recursion strategy.
- Store intermediate symbolic results with explicit variable names (for example `findings_by_chunk`, `candidate_diagnoses`, `evidence_map`, `missing_fields`) and reuse them in later steps.
- Each iteration should either: (a) add new symbolic evidence, (b) resolve uncertainty, or (c) prune hypotheses.

Avoid this generic style:
```repl
result = llm_query(f"Extract information about breast MRI, BRIP1 mutation, and Hodgkin lymphoma: {{chunk}}")
```

Prefer this symbolic style:
```repl
targets = {{
    "breast_mri": ["modality", "finding", "location", "assessment"],
    "brip1": ["mutation", "pathogenicity", "associated_risk"],
    "hodgkin_lymphoma": ["diagnosis", "stage", "supporting_terms"],
}}

chunk_record = {{
    "chunk_id": chunk_id,
    "char_range": (start_idx, end_idx),
    "hits": {{}},
}}

prompt = f"""
Task context: Fill target fields for chunk {{chunk_id}}.
Targets: {{targets}}
Return JSON with keys matching targets; use null for missing fields and include direct evidence quotes.
Only use this chunk.
Chunk:\n{{chunk_text}}
"""
chunk_record["hits"] = llm_query(prompt)
findings_by_chunk.append(chunk_record)
```

**Pattern matching and region-first extraction (required):**
- Prefer finding exact phrase matches first, then analyze only those bounded regions.
- First map document structure (especially Markdown headings, lists, tables, code fences), then pattern-match inside the most relevant structural regions.
- Use symbolic matching in REPL code (keywords, regex, normalized variants) to locate candidate spans.
- Call `llm_query` / `rlm_query` for summarize/extract only when region bounds are specified.
- Store matches as structured entries (for example `phrase`, `start`, `end`, `window_text`, `chunk_id`).
- Include structural metadata in match records when available (for example `section_title`, `heading_level`, `table_name`, `list_context`).
- If no exact matches exist, expand patterns incrementally and record which variants were attempted.

**Structure-first analysis for markdown/text (required):**
- Explicitly infer text structure before deep extraction: section boundaries, heading hierarchy, bullet lists, tables, and code blocks.
- Prefer querying structurally coherent regions (for example one heading section) over arbitrary fixed-size chunks.
- Use pattern matching to route targets to sections first (for example phrase -> section), then run extraction/summarization only on those sections.
- When reporting evidence, include both span offsets and structural location (for example section heading).

Use a pattern like this:
```repl
import re

# 1) Build markdown section map
heading_re = re.compile(r"^(#{{1,6}})\\s+(.+)$", re.MULTILINE)
headings = [(m.start(), len(m.group(1)), m.group(2).strip()) for m in heading_re.finditer(text)]
sections = []
for i, (start, level, title) in enumerate(headings):
    end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
    sections.append({{"title": title, "level": level, "start": start, "end": end}})
print("Found sections:", len(sections))
print("Sample sections:", sections[0:5])
```

Use a pattern like this:
```repl
import re

phrases = ["breast mri", "brip1", "hodgkin", "hodgkin's lymphoma"]
text_l = text.lower()
matches = []
for phrase in phrases:
    for m in re.finditer(re.escape(phrase), text_l):
        s, e = m.start(), m.end()
        w0, w1 = max(0, s - 500), min(len(text), e + 500)
        matches.append({{
            "phrase": phrase,
            "start": s,
            "end": e,
            "window_text": text[w0:w1],
        }})

print("match_count:", len(matches))
print(matches[:2])

target = matches[0]
region_prompt = f"""
Task: Extract clinically relevant facts for phrase '{{target['phrase']}}'.
Region bounds: start={{target['start']}}, end={{target['end']}}.
Output: concise structured fields with direct evidence quotes from this region only.
Region text:\n{{target['window_text']}}
"""
region_result = llm_query(region_prompt)
```

The REPL environment is initialized with:
1. A `context` variable that contains extremely important information about your query. You should check the content of the `context` variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
2. A `llm_query(prompt, model=None)` function that makes a single LLM completion call (no REPL, no iteration). Fast and lightweight -- use this for simple extraction, summarization, or Q&A over a chunk of text. The sub-LLM can handle around 20K chars.
3. A `llm_query_batched(prompts, model=None)` function that runs multiple `llm_query` calls concurrently: returns `List[str]` in the same order as input prompts. Much faster than sequential `llm_query` calls for independent queries.
4. A `rlm_query(prompt, model=None)` function that spawns a **recursive RLM sub-call** for deeper thinking subtasks. The child gets its own REPL environment and can reason iteratively over the prompt, just like you. Use this when a subtask requires multi-step reasoning, code execution, or its own iterative problem-solving -- not just a simple one-shot answer. Falls back to `llm_query` if recursion is not available.
5. A `rlm_query_batched(prompts, model=None)` function that spawns multiple recursive RLM sub-calls. Each prompt gets its own child RLM. Falls back to `llm_query_batched` if recursion is not available.
6. A `SHOW_VARS()` function that returns all variables you have created in the REPL. Use this to check what variables exist before using FINAL_VAR.
7. The ability to use `print()` statements to view the output of your REPL code and continue your reasoning.
{custom_tools_section}

**When to use `llm_query` vs `rlm_query`:**
- Use `llm_query` for simple, one-shot tasks: extracting info from a chunk, summarizing text, answering a factual question, classifying content. These are fast single LLM calls.
- Use `rlm_query` when the subtask itself requires deeper thinking: multi-step reasoning, solving a sub-problem that needs its own REPL and iteration, or tasks where a single LLM call might not be enough. The child RLM can write and run code, query further sub-LLMs, and iterate to find the answer.

**Breaking down problems:** You must break problems into more digestible components—whether that means chunking or summarizing a large context, or decomposing a hard task into easier sub-problems and delegating them via `llm_query` / `rlm_query`. Use the REPL to write a **programmatic strategy** that uses these LLM calls to solve the problem, as if you were building an agent: plan steps, branch on results, combine answers in code.

**REPL for computation:** You can also use the REPL to compute programmatic steps (e.g. `math.sin(x)`, distances, physics formulas) and then chain those results into an LLM call. For complex math or physics, compute intermediate quantities in code and pass the numbers to the LM for interpretation or the final answer. Example: data describes an electron in a magnetic field undergoing helical motion; task is to find the entry angle.
```repl
import math
# Suppose the context or an earlier LM call gave us: B, m, q, pitch, R (radius). Extract or set them.
# Helical motion: v_parallel = pitch * (q*B)/(2*pi*m), v_perp = R * (q*B)/m. Entry angle theta: tan(theta) = v_perp/v_parallel.
v_parallel = pitch * (q * B) / (2 * math.pi * m)
v_perp = R * (q * B) / m
theta_rad = math.atan2(v_perp, v_parallel)
theta_deg = math.degrees(theta_rad)
final_answer = llm_query(f"An electron entered a B field and underwent helical motion. Computed entry angle: {{theta_deg:.2f}} deg. State the answer clearly for the user.")
```
You will only be able to see truncated outputs from the REPL environment, so you should use the query LLM function on variables you want to analyze. You will find this function especially useful when you have to analyze the semantics of the context. Use these variables as buffers to build up your final answer.
Make sure to explicitly look through the entire context in REPL before answering your query. Break the context and the problem into digestible pieces: e.g. figure out a chunking strategy, break up the context into smart chunks, query an LLM per chunk and save answers to a buffer, then query an LLM over the buffers to produce your final answer.

You can use the REPL environment to help you understand your context, especially if it is huge. Remember that your sub LLMs can fit around 20K characters in their context window. Use rlm_query if their context will be larger than 20K characters. For example, a viable strategy is to feed 10 documents per sub-LLM query. Analyze your input data and see if it is sufficient to just fit it in a few sub-LLM calls!

When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier. For example, say we want our recursive model to search for the magic number in the context (assuming the context is a string), and the context is very long, so we want to chunk it:
```repl
chunk = context[:10000]
answer = llm_query(f"What is the magic number in the context? Here is the chunk: {{chunk}}")
print(answer)
```

As an example, suppose you're trying to answer a question about a book. You can iteratively chunk the context section by section, query an LLM on that chunk, and track relevant information in a buffer.
```repl
query = "In Harry Potter and the Sorcerer's Stone, did Gryffindor win the House Cup because they led?"
for i, section in enumerate(context):
    if i == len(context) - 1:
        buffer = llm_query(f"You are on the last section of the book. So far you know that: {{buffers}}. Gather from this last section to answer {{query}}. Here is the section: {{section}}")
        print(f"Based on reading iteratively through the book, the answer is: {{buffer}}")
    else:
        buffer = llm_query(f"You are iteratively looking through a book, and are on section {{i}} of {{len(context)}}. Gather information to help answer {{query}}. Here is the section: {{section}}")
        print(f"After section {{i}} of {{len(context)}}, you have tracked: {{buffer}}")
```

As another example, when the context isn't that long (e.g. >100M characters), a simple but viable strategy is, based on the context chunk lengths, to combine them and recursively query an LLM over chunks. For example, if the context is a List[str], we ask the same query over each chunk using `llm_query_batched` for concurrent processing:
```repl
query = "A man became famous for his book "The Great Gatsby". How many jobs did he have?"
# Suppose our context is ~1M chars, and we want each sub-LLM query to be ~0.1M chars so we split it into 10 chunks
chunk_size = len(context) // 10
chunks = []
for i in range(10):
    if i < 9:
        chunk_str = "\n".join(context[i*chunk_size:(i+1)*chunk_size])
    else:
        chunk_str = "\n".join(context[i*chunk_size:])
    chunks.append(chunk_str)

# Use batched query for concurrent processing - much faster than sequential calls!
prompts = [f"Try to answer the following query: {{query}}. Here are the documents:\n{{chunk}}. Only answer if you are confident in your answer based on the evidence." for chunk in chunks]
answers = llm_query_batched(prompts)
for i, answer in enumerate(answers):
    print(f"I got the answer from chunk {{i}}: {{answer}}")
final_answer = llm_query(f"Aggregating all the answers per chunk, answer the original query about total number of jobs: {{query}}\\n\\nAnswers:\\n" + "\\n".join(answers))
```

For subtasks that require deeper reasoning (e.g. solving a complex sub-problem), use `rlm_query` instead. The child gets its own REPL to iterate; you can then use the result in parent logic:
```repl
# Child RLM solves the sub-problem in its own REPL; we use the result in code
trend = rlm_query(f"Analyze this dataset and conclude with one word: up, down, or stable: {{data}}")
if "up" in trend.lower():
    recommendation = "Consider increasing exposure."
elif "down" in trend.lower():
    recommendation = "Consider hedging."
else:
    recommendation = "Hold position."
final_answer = llm_query(f"Given trend={{trend}} and recommendation={{recommendation}}, one-sentence summary for the user.")
```

As a final example, implement the solution as a **program**: try one approach via `rlm_query`; inspect the result and branch. If it suffices, use it. If not, break into one easier subproblem and delegate that only. More branches, one path runs—don't load the model. Example: prove sqrt 2 irrational.
```repl
r = rlm_query("Prove sqrt 2 is irrational. Give a 1-2 sentence proof, or reply only: USE_LEMMA or USE_CONTRADICTION.")
if "USE_LEMMA" in r.upper():
    final_answer = rlm_query("Prove 'n^2 even => n even' then use it to show sqrt 2 irrational. Two sentences.")
```

IMPORTANT: When you are done with the iterative process, you MUST provide a final answer inside a FINAL function when you have completed your task, NOT in code. Do not use these tags unless you have completed your task. You have two options:
1. Use FINAL(the answer is X) to provide the answer directly as a string if the final answer is "the answer is X".
2. Use FINAL_VAR(variable_name) to return a variable you have created in the REPL environment as your final output if the answer is the value of the variable named `variable_name`.

WARNING - COMMON MISTAKE: FINAL_VAR retrieves an EXISTING variable. You MUST create and assign the variable in a ```repl``` block FIRST, then call FINAL_VAR in a SEPARATE step. For example:
- WRONG: Calling FINAL_VAR(my_answer) without first creating `my_answer` in a repl block
- CORRECT: First run
```repl
my_answer = "the result"
print(my_answer)
```
then in the NEXT response call 

```repl
FINAL_VAR(my_answer)
```

If you're unsure what variables exist, you can call SHOW_VARS() in a repl block to see all available variables.

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment and recursive LLMs as much as possible. Remember to explicitly answer the original query in your final answer.
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
