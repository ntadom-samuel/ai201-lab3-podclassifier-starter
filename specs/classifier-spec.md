# Classifier Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 2.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `build_few_shot_prompt()` and
`classify_episode()` in `classifier.py`.

---

## build_few_shot_prompt(labeled_examples, description)

### What it does

Constructs a prompt string for the LLM that includes the task instructions,
all labeled training examples, and the new episode description to classify.

### Inputs

| Parameter          | Type         | Description                                                                                                          |
| ------------------ | ------------ | -------------------------------------------------------------------------------------------------------------------- |
| `labeled_examples` | `list[dict]` | Each dict has `"title"`, `"description"`, `"label"` (and others). These are the examples you labeled in Milestone 1. |
| `description`      | `str`        | The episode description to classify.                                                                                 |

### Output

| Return value | Type  | Description                                        |
| ------------ | ----- | -------------------------------------------------- |
| prompt       | `str` | A complete prompt string ready to send to the LLM. |

---

### Spec fields — fill these in before writing code

**Task instruction (what should the LLM know about the task?):**

```
You are classifying podcast episodes by their format. Classify the episode
into exactly one of these four labels:

- interview: a conversation between a host and one or more guests
- solo: a single host speaking from memory, experience, or opinion — no guests,
  no assembled external sources
- panel: multiple guests with roughly equal speaking time, often debating or
  discussing a topic together
- narrative: a story assembled from external sources — interviews, archival
  audio, reporting — with a clear narrative arc

Return only the label and your reasoning. Do not explain the taxonomy.
```

---

**How should labeled examples be formatted in the prompt?**

```
Each example should include the episode title, a brief excerpt or the full
description, and the correct label. Separate examples with a blank line or
a delimiter like "---". Include all fields that help the model see why the
label was applied — title and description are both useful; other fields
(like episode ID) are not needed.
```

---

**Example block sketch (write one concrete example):**

```
Title: {title}
Description: {description}
Label: {label}
```

---

**How should the new episode (to be classified) be presented?**

```
Present it in the same format as the labeled examples, but omit the Label
line and replace it with an instruction to classify. For example:

Title: {title}
Description: {description}
Label: ?

Then add a line like: "Classify the episode above. Return your answer in
the format below:" followed by the output format you chose.
```

---

**What output format should you request from the LLM?**

```
Label and reasoning on separate lines. E.g.

"
interview
Reasoning: clear host-guest Q&A with a single guest...

"
```

---

**Edge cases to handle in the prompt:**

```
labeled_examples empty: build a valid zero-shot prompt — keep the task
instruction and label definitions, and conditionally omit the examples
section entirely (no dangling header or stray "---" delimiters). The label
definitions alone are enough for the model to classify.

description short/empty: still build a valid prompt; place whatever text
exists on the Description line (blank if empty). Don't try to judge
sufficiency here — formatting is this function's only job. If the resulting
classification is unreliable, classify_episode()'s validation + "unknown"
fallback (Step 4) is the safety net.
```

---

## classify_episode(description, labeled_examples)

### What it does

Classifies a single podcast episode description using the few-shot LLM classifier.
Returns a dict with a label and reasoning.

### Inputs

| Parameter          | Type         | Description                                               |
| ------------------ | ------------ | --------------------------------------------------------- |
| `description`      | `str`        | The episode description to classify.                      |
| `labeled_examples` | `list[dict]` | Labeled training examples from `load_labeled_examples()`. |

### Output

| Return value | Type   | Description                                                                                         |
| ------------ | ------ | --------------------------------------------------------------------------------------------------- |
| result       | `dict` | Must have keys `"label"` and `"reasoning"`. `"label"` must be one of `VALID_LABELS` or `"unknown"`. |

---

### Spec fields — fill these in before writing code

**Step 1 — Build the prompt:**

```
Call build_few_shot_prompt(labeled_examples, description) and store the
returned string in a variable (e.g., prompt). Pass through both arguments
exactly as received — no modification needed before calling.
```

---

**Step 2 — Send to the LLM:**

```
Call _client.chat.completions.create() with:
  - model: the model name from config (LLM_MODEL)
  - messages: a list with one dict — {"role": "user", "content": prompt}
    (system-design.md shows an optional system message too — either shape works)
  - max_tokens: a reasonable limit (e.g., 200–300) to keep responses concise

Extract the response text from:
  response.choices[0].message.content
```

---

**Step 3 — Parse the response:**

```
Strip the response, split into lines (text.strip().splitlines()).
- label  = lines[0].strip().lower()   (first non-empty line)
- reasoning = "\n".join(lines[1:]).strip(), with an optional
  .removeprefix("Reasoning:").strip()

Normalize label to lowercase so it matches VALID_LABELS. If the response is
empty or has no parseable first line, fall through to the "unknown" path
(Step 4). Keep parsing of the reasoning loose — first line is the contract,
the rest is free text.

```

---

**Step 4 — Validate the label:**

```
After parsing, check: if label not in VALID_LABELS, set label = "unknown".
Keep the parsed reasoning regardless, so unknown results stay debuggable.

Don't guess a "closest" label or default to a common one — that hides model
failures and corrupts the accuracy evaluation. "unknown" is an honest
signal, distinct from a confident wrong answer.

Assumes Step 3 already normalized the label (strip, lower, drop trailing
punctuation); then a plain membership check is safe.
```

```python
if label not in VALID_LABELS:
    label = "unknown"
```

---

**Step 5 — Handle errors gracefully:**

```
Wrap the API call and parsing in try/except. classify_episode() must never
raise — the 20-call evaluation loop depends on each call returning a dict.

What can fail: network/API errors (timeout, rate limit, auth, 5xx);
empty or None message.content; a response that ignores the format. The
last case already resolves to "unknown" via Step 4.

On any exception, return {"label": "unknown", "reasoning": f"error: {e}"} —
same shape as success. Catch broadly here on purpose: the contract is
"always return a valid dict," and putting str(e) in reasoning keeps failures
debuggable. "unknown" is the single bucket for both invalid labels and hard
errors, so one bad call never crashes the rest.
```

```python
try:
    response = _client.chat.completions.create(...)
    text = response.choices[0].message.content
    # ... Step 3 parse, Step 4 validate ...
    return {"label": label, "reasoning": reasoning}
except Exception as e:
    return {"label": "unknown", "reasoning": f"error: {e}"}
```

---

### Return value structure

```python
{
    "label": str,      # one of VALID_LABELS, or "unknown" if invalid/error
    "reasoning": str,  # brief explanation from the LLM
}
```

---

## Notes on label quality

The classifier is only as good as your labels. If your training examples have
inconsistent or ambiguous labels, the LLM will learn the wrong pattern.

Before implementing the classifier, re-read `data/taxonomy.md` and double-check
any labels you're unsure about. Annotation quality is part of the lab.

---

## Implementation Notes

_Fill this in after implementing and testing both functions._

**Test: what does the raw LLM response look like for one episode?**

```
Episode tested: [title]
Raw response text: [paste it here]
```

**How did you parse the label out of the response?**

```
[describe the string operations — strip, split, lower, etc.]
```

**Did any episodes return `"unknown"`? If so, why?**

```
[yes / no — if yes, what did the raw response look like?]
```

**One thing about the output format that surprised you:**

```
[your answer here]
```
