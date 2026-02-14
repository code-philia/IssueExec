# test_selection_prompt
obtain_relevant_tests_prompt = """
You are identifying test functions that currently exhibit **FALSE NEGATIVE** behavior: tests that should have prevented this bug but failed to do so because they did not validate the real requirement under the triggering condition.

Core principle:
- If Component X produces incorrect data/state that breaks Component Y, the primary fault is in **Component X's tests** not validating X's behavior correctly. Component Y is often the downstream victim.

Task:
- Given a GitHub issue description and a list of available test functions (with their enriched representations), select tests that are most likely to be **requirement-relevant** to the root cause component and thus most useful for downstream trace-guided localization.

Selection rules (keep recall high):
1. Identify the **root-cause component** (the component whose behavior is incorrect under some condition).
2. Prefer tests that **directly exercise** that component or its problematic method(s).
3. Include tests that cover the component under **different configurations / edge conditions** related to the issue.
4. Deprioritize pure downstream consumers or observers unless they are needed for coverage breadth.
5. Maintain **HIGH RECALL**: it's better to include extra potentially relevant tests than to miss a few critical ones.

### GitHub Issue Description ###
{problem_statement}

### Test Functions ###
{test_functions}

Select up to {max_tests} test functions.

Return ONLY the selected test functions in this exact format:
```
test_path_1::test_function_1
test_path_2::test_function_2
```

Requirements:
- Output must be wrapped in triple backticks
- One test per line in format: file_path::function_name
- No numbering, bullets, markdown, or headers
- No explanations or additional text

"""


# trace_analysis_prompt
analyze_blind_spots_prompt = """
# Test False Negative Root Cause Analysis Framework

## Core Principle
In projects with good test coverage, new bugs often indicate **test false negatives**: related tests pass even though behavior is incorrect. This mismatch helps pinpoint where production code violates requirements that tests fail to encode.

## Task Description
**Issue Description**:
{problem_statement}

**Related Test Functions and Coverage Information**:
{test_functions}

## Objective
Explain why the related tests can pass (false negatives), and infer which requirement/assumption is missing. Focus on locating **problematic production code** (not test code changes).

## Analysis Requirements
1. Identify how current assertions are insufficient or misaligned with the issue.
2. Identify test input/data assumptions that differ from the issue’s trigger condition(s).
3. Describe the likely implementation assumptions that let incorrect behavior pass tests.
4. Derive actionable hints for where the production logic is wrong and how it propagates.

## Output Format Requirements
Wrap the entire output in triple backticks and follow this exact schema:

```
BLIND_SPOTS_ANALYSIS:

false_negative_mechanism:
- assertion_deviation: [difference between what tests validate vs what issue requires]
- data_assumption_defect: [gap between test inputs/conditions vs real trigger conditions]
- logic_assumption_error: [implementation assumption implied by tests that fails in reality]

root_cause_hypothesis:
- likely_root_cause_behavior: [what behavior is wrong, under what condition]
- likely_missing_requirement: [what requirement is not enforced by tests]
- propagation_summary: [how wrong behavior leads to the observed symptom]

systematic_risk_areas:
- symmetric_operation_risk: [paired/inverse operations likely to share the same blind spot]
- shared_component_risk: [shared utilities/data structures likely affected]
- similar_logic_risk: [other similar logic patterns likely to contain the same assumption]

modification_priority:
1. [Highest priority: most likely root-cause logic to inspect/fix]
2. [High priority: critical dependent logic that must stay consistent]
3. [Medium priority: preventive checks for symmetric/shared areas]
```
"""

# trace_analysis_prompt
localize_suspicious_locations_prompt = """
# Test False Negative Code Mapping Framework

## Core Principle
False negatives happen when **incorrect implementations still satisfy incomplete assertions**. Use the false-negative mechanisms to map onto concrete production code locations.

## Inputs

**Issue Description**:
{problem_statement}

**Test False Negative Analysis Results**:
{blind_spots_analysis}

**Code Locations Covered by Related Tests**:
{test_coverage_locations}

## Mapping Objective
Produce a ranked set of suspicious code locations that are likely to require modification.

## CRITICAL CONSTRAINT
You MUST ONLY reference functions/methods/classes that appear in **Code Locations Covered by Related Tests**.
If the issue strongly points to code not in the coverage list, write:
- `NOT_IN_COVERAGE: ...` and explain the limitation.

## Output Format Requirements
Wrap the entire output in triple backticks and follow this exact schema:

```
FALSE_NEGATIVE_CODE_MAPPING:

direct_false_negative_sources:
- file_path::function_name [MECHANISM: assertion_deviation | data_assumption_violation | logic_assumption_error - brief reason]
- file_path::ClassName.method_name [MECHANISM: ... - brief reason]
- NOT_IN_COVERAGE: [if applicable]

assumption_propagation_risks:
- file_path::function_name [RISK: shared_utility_propagation | data_structure_propagation | symmetric_operation - brief reason]
- NOT_IN_COVERAGE: [if applicable]

systematic_risk_areas:
- file_path::function_name [PATTERN: similar_logic_pattern | untested_branches | similar_signature - brief reason]
- NOT_IN_COVERAGE: [if applicable]

modification_priority_ranking:
1. CRITICAL: [locations most likely to be the root cause; or NOT_IN_COVERAGE]
2. HIGH: [locations likely needing coordinated changes (pairs/shared state)]
3. MEDIUM: [preventive review locations]

mapping_reasoning:
- mechanism_to_code: [how the blind-spot mechanisms map to these locations]
- clustering_notes: [paired operations / shared state that should be kept consistent]
- coverage_limitations: [only if applicable]
```
"""



# refinement prompt
suppletory_localize_prompt = """
Please look through the following GitHub Problem Description and Relevant Files.
Identify all locations that need inspection or editing to fix the problem, including directly related areas as well as any potentially related global variables, functions, and classes.
For each location you provide, either give the name of the class, the name of a method in a class, the name of a function, or the name of a global variable.

### GitHub Problem Description ###
{problem_statement}

### Relevant Files ###
{file_contents}

###

Please provide the complete set of locations as either a class name, a function name, or a variable name.
Note that if you include a class, you do not need to list its specific methods.
You can include either the entire class or don't include the class name and instead include specific methods in the class.

### Output Format ###
CRITICAL: Output ONLY the code block with locations. Do NOT include any explanations, analysis, or additional text before or after the code block.

Format requirements:
- Use format: `file_path::location_name`
- Each location on a separate line
- Wrap all locations in triple backticks (```)
- Do NOT add any text outside the code block
- If no locations found, output empty code block: ```\n```

Example output (copy this exact format):
```
sklearn/model_selection/_split.py::BaseCrossValidator.__repr__
sklearn/model_selection/_split.py::BaseShuffleSplit.split
sklearn/model_selection/_split.py::_BaseKFold
sklearn/model_selection/_split.py::func1
```

Remember: Output ONLY the code block, nothing else.
"""


# rerank_prompt
rerank_prompt_with_context = """
# Code Location Reranking Task

## Objective
Rerank candidate code locations by likelihood of containing the **root cause requiring modification**.
- Identify **clusters** that must be modified together to maintain consistency.
- Prefer **root cause** over symptom-only locations.
- **Filter out** locations that do not require code changes.

**Issue Description**:
{problem_statement}

**Candidate Code Locations**:
{candidate_locations}

## Input Format
Part 1: Location overview (numbered list of ALL candidates):
```
(1) file_path::identifier
(2) file_path::identifier
...
```

Part 2: Location details for each candidate (separated by ---):
```
### Location N: file_path::identifier ###
```python
<code content>
```
```

Identifier may be:
- `ClassName`
- `function_name`
- `ClassName.method_name`

For large classes, only a skeleton may be provided.

## Ranking Criteria
1. Root cause vs symptom (highest weight)
- Prefer producers/transformers of incorrect data/state.
- Deprioritize pure downstream consumers.
- Exclude pure observers (logging, reporting, display-only) unless they must change for correctness.

2. Consistency clustering (critical)
- Paired/symmetric operations (encode/decode, serialize/deserialize, add/remove, etc.) should be kept consistent.
- Same-class methods sharing state may need coordinated edits.
- Shared data format/structure changes can require multiple locations to update.

3. Architectural appropriateness
- Favor fixes at the correct abstraction level with minimal side effects.

4. Causal chain position
- Upstream cause > midstream transform > downstream consume > observer (exclude).

## Filtering Rule
Exclude a location if it is:
- Unrelated to the issue, OR
- Only a symptom manifestation point, OR
- A passive observer (logger/monitor/UI renderer) that does not affect buggy behavior.

When uncertain, err on the side of inclusion.

## Output Format Requirements
Output ONLY the final reranked list wrapped in triple backticks:
```
(1) file_path::identifier
(2) file_path::identifier
...
```

Requirements:
- Sequential numbering (1), (2), (3)...
- Include ONLY locations that require modification (may be fewer than the overview)
- Use EXACT identifiers as provided in the input
- No explanations or extra text inside the code block
"""