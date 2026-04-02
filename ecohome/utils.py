import json
import re
import os
from langchain_core.messages import HumanMessage
from ragas import SingleTurnSample
from ragas.metrics import (
    AnswerCorrectness,
    AnswerSimilarity,
    AspectCritic,
    AnswerRelevancy,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def _build_metric_feedback(
    question: str,
    response: str,
    expected: str,
    scores: dict,
    llm,
) -> dict:
    """
    Call the LLM once to produce per-dimension feedback and a summary.
    Returns a dict: {dimension: explanation, ..., "summary": str}
    """
    prompt = f"""
You are an expert evaluator for an AI energy advisor (EcoHome). Given:

QUESTION: {question}
AGENT RESPONSE: {response}
EXPECTED RESPONSE: {expected}

SCORES (0–1 for floats, 0/1 for binary):
- answer_correctness : {scores["answer_correctness"]}
- answer_relevancy   : {scores["answer_relevancy"]}
- completeness       : {scores["completeness"]}
- actionability      : {scores["actionability"]}
- usefulness         : {scores["usefulness"]}

For EACH dimension, write 1–2 sentences explaining WHY the score is what it is.
Then write a 2–3 sentence overall summary with the most important improvement to make.

Return ONLY valid JSON in exactly this shape:
{{
  "answer_correctness": "...",
  "answer_relevancy":   "...",
  "completeness":       "...",
  "actionability":      "...",
  "usefulness":         "...",
  "summary":            "..."
}}
"""
    try:
        raw = llm.invoke([HumanMessage(content=prompt)])
        match = re.search(r"\{.*\}", raw.content, re.DOTALL)
        if match:
            return json.loads(match.group())
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: rule-based feedback if LLM call fails
    return {
        "answer_correctness": "High"
        if scores["answer_correctness"] >= 0.7
        else "Low — factual gaps detected.",
        "answer_relevancy": "On-topic."
        if scores["answer_relevancy"] >= 0.7
        else "Response drifted from the question.",
        "completeness": "All key points covered."
        if scores["completeness"]
        else "Key points from the expected answer are missing.",
        "actionability": "Concrete steps provided."
        if scores["actionability"]
        else "Response lacks specific, actionable guidance.",
        "usefulness": "Helpful to the user."
        if scores["usefulness"]
        else "Response is not practically helpful.",
        "summary": (
            f"Aggregate score: {scores['aggregate']}. "
            + ("Correctness needs work. " if scores["answer_correctness"] < 0.5 else "")
            + ("Completeness needs work. " if not scores["completeness"] else "")
        ),
    }


def evaluate_response(
    question: str,
    final_response: str,
    expected_response: str,
    metrics: dict,
) -> dict:
    """
    Evaluate a single response against expected response using Ragas.
    Returns a dict with per-metric scores, a weighted aggregate (0–1),
    and comprehensive per-dimension feedback strings.
    """

    # ── Guard: skip errored responses ────────────────────────────────────────
    last_message = final_response["messages"][-1].content
    if last_message.startswith("Error:"):
        return {
            "answer_correctness": 0.0,
            "answer_relevancy": 0.0,
            "completeness": 0,
            "actionability": 0,
            "usefulness": 0,
            "aggregate": 0.0,
            "feedback": {
                "answer_correctness": "Skipped — agent returned an error.",
                "answer_relevancy": "Skipped — agent returned an error.",
                "completeness": "Skipped — agent returned an error.",
                "actionability": "Skipped — agent returned an error.",
                "usefulness": "Skipped — agent returned an error.",
                "summary": f"Response was skipped due to error: {last_message}",
            },
            "skipped": True,
            "skip_reason": last_message,
        }

    # ── Build Ragas sample ────────────────────────────────────────────────────
    sample = SingleTurnSample(
        user_input=question,
        response=last_message,
        reference=expected_response,
    )

    # ── Score all metrics ─────────────────────────────────────────────────────
    correctness = round(metrics["answer_correctness"].single_turn_score(sample), 3)
    relevancy = round(metrics["answer_relevancy"].single_turn_score(sample), 3)
    completeness = int(metrics["completeness"].single_turn_score(sample))
    actionability = int(metrics["actionability"].single_turn_score(sample))
    usefulness = int(
        metrics["usefulness"].single_turn_score(sample)
    )  # ← NEW dedicated metric

    scores = {
        "answer_correctness": correctness,
        "answer_relevancy": relevancy,
        "completeness": completeness,
        "actionability": actionability,
        "usefulness": usefulness,  # ← NEW
        "skipped": False,
        "skip_reason": None,
    }

    # Weighted aggregate — usefulness absorbs part of actionability's old weight
    scores["aggregate"] = round(
        0.35 * correctness
        + 0.20 * relevancy
        + 0.20 * completeness
        + 0.15 * usefulness  # ← usefulness as first-class dimension
        + 0.10 * actionability,
        3,
    )

    # ── Comprehensive feedback ────────────────────────────────────────────────
    scores["feedback"] = _build_metric_feedback(
        question=question,
        response=last_message,
        expected=expected_response,
        scores=scores,
        llm=metrics["raw_llm"],  # raw ChatOpenAI (not the Ragas wrapper)
    )

    return scores


def _build_tool_feedback(
    called: set,
    expected: set,
    true_positives: set,
    false_positives: set,
    false_negatives: set,
    tool_appropriateness: float,
    tool_completeness: float,
    f1: float,
    llm=None,
) -> dict:
    """
    Generate per-dimension qualitative feedback.
    Uses the LLM when available; falls back to deterministic rule-based messages.
    """

    # ── LLM-based feedback (preferred) ───────────────────────────────────────
    if llm is not None:
        prompt = f"""
You are an expert evaluator for an AI energy advisor agent (EcoHome).
The agent handles EV charging, solar forecasting, HVAC scheduling,
battery dispatch, demand response, and appliance optimization.

=== TOOL USAGE RESULTS ===
- Tools called    : {sorted(called) or "none"}
- Tools expected  : {sorted(expected) or "none"}
- Correct tools   : {sorted(true_positives) or "none"}
- Unnecessary tools (false positives): {sorted(false_positives) or "none"}
- Missing tools   (false negatives)  : {sorted(false_negatives) or "none"}

=== SCORES ===
- Tool Appropriateness (precision): {tool_appropriateness}  — were the right tools selected?
- Tool Completeness    (recall)   : {tool_completeness}     — were all necessary tools used?
- F1                              : {f1}

=== YOUR TASK ===
Write feedback for each dimension. Be specific to energy-advisor tools.

Return ONLY valid JSON in exactly this shape:
{{
  "tool_appropriateness": "1-2 sentences: why is appropriateness score {tool_appropriateness}? Name specific unnecessary tools if any.",
  "tool_completeness":    "1-2 sentences: why is completeness score {tool_completeness}? Name specific missing tools if any.",
  "summary":              "2-3 sentences: overall tool selection quality and the single most important fix."
}}
"""
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            match = re.search(r"\{.*\}", response.content, re.DOTALL)
            if match:
                return json.loads(match.group())
        except (json.JSONDecodeError, AttributeError):
            pass  # fall through to rule-based

    # ── Rule-based fallback ───────────────────────────────────────────────────
    if tool_appropriateness == 1.0:
        appropriateness_msg = (
            "All called tools were expected — no unnecessary tools used."
        )
    elif tool_appropriateness == 0.0:
        appropriateness_msg = (
            f"None of the called tools were appropriate. "
            f"Unnecessary tools used: {sorted(false_positives)}."
        )
    else:
        appropriateness_msg = (
            f"Some unnecessary tools were called: {sorted(false_positives)}. "
            f"These added noise without contributing to the correct answer."
        )

    if tool_completeness == 1.0:
        completeness_msg = "All required tools were called — nothing was missed."
    elif tool_completeness == 0.0:
        completeness_msg = (
            f"No required tools were called. "
            f"All expected tools were missed: {sorted(false_negatives)}."
        )
    else:
        completeness_msg = (
            f"Some required tools were not called: {sorted(false_negatives)}. "
            f"This likely caused an incomplete or incorrect answer."
        )

    if f1 >= 0.8:
        summary = f"Strong tool usage overall (F1={f1}). " + (
            f"Minor gaps: {sorted(false_negatives or false_positives)}."
            if (false_negatives or false_positives)
            else ""
        )
    elif f1 >= 0.5:
        summary = (
            f"Partial tool match (F1={f1}). The most impactful fix is to ensure "
            f"the agent calls: {sorted(false_negatives)}."
        )
    else:
        summary = (
            f"Poor tool selection (F1={f1}). The agent called {sorted(called) or 'nothing'} "
            f"but needed {sorted(expected)}. Review tool descriptions and agent routing logic."
        )

    return {
        "tool_appropriateness": appropriateness_msg,
        "tool_completeness": completeness_msg,
        "summary": summary,
    }


def evaluate_tool_usage(messages: list, expected_tools: list[str], llm=None) -> dict:
    """
    Evaluate if the right tools were used by inspecting LangGraph message history.

    Args:
        messages:       response["messages"] from ecohome_agent.invoke()
        expected_tools: list of expected tool name strings from the test case
        llm:            optional raw ChatOpenAI instance for qualitative feedback
                        (metrics["raw_llm"]). If None, rule-based feedback is used.

    Returns:
        dict with precision, recall, f1, named appropriateness/completeness scores,
        per-tool match details, and comprehensive feedback.
    """

    # ── 1. Extract all tool names actually called ─────────────────────────────
    called_tools = []

    for msg in messages:
        # AIMessage with tool_calls — primary source
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_name = tc["name"] if isinstance(tc, dict) else tc.name
                called_tools.append(tool_name)

        # ToolMessage — backup source (avoid double-counting with elif)
        elif hasattr(msg, "name") and msg.name:
            if msg.__class__.__name__ == "ToolMessage":
                called_tools.append(msg.name)

    called_tools_set = set(called_tools)
    expected_tools_set = set(expected_tools)

    # ── 2. Set algebra ────────────────────────────────────────────────────────
    true_positives = called_tools_set & expected_tools_set  # correct
    false_positives = called_tools_set - expected_tools_set  # extra / inappropriate
    false_negatives = expected_tools_set - called_tools_set  # missed / incomplete

    # ── 3. Core metrics ───────────────────────────────────────────────────────
    precision = len(true_positives) / len(called_tools_set) if called_tools_set else 0.0
    recall = (
        len(true_positives) / len(expected_tools_set) if expected_tools_set else 0.0
    )
    f1 = (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    )

    # ── 4. Named semantic scores (explicit, not just aliases) ─────────────────
    # tool_appropriateness  = precision:  of tools called, how many were correct?
    # tool_completeness     = recall:     of tools needed, how many were called?
    tool_appropriateness = round(precision, 3)
    tool_completeness = round(recall, 3)

    # ── 5. Comprehensive feedback ─────────────────────────────────────────────
    feedback = _build_tool_feedback(
        called=called_tools_set,
        expected=expected_tools_set,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        tool_appropriateness=tool_appropriateness,
        tool_completeness=tool_completeness,
        f1=round(f1, 3),
        llm=llm,
    )

    return {
        # Tool sets (actionable for debugging)
        "called_tools": sorted(called_tools_set),
        "expected_tools": sorted(expected_tools_set),
        "false_positives": sorted(false_positives),  # unnecessary tools used
        "false_negatives": sorted(false_negatives),  # required tools missed
        # Numeric metrics
        "tool_appropriateness": tool_appropriateness,  # precision: right tools selected?
        "tool_completeness": tool_completeness,  # recall: all needed tools used?
        "f1": round(f1, 3),
        # Summary
        "exact_match": called_tools_set == expected_tools_set,
        "feedback": feedback,
    }


# ── Ragas metric setup ────────────────────────────────────────────────────────


def build_ragas_metrics(model: str = "gpt-4o"):
    api_key = os.getenv("VOCAREUM_API_KEY")
    base_url = os.getenv("VOCAREUM_BASE_URL")

    # ✅ Keep a reference to the raw LangChain LLM
    raw_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        base_url=base_url,
        api_key=api_key,
    )

    evaluator_llm = LangchainLLMWrapper(raw_llm)  # used by Ragas metrics
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model="text-embedding-3-small",
            base_url=base_url,
            api_key=api_key,
        )
    )

    answer_similarity = AnswerSimilarity(embeddings=evaluator_embeddings)

    answer_correctness = AnswerCorrectness(
        llm=evaluator_llm,
        answer_similarity=answer_similarity,
        weights=[0.4, 0.6],
    )
    answer_relevancy = AnswerRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    completeness_critic = AspectCritic(
        name="completeness",
        llm=evaluator_llm,
        definition=(
            "Does the response cover all the key components mentioned in the "
            "expected response? Score 1 if fully covered, 0 if key parts are missing."
        ),
    )
    actionability_critic = AspectCritic(
        name="actionability",
        llm=evaluator_llm,
        definition=(
            "Does the response provide concrete, actionable recommendations "
            "(e.g., specific times, numbers, steps)? Score 1 if yes, 0 if only generic advice."
        ),
    )
    usefulness_critic = AspectCritic(
        name="usefulness",
        llm=evaluator_llm,
        definition=(
            "Is the response genuinely helpful to the user — even if it does not give "
            "step-by-step actions? Score 1 if the response answers the user's need "
            "clearly and usefully, 0 if it is vague, misleading, or unhelpful."
        ),
    )

    return {
        "llm": evaluator_llm,  # LangchainLLMWrapper — for Ragas metrics
        "raw_llm": raw_llm,  # ✅ raw ChatOpenAI — for direct .invoke() calls
        "answer_correctness": answer_correctness,
        "answer_relevancy": answer_relevancy,
        "completeness": completeness_critic,
        "actionability": actionability_critic,
        "usefulness": usefulness_critic,
    }


# ── Private helpers ───────────────────────────────────────────────────────────


def _highlight(s: dict) -> str:
    dims = {
        "correctness": s.get("answer_correctness", 0),
        "relevancy": s.get("answer_relevancy", 0),
        "completeness": s.get("completeness", 0),
        "actionability": s.get("actionability", 0),
    }
    return max(dims, key=dims.get)


def _weak_dimensions(s: dict) -> list[str]:
    weak = []
    if s.get("answer_correctness", 1) < 0.5:
        weak.append("correctness")
    if s.get("answer_relevancy", 1) < 0.5:
        weak.append("relevancy")
    if s.get("completeness", 1) == 0:
        weak.append("completeness")
    if s.get("actionability", 1) == 0:
        weak.append("actionability")
    return weak


def _build_recommendations(overall: dict, non_skipped: list[dict], llm) -> list[str]:
    """Use the evaluator LLM to generate contextual, actionable recommendations."""
    per_test_summary = "\n".join(
        [
            f"- {s['test_id']}: aggregate={s.get('aggregate')}, "
            f"correctness={s.get('answer_correctness')}, "
            f"relevancy={s.get('answer_relevancy')}, "
            f"completeness={s.get('completeness')}, "
            f"actionability={s.get('actionability')}, "
            f"usefulness={s.get('usefulness')}, "
            f"tool_f1={s['tool_eval']['f1'] if s.get('tool_eval') else 'N/A'}, "
            f"missed_tools={s['tool_eval']['false_negatives'] if s.get('tool_eval') else []}, "
            f"extra_tools={s['tool_eval']['false_positives'] if s.get('tool_eval') else []}, "
            f"weak_dims={_weak_dimensions(s)}"
            for s in non_skipped
        ]
    )

    prompt = f"""
You are an expert LLM evaluation analyst reviewing an AI energy advisor agent (EcoHome).
The agent answers questions about EV charging, solar forecasting, HVAC scheduling,
battery dispatch, demand response, and appliance optimization.

=== OVERALL METRICS ===
- Total evaluated        : {overall["evaluated"]}
- Pass rate              : {overall["passed"]}/{overall["evaluated"]}
- Mean aggregate         : {overall["mean_aggregate"]}
- Mean correctness       : {overall["mean_correctness"]}
- Mean relevancy         : {overall["mean_relevancy"]}
- Mean completeness      : {overall["mean_completeness"]}
- Mean actionability     : {overall["mean_actionability"]}
- Mean usefulness        : {overall["mean_usefulness"]}
- Mean tool F1           : {overall["mean_tool_f1"]}
- Tool appropriateness   : {overall["mean_tool_appropriateness"]}
- Tool completeness      : {overall["mean_tool_completeness"]}     
- Tool exact match rate  : {overall["tool_exact_match_rate"]}

=== PER-TEST BREAKDOWN ===
{per_test_summary}

=== YOUR TASK ===
Provide 4–6 specific, actionable recommendations to improve the agent. Each must:
1. Reference specific test IDs or metrics where relevant
2. Explain the ROOT CAUSE (not just re-state the score)
3. Suggest a CONCRETE fix (prompt change, tool description, new tool, etc.)
4. Be specific to an energy advisor agent — avoid generic LLM advice

Return ONLY a JSON array of strings, one recommendation per item.
["Recommendation 1 ...", "Recommendation 2 ...", ...]
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        match = re.search(r"\[.*\]", response.content, re.DOTALL)
        if match:
            return json.loads(match.group())
    except (json.JSONDecodeError, AttributeError):
        pass

    return [
        line.strip().lstrip("-•").strip()
        for line in response.content.splitlines()
        if line.strip()
    ]


def _print_report(report: dict):
    o = report["overall"]
    print("\n" + "=" * 65)
    print("         ECOHOME AGENT — EVALUATION REPORT")
    print("=" * 65)

    # ── Overall scores ────────────────────────────────────────────────────────
    print(
        f"\n📊 OVERALL  ({o['evaluated']} evaluated / {o['skipped']} skipped / {o['total_tests']} total)"
    )
    print(
        f"   Pass rate            : {o['passed']}/{o['evaluated']}  ({100 * o['passed'] // max(o['evaluated'], 1)}%)"
    )
    print(f"   Aggregate            : {o['mean_aggregate']:.3f}")
    print()
    print(f"   Response Metrics")
    print(f"     Correctness        : {o['mean_correctness']:.3f}")
    print(f"     Relevancy          : {o['mean_relevancy']:.3f}")
    print(f"     Completeness       : {o['mean_completeness']:.3f}")
    print(f"     Usefulness         : {o['mean_usefulness']:.3f}")
    print(f"     Actionability      : {o['mean_actionability']:.3f}")
    print()
    print(f"   Tool Metrics")
    print(f"     Appropriateness    : {o['mean_tool_appropriateness']:.3f}")
    print(f"     Completeness       : {o['mean_tool_completeness']:.3f}")
    print(f"     F1                 : {o['mean_tool_f1']:.3f}")
    print(f"     Exact match        : {o['tool_exact_match_rate']:.1%}")

    # ── Strengths ─────────────────────────────────────────────────────────────
    print("\n✅ STRENGTHS (top 3)")
    for s in report["strengths"]:
        print(
            f"   {s['test_id']:<30} aggregate={s['aggregate']:.3f}  best={s['highlight']}"
        )

    # ── Weaknesses ────────────────────────────────────────────────────────────
    print("\n❌ WEAKNESSES (bottom 3)")
    for w in report["weaknesses"]:
        weak = ", ".join(w["weak_on"]) or "—"
        missed = ", ".join(w["missed_tools"]) or "—"
        extra = ", ".join(w["extra_tools"]) or "—"
        print(f"   {w['test_id']:<30} aggregate={w['aggregate']:.3f}")
        print(f"     weak=[{weak}]  missed=[{missed}]  extra=[{extra}]")

    # ── Recommendations ───────────────────────────────────────────────────────
    print("\n💡 RECOMMENDATIONS")
    for i, r in enumerate(report["recommendations"], 1):
        print(f"   {i}. {r}")

    # ── Per-test breakdown ────────────────────────────────────────────────────
    print("\n📋 PER-TEST BREAKDOWN")
    print("   " + "-" * 61)

    for t in report["per_test"]:
        if t["skipped"]:
            status = "⚠️  SKIPPED"
        else:
            status = "✅ PASS" if t["pass"] else "❌ FAIL"

        agg = f"{t['aggregate']:.3f}" if t["aggregate"] is not None else "—"

        # response metrics
        corr = str(t["correctness"] or "—")
        rel = str(t["relevancy"] or "—")
        comp = str(t["completeness"] or "—")
        use = str(t["usefulness"] or "—")
        act = str(t["actionability"] or "—")

        # tool metrics
        t_apr = str(t["tool_appropriateness"] or "—")
        t_cmp = str(t["tool_completeness"] or "—")
        t_f1 = str(t["tool_f1"] or "—")
        t_em = "yes" if t["tool_exact_match"] else "no"

        print(f"\n   {t['test_id']}  [{status}]  aggregate={agg}")
        print(
            f"     Response  corr={corr:<6} rel={rel:<6} comp={comp:<4} use={use:<4} act={act:<4}"
        )
        print(f"     Tools     apr={t_apr:<6} cmp={t_cmp:<6} f1={t_f1:<6} exact={t_em}")

    print("\n   " + "-" * 61)

    # ── Skipped tests ─────────────────────────────────────────────────────────
    if report["skipped_tests"]:
        print(f"\n⚠️  SKIPPED: {', '.join(report['skipped_tests'])}")

    print("=" * 65)


# ── Main report generator ─────────────────────────────────────────────────────


def generate_evaluation_report(test_results: list[dict], metrics: dict) -> dict:
    """
    Generate a comprehensive evaluation report using evaluate_response
    and evaluate_tool_usage across all test results.

    Args:
        test_results: pre-collected list of result dicts from the test loop
        metrics:      dict returned by build_ragas_metrics()

    Returns:
        Structured report dict and prints a human-readable summary via _print_report().
    """

    def score_single(result: dict) -> dict:
        ragas_scores = evaluate_response(
            question=result["question"],
            final_response=result["response"],
            expected_response=result["expected_response"],
            metrics=metrics,
        )
        raw = result["response"]
        tool_scores = (
            evaluate_tool_usage(
                raw["messages"],
                result.get("expected_tools", []),
                llm=metrics.get("raw_llm"),
            )
            if raw and "messages" in raw
            else None
        )
        return {**result, **ragas_scores, "tool_eval": tool_scores}

    scored = [score_single(r) for r in test_results]
    non_skipped = [s for s in scored if not s.get("skipped")]
    skipped = [s for s in scored if s.get("skipped")]

    def avg(lst, key):
        vals = [x[key] for x in lst if x.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    tool_evals = [s["tool_eval"] for s in non_skipped if s["tool_eval"]]

    # ── Overall scores & metrics ──────────────────────────────────────────────
    overall = {
        "total_tests": len(scored),
        "skipped": len(skipped),
        "evaluated": len(non_skipped),
        "passed": sum(1 for s in non_skipped if s.get("aggregate", 0) >= 0.65),
        "mean_aggregate": avg(non_skipped, "aggregate"),
        "mean_correctness": avg(non_skipped, "answer_correctness"),
        "mean_relevancy": avg(non_skipped, "answer_relevancy"),
        "mean_completeness": avg(non_skipped, "completeness"),
        "mean_actionability": avg(non_skipped, "actionability"),
        "mean_usefulness": avg(non_skipped, "usefulness"),
        "mean_tool_appropriateness": avg(tool_evals, "tool_appropriateness"),
        "mean_tool_completeness": avg(tool_evals, "tool_completeness"),
        "mean_tool_f1": avg(tool_evals, "f1"),
        "tool_exact_match_rate": round(
            sum(
                1
                for s in non_skipped
                if s.get("tool_eval") and s["tool_eval"]["exact_match"]
            )
            / len(non_skipped),
            3,
        )
        if non_skipped
        else 0.0,
    }

    # ── Strengths & weaknesses ────────────────────────────────────────────────
    sorted_scored = sorted(
        non_skipped, key=lambda x: x.get("aggregate", 0), reverse=True
    )

    strengths = [
        {
            "test_id": s["test_id"],
            "aggregate": s.get("aggregate"),
            "highlight": _highlight(s),
        }
        for s in sorted_scored[:3]
    ]

    weaknesses = [
        {
            "test_id": s["test_id"],
            "aggregate": s.get("aggregate"),
            "weak_on": _weak_dimensions(s),
            "missed_tools": s["tool_eval"]["false_negatives"]
            if s.get("tool_eval")
            else [],
            "extra_tools": s["tool_eval"]["false_positives"]
            if s.get("tool_eval")
            else [],
        }
        for s in sorted_scored[-3:]
    ]

    # ── Recommendations ───────────────────────────────────────────────────────
    recommendations = _build_recommendations(overall, non_skipped, metrics["raw_llm"])

    # ── Per-test breakdown ────────────────────────────────────────────────────
    per_test = [
        {
            "test_id": s["test_id"],
            "aggregate": s.get("aggregate"),
            "correctness": s.get("answer_correctness"),
            "relevancy": s.get("answer_relevancy"),
            "completeness": s.get("completeness"),
            "actionability": s.get("actionability"),
            "usefulness": s.get("usefulness"),
            "tool_appropriateness": s["tool_eval"]["tool_appropriateness"]
            if s.get("tool_eval")
            else None,
            "tool_completeness": s["tool_eval"]["tool_completeness"]
            if s.get("tool_eval")
            else None,
            "tool_f1": s["tool_eval"]["f1"] if s.get("tool_eval") else None,
            "tool_exact_match": s["tool_eval"]["exact_match"]
            if s.get("tool_eval")
            else None,
            "tool_feedback": s["tool_eval"]["feedback"] if s.get("tool_eval") else None,
            "response_feedback": s.get("feedback"),
            "pass": s.get("aggregate", 0) >= 0.65,
            "skipped": s.get("skipped", False),
        }
        for s in scored
    ]

    # ── Assemble & display ────────────────────────────────────────────────────
    report = {
        "overall": overall,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "recommendations": recommendations,
        "per_test": per_test,
        "skipped_tests": [s["test_id"] for s in skipped],
    }

    _print_report(report)
    return report
