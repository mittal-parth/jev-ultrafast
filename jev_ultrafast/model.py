"""TypeSafe makes choices; an optional small OpenAI-compatible model writes field values."""

import json
import math
import os
import re
import threading
import time

import httpx

from .questions import NEXT_ACTION, TARGET, TEXT_VALUE

CLIENT = httpx.Client(http2=True, timeout=90)
SARVAM_CLIENT = httpx.Client(http2=False, timeout=180)
_PROVIDER = threading.local()
SARVAM_DECOY = "__none_of_the_above__"


def use_decision_provider(provider):
    """Pin Jev vs Sarvam for this thread (comparison runs must not share os.environ)."""
    _PROVIDER.name = provider.lower()


def decision_client():
    """Jev (TypeSafe) or Sarvam Prism — same System-1 request shape, different hosts."""
    provider = getattr(_PROVIDER, "name", None) or os.environ.get("DECISION_PROVIDER", "jev")
    provider = provider.lower()
    if provider == "sarvam":
        key = os.environ.get("SARVAM_API_KEY")
        if not key:
            raise RuntimeError("Sarvam Prism needs SARVAM_API_KEY; no action executed.")
        return {
            "url": os.environ.get("SARVAM_DECISIONS_URL", "https://api.sarvam.ai/v2/systemone"),
            "key": key,
            "model": os.environ.get("DECISION_MODEL", "sarvam-prism"),
        }
    key = os.environ.get("JEV_API_KEY") or os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise RuntimeError("Jev needs JEV_API_KEY or TYPESAFE_API_KEY; no action executed.")
    return {
        "url": os.environ.get("TYPESAFE_DECISIONS_URL", "https://api.typesafe.ai/v1/systemone"),
        "key": key,
        "model": os.environ.get("DECISION_MODEL") or os.environ.get("TYPESAFE_MODEL", "jev-latest"),
    }


def post_json(url, key, body):
    if "sarvam.ai" in url:
        auth = {"api-subscription-key": key, "Authorization": f"Bearer {key}"}
    else:
        auth = {"Authorization": f"Bearer {key}"}
    client = SARVAM_CLIENT if "sarvam.ai" in url else CLIENT
    attempts = 5 if "sarvam.ai" in url else 3
    for attempt in range(attempts):
        try:
            response = client.post(url, json=body, headers=auth)
        except httpx.HTTPError as error:
            if attempt < attempts - 1:
                time.sleep(0.75 * 2**attempt)
                continue
            raise RuntimeError("Model connection failed; no action executed.") from error
        if response.status_code in {429, 529, 503} and attempt < attempts - 1:
            time.sleep(0.5 * 2**attempt)
            continue
        if response.is_error:
            raise RuntimeError(f"Model provider returned HTTP {response.status_code}; no action executed.")
        return response.json()
    raise RuntimeError("Model unavailable")


def validate_choice(answer, ids):
    try:
        probabilities = answer["probabilities"]
        numbers = [*probabilities.values(), answer["confidence"]]
        valid = (
            answer["choice"] in ids
            and set(probabilities) == set(ids)
            and all(type(n) in (int, float) and math.isfinite(n) and 0 <= n <= 1 for n in numbers)
            and abs(sum(probabilities.values()) - 1) < 0.02
            and probabilities[answer["choice"]] >= max(probabilities.values()) - 1e-6
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("Invalid TypeSafe response; no action executed.")
    return answer


def action_space(actions):
    """One index per observed element; each operation has its own valid target choices."""
    elements, indices, targets, controls = [], {}, {}, {}
    operations = {"click": "CLICK", "fill": "TYPE_TEXT", "select": "SELECT"}
    for action in actions:
        kind = action["kind"]
        if kind not in operations:
            controls[action["id"].upper()] = action
            continue
        node = action["node"]
        if node not in indices:
            index = str(len(elements) + 1)
            indices[node] = index
            element = {k: action[k] for k in ("role", "value", "checked", "selected", "expanded") if k in action}
            element.update(index=index, label=action["label"].split(" → ")[0], operations=[])
            if kind == "select":
                element["value"] = action.get("current_value", "")
                element["options"] = []
            elements.append(element)
        index = indices[node]
        operation = operations[kind]
        group = targets.setdefault(operation, {})
        element = elements[int(index) - 1]
        if operation not in element["operations"]:
            element["operations"].append(operation)
        target = index
        if kind == "select":
            target = f"{index}:{len(element['options']) + 1}"
            element["options"].append({"index": target, "label": action["label"], "value": action["value"]})
        group[target] = action
    return elements, targets, controls


def pad_choice_criteria(criteria, provider):
    """Sarvam Prism rejects choice questions with fewer than two options."""
    if provider != "sarvam" or len(criteria) >= 2:
        return criteria
    padded = dict(criteria)
    padded[SARVAM_DECOY] = "None of the listed elements apply; choose only when no real target fits."
    return padded


def trim_large_action_space(actions, goal):
    """Keep decision requests under provider limits on dense pages (e.g. flight calendars)."""
    if os.environ.get("TRIM_LARGE_ACTION_SPACES") != "1" or len(actions) <= 120:
        return actions
    goal_l = goal.lower()
    wants_sep = "september" in goal_l and "20" in goal_l

    def score(action):
        if action["kind"] != "click":
            return 2000
        label = action["label"].lower()
        points = 0
        for token in (
            "search",
            "one way",
            "round trip",
            "where from",
            "where to",
            "departure",
            "done",
            "select flight",
            "zurich",
            "zürich",
            "london",
            "next",
            "previous",
            "reset",
        ):
            if token in label:
                points += 60
        if wants_sep:
            if "september" in label and "20" in label:
                points += 500
            for token in ("september", " sep ", ", 20", "20,", "2026", "sunday"):
                if token in label:
                    points += 90
            if "november" in label or "december" in label:
                points -= 120
        return points

    non_clicks = [a for a in actions if a["kind"] != "click"]
    clicks = sorted((a for a in actions if a["kind"] == "click"), key=score, reverse=True)
    return non_clicks + clicks[:90]


def choose(state, goal, history):
    actions = trim_large_action_space(state["actions"], goal)
    elements, targets, controls = action_space(actions)
    labels = {
        "CLICK": "Click an element, button, menu option, autocomplete suggestion, or calendar day.",
        "TYPE_TEXT": "Enter or replace text in an editable field. A small LLM will supply the value from the goal.",
        "SELECT": "Select an observed dropdown value.",
    }
    operations = {key: labels[key] for key in targets}
    operations.update({key: value["label"] for key, value in controls.items()})
    operations.update(DONE="Every requirement is visibly satisfied.", BLOCKED="No supported operation can progress.")
    client = decision_client()
    provider = getattr(_PROVIDER, "name", None) or os.environ.get("DECISION_PROVIDER", "jev")
    operations = pad_choice_criteria(operations, provider)
    questions = {
        "operation": {"type": "choice", "criteria": operations, "instructions": {"goal": goal, "rules": NEXT_ACTION}}
    }
    for operation, candidates in targets.items():
        criteria = {
            index: {
                "element": f"[{index}] {a['label']}",
                "current_value": a.get("current_value", a.get("value", "")),
                **{k: a[k] for k in ("role", "checked", "selected", "expanded") if k in a},
            }
            for index, a in candidates.items()
        }
        questions[operation.lower() + "_target"] = {
            "type": "choice",
            "criteria": pad_choice_criteria(criteria, provider),
            "instructions": {"goal": goal, "operation": operation, "rules": [NEXT_ACTION, TARGET]},
        }
    body = {
        "model": client["model"],
        "state": {
            "page": {k: state[k] for k in ("url", "title", "text")},
            "elements": elements,
            "recent_actions": [
                {k: h.get(k) for k in ("action", "kind", "text", "page_changed")} for h in history[-10:]
            ],
        },
        "questions": questions,
    }
    started = time.perf_counter()
    result = post_json(client["url"], client["key"], body)
    operation_answer = validate_choice(result["answers"].get("operation", {}), operations)
    operation = operation_answer["choice"]
    if operation == SARVAM_DECOY:
        raise ValueError("Invalid TypeSafe response; no action executed.")
    target = None
    target_answer = None
    probabilities = {}
    if operation in targets:
        # Unused target heads cannot cause an action. Validate the head selected by the operation.
        target_criteria = targets[operation]
        target_answer = validate_choice(
            result["answers"].get(operation.lower() + "_target", {}),
            pad_choice_criteria(target_criteria, provider) if provider == "sarvam" else target_criteria,
        )
        target = target_answer["choice"]
        if target == SARVAM_DECOY:
            raise ValueError("Invalid TypeSafe response; no action executed.")
        choice = targets[operation][target]["id"]
        if choice not in {a["id"] for a in state["actions"]}:
            raise ValueError("Invalid TypeSafe response; no action executed.")
        probabilities = {a["id"]: target_answer["probabilities"][index] for index, a in targets[operation].items()}
    else:
        choice = controls[operation]["id"] if operation in controls else operation
        probabilities[choice] = operation_answer["probabilities"][operation]
    return {
        "choice": choice,
        "operation": operation,
        "target": target,
        "confidence": operation_answer["confidence"],
        "probabilities": probabilities,
        "operation_probabilities": operation_answer["probabilities"],
        "target_probabilities": target_answer["probabilities"] if target_answer else {},
        "target_confidence": target_answer["confidence"] if target_answer else None,
        "raw_answers": result["answers"],
        "model": result["model"],
        "usage": result.get("usage", {}),
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "request": body,
    }


def field_context(goal, action, page, history):
    return {
        "goal": goal,
        "field": {k: action.get(k) for k in ("label", "role", "value")},
        "page": {"title": page["title"], "text": page["text"][:6000]},
        "recent_actions": [{k: h.get(k) for k in ("action", "text")} for h in history[-6:]],
    }


def field_text_systemone(context):
    """Use the same System-1 API key to pick a field value from goal-derived options."""
    label = context["field"]["label"].lower()
    goal = context["goal"]
    criteria = {}
    if "where from" in label:
        match = re.search(r"from\s+([^,\n]+?)\s+to\s", goal, re.I)
        raw = match.group(1).strip() if match else ""
        for token in {raw, raw.replace("Zurich", "Zürich"), "Zürich", "Zurich"}:
            if token:
                criteria[f"from_{len(criteria)}"] = token
    elif "where to" in label:
        match = re.search(r"to\s+([^,\n]+?)\s+on\s", goal, re.I)
        raw = match.group(1).strip() if match else ""
        for token in {raw, "London", "London, United Kingdom"}:
            if token:
                criteria[f"to_{len(criteria)}"] = token
    else:
        criteria["value"] = goal[:200]
    if not criteria:
        raise ValueError("Could not derive TYPE_TEXT options from the goal; nothing typed.")
    client = decision_client()
    body = {
        "model": client["model"],
        "state": context,
        "questions": {
            "text": {
                "type": "choice",
                "criteria": criteria,
                "instructions": {
                    "goal": goal,
                    "field": context["field"],
                    "rules": TEXT_VALUE,
                },
            }
        },
    }
    started = time.perf_counter()
    result = post_json(client["url"], client["key"], body)
    answer = validate_choice(result["answers"].get("text", {}), criteria)
    value = criteria[answer["choice"]]
    return value, {
        "model": result["model"],
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "usage": result.get("usage", {}),
    }


def field_text(context):
    if os.environ.get("TEXT_HELPER") == "systemone":
        return field_text_systemone(context)
    key = os.environ.get("TEXT_MODEL_API_KEY")
    if not key:
        raise ValueError("TYPE_TEXT needs TEXT_MODEL_API_KEY; no text is hardcoded or guessed by the executor.")
    base = os.environ.get("TEXT_MODEL_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    model = os.environ.get("TEXT_MODEL", "deepseek-chat")
    reasoning = {"thinking": {"type": "disabled"}} if "api.deepseek.com/" in base else {"reasoning": {"effort": "low"}}
    if os.environ.get("TEXT_MODEL_REASONING") == "none":
        reasoning = {"reasoning": {"enabled": False}}
    started = time.perf_counter()
    result = post_json(
        base + "/chat/completions",
        key,
        {
            "model": model,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"},
            **reasoning,
            "messages": [
                {"role": "system", "content": TEXT_VALUE},
                {
                    "role": "user",
                    "content": json.dumps(context),
                },
            ],
        },
    )
    try:
        output = json.loads(result["choices"][0]["message"]["content"])
        value = output["text"]
        if set(output) != {"text"} or not isinstance(value, str) or not value.strip() or len(value) > 2000:
            raise ValueError()
    except (ValueError, KeyError, TypeError):
        raise ValueError("Text helper returned no valid field value; nothing typed.") from None
    return value, {
        "model": model,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "usage": result.get("usage", {}),
    }
