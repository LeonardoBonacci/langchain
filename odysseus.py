import json
import random
import difflib
from typing import Any, Dict, List, TypedDict

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

# -----------------------------
# Types
# -----------------------------
class OdysseusState(TypedDict, total=False):
    tasks: List[Dict[str, Any]]
    completed: List[str]
    next_task: str
    summary: str
    plan: str
    explanations: str
    event: str

# -----------------------------
# Helpers
# -----------------------------
def format_tasks(tasks: List[Dict[str, Any]]) -> str:
    return "\n".join(
        [f"- {t.get('task')} (difficulty {t.get('difficulty', '?')}): {t.get('description')}" for t in tasks]
    )

# -----------------------------
# Load tasks
# -----------------------------
with open("odysseus_tasks.json", "r", encoding="utf-8") as f:
    tasks = json.load(f)

# -----------------------------
# Initialize LLM
# -----------------------------
llm = ChatOllama(model="qwen3:8b", temperature=0.3)

# -----------------------------
# Nodes
# -----------------------------
# 1) Summarize all tasks
def summarize_tasks(state: OdysseusState) -> OdysseusState:
    summary = "\n".join([f"{x['task']}: {x['description']}" for x in state["tasks"]])
    return {"summary": summary, "completed": []}

# 2) Pick next task dynamically
def choose_path(state: OdysseusState) -> OdysseusState:
    remaining = state["tasks"]
    task_names = [t.get("task", "") for t in remaining if t.get("task")]
    prompt = PromptTemplate.from_template(
        """Odysseus is at a crossroads.

Pick the single NEXT task to do.

Rules:
- You MUST choose exactly one task name from the list below.
- Output ONLY the task name exactly as written (no numbering, no markdown, no quotes).

Task names:
{task_names}

Context (descriptions):
{tasks}
"""
    ).format(task_names="\n".join([f"- {n}" for n in task_names]), tasks=format_tasks(remaining))

    message = llm.invoke(prompt)
    raw = getattr(message, "content", str(message)).strip().split("\n")[0]

    # Normalize common formatting artifacts (bullets, numbering, markdown).
    candidate = raw.strip()
    for prefix in ("- ", "* "):
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix):].strip()
    if "." in candidate[:4]:
        # e.g. "1. Task"
        left, right = candidate.split(".", 1)
        if left.strip().isdigit():
            candidate = right.strip()
    candidate = candidate.strip(" \t\"'`*")

    if candidate in task_names:
        return {"next_task": candidate}

    # Fuzzy fallback: map to the closest task name.
    matches = difflib.get_close_matches(candidate, task_names, n=1, cutoff=0.55)
    if matches:
        return {"next_task": matches[0]}

    # Final fallback: pick something valid so we still make progress.
    return {"next_task": task_names[0] if task_names else ""}

# 3) Update state after completing a task
def complete_task(state: OdysseusState) -> OdysseusState:
    task_done = state["next_task"]
    remaining_tasks = [t for t in state["tasks"] if t["task"] != task_done]
    completed = state.get("completed", [])
    completed.append(task_done)
    return {"tasks": remaining_tasks, "completed": completed, "next_task": task_done}

# 4) Optional random event
events = ["Storm at sea", "Crew mutiny", "Helpful nymph", "Monster attack"]
def random_event(state: OdysseusState) -> OdysseusState:
    if random.random() < 0.5:  # 50% chance
        event = random.choice(events)
        print(f"🌊 Event occurs: {event}")
        return {"event": event}
    return {}

# 5) Explain final plan
def explain_plan(state: OdysseusState) -> OdysseusState:
    prompt = f"""
    Here is the sequence of tasks Odysseus completed:
    {', '.join(state.get('completed', []))}

    For each task, provide one short sentence explaining why it was done in this order.
    """
    message = llm.invoke(prompt)
    explanations = getattr(message, "content", str(message))
    return {"explanations": explanations}

# -----------------------------
# Build StateGraph
# -----------------------------
graph = StateGraph(OdysseusState)
graph.add_node("SummarizeTasks", summarize_tasks)
graph.add_node("ChooseNextTask", choose_path)
graph.add_node("CompleteTask", complete_task)
graph.add_node("RandomEvent", random_event)
graph.add_node("ExplainPlan", explain_plan)

# Entry point
graph.set_entry_point("SummarizeTasks")

# Flow: summarize -> choose -> complete -> random event -> loop until done
graph.add_edge("SummarizeTasks", "ChooseNextTask")
graph.add_edge("ChooseNextTask", "CompleteTask")
graph.add_edge("CompleteTask", "RandomEvent")

# Route depending on whether tasks remain
def route_after_event(state: OdysseusState) -> str:
    return "ExplainPlan" if len(state.get("tasks", [])) == 0 else "ChooseNextTask"

graph.add_conditional_edges(
    "RandomEvent",
    route_after_event,
    {
        "ChooseNextTask": "ChooseNextTask",
        "ExplainPlan": "ExplainPlan",
    },
)
graph.add_edge("ExplainPlan", END)

# -----------------------------
# Run
# -----------------------------
app = graph.compile()
result = app.invoke({"tasks": tasks})

# -----------------------------
# Print results
# -----------------------------
print("\n📝 TASKS SUMMARY\n" + "=" * 40)
print(result.get("summary", ""))
print("\n🏹 COMPLETED TASKS\n" + "=" * 40)
print("\n".join(result.get("completed", [])))
print("\n📜 EXPLANATIONS\n" + "=" * 40)
print(result.get("explanations", ""))