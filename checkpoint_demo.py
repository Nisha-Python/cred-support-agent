import os
import sqlite3

from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver


DB_PATH = "langgraph_checkpoint.db"


class AgentState(TypedDict):
    query: str
    step1_done: bool
    step2_done: bool
    step3_done: bool
    events: list[str]


def node_one(state: AgentState):
    print("NODE 1: executed")

    events = list(state.get("events", []))
    events.append("node_1")

    return {
        "step1_done": True,
        "events": events,
    }


def node_two(state: AgentState):
    print("NODE 2: executed")

    events = list(state.get("events", []))
    events.append("node_2")

    return {
        "step2_done": True,
        "events": events,
    }


def node_three(state: AgentState):
    print("NODE 3: executed")

    events = list(state.get("events", []))
    events.append("node_3")

    return {
        "step3_done": True,
        "events": events,
    }


def build_graph(checkpointer):
    graph = StateGraph(AgentState)

    graph.add_node("node_one", node_one)
    graph.add_node("node_two", node_two)
    graph.add_node("node_three", node_three)

    graph.add_edge(START, "node_one")
    graph.add_edge("node_one", "node_two")
    graph.add_edge("node_two", "node_three")
    graph.add_edge("node_three", END)

    return graph.compile(checkpointer=checkpointer)


def main():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    connection = sqlite3.connect(
        DB_PATH,
        check_same_thread=False,
    )

    checkpointer = SqliteSaver(connection)

    graph = build_graph(checkpointer)

    thread_id = "task15-thread"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    print("TASK 15 - SQLITE LANGGRAPH CHECKPOINT")
    print("=" * 65)
    print(f"SQLite database: {DB_PATH}")
    print(f"Thread ID: {thread_id}")
    print("=" * 65)

    print("\nPHASE 1: Execute first two nodes and simulate interruption")
    print("-" * 65)

    state = {
        "query": "Check my loan application",
        "step1_done": False,
        "step2_done": False,
        "step3_done": False,
        "events": [],
    }

    # Execute node 1.
    state = {
        **state,
        **node_one(state),
    }

    # Save checkpoint after node 1.
    graph.update_state(config, state)

    # Execute node 2.
    state = {
        **state,
        **node_two(state),
    }

    # Save checkpoint after node 2.
    graph.update_state(config, state)

    print("\nCheckpoint saved after NODE 2.")
    print("Simulating interruption now...")

    print("\nSaved state:")
    checkpoint_state = graph.get_state(config)

    print(checkpoint_state.values)

    assert checkpoint_state.values["step1_done"] is True
    assert checkpoint_state.values["step2_done"] is True
    assert checkpoint_state.values["step3_done"] is False

    print("\nPHASE 1 VALIDATION: PASSED")

    print("\nPHASE 2: Resume same thread from SQLite checkpoint")
    print("-" * 65)

    resumed_state = dict(checkpoint_state.values)

    # Resume directly from the checkpoint.
    resumed_state = {
        **resumed_state,
        **node_three(resumed_state),
    }

    graph.update_state(config, resumed_state)

    final_state = graph.get_state(config)

    print("\nFinal resumed state:")
    print(final_state.values)

    events = final_state.values["events"]

    assert events.count("node_1") == 1
    assert events.count("node_2") == 1
    assert events.count("node_3") == 1

    assert final_state.values["step1_done"] is True
    assert final_state.values["step2_done"] is True
    assert final_state.values["step3_done"] is True

    print("\nNodes executed across complete workflow:")
    print(events)

    print("\n" + "=" * 65)
    print("TASK 15 VALIDATION: PASSED")
    print("=" * 65)

    connection.close()


if __name__ == "__main__":
    main()