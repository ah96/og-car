"""Generate paper figures and qualitative narrative examples for MACE.

Outputs
-------
figures/hospital_<robot>.pdf      — navigation graph with chosen path + impassable edges
figures/warehouse_<robot>.pdf     — same for warehouse scenario
figures/narrative_example.txt     — full 3-layer explanation for one representative case

Usage
-----
    python experiments/generate_figures.py --output-dir figures/

The script runs headless (no GUI window), writing PDF/PNG files directly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mace.semantic import (
    AffordanceReasoner,
    OntofactNavigator,
    draw_navigation_graph,
)
from mace.semantic.navigation import AStarPlanner
from scenarios.hospital import build_hospital_world
from scenarios.warehouse import build_warehouse_world


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate MACE paper figures.")
    p.add_argument("--output-dir", default="figures", help="Directory for output files.")
    p.add_argument("--format",     default="pdf",     help="Figure format: pdf or png.")
    p.add_argument("--dpi",        type=int, default=200)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-scenario graph figures
# ---------------------------------------------------------------------------

def figure_hospital(out: Path, fmt: str, dpi: int) -> None:
    onto, graph, agents = build_hospital_world()
    reasoner = AffordanceReasoner()
    planner  = AStarPlanner(graph, reasoner)

    tasks = [
        ("delivery_bot", "entrance", "icu_main",
         "Hospital — delivery_bot: ICU delivery (feasible, staff route blocked)"),
        ("cargo_bot",    "entrance", "icu_main",
         "Hospital — cargo_bot: ICU delivery (infeasible — narrow corridor + closed door)"),
        ("legged_bot",   "entrance", "icu_main",
         "Hospital — legged_bot: ICU delivery (infeasible — closed door, cannot open)"),
    ]

    for robot_name, start, goal, title in tasks:
        agent        = agents[robot_name]
        chosen       = planner.find_path(start, goal, agent)

        # Collect feasible alternative paths for context
        alternatives = planner.find_k_paths(start, goal, agent, k=3)
        if alternatives and alternatives[0].nodes == chosen.nodes:
            alternatives = alternatives[1:]

        save = str(out / f"hospital_{robot_name}.{fmt}")
        draw_navigation_graph(
            graph,
            chosen_path       = chosen if chosen.is_feasible else None,
            alternative_paths = alternatives if alternatives else None,
            reasoner          = reasoner,
            agent             = agent,
            title             = title,
            save_path         = save,
            show              = False,
            paper_mode        = True,
            dpi               = dpi,
        )
        plt.close("all")
        print(f"  Saved {save}")


def figure_warehouse(out: Path, fmt: str, dpi: int) -> None:
    onto, graph, agents = build_warehouse_world()
    reasoner = AffordanceReasoner()
    planner  = AStarPlanner(graph, reasoner)

    tasks = [
        ("picker_bot",   "loading_bay", "storage_A",
         "Warehouse — picker_bot: storage_A (feasible)"),
        ("picker_bot",   "loading_bay", "mezzanine",
         "Warehouse — picker_bot: mezzanine (infeasible — ramp too steep for wheeled robot)"),
        ("forklift_bot", "loading_bay", "narrow_aisle",
         "Warehouse — forklift_bot: narrow_aisle (infeasible — too wide for aisle)"),
        ("tracked_bot",  "loading_bay", "mezzanine",
         "Warehouse — tracked_bot: mezzanine (feasible via ramp)"),
    ]

    for robot_name, start, goal, title in tasks:
        agent        = agents[robot_name]
        chosen       = planner.find_path(start, goal, agent)
        alternatives = planner.find_k_paths(start, goal, agent, k=3)
        if alternatives and chosen.is_feasible and alternatives[0].nodes == chosen.nodes:
            alternatives = alternatives[1:]

        save = str(out / f"warehouse_{robot_name}_{goal.replace('_','-')}.{fmt}")
        draw_navigation_graph(
            graph,
            chosen_path       = chosen if chosen.is_feasible else None,
            alternative_paths = alternatives if alternatives else None,
            reasoner          = reasoner,
            agent             = agent,
            title             = title,
            save_path         = save,
            show              = False,
            paper_mode        = True,
            dpi               = dpi,
        )
        plt.close("all")
        print(f"  Saved {save}")


# ---------------------------------------------------------------------------
# Qualitative narrative examples
# ---------------------------------------------------------------------------

def generate_narratives(out: Path) -> None:
    txt_path = out / "narrative_example.txt"

    lines: list[str] = []

    def section(header: str) -> None:
        lines.append("\n" + "=" * 72)
        lines.append(header)
        lines.append("=" * 72)

    # ── Case 1: Hospital / cargo_bot (infeasible, actionable fix exists) ──────
    section("CASE 1 — Hospital: cargo_bot → ICU main (infeasible path)")
    lines.append(
        "Scenario:  cargo_bot (width 1.1 m, no arm) navigating from\n"
        "           'entrance' to 'icu_main' in a hospital.\n"
        "Challenge: The direct corridor is too narrow (0.9 m) AND the ICU\n"
        "           doorway is closed (cargo_bot cannot open doors).\n"
    )

    onto_h, graph_h, agents_h = build_hospital_world()
    nav_h  = OntofactNavigator(onto_h, graph_h)
    agent  = agents_h["cargo_bot"]
    path_h, exp_h = nav_h.navigate("entrance", "icu_main", agent, k_alternatives=3)

    lines.append(f"Planner result:  {'INFEASIBLE — no reachable path' if not path_h.is_feasible else path_h.summary()}")

    if exp_h:
        lines.append("\n--- MACE Explanation Report ---")
        lines.append(nav_h.explainer.format_report(exp_h))
    else:
        lines.append("[No explanation generated — infeasible path, no explanation produced by orchestrator]")

    # query_why_not for the most obvious alternative
    # Three routes to icu_main — each blocked for a different reason
    cf_queries = [
        (
            ["entrance", "lobby", "corridor_a", "icu_entrance", "icu_main"],
            "Direct route via corridor A (blocked: corridor too narrow for cargo_bot)",
        ),
        (
            ["entrance", "lobby", "corridor_a", "corridor_b", "icu_entrance", "icu_main"],
            "Detour via corridor B (still blocked: closed door, cargo_bot cannot open)",
        ),
        (
            ["entrance", "lobby", "corridor_a", "corridor_b", "staff_corridor", "icu_main"],
            "Staff shortcut (blocked: restricted access area)",
        ),
    ]
    for alt_nodes, description in cf_queries:
        lines.append(f"\n--- query_why_not: '{description}' ---")
        cf = nav_h.query_why_not("entrance", "icu_main", agent, alt_nodes=alt_nodes)
        lines.append(f"Query: {cf.query}")
        lines.append(f"Answer: {cf.explanation}")
        lines.append(f"Proposed changes ({len(cf.changes)}):")
        for c in cf.changes:
            lines.append(
                f"  • {c.individual_name}.{c.property_name}: "
                f"{c.original_value!r} → {c.counterfactual_value!r}  "
                f"[{c.effort()} effort, {'actionable' if c.is_actionable() else 'structural'}]"
            )
            lines.append(f"    {c.rationale}")

    # ── Case 2: Hospital / delivery_bot (feasible, staff shortcut blocked) ────
    section("CASE 2 — Hospital: delivery_bot → ICU main (feasible, deviation explained)")
    lines.append(
        "Scenario:  delivery_bot (width 0.6 m, has arm, can open doors) navigating\n"
        "           from 'entrance' to 'icu_main'.\n"
        "Challenge: The robot takes the longer corridor route instead of the\n"
        "           shorter staff corridor (restricted access).\n"
    )
    onto_h2, graph_h2, agents_h2 = build_hospital_world()
    nav_h2   = OntofactNavigator(onto_h2, graph_h2)
    agent_d  = agents_h2["delivery_bot"]
    path_d, exp_d = nav_h2.navigate("entrance", "icu_main", agent_d, k_alternatives=3)
    lines.append(f"Planner result:  {path_d.summary()}")

    if exp_d:
        lines.append("\n--- MACE Explanation Report ---")
        lines.append(nav_h2.explainer.format_report(exp_d))

    lines.append("\n--- query_why_not: 'Why not the staff corridor shortcut?' ---")
    cf2 = nav_h2.query_why_not(
        "entrance", "icu_main", agent_d,
        alt_nodes=["entrance", "lobby", "corridor_a", "corridor_b", "staff_corridor", "icu_main"],
    )
    lines.append(f"Query: {cf2.query}")
    lines.append(f"Answer: {cf2.explanation}")
    lines.append(f"Proposed changes ({len(cf2.changes)}):")
    for c in cf2.changes:
        lines.append(
            f"  • {c.individual_name}.{c.property_name}: "
            f"{c.original_value!r} → {c.counterfactual_value!r}  "
            f"[{c.effort()} effort, {'actionable' if c.is_actionable() else 'structural'}]"
        )
        lines.append(f"    {c.rationale}")

    # ── Case 3: Warehouse / picker_bot (infeasible, structural constraint) ────
    section("CASE 3 — Warehouse: picker_bot → mezzanine (infeasible, structural)")
    lines.append(
        "Scenario:  picker_bot (wheeled, max slope 8°) navigating from\n"
        "           'loading_bay' to 'mezzanine'.\n"
        "Challenge: The only route to the mezzanine crosses a 18° ramp, far\n"
        "           exceeding the robot's slope tolerance.\n"
    )
    onto_w, graph_w, agents_w = build_warehouse_world()
    nav_w  = OntofactNavigator(onto_w, graph_w)
    agent_p = agents_w["picker_bot"]
    path_w, exp_w = nav_w.navigate("loading_bay", "mezzanine", agent_p, k_alternatives=3)
    lines.append(f"Planner result:  {'INFEASIBLE — no reachable path' if not path_w.is_feasible else path_w.summary()}")

    lines.append("\n--- query_why_not: 'Why not loading_bay→main_aisle→cross_aisle→ramp_zone→mezzanine?' ---")
    cf3 = nav_w.query_why_not(
        "loading_bay", "mezzanine", agent_p,
        alt_nodes=["loading_bay", "main_aisle", "cross_aisle", "ramp_zone", "mezzanine"],
    )
    lines.append(f"Query: {cf3.query}")
    lines.append(f"Answer: {cf3.explanation}")
    lines.append(f"Proposed changes ({len(cf3.changes)}):")
    for c in cf3.changes:
        lines.append(
            f"  • {c.individual_name}.{c.property_name}: "
            f"{c.original_value!r} → {c.counterfactual_value!r}  "
            f"[{c.effort()} effort, {'actionable' if c.is_actionable() else 'structural'}]"
        )
        lines.append(f"    {c.rationale}")

    txt = "\n".join(lines)
    txt_path.write_text(txt)
    print(f"  Saved {txt_path}")
    print()
    # Also print to stdout so the user can see it immediately
    print(txt)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Hospital navigation graphs …")
    figure_hospital(out, args.format, args.dpi)

    print("\n[2/3] Warehouse navigation graphs …")
    figure_warehouse(out, args.format, args.dpi)

    print("\n[3/3] Qualitative narrative examples …")
    generate_narratives(out)

    print(f"\nAll outputs in {out}/")


if __name__ == "__main__":
    main()
