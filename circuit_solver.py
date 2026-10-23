 
"""
Circuit Solver AI — single-file Streamlit application.

What it does:
1) Upload a clear circuit image.
2) Uses a vision-capable OpenAI model to extract a strict JSON circuit/netlist.
3) Converts the circuit to Modified Nodal Analysis (MNA).
4) Solves linear DC/AC circuits numerically.
5) Supports ideal diode/switch state search for small numbers of devices.
6) Verifies KCL, KVL, component laws, and power balance.
7) Refuses to call a result "verified" when the image interpretation is ambiguous.

Run:
    pip install streamlit openai numpy pillow
    streamlit run circuit_solver.py

Environment:
    Windows PowerShell:
        $env:OPENAI_API_KEY="YOUR_KEY"
        $env:VISION_MODEL="gpt-4.1"
    macOS/Linux:
        export OPENAI_API_KEY="YOUR_KEY"
        export VISION_MODEL="gpt-4.1"

Important:
- This is a strong MVP for clear linear DC/AC schematics.
- It is not honest to claim arbitrary photographed circuits can be solved with
  guaranteed 100% accuracy. The app therefore uses an uncertainty gate and
  independent mathematical validation instead of guessing.
"""

from __future__ import annotations

import base64
import json
import math
import os
import re
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_TITLE = "Circuit Solver AI"
DEFAULT_MODEL = os.getenv("VISION_MODEL", "gpt-4.1")
MAX_IMAGE_SIDE = 1800
CONFIDENCE_THRESHOLD = 0.92
VALIDATION_TOLERANCE = 1e-7

SUPPORTED_TYPES = {
    "resistor",
    "capacitor",
    "inductor",
    "voltage_source",
    "current_source",
    "diode",
    "switch",
}

LINEAR_TYPES = {"resistor", "capacitor", "inductor", "voltage_source", "current_source"}

# Current Responses API supports image input via input_image and structured
# output via text.format/json_schema. The schema is deliberately explicit.
CIRCUIT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "analysis_mode": {
            "type": "string",
            "enum": ["dc", "ac", "transient", "symbolic"]
        },
        "frequency_hz": {"type": ["number", "null"]},
        "reference_node": {"type": "string"},
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"}
                },
                "required": ["id", "label"]
            }
        },
        "components": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": [
                            "resistor",
                            "capacitor",
                            "inductor",
                            "voltage_source",
                            "current_source",
                            "diode",
                            "switch"
                        ]
                    },
                    "node_a": {"type": "string"},
                    "node_b": {"type": "string"},
                    "value": {"type": ["number", "null"]},
                    "unit": {"type": "string"},
                    "polarity": {"type": "string"},
                    "state": {"type": "string"},
                    "confidence": {"type": "number"}
                },
                "required": [
                    "id",
                    "type",
                    "node_a",
                    "node_b",
                    "value",
                    "unit",
                    "polarity",
                    "state",
                    "confidence"
                ]
            }
        },
        "target": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "all",
                        "voltage",
                        "current",
                        "power",
                        "equivalent_resistance",
                        "thevenin",
                        "norton"
                    ]
                },
                "component_id": {"type": "string"},
                "node_a": {"type": "string"},
                "node_b": {"type": "string"},
                "description": {"type": "string"}
            },
            "required": [
                "operation",
                "component_id",
                "node_a",
                "node_b",
                "description"
            ]
        },
        "overall_confidence": {"type": "number"},
        "ambiguities": {
            "type": "array",
            "items": {"type": "string"}
        },
        "unsupported_items": {
            "type": "array",
            "items": {"type": "string"}
        },
        "visible_question": {"type": "string"},
        "notes": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": [
        "analysis_mode",
        "frequency_hz",
        "reference_node",
        "nodes",
        "components",
        "target",
        "overall_confidence",
        "ambiguities",
        "unsupported_items",
        "visible_question",
        "notes"
    ]
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BranchResult:
    component_id: str
    component_type: str
    node_a: str
    node_b: str
    voltage: complex
    current: complex
    power: complex


@dataclass
class SolveResult:
    voltages: Dict[str, complex]
    branches: Dict[str, BranchResult]
    matrix_a: np.ndarray
    vector_b: np.ndarray
    unknown_names: List[str]
    residual: float
    kcl_error: float
    kvl_error: float
    power_error: float
    reference_node: str
    mode: str
    frequency_hz: Optional[float]
    verification_ok: bool
    verification_messages: List[str]
    raw_solution: np.ndarray


# ---------------------------------------------------------------------------
# UI styling
# ---------------------------------------------------------------------------

def apply_css() -> None:
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2.25rem;
            font-weight: 800;
            margin-bottom: 0.15rem;
        }
        .subtitle {
            color: #666;
            margin-bottom: 1.25rem;
        }
        .verified {
            padding: 0.8rem 1rem;
            border-radius: 12px;
            background: #eaf7ee;
            border: 1px solid #9bd0a7;
        }
        .warning {
            padding: 0.8rem 1rem;
            border-radius: 12px;
            background: #fff6e5;
            border: 1px solid #efc36f;
        }
        .danger {
            padding: 0.8rem 1rem;
            border-radius: 12px;
            background: #fdecec;
            border: 1px solid #e6a0a0;
        }
        code {
            white-space: pre-wrap !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------------

def prepare_image(uploaded_file) -> Tuple[bytes, str, Image.Image]:
    raw = uploaded_file.getvalue()
    mime = uploaded_file.type or "image/jpeg"

    image = Image.open(BytesIO(raw))
    image = ImageOps.exif_transpose(image).convert("RGB")

    scale = min(1.0, MAX_IMAGE_SIDE / max(image.size))
    if scale < 1.0:
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        )

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=92, optimize=True)
    return buffer.getvalue(), "image/jpeg", image


def image_to_data_url(image_bytes: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode('utf-8')}"


# ---------------------------------------------------------------------------
# Vision extraction
# ---------------------------------------------------------------------------

VISION_INSTRUCTIONS = """
You are a meticulous electrical-engineering circuit parser.

Your job is NOT to solve the circuit. Your job is to reconstruct the exact
electrical circuit visible in the supplied image as a machine-readable graph.

Rules:
1. Inspect the WHOLE image, including visible question text.
2. Identify every visible component and every electrical node.
3. Assign stable node IDs such as N1, N2, N3. Use GND for an explicit ground.
4. Two points are the same electrical node only when the drawing connects them.
5. A crossing without a junction dot must NOT automatically be considered connected.
6. A junction dot means connected.
7. Preserve component orientation:
      voltage = V(node_a) - V(node_b)
      current = current flowing from node_a to node_b.
8. For voltage sources, `value` is the signed source voltage from node_a to node_b.
9. For current sources, `value` is the signed current from node_a to node_b.
10. For resistors/capacitors/inductors, value must be in SI base units:
      ohm, farad, henry.
11. Use `polarity` to describe visible source polarity, but also make the
    node_a/node_b ordering consistent with the physical polarity.
12. For AC, return the frequency in Hz when visible or stated.
13. `analysis_mode` must be:
      dc, ac, transient, or symbolic.
14. If the requested problem is transient and the image contains initial
    conditions, record them in notes/visible_question, but do not invent
    missing values.
15. Diodes and switches must preserve orientation/state when visibly known.
16. If a value, connection, orientation, or symbol is genuinely ambiguous,
    put a concise explanation in `ambiguities`. DO NOT GUESS.
17. `overall_confidence` is your confidence in the entire reconstruction,
    not your confidence that the circuit is solvable.
18. `target` describes what the visible/user question asks for. If no question
    is supplied, use operation "all".
19. Use empty strings for target fields that are not applicable.
20. Only report components you can actually see or strongly infer from the
    circuit drawing. Do not add invisible components.
21. Unsupported devices such as BJT, MOSFET, op-amp, dependent source,
    transformer, transmission line, etc. must go into unsupported_items.
22. Be conservative: a wrong connection is worse than an ambiguity flag.
"""

def get_openai_client() -> Any:
    if OpenAI is None:
        raise RuntimeError("The OpenAI Python package is not installed.")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in your environment before running."
        )
    return OpenAI(api_key=api_key)


def extract_circuit(image_bytes: bytes, mime: str, question: str, model: str) -> Dict[str, Any]:
    client = get_openai_client()
    data_url = image_to_data_url(image_bytes, mime)

    user_text = """
Analyze the supplied circuit image.

User question:
{question}

Return only the required structured circuit JSON. Build the netlist from the
actual drawing. Validate visually before deciding nodes and component values.
""".format(question=question.strip() or "(No additional text question supplied.)")

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": VISION_INSTRUCTIONS,
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_text},
                    {
                        "type": "input_image",
                        "detail": "high",
                        "image_url": data_url,
                    },
                ],
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "circuit_extraction",
                "strict": True,
                "schema": CIRCUIT_SCHEMA,
            }
        },
        max_output_tokens=5000,
    )

    raw = response.output_text
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Vision model returned invalid JSON: {exc}") from exc


# ---------------------------------------------------------------------------
# Parsing and normalization
# ---------------------------------------------------------------------------

def clean_id(value: str, fallback: str) -> str:
    value = str(value or "").strip()
    return value if value else fallback


def normalize_circuit(circuit: Dict[str, Any]) -> Dict[str, Any]:
    nodes = circuit.get("nodes", [])
    components = circuit.get("components", [])
    ref = str(circuit.get("reference_node", "") or "").strip() or "GND"

    node_ids = []
    for i, node in enumerate(nodes):
        nid = clean_id(node.get("id"), f"N{i+1}")
        if nid not in node_ids:
            node_ids.append(nid)

    if ref not in node_ids:
        node_ids.append(ref)

    normalized_components = []
    for i, comp in enumerate(components):
        c = dict(comp)
        c["id"] = clean_id(c.get("id"), f"X{i+1}")
        c["type"] = str(c.get("type", "")).strip().lower()
        c["node_a"] = clean_id(c.get("node_a"), "")
        c["node_b"] = clean_id(c.get("node_b"), "")
        c["value"] = c.get("value")
        c["unit"] = str(c.get("unit", "") or "").lower()
        c["polarity"] = str(c.get("polarity", "") or "")
        c["state"] = str(c.get("state", "") or "").lower()
        c["confidence"] = float(c.get("confidence", 0.0) or 0.0)
        normalized_components.append(c)

    circuit["reference_node"] = ref
    circuit["nodes"] = [{"id": n, "label": n} for n in node_ids]
    circuit["components"] = normalized_components
    return circuit


def validate_netlist(circuit: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    nodes = {n["id"] for n in circuit.get("nodes", [])}
    seen = set()

    if not nodes:
        errors.append("No electrical nodes were detected.")

    ref = circuit.get("reference_node", "")
    if ref not in nodes:
        errors.append(f"Reference node {ref!r} is missing.")

    for comp in circuit.get("components", []):
        cid = comp["id"]
        if cid in seen:
            errors.append(f"Duplicate component ID: {cid}")
        seen.add(cid)

        if comp["type"] not in SUPPORTED_TYPES:
            errors.append(f"Unsupported component type: {comp['type']} ({cid}).")

        if comp["node_a"] not in nodes:
            errors.append(f"{cid}: node_a {comp['node_a']!r} does not exist.")
        if comp["node_b"] not in nodes:
            errors.append(f"{cid}: node_b {comp['node_b']!r} does not exist.")

        if comp["type"] in {"resistor", "capacitor", "inductor"}:
            value = comp.get("value")
            if value is None or not np.isfinite(float(value)) or float(value) <= 0:
                errors.append(f"{cid}: invalid positive component value.")

        if comp["type"] in {"voltage_source", "current_source"}:
            if comp.get("value") is None or not np.isfinite(float(comp["value"])):
                errors.append(f"{cid}: invalid source value.")

        if comp["confidence"] < 0 or comp["confidence"] > 1.0:
            errors.append(f"{cid}: confidence must be between 0 and 1.")

    return errors


# ---------------------------------------------------------------------------
# MNA engine
# ---------------------------------------------------------------------------

def edge_value_to_complex(value: float) -> complex:
    return complex(float(value), 0.0)


def build_mna(
    circuit: Dict[str, Any],
    *,
    mode: str,
    frequency_hz: Optional[float],
    zero_independent_sources: bool = False,
    extra_voltage_sources: Optional[List[Dict[str, Any]]] = None,
    diode_states: Optional[Dict[str, str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, int], List[Dict[str, Any]]]:

    ref = circuit["reference_node"]
    all_nodes = [n["id"] for n in circuit["nodes"]]
    analysis_nodes = [n for n in all_nodes if n != ref]
    node_index = {n: i for i, n in enumerate(analysis_nodes)}

    branch_specs: List[Dict[str, Any]] = []

    def add_branch(cid: str, ctype: str, a: str, b: str, value: complex) -> None:
        branch_specs.append(
            {
                "id": cid,
                "type": ctype,
                "node_a": a,
                "node_b": b,
                "value": value,
            }
        )

    for comp in circuit["components"]:
        cid = comp["id"]
        typ = comp["type"]
        a = comp["node_a"]
        b = comp["node_b"]
        value = comp.get("value")

        if typ == "voltage_source":
            v = 0.0 if zero_independent_sources else float(value)
            add_branch(cid, typ, a, b, complex(v))
        elif typ == "inductor" and mode == "dc":
            # Ideal DC inductor = short, modeled as a 0-V MNA branch.
            add_branch(cid, "inductor_short", a, b, 0j)
        elif typ == "diode":
            state = (diode_states or {}).get(cid, comp.get("state", "unknown"))
            if state == "on":
                add_branch(cid, "diode_on", a, b, 0j)
        elif typ == "switch":
            state = comp.get("state", "unknown")
            if state in {"closed", "on"}:
                add_branch(cid, "switch_closed", a, b, 0j)

    for extra in extra_voltage_sources or []:
        add_branch(
            extra["id"],
            "test_voltage_source",
            extra["node_a"],
            extra["node_b"],
            complex(extra["value"]),
        )

    n = len(analysis_nodes)
    m = len(branch_specs)
    size = n + m

    dtype = complex
    A = np.zeros((size, size), dtype=dtype)
    z = np.zeros(size, dtype=dtype)

    def stamp_admittance(a: str, b: str, y: complex) -> None:
        if a != ref:
            ia = node_index[a]
            A[ia, ia] += y
        if b != ref:
            ib = node_index[b]
            A[ib, ib] += y
        if a != ref and b != ref:
            ia = node_index[a]
            ib = node_index[b]
            A[ia, ib] -= y
            A[ib, ia] -= y

    def stamp_current_source(a: str, b: str, current: complex) -> None:
        # Current defined from a -> b.
        if a != ref:
            z[node_index[a]] -= current
        if b != ref:
            z[node_index[b]] += current

    # Passive/source components
    for comp in circuit["components"]:
        typ = comp["type"]
        a = comp["node_a"]
        b = comp["node_b"]
        value = comp.get("value")

        if typ == "resistor":
            stamp_admittance(a, b, 1.0 / float(value))

        elif typ == "capacitor":
            if mode == "ac":
                if not frequency_hz or frequency_hz <= 0:
                    raise ValueError("AC mode requires a positive frequency in Hz.")
                omega = 2 * math.pi * float(frequency_hz)
                y = 1j * omega * float(value)
                stamp_admittance(a, b, y)
            # DC steady state capacitor = open circuit.

        elif typ == "inductor":
            if mode == "ac":
                if not frequency_hz or frequency_hz <= 0:
                    raise ValueError("AC mode requires a positive frequency in Hz.")
                omega = 2 * math.pi * float(frequency_hz)
                y = 1.0 / (1j * omega * float(value))
                stamp_admittance(a, b, y)
            # DC already stamped above as 0-V branch.

        elif typ == "current_source":
            current = 0.0 if zero_independent_sources else complex(float(value))
            stamp_current_source(a, b, current)

        # Voltage sources, inductor shorts, diode-on shorts and closed switches
        # are handled as branch variables below.

    # Stamp voltage-source-like branches.
    for branch_i, branch in enumerate(branch_specs):
        k = n + branch_i
        a = branch["node_a"]
        b = branch["node_b"]
        v = branch["value"]

        if a != ref:
            ia = node_index[a]
            A[ia, k] += 1
            A[k, ia] += 1

        if b != ref:
            ib = node_index[b]
            A[ib, k] -= 1
            A[k, ib] -= 1

        z[k] += v

    unknown_names = [f"V({node})" for node in analysis_nodes]
    unknown_names.extend(f"I({b['id']})" for b in branch_specs)
    return A, z, unknown_names, node_index, branch_specs


def solve_mna_once(
    circuit: Dict[str, Any],
    *,
    mode: str,
    frequency_hz: Optional[float],
    zero_independent_sources: bool = False,
    extra_voltage_sources: Optional[List[Dict[str, Any]]] = None,
    diode_states: Optional[Dict[str, str]] = None,
) -> SolveResult:

    A, z, unknown_names, node_index, branch_specs = build_mna(
        circuit,
        mode=mode,
        frequency_hz=frequency_hz,
        zero_independent_sources=zero_independent_sources,
        extra_voltage_sources=extra_voltage_sources,
        diode_states=diode_states,
    )

    if A.shape[0] == 0:
        raise ValueError("The circuit has no unknowns to solve.")

    try:
        x = np.linalg.solve(A, z)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "The circuit equations are singular. This usually means the image "
            "was interpreted incorrectly, a node is floating, or the circuit "
            "contains an unsupported topology."
        ) from exc

    residual = float(np.max(np.abs(A @ x - z))) if len(x) else 0.0

    ref = circuit["reference_node"]
    voltages: Dict[str, complex] = {ref: 0j}
    for node, idx in node_index.items():
        voltages[node] = complex(x[idx])

    branch_currents: Dict[str, complex] = {}
    for branch_i, branch in enumerate(branch_specs):
        branch_currents[branch["id"]] = complex(x[len(node_index) + branch_i])

    # Compute branch values from the solved node voltages.
    branches: Dict[str, BranchResult] = {}
    for comp in circuit["components"]:
        cid = comp["id"]
        typ = comp["type"]
        a = comp["node_a"]
        b = comp["node_b"]
        va = voltages[a]
        vb = voltages[b]
        v = va - vb

        if typ == "resistor":
            current = v / float(comp["value"])
        elif typ == "capacitor":
            if mode == "ac":
                omega = 2 * math.pi * float(frequency_hz or 0.0)
                current = 1j * omega * float(comp["value"]) * v
            else:
                current = 0j
        elif typ == "inductor":
            if mode == "ac":
                omega = 2 * math.pi * float(frequency_hz or 0.0)
                current = v / (1j * omega * float(comp["value"]))
            else:
                current = branch_currents.get(cid, 0j)
        elif typ == "current_source":
            current = 0j if zero_independent_sources else complex(float(comp["value"]))
        elif typ == "voltage_source":
            current = branch_currents.get(cid, 0j)
        elif typ == "diode":
            if (diode_states or {}).get(cid, comp.get("state", "unknown")) == "on":
                current = branch_currents.get(cid, 0j)
            else:
                current = 0j
        elif typ == "switch":
            if comp.get("state", "unknown") in {"closed", "on"}:
                current = branch_currents.get(cid, 0j)
            else:
                current = 0j
        else:
            current = 0j

        power = v * np.conjugate(current) if mode == "ac" else v * current

        branches[cid] = BranchResult(
            component_id=cid,
            component_type=typ,
            node_a=a,
            node_b=b,
            voltage=v,
            current=current,
            power=power,
        )

    # Add extra branches to the result dictionary (e.g. test source).
    for branch_id, current in branch_currents.items():
        if branch_id in branches:
            continue
        branch = next(b for b in branch_specs if b["id"] == branch_id)
        va = voltages[branch["node_a"]]
        vb = voltages[branch["node_b"]]
        vv = va - vb
        pp = vv * np.conjugate(current) if mode == "ac" else vv * current
        branches[branch_id] = BranchResult(
            component_id=branch_id,
            component_type=branch["type"],
            node_a=branch["node_a"],
            node_b=branch["node_b"],
            voltage=vv,
            current=current,
            power=pp,
        )

    kcl_error = calculate_kcl_error(circuit, branches)
    kvl_error = calculate_kvl_error(circuit, branches)
    power_error = calculate_power_error(branches)

    messages = [
        f"MNA residual: {residual:.3e}",
        f"Maximum KCL residual: {kcl_error:.3e}",
        f"Maximum KVL residual: {kvl_error:.3e}",
        f"Power-balance residual: {power_error:.3e}",
    ]

    ok = (
        np.isfinite(residual)
        and np.isfinite(kcl_error)
        and np.isfinite(kvl_error)
        and np.isfinite(power_error)
        and residual <= 1e-6
        and kcl_error <= 1e-6
        and kvl_error <= 1e-6
        and power_error <= 1e-5
    )

    return SolveResult(
        voltages=voltages,
        branches=branches,
        matrix_a=A,
        vector_b=z,
        unknown_names=unknown_names,
        residual=residual,
        kcl_error=kcl_error,
        kvl_error=kvl_error,
        power_error=power_error,
        reference_node=ref,
        mode=mode,
        frequency_hz=frequency_hz,
        verification_ok=ok,
        verification_messages=messages,
        raw_solution=x,
    )


def calculate_kcl_error(
    circuit: Dict[str, Any],
    branches: Dict[str, BranchResult],
) -> float:
    node_sum: Dict[str, complex] = {n["id"]: 0j for n in circuit["nodes"]}

    for br in branches.values():
        if br.component_id.startswith("__"):
            continue
        node_sum[br.node_a] = node_sum.get(br.node_a, 0j) + br.current
        node_sum[br.node_b] = node_sum.get(br.node_b, 0j) - br.current

    return float(max((abs(v) for v in node_sum.values()), default=0.0))


def find_path(
    adjacency: Dict[str, List[Tuple[str, str]]],
    start: str,
    goal: str,
    excluded_edge: Optional[str] = None,
) -> Optional[List[Tuple[str, str]]]:
    # Returns directed (component_id, direction), where direction is +1 when
    # moving component node_a -> node_b and -1 for node_b -> node_a.
    queue = [start]
    prev: Dict[str, Tuple[str, str, int]] = {}
    visited = {start}

    while queue:
        u = queue.pop(0)
        if u == goal:
            break

        for cid, v in adjacency.get(u, []):
            if cid == excluded_edge or v in visited:
                continue
            # Direction from u to v is discovered from the stored branch.
            prev[v] = (u, cid, 1)
            visited.add(v)
            queue.append(v)

    if goal not in prev and goal != start:
        return None

    path: List[Tuple[str, str]] = []
    cur = goal
    while cur != start:
        u, cid, _ = prev[cur]
        path.append((cid, u))
        cur = u
    path.reverse()

    result: List[Tuple[str, str]] = []
    for cid, from_node in path:
        # Determine orientation by branch record later; caller will translate.
        result.append((cid, from_node))
    return result


def calculate_kvl_error(
    circuit: Dict[str, Any],
    branches: Dict[str, BranchResult],
) -> float:
    # Build a spanning forest from the component graph.
    # Every non-tree edge creates a fundamental cycle.
    components = [
        c for c in circuit["components"]
        if c["id"] in branches
    ]

    parent: Dict[str, str] = {}
    rank: Dict[str, int] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        rank.setdefault(ra, 0)
        rank.setdefault(rb, 0)
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        return True

    tree_ids = set()
    for comp in components:
        if union(comp["node_a"], comp["node_b"]):
            tree_ids.add(comp["id"])

    adjacency: Dict[str, List[Tuple[str, str]]] = {}
    for comp in components:
        if comp["id"] not in tree_ids:
            continue
        adjacency.setdefault(comp["node_a"], []).append((comp["id"], comp["node_b"]))
        adjacency.setdefault(comp["node_b"], []).append((comp["id"], comp["node_a"]))

    errors = []

    for comp in components:
        if comp["id"] in tree_ids:
            continue

        # Find path from chord.b to chord.a.
        start = comp["node_b"]
        goal = comp["node_a"]

        queue = [start]
        prev: Dict[str, Tuple[str, str]] = {}
        visited = {start}

        while queue and goal not in visited:
            u = queue.pop(0)
            for cid, v in adjacency.get(u, []):
                if v in visited:
                    continue
                prev[v] = (u, cid)
                visited.add(v)
                queue.append(v)

        if goal not in visited:
            continue

        # Sum voltage drops along the tree path from b -> a.
        total = 0j
        cur = goal
        path = []
        while cur != start:
            u, cid = prev[cur]
            path.append((u, cur, cid))
            cur = u
        path.reverse()

        for u, v, cid in path:
            br = branches[cid]
            # br.voltage = V(node_a) - V(node_b)
            if u == br.node_a and v == br.node_b:
                total += br.voltage
            elif u == br.node_b and v == br.node_a:
                total -= br.voltage

        # Add chord from a -> b.
        total += branches[comp["id"]].voltage
        errors.append(abs(total))

    return float(max(errors, default=0.0))


def calculate_power_error(branches: Dict[str, BranchResult]) -> float:
    total = 0j
    scale = 0.0
    for br in branches.values():
        if br.component_id.startswith("__"):
            continue
        total += br.power
        scale += abs(br.power)

    # Relative-to-scale with absolute floor to make near-zero circuits stable.
    denom = max(scale, 1.0)
    return float(abs(total) / denom)


# ---------------------------------------------------------------------------
# Diode state search
# ---------------------------------------------------------------------------

def find_diode_states_and_solve(
    circuit: Dict[str, Any],
    mode: str,
    frequency_hz: Optional[float],
) -> Tuple[SolveResult, Dict[str, str]]:

    diodes = [c for c in circuit["components"] if c["type"] == "diode"]

    # If there are no unknown diodes, solve directly.
    unknown = [
        d for d in diodes
        if d.get("state", "unknown") not in {"on", "off"}
    ]

    if len(unknown) > 8:
        raise ValueError(
            "More than 8 unknown ideal diodes were detected. "
            "The single-file engine stops instead of brute-forcing an unsafe "
            "state space."
        )

    if not unknown:
        states = {
            d["id"]: d.get("state", "off")
            for d in diodes
        }
        return solve_mna_once(
            circuit,
            mode=mode,
            frequency_hz=frequency_hz,
            diode_states=states,
        ), states

    best: Optional[Tuple[SolveResult, Dict[str, str]]] = None

    for mask in range(2 ** len(unknown)):
        states = {
            d["id"]: d.get("state", "off")
            for d in diodes
        }

        for i, d in enumerate(unknown):
            states[d["id"]] = "on" if (mask & (1 << i)) else "off"

        try:
            result = solve_mna_once(
                circuit,
                mode=mode,
                frequency_hz=frequency_hz,
                diode_states=states,
            )
        except ValueError:
            continue

        if not result.verification_ok:
            continue

        consistent = True

        for d in diodes:
            cid = d["id"]
            br = result.branches[cid]
            vd = br.voltage.real
            current = br.current.real

            if states[cid] == "on":
                # Ideal diode:
                # on => Vd ~= 0 and Id >= 0.
                if abs(vd) > 1e-5 or current < -1e-7:
                    consistent = False
                    break
            else:
                # off => Id ~= 0 and Vd <= 0.
                if abs(current) > 1e-6 or vd > 1e-5:
                    consistent = False
                    break

        if consistent:
            best = (result, states)
            break

    if best is None:
        raise ValueError(
            "No self-consistent ideal-diode state could be found. "
            "The circuit may require a real diode model or may have been "
            "interpreted incorrectly."
        )

    return best


# ---------------------------------------------------------------------------
# Target calculations
# ---------------------------------------------------------------------------

def get_component(
    circuit: Dict[str, Any],
    component_id: str,
) -> Dict[str, Any]:
    for c in circuit["components"]:
        if c["id"].lower() == component_id.lower():
            return c
    raise ValueError(f"Target component {component_id!r} was not found.")


def solve_equivalent_resistance(
    circuit: Dict[str, Any],
    node_a: str,
    node_b: str,
    mode: str = "dc",
    frequency_hz: Optional[float] = None,
) -> Tuple[float, SolveResult]:

    if not node_a or not node_b or node_a == node_b:
        raise ValueError("Equivalent resistance requires two distinct terminal nodes.")

    test_source = {
        "id": "__RTEST",
        "node_a": node_a,
        "node_b": node_b,
        "value": 1.0,
    }

    result = solve_mna_once(
        circuit,
        mode=mode,
        frequency_hz=frequency_hz,
        zero_independent_sources=True,
        extra_voltage_sources=[test_source],
    )

    i_test = result.branches["__RTEST"].current.real
    i_network = -i_test

    if abs(i_network) < 1e-12:
        raise ValueError(
            "The test source sees an open circuit, so the equivalent resistance "
            "is infinite or undefined."
        )

    req = 1.0 / i_network
    return float(req), result


def solve_thevenin(
    circuit: Dict[str, Any],
    node_a: str,
    node_b: str,
    mode: str,
    frequency_hz: Optional[float],
) -> Dict[str, Any]:

    open_result, _ = find_diode_states_and_solve(
        circuit, mode, frequency_hz
    )
    vth = (
        open_result.voltages[node_a] - open_result.voltages[node_b]
    )

    rth, r_result = solve_equivalent_resistance(
        circuit, node_a, node_b, mode, frequency_hz
    )

    return {
        "vth": vth,
        "rth": rth,
        "result": open_result,
        "r_result": r_result,
    }


def solve_norton(
    circuit: Dict[str, Any],
    node_a: str,
    node_b: str,
    mode: str,
    frequency_hz: Optional[float],
) -> Dict[str, Any]:

    # Norton current is obtained using a 0-V test source across the terminals.
    short_source = {
        "id": "__SHORT",
        "node_a": node_a,
        "node_b": node_b,
        "value": 0.0,
    }

    short_result = solve_mna_once(
        circuit,
        mode=mode,
        frequency_hz=frequency_hz,
        extra_voltage_sources=[short_source],
    )

    inorton = -short_result.branches["__SHORT"].current

    rth, _ = solve_equivalent_resistance(
        circuit, node_a, node_b, mode, frequency_hz
    )

    return {
        "in": inorton,
        "rn": rth,
        "result": short_result,
    }


def target_answer(
    circuit: Dict[str, Any],
    solve_result: SolveResult,
) -> Tuple[str, Dict[str, Any]]:
    target = circuit["target"]
    op = target["operation"]
    cid = target.get("component_id", "")
    a = target.get("node_a", "")
    b = target.get("node_b", "")

    if op == "all":
        return "all", {}

    if op in {"voltage", "current", "power"}:
        comp = get_component(circuit, cid)
        br = solve_result.branches[cid]
        if op == "voltage":
            return "voltage", {"component": comp, "value": br.voltage}
        if op == "current":
            return "current", {"component": comp, "value": br.current}
        return "power", {"component": comp, "value": br.power}

    if op == "equivalent_resistance":
        req, rr = solve_equivalent_resistance(
            circuit,
            a,
            b,
            solve_result.mode,
            solve_result.frequency_hz,
        )
        return "equivalent_resistance", {"node_a": a, "node_b": b, "value": req, "result": rr}

    if op == "thevenin":
        th = solve_thevenin(
            circuit,
            a,
            b,
            solve_result.mode,
            solve_result.frequency_hz,
        )
        return "thevenin", th

    if op == "norton":
        no = solve_norton(
            circuit,
            a,
            b,
            solve_result.mode,
            solve_result.frequency_hz,
        )
        return "norton", no

    raise ValueError(f"Unsupported target operation: {op}")


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def is_close_zero(z: complex, tol: float = 1e-9) -> bool:
    return abs(z) <= tol


def format_complex(z: complex, unit: str = "") -> str:
    z = complex(z)
    if abs(z.imag) < 1e-9:
        return f"{z.real:.8g} {unit}".strip()

    mag = abs(z)
    phase = math.degrees(math.atan2(z.imag, z.real))
    return f"{mag:.8g} ∠ {phase:.6g}° {unit}".strip()


def format_power(z: complex, mode: str) -> str:
    if mode == "ac":
        return (
            f"P={z.real:.8g} W, "
            f"Q={z.imag:.8g} var, "
            f"|S|={abs(z):.8g} VA"
        )
    return f"{z.real:.8g} W"


def matrix_text(A: np.ndarray, b: np.ndarray, names: List[str]) -> str:
    lines = []
    lines.append("Unknown vector:")
    lines.append("  " + ", ".join(names))
    lines.append("")
    lines.append("MNA system  A · x = b")
    lines.append("A =")
    lines.append(np.array2string(A, precision=6, suppress_small=True))
    lines.append("")
    lines.append("b =")
    lines.append(np.array2string(b.reshape(-1, 1), precision=6, suppress_small=True))
    return "\n".join(lines)


def generate_deterministic_explanation(
    circuit: Dict[str, Any],
    result: SolveResult,
    answer_kind: str,
    answer_data: Dict[str, Any],
) -> List[str]:

    lines: List[str] = []

    lines.append(
        f"Reference node: {result.reference_node}. "
        f"Analysis: {result.mode.upper()}"
        + (
            f" at {result.frequency_hz:g} Hz."
            if result.frequency_hz
            else "."
        )
    )

    lines.append("")

    for c in circuit["components"]:
        cid = c["id"]
        br = result.branches.get(cid)
        if not br:
            continue

        if c["type"] == "resistor":
            lines.append(
                f"{cid}: V({br.node_a}) − V({br.node_b}) = "
                f"{format_complex(br.voltage, 'V')}; "
                f"I = V/R = {format_complex(br.current, 'A')}."
            )
        elif c["type"] == "capacitor" and result.mode == "ac":
            lines.append(
                f"{cid}: I = jωCV = {format_complex(br.current, 'A')}."
            )
        elif c["type"] == "inductor" and result.mode == "ac":
            lines.append(
                f"{cid}: I = V/(jωL) = {format_complex(br.current, 'A')}."
            )
        elif c["type"] == "voltage_source":
            lines.append(
                f"{cid}: V = {format_complex(br.voltage, 'V')}; "
                f"I = {format_complex(br.current, 'A')}."
            )
        elif c["type"] == "current_source":
            lines.append(
                f"{cid}: source current = {format_complex(br.current, 'A')} "
                f"from {br.node_a} to {br.node_b}."
            )
        elif c["type"] == "diode":
            lines.append(
                f"{cid}: ideal-diode state = {c.get('state', 'detected')}."
            )

    lines.append("")
    lines.append("Validation:")
    lines.append(f"• MNA residual ≤ {result.residual:.3e}")
    lines.append(f"• KCL maximum residual ≤ {result.kcl_error:.3e}")
    lines.append(f"• KVL maximum residual ≤ {result.kvl_error:.3e}")
    lines.append(f"• Relative power-balance residual ≤ {result.power_error:.3e}")

    if answer_kind == "voltage":
        c = answer_data["component"]
        lines.append("")
        lines.append(
            f"Requested voltage across {c['id']}: "
            f"{format_complex(answer_data['value'], 'V')} "
            f"(node_a → node_b)."
        )

    elif answer_kind == "current":
        c = answer_data["component"]
        lines.append("")
        lines.append(
            f"Requested current through {c['id']}: "
            f"{format_complex(answer_data['value'], 'A')} "
            f"from {c['node_a']} to {c['node_b']}."
        )

    elif answer_kind == "power":
        c = answer_data["component"]
        lines.append("")
        lines.append(
            f"Requested absorbed power of {c['id']}: "
            f"{format_power(answer_data['value'], result.mode)}."
        )

    elif answer_kind == "equivalent_resistance":
        lines.append("")
        lines.append(
            f"Equivalent resistance between {answer_data['node_a']} and "
            f"{answer_data['node_b']} = {answer_data['value']:.8g} Ω."
        )

    elif answer_kind == "thevenin":
        lines.append("")
        lines.append(
            f"Thevenin voltage = {format_complex(answer_data['vth'], 'V')}; "
            f"Rth = {answer_data['rth']:.8g} Ω."
        )

    elif answer_kind == "norton":
        lines.append("")
        lines.append(
            f"Norton current = {format_complex(answer_data['in'], 'A')}; "
            f"Rn = {answer_data['rn']:.8g} Ω."
        )

    return lines


# ---------------------------------------------------------------------------
# Demo circuit
# ---------------------------------------------------------------------------

def demo_circuit() -> Dict[str, Any]:
    return {
        "analysis_mode": "dc",
        "frequency_hz": None,
        "reference_node": "GND",
        "nodes": [
            {"id": "N1", "label": "N1"},
            {"id": "N2", "label": "N2"},
            {"id": "GND", "label": "GND"},
        ],
        "components": [
            {
                "id": "V1",
                "type": "voltage_source",
                "node_a": "N1",
                "node_b": "GND",
                "value": 12,
                "unit": "V",
                "polarity": "+ at N1",
                "state": "active",
                "confidence": 1,
            },
            {
                "id": "R1",
                "type": "resistor",
                "node_a": "N1",
                "node_b": "N2",
                "value": 10,
                "unit": "ohm",
                "polarity": "",
                "state": "active",
                "confidence": 1,
            },
            {
                "id": "R2",
                "type": "resistor",
                "node_a": "N2",
                "node_b": "GND",
                "value": 20,
                "unit": "ohm",
                "polarity": "",
                "state": "active",
                "confidence": 1,
            },
        ],
        "target": {
            "operation": "current",
            "component_id": "R2",
            "node_a": "",
            "node_b": "",
            "description": "Find current through R2",
        },
        "overall_confidence": 1.0,
        "ambiguities": [],
        "unsupported_items": [],
        "visible_question": "Find current through R2",
        "notes": ["12 V source with 10 Ω and 20 Ω divider."],
    }


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="⚡",
        layout="wide",
    )
    apply_css()

    st.markdown('<div class="main-title">⚡ Circuit Solver AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">'
        'Upload a circuit image. The AI reconstructs the circuit; the mathematics '
        'engine solves and verifies it.'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Settings")
        model = st.text_input("Vision model", value=DEFAULT_MODEL)
        strict_mode = st.checkbox(
            "Strict verification",
            value=True,
            help="Stop instead of presenting an unverified result."
        )
        show_matrix = st.checkbox("Show MNA matrix", value=False)
        show_raw_json = st.checkbox("Show extracted JSON", value=False)

        st.divider()

        st.caption(
            "Supported in this single-file engine: linear DC/AC circuits, "
            "R/C/L, independent V/I sources, basic switches, and ideal diodes."
        )
        st.caption(
            "Unsupported/complex devices are flagged rather than guessed."
        )

        if st.button("Run built-in solver test"):
            demo = demo_circuit()
            demo = normalize_circuit(demo)
            errors = validate_netlist(demo)
            if errors:
                st.error("\n".join(errors))
            else:
                result, diode_states = find_diode_states_and_solve(
                    demo,
                    demo["analysis_mode"],
                    demo["frequency_hz"],
                )
                st.session_state["demo_result"] = result
                st.session_state["demo_circuit"] = demo
                st.session_state["demo_diode_states"] = diode_states

    uploaded = st.file_uploader(
        "Upload a clear circuit diagram/photo",
        type=["png", "jpg", "jpeg", "webp"],
    )

    question = st.text_input(
        "Question (optional)",
        placeholder="Example: Find the current through R2",
    )

    col_a, col_b = st.columns([1, 1])

    if uploaded is not None:
        image_bytes, mime, image = prepare_image(uploaded)
        with col_a:
            st.image(image, caption="Uploaded circuit", use_container_width=True)

        solve_clicked = st.button("🔍 Analyze and Solve", type="primary")

        if solve_clicked:
            if not os.getenv("OPENAI_API_KEY"):
                st.error(
                    "OPENAI_API_KEY is missing. Set the API key in your environment "
                    "and restart Streamlit."
                )
                return

            with st.spinner("Reading the circuit and building the electrical graph..."):
                try:
                    circuit = extract_circuit(
                        image_bytes=image_bytes,
                        mime=mime,
                        question=question,
                        model=model.strip() or DEFAULT_MODEL,
                    )
                except Exception as exc:
                    st.exception(exc)
                    return

            circuit = normalize_circuit(circuit)

            with st.expander("1. Recognized circuit", expanded=True):
                st.write(
                    f"Analysis mode: **{circuit['analysis_mode'].upper()}**"
                    + (
                        f" | Frequency: **{circuit['frequency_hz']} Hz**"
                        if circuit.get("frequency_hz")
                        else ""
                    )
                )
                st.write(
                    f"Reference node: **{circuit['reference_node']}** | "
                    f"Model confidence: **{circuit['overall_confidence']:.1%}**"
                )
                st.write(
                    f"Detected target: **{circuit['target']['description']}**"
                )

                component_rows = []
                for c in circuit["components"]:
                    component_rows.append(
                        {
                            "ID": c["id"],
                            "Type": c["type"],
                            "Node A": c["node_a"],
                            "Node B": c["node_b"],
                            "Value": c["value"],
                            "Unit": c["unit"],
                            "State": c["state"],
                            "Confidence": f"{c['confidence']:.1%}",
                        }
                    )
                st.dataframe(component_rows, use_container_width=True)

            if show_raw_json:
                st.subheader("Extracted netlist JSON")
                st.json(circuit)

            errors = validate_netlist(circuit)

            if circuit.get("ambiguities"):
                st.markdown(
                    '<div class="warning"><strong>Ambiguities detected.</strong>'
                    '<br>' + "<br>".join(
                        f"• {x}" for x in circuit["ambiguities"]
                    ) + "</div>",
                    unsafe_allow_html=True,
                )

            if circuit.get("unsupported_items"):
                st.markdown(
                    '<div class="warning"><strong>Unsupported items.</strong>'
                    '<br>' + "<br>".join(
                        f"• {x}" for x in circuit["unsupported_items"]
                    ) + "</div>",
                    unsafe_allow_html=True,
                )

            if errors:
                st.markdown(
                    '<div class="danger"><strong>Netlist validation failed.</strong>'
                    '<br>' + "<br>".join(
                        f"• {x}" for x in errors
                    ) + "</div>",
                    unsafe_allow_html=True,
                )
                return

            if (
                strict_mode
                and (
                    circuit["overall_confidence"] < CONFIDENCE_THRESHOLD
                    or circuit.get("ambiguities")
                    or circuit.get("unsupported_items")
                    or any(
                        c["confidence"] < CONFIDENCE_THRESHOLD
                        for c in circuit["components"]
                    )
                )
            ):
                st.markdown(
                    '<div class="danger"><strong>Strict mode stopped the solver.</strong>'
                    '<br>The input was not interpreted with sufficiently high '
                    'confidence to justify a verified numerical answer.</div>',
                    unsafe_allow_html=True,
                )
                return

            mode = circuit["analysis_mode"]
            if mode in {"transient", "symbolic"}:
                st.warning(
                    f"The detected mode is {mode!r}. This single-file runtime "
                    "focuses on numeric linear DC/AC MNA. It will not invent "
                    "transient or symbolic data."
                )
                return

            try:
                with st.spinner("Solving equations and independently validating the result..."):
                    result, diode_states = find_diode_states_and_solve(
                        circuit,
                        mode,
                        circuit.get("frequency_hz"),
                    )

                    answer_kind, answer_data = target_answer(circuit, result)

            except Exception as exc:
                st.error(f"Solver stopped safely: {exc}")
                st.info(
                    "This means the engine did not find a mathematically "
                    "trustworthy result. Re-check the image, node connections, "
                    "component values, source polarities, and requested mode."
                )
                return

            st.subheader("2. Result")

            if result.verification_ok:
                st.markdown(
                    '<div class="verified"><strong>✓ MATHEMATICALLY VERIFIED</strong>'
                    '<br>KCL, KVL, MNA residual and power balance passed.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="danger"><strong>UNVERIFIED RESULT</strong>'
                    '<br>The mathematical checks did not all pass.</div>',
                    unsafe_allow_html=True,
                )

            if answer_kind == "current":
                st.metric(
                    f"Current through {answer_data['component']['id']}",
                    format_complex(answer_data["value"], "A"),
                )
            elif answer_kind == "voltage":
                st.metric(
                    f"Voltage across {answer_data['component']['id']}",
                    format_complex(answer_data["value"], "V"),
                )
            elif answer_kind == "power":
                st.metric(
                    f"Power of {answer_data['component']['id']}",
                    format_power(answer_data["value"], result.mode),
                )
            elif answer_kind == "equivalent_resistance":
                st.metric(
                    f"Equivalent resistance {answer_data['node_a']} → {answer_data['node_b']}",
                    f"{answer_data['value']:.8g} Ω",
                )
            elif answer_kind == "thevenin":
                c1, c2 = st.columns(2)
                c1.metric("Thevenin voltage", format_complex(answer_data["vth"], "V"))
                c2.metric("Thevenin resistance", f"{answer_data['rth']:.8g} Ω")
            elif answer_kind == "norton":
                c1, c2 = st.columns(2)
                c1.metric("Norton current", format_complex(answer_data["in"], "A"))
                c2.metric("Norton resistance", f"{answer_data['rn']:.8g} Ω")
            else:
                st.info("No single target was supplied. Full solved values are shown below.")

            with st.expander("3. Solved node voltages", expanded=True):
                rows = []
                for node, voltage in result.voltages.items():
                    rows.append(
                        {
                            "Node": node,
                            "Voltage": format_complex(voltage, "V"),
                        }
                    )
                st.dataframe(rows, use_container_width=True)

            with st.expander("4. Solved component currents, voltages and power", expanded=True):
                rows = []
                for cid, br in result.branches.items():
                    if cid.startswith("__"):
                        continue
                    rows.append(
                        {
                            "Component": cid,
                            "Type": br.component_type,
                            "Voltage": format_complex(br.voltage, "V"),
                            "Current": format_complex(br.current, "A"),
                            "Power": format_power(br.power, result.mode),
                            "Direction": f"{br.node_a} → {br.node_b}",
                        }
                    )
                st.dataframe(rows, use_container_width=True)

            with st.expander("5. Derivation / mathematical model", expanded=True):
                explanation = generate_deterministic_explanation(
                    circuit,
                    result,
                    answer_kind,
                    answer_data,
                )
                for line in explanation:
                    st.write(line)

            with st.expander("6. Validation checks", expanded=True):
                st.write("PASS" if result.residual <= 1e-6 else "FAIL", "— MNA residual")
                st.write("PASS" if result.kcl_error <= 1e-6 else "FAIL", "— KCL")
                st.write("PASS" if result.kvl_error <= 1e-6 else "FAIL", "— KVL")
                st.write("PASS" if result.power_error <= 1e-5 else "FAIL", "— Power balance")

            if show_matrix:
                with st.expander("7. MNA matrix"):
                    st.code(
                        matrix_text(
                            result.matrix_a,
                            result.vector_b,
                            result.unknown_names,
                        ),
                        language="text",
                    )

            with st.expander("8. Engineering notes"):
                st.write(
                    "Positive current means current flows from node_a to node_b. "
                    "Positive component power means absorbed power under the passive "
                    "sign convention."
                )
                st.write(
                    "AC results are phasors. Capacitors/inductors use their "
                    "frequency-domain impedances."
                )
                if circuit.get("notes"):
                    for note in circuit["notes"]:
                        st.write(f"• {note}")
                if diode_states:
                    st.write(
                        "Detected ideal diode states: "
                        + ", ".join(f"{k}={v}" for k, v in diode_states.items())
                    )

    elif "demo_result" in st.session_state:
        demo = st.session_state["demo_result"]
        demo_circuit_obj = st.session_state["demo_circuit"]
        st.subheader("Built-in solver test")
        st.success("The deterministic solver is running independently of image recognition.")
        st.metric("Current through R2", format_complex(demo.branches["R2"].current, "A"))
        st.code(
            matrix_text(demo.matrix_a, demo.vector_b, demo.unknown_names),
            language="text",
        )
        st.write(
            f"KCL={demo.kcl_error:.3e}, "
            f"KVL={demo.kvl_error:.3e}, "
            f"Power={demo.power_error:.3e}"
        )


if __name__ == "__main__":
    main()
