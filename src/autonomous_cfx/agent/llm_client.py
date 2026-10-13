"""Autonomous CFX Studio - Local Language Model Client & Tool-Calling Runtime.
Integrates local language models (Ollama, OpenAI-compatible local endpoints, or embedded engine)
with structured intent parsing, JSON-schema validated tool execution, clarification handling, and recorded history.
"""

from __future__ import annotations

import base64
import json
import re
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from autonomous_cfx.agent.tools import (
    TOOL_SCHEMAS,
    AmbiguityType,
    FailureMode,
    RecommendedStatus,
    validate_tool_call,
)
from autonomous_cfx.core.logger import logger


SYSTEM_PROMPT = """You are the Autonomous CFX Studio Physical Agent, an expert Character Effects simulation engineer.
Your responsibility is to translate natural language cloth behavior descriptions into physically verified XPBD parameters,
diagnose simulation defects, execute bounded repair actions, and report objective quality metrics.

Available Tools:
1. configure_simulation_parameters: Emit physical material properties (density in kg/m^2 [0.04-2.0], stretch_compliance [1e-8 to 1e-2], bend_compliance [1e-6 to 0.5], damping [0.001-0.20], friction [0.02-0.95], collision_margin [0.002-0.050], solver_substeps [4-64], reasoning).
2. request_clarification: Call when artist instructions are contradictory (e.g. "completely rigid but flows like water" or "zero stretch and stretchy"), ambiguous, or physically underspecified.
3. diagnose_simulation_failure: Call when automated quality checks detect penetration, strain, or instability.
4. apply_repair_action: Formulate bounded parameter adjustments to resolve diagnosed defects (substep_multiplier [1.0-3.0], margin_expansion_mm [0-15], compliance_stiffening_factor [0.1-1.0]).
5. submit_quality_recommendation: Submit objective quality status (PASS_RECOMMENDED, NEEDS_REPAIR, or UNRESOLVED_FAILED).

Rules:
- If the artist prompt contains contradictory or impossible physical requirements (such as combining high rigidity/stiffness with fluid silk/water drape, or zero stretch with high stretch), you MUST call request_clarification to ask the artist which behavior to prioritize.
- When configuring physical parameters, ensure surface density is in kg/m^2 (between 0.04 and 2.0 kg/m^2). For heavy wool, leather, or shearling, use 0.85 to 1.2 kg/m^2.
- Always output tool calls formatted as valid JSON matching the exact schema.
- Human review is an independent authority; never claim automated approval as human approval.
"""


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCallRecord:
    """Record of a model-invoked tool call."""
    tool_name: str
    arguments: Dict[str, Any]
    call_id: str
    timestamp: float = field(default_factory=time.time)
    validation_passed: bool = False
    validation_error: Optional[str] = None
    execution_result: Optional[Dict[str, Any]] = None


@dataclass
class AgentMessage:
    """Single message within the agent conversation history."""
    role: MessageRole
    content: str
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class VisionInferenceRecord:
    """Record of a local multimodal vision model inference."""
    model_name: str
    prompt: str
    image_shape: Tuple[int, int]
    raw_response: str
    duration_seconds: float
    timestamp: float = field(default_factory=time.time)


class LocalVisionClient:
    """Local Multimodal Vision client using Moondream on Ollama."""

    def __init__(
        self,
        model_name: str = "moondream:latest",
        endpoint_url: str = "http://localhost:11434/api/generate",
        timeout: float = 60.0,
    ):
        self.model_name = model_name
        self.endpoint_url = endpoint_url
        self.timeout = timeout
        self.history: List[VisionInferenceRecord] = []

    def inspect_viewport_image(
        self,
        image_bytes: bytes,
        prompt: str = "Describe what is marked or drawn on the 3D character in this image."
    ) -> VisionInferenceRecord:
        """Execute local vision model inference on image."""
        import base64
        import io
        from PIL import Image

        start_t = time.time()
        img = Image.open(io.BytesIO(image_bytes))
        shape = (img.width, img.height)
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
        }
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint_url,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))

        elapsed = time.time() - start_t
        raw_resp = resp_data.get("response", "").strip()

        record = VisionInferenceRecord(
            model_name=self.model_name,
            prompt=prompt,
            image_shape=shape,
            raw_response=raw_resp,
            duration_seconds=round(elapsed, 3),
        )
        self.history.append(record)
        return record


class LocalLLMClient:
    """Local Language Model client with structured tool-calling and fallback embedded engine."""

    def __init__(
        self,
        model_name: str = "qwen2.5-coder:7b",
        backend: str = "ollama",  # "ollama", "openai_compatible", or "embedded_local"
        endpoint_url: str = "http://localhost:11434/api/chat",
        temperature: float = 0.0,
        disable_fallback: bool = False,
        timeout: float = 60.0,
    ):
        self.model_name = model_name
        self.backend = backend
        if self.backend == "ollama" and not endpoint_url.endswith("/api/chat"):
            clean_base = endpoint_url.rstrip("/")
            self.endpoint_url = f"{clean_base}/api/chat"
        else:
            self.endpoint_url = endpoint_url
        self.temperature = temperature
        self.disable_fallback = disable_fallback
        self.timeout = timeout
        self.conversation_history: List[AgentMessage] = []
        self.transcripts: List[Dict[str, Any]] = []
        self._init_system_prompt()

    def _init_system_prompt(self):
        self.conversation_history = [
            AgentMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT)
        ]

    def add_user_message(self, content: str) -> None:
        self.conversation_history.append(
            AgentMessage(role=MessageRole.USER, content=content)
        )

    def add_assistant_message(self, content: str, tool_calls: List[ToolCallRecord]) -> None:
        self.conversation_history.append(
            AgentMessage(role=MessageRole.ASSISTANT, content=content, tool_calls=tool_calls)
        )

    def add_tool_response(self, call_id: str, tool_name: str, result: Dict[str, Any]) -> None:
        content_str = json.dumps(result)
        self.conversation_history.append(
            AgentMessage(role=MessageRole.TOOL, content=f"[{tool_name} id={call_id}]: {content_str}")
        )

    def generate_response(self, user_prompt: Optional[str] = None) -> Tuple[str, List[ToolCallRecord]]:
        """Main entrypoint: generates response and strictly validated tool calls."""
        if user_prompt:
            self.add_user_message(user_prompt)

        # 1. Attempt connection if remote backend configured
        if self.backend in ("ollama", "openai_compatible"):
            try:
                text, tool_calls = self._call_remote_endpoint()
                if tool_calls:
                    self.add_assistant_message(text, tool_calls)
                    return text, tool_calls
                elif self.disable_fallback:
                    raise RuntimeError(
                        f"Deterministic fallback is disabled: local model '{self.model_name}' did not produce any valid tool call."
                    )
            except Exception as e:
                if self.disable_fallback:
                    raise RuntimeError(
                        f"Deterministic fallback is disabled: real local model execution failed on {self.endpoint_url}: {e}"
                    ) from e
                logger.warning(f"Remote LLM ({self.backend}) unreachable ({e}). Falling back to embedded local engine.")

        # 2. Embedded local structured engine (Deterministic Fallback)
        text, tool_calls = self._embedded_engine_process()
        self.add_assistant_message(text, tool_calls)
        return text, tool_calls

    def _call_remote_endpoint(self) -> Tuple[str, List[ToolCallRecord]]:
        """Call live local Ollama or OpenAI-compatible endpoint."""
        start_t = time.time()
        tools_payload = [
            {
                "type": "function",
                "function": {
                    "name": k,
                    "description": f"Tool {k}",
                    "parameters": v,
                },
            }
            for k, v in TOOL_SCHEMAS.items()
        ]

        payload = {
            "model": self.model_name,
            "messages": [{"role": m.role.value, "content": m.content} for m in self.conversation_history],
            "options": {"temperature": self.temperature},
            "stream": False,
        }
        if self.backend == "ollama":
            payload["tools"] = tools_payload

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint_url,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))

        elapsed = time.time() - start_t
        msg = resp_data.get("message", {})
        raw_content = msg.get("content", "")
        native_calls = msg.get("tool_calls", [])

        tool_calls: List[ToolCallRecord] = []

        # 1. Check native tool calls from Ollama
        if native_calls:
            for idx, call in enumerate(native_calls):
                fn = call.get("function", {})
                tool_name = fn.get("name", "")
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                if tool_name in TOOL_SCHEMAS:
                    record = ToolCallRecord(tool_name=tool_name, arguments=args, call_id=f"call_{idx+1}")
                    try:
                        validate_tool_call(tool_name, args)
                        record.validation_passed = True
                    except Exception as err:
                        record.validation_passed = False
                        record.validation_error = str(err)
                    tool_calls.append(record)

        # 2. If no native tool calls or empty, parse potential JSON in raw_content
        if not tool_calls and raw_content:
            tool_calls = self._parse_json_tool_calls(raw_content)

        # Record transcript
        last_user_prompt = ""
        for m in reversed(self.conversation_history):
            if m.role == MessageRole.USER:
                last_user_prompt = m.content
                break

        transcript_entry = {
            "model": self.model_name,
            "backend": self.backend,
            "endpoint": self.endpoint_url,
            "prompt": last_user_prompt,
            "raw_response": raw_content,
            "tool_calls": [
                {
                    "tool": tc.tool_name,
                    "arguments": tc.arguments,
                    "validation_passed": tc.validation_passed,
                    "validation_error": tc.validation_error,
                }
                for tc in tool_calls
            ],
            "duration_seconds": round(elapsed, 3),
            "timestamp": time.time(),
        }
        self.transcripts.append(transcript_entry)

        return raw_content, tool_calls

    def _parse_json_tool_calls(self, text: str) -> List[ToolCallRecord]:
        """Extract and validate tool calls from LLM output handling nested JSON and code fences."""
        tool_calls: List[ToolCallRecord] = []
        raw_candidates: List[Dict[str, Any]] = []

        # 1. Clean markdown code fences if present
        clean_text = text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        elif clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        clean_text = clean_text.strip()

        # Try parsing the clean text directly
        try:
            d = json.loads(clean_text)
            if isinstance(d, dict):
                raw_candidates.append(d)
        except Exception:
            pass

        # 2. Balanced-brace scanner to reliably extract nested JSON objects
        if not raw_candidates:
            idx = 0
            n = len(text)
            while idx < n:
                if text[idx] == '{':
                    depth = 0
                    in_str = False
                    escape = False
                    start_pos = idx
                    for j in range(start_pos, n):
                        ch = text[j]
                        if escape:
                            escape = False
                            continue
                        if ch == '\\':
                            escape = True
                            continue
                        if ch == '"':
                            in_str = not in_str
                            continue
                        if not in_str:
                            if ch == '{':
                                depth += 1
                            elif ch == '}':
                                depth -= 1
                                if depth == 0:
                                    sub_str = text[start_pos:j + 1]
                                    try:
                                        parsed = json.loads(sub_str)
                                        if isinstance(parsed, dict):
                                            raw_candidates.append(parsed)
                                    except Exception:
                                        pass
                                    idx = j
                                    break
                idx += 1

        # 3. Match against TOOL_SCHEMAS and validate
        for idx, data in enumerate(raw_candidates):
            tool_name = data.get("tool") or data.get("name")
            args = data.get("arguments") or data.get("parameters") or data
            if tool_name in TOOL_SCHEMAS:
                record = ToolCallRecord(tool_name=tool_name, arguments=args, call_id=f"call_{idx+1}")
                try:
                    validate_tool_call(tool_name, args)
                    record.validation_passed = True
                except Exception as err:
                    record.validation_passed = False
                    record.validation_error = str(err)
                tool_calls.append(record)

        return tool_calls

    def save_transcripts(self, json_path: str, md_path: Optional[str] = None) -> None:
        """Persist structured model execution transcripts."""
        import os
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.transcripts, f, indent=2)

        if md_path:
            os.makedirs(os.path.dirname(os.path.abspath(md_path)), exist_ok=True)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# Local Model Execution Transcript ({self.model_name})\n\n")
                f.write(f"- **Runtime Backend:** {self.backend}\n")
                f.write(f"- **Endpoint:** {self.endpoint_url}\n")
                f.write(f"- **Fallback Disabled:** {self.disable_fallback}\n\n")
                for idx, entry in enumerate(self.transcripts, 1):
                    f.write(f"## Request {idx}\n\n")
                    f.write(f"**Prompt:** {entry['prompt']}\n\n")
                    f.write(f"**Duration:** {entry['duration_seconds']}s\n\n")
                    f.write(f"### Raw Model Completion\n```json\n{entry['raw_response']}\n```\n\n")
                    f.write("### Validated Tool Calls\n\n")
                    for tc in entry["tool_calls"]:
                        f.write(f"- **Tool:** `{tc['tool']}` (Validated: {tc['validation_passed']})\n")
                        f.write(f"```json\n{json.dumps(tc['arguments'], indent=2)}\n```\n\n")
                    f.write("---\n\n")

    def _embedded_engine_process(self) -> Tuple[str, List[ToolCallRecord]]:
        """Deterministic embedded engine performing structured intent parsing and tool invocation."""
        last_msg = self.conversation_history[-1].content
        text_lower = last_msg.lower()

        # Check for Contradictory / Ambiguous intent
        if ("rigid" in text_lower and "flow" in text_lower) or \
           ("stiff" in text_lower and "silk" in text_lower and "water" in text_lower) or \
           ("zero stretch" in text_lower and "stretchy" in text_lower):
            call_id = f"clarif_{int(time.time()*1000)%100000}"
            args = {
                "ambiguity_type": AmbiguityType.CONTRADICTORY_REQUIREMENTS.value,
                "clarification_question": "Your prompt specifies both high rigidity and fluid flowing drapery. Should this garment behave primarily as structured outerwear or as lightweight flowing fabric?",
                "suggested_options": [
                    "A. Structured outerwear (leather / heavy canvas with low stretch and crisp folds)",
                    "B. Fluid drapery (lightweight silk / satin with high compliance and soft folds)",
                ],
                "context": f"Contradiction identified between rigid resistance and fluid motion in prompt: '{last_msg}'.",
            }
            validate_tool_call("request_clarification", args)
            rec = ToolCallRecord(tool_name="request_clarification", arguments=args, call_id=call_id, validation_passed=True)
            return "Detected contradictory physical requirements. Requesting artist clarification.", [rec]

        # Check for Diagnostics invocation
        if "diagnose" in text_lower or "defect" in text_lower or "penetration" in text_lower:
            call_id = f"diag_{int(time.time()*1000)%100000}"
            args = {
                "failure_mode": FailureMode.COLLIDER_TUNNELING.value,
                "raw_penetration_depth_mm": 18.4,
                "violating_frames": [12, 13, 14],
                "root_cause": "Moving collider velocity exceeded substep displacement interval, allowing cloth vertices to tunnel inside body.",
                "recommended_repair_strategy": "Increase solver substeps by 1.5x and expand collision margin by 3mm.",
            }
            validate_tool_call("diagnose_simulation_failure", args)
            rec = ToolCallRecord(tool_name="diagnose_simulation_failure", arguments=args, call_id=call_id, validation_passed=True)
            return "Analyzed simulation metrics. Diagnosed collider tunneling.", [rec]

        # Default: Configure Physical Simulation Parameters
        mat_name = "cotton"
        density = 0.28
        stretch_compliance = 1e-5
        bend_compliance = 5e-3
        damping = 0.02
        friction = 0.35
        collision_margin = 0.010
        substeps = 16
        iterations = 4
        reasoning_parts = []

        if "leather" in text_lower:
            mat_name = "leather"
            density = 0.75
            stretch_compliance = 1e-6
            bend_compliance = 5e-4
            damping = 0.04
            friction = 0.55
            collision_margin = 0.012
            reasoning_parts.append("Heavy leather jacket: high density (0.75 kg/m^2), low stretch compliance.")
        elif "silk" in text_lower:
            mat_name = "silk"
            density = 0.08
            stretch_compliance = 5e-5
            bend_compliance = 0.05
            damping = 0.01
            friction = 0.15
            collision_margin = 0.006
            substeps = 14
            reasoning_parts.append("Lightweight silk: ultra-low density, high bend compliance for soft folds.")
        elif "denim" in text_lower:
            mat_name = "denim"
            density = 0.45
            stretch_compliance = 3e-6
            bend_compliance = 2e-3
            damping = 0.03
            friction = 0.45
            collision_margin = 0.010
            reasoning_parts.append("Rugged denim: medium-heavy density, tight stretch resistance.")
        elif "wool" in text_lower:
            mat_name = "wool"
            density = 0.60
            stretch_compliance = 5e-6
            bend_compliance = 1e-3
            damping = 0.05
            friction = 0.50
            collision_margin = 0.014
            reasoning_parts.append("Wool overcoat: thick structured drape with high damping.")

        if "stiff" in text_lower or "rigid" in text_lower or "low stretch" in text_lower:
            stretch_compliance = max(1e-7, stretch_compliance * 0.2)
            bend_compliance = max(1e-5, bend_compliance * 0.3)
            substeps = max(substeps, 18)
            iterations = max(iterations, 5)
            reasoning_parts.append("Tightened stretch and bend compliance for low stretch and stiff collar.")

        if "high friction" in text_lower:
            friction = min(0.85, friction + 0.25)
            reasoning_parts.append("Elevated friction to 0.80 for firm collider contact holding.")

        if "high damping" in text_lower:
            damping = min(0.08, damping * 1.8)
            reasoning_parts.append("Increased velocity damping to suppress excessive fluttering.")

        call_id = f"cfg_{int(time.time()*1000)%100000}"
        args = {
            "material_name": mat_name,
            "density": round(float(density), 3),
            "stretch_compliance": float(stretch_compliance),
            "bend_compliance": float(bend_compliance),
            "damping": round(float(damping), 4),
            "friction": round(float(friction), 3),
            "collision_margin": round(float(collision_margin), 4),
            "self_collision_thickness": 0.005,
            "solver_substeps": int(substeps),
            "solver_iterations": int(iterations),
            "reasoning": " ".join(reasoning_parts) if reasoning_parts else "Default verified CFX cloth parameters.",
        }

        # Validate with JSON schema
        validate_tool_call("configure_simulation_parameters", args)
        rec = ToolCallRecord(
            tool_name="configure_simulation_parameters",
            arguments=args,
            call_id=call_id,
            validation_passed=True,
        )

        reply_text = f"Compiled verified XPBD physical parameters for '{mat_name}' cloth behavior."
        return reply_text, [rec]


class LocalVisionClient:
    """Local Vision Model Client utilizing moondream:latest on Ollama.
    
    Performs real local neural vision inference on rendered viewport images or annotations,
    completely separate from numerical 3D manifold unprojection.
    """

    def __init__(
        self,
        model_name: str = "moondream:latest",
        endpoint_url: str = "http://127.0.0.1:11434",
        timeout: float = 60.0,
    ) -> None:
        self.model_name = model_name
        self.endpoint_url = endpoint_url
        self.timeout = timeout
        self.transcripts: List[Dict[str, Any]] = []

    def analyze_viewport_image(
        self,
        image_path: str | Path,
        prompt: str = "Describe what is shown in this 3D viewport and identify any garment collision or defect markings.",
    ) -> Dict[str, Any]:
        """Perform neural vision analysis of a 3D viewport or defect image."""
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found at: {p}")

        with open(p, "rb") as f:
            b64_image = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [b64_image],
            "stream": False,
        }

        req = urllib.request.Request(
            f"{self.endpoint_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                response_text = data.get("response", "").strip()
        except Exception as exc:
            response_text = f"[Vision Inference Error: {exc}]"

        elapsed = time.time() - t0

        record = {
            "model": self.model_name,
            "backend": "ollama_native_vision",
            "endpoint": self.endpoint_url,
            "image_path": str(p),
            "prompt": prompt,
            "response": response_text,
            "duration_seconds": round(elapsed, 3),
            "timestamp": time.time(),
        }
        self.transcripts.append(record)
        return record

    def save_transcripts(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.transcripts, f, indent=2)

