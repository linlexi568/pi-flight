"""Serialization utilities for PI-Flight DSL programs.

This module converts AST nodes (TerminalNode, UnaryOpNode, BinaryOpNode, IfNode,
and advanced nodes) to and from JSON-serializable dictionaries so that
searched programs can be persisted and later reloaded for evaluation or
dataset collection.
"""
from __future__ import annotations
from typing import Any, Dict, List
from .dsl import (
    ProgramNode,
    TerminalNode,
    UnaryOpNode,
    BinaryOpNode,
    IfNode,
)

ASTDict = Dict[str, Any]

def serialize_ast(node: ProgramNode) -> ASTDict:
    if isinstance(node, TerminalNode):
        return {"type": "Terminal", "value": node.value}
    if isinstance(node, UnaryOpNode):
        return {"type": "Unary", "op": node.op, "child": serialize_ast(node.child)}
    if isinstance(node, BinaryOpNode):
        return {"type": "Binary", "op": node.op, "left": serialize_ast(node.left), "right": serialize_ast(node.right)}
    if isinstance(node, IfNode):
        return {"type": "If", "condition": serialize_ast(node.condition), "then": serialize_ast(node.then_branch), "else": serialize_ast(node.else_branch)}
    raise TypeError(f"Cannot serialize unknown node type: {type(node)}")

def deserialize_ast(obj: ASTDict) -> ProgramNode:
    t = obj.get("type")
    if t == "Terminal":
        return TerminalNode(obj["value"])
    if t == "Unary":
        return UnaryOpNode(obj["op"], deserialize_ast(obj["child"]))
    if t == "Binary":
        return BinaryOpNode(obj["op"], deserialize_ast(obj["left"]), deserialize_ast(obj["right"]))
    if t == "If":
        return IfNode(deserialize_ast(obj["condition"]), deserialize_ast(obj["then"]), deserialize_ast(obj["else"]))
    raise ValueError(f"Unknown AST dict type: {t}")

def serialize_program(rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    serial_rules: List[Dict[str, Any]] = []
    for r in rules:
        condition = r.get('condition')
        if condition is None:
            continue
        action_list = r.get('action', [])
        serial_rules.append({
            'condition': serialize_ast(condition),
            'action': [serialize_ast(a) for a in action_list]
        })
    return {"rules": serial_rules}

def deserialize_program(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    rules_out: List[Dict[str, Any]] = []
    for r in obj.get('rules', []):
        cond_ast = deserialize_ast(r['condition'])
        action_asts = [deserialize_ast(a) for a in r.get('action', [])]
        rules_out.append({'condition': cond_ast, 'action': action_asts})
    return rules_out

def save_program_json(rules: List[Dict[str, Any]], path: str, meta: Dict[str, Any] | None = None):
    import json, os, time
    payload = serialize_program(rules)
    if meta:
        payload['meta'] = meta
    payload.setdefault('meta', {})['saved_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def load_program_json(path: str) -> List[Dict[str, Any]]:
    import json
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return deserialize_program(data)

def save_search_history(history: List[Dict[str, Any]], path: str):
    import json, os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'history': history}, f, ensure_ascii=False, indent=2)

__all__ = [
    'serialize_ast','deserialize_ast','serialize_program','deserialize_program',
    'save_program_json','load_program_json','save_search_history'
]
