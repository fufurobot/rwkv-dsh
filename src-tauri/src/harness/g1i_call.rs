use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{json, Map, Value};
use std::collections::HashSet;
use std::fmt;

use crate::agent::protocol::ToolSpec;

#[derive(Debug, Clone, PartialEq)]
pub struct G1iFunctionCall {
    pub name: String,
    pub arguments: Value,
}

pub fn parse_function_call_json(raw: &str) -> Result<G1iFunctionCall, String> {
    normalize_function_call(&parse_strict_json_value(raw)?)
}

pub fn parse_function_call_json_with_tools(
    raw: &str,
    tools: &[ToolSpec],
) -> Result<G1iFunctionCall, String> {
    normalize_function_call_with_tools(&parse_strict_json_value(raw)?, tools)
}

pub fn normalize_function_call(value: &Value) -> Result<G1iFunctionCall, String> {
    normalize_function_call_with_tools(value, &[])
}

pub fn normalize_function_call_with_tools(
    value: &Value,
    tools: &[ToolSpec],
) -> Result<G1iFunctionCall, String> {
    let call = value
        .as_object()
        .ok_or_else(|| "g1i function call must be a JSON object".to_string())?;
    if looks_like_legacy_planner(call) {
        return Err(
            "g1i legacy planner JSON is not a function call; output exactly one {\"name\":\"TOOL_NAME\",\"arguments\":{...}} object"
                .into(),
        );
    }
    let mut names = Vec::new();
    collect_name_fields(call, "", &mut names)?;
    let mut shorthand_arguments = None;
    if names.is_empty() && call.len() == 1 {
        if let Some((name, arguments)) = call.iter().next() {
            if arguments.is_object()
                && !matches!(
                    name.as_str(),
                    "name"
                        | "tool_name"
                        | "command"
                        | "cmd"
                        | "tool"
                        | "function"
                        | "function_call"
                        | "tool_call"
                        | "tool_calls"
                        | "arguments"
                        | "args"
                        | "params"
                        | "parameters"
                        | "type"
                        | "id"
                        | "index"
                )
            {
                collect_name(Some(&Value::String(name.clone())), "shorthand", &mut names)?;
                shorthand_arguments = Some(arguments.clone());
            }
        }
    }

    let nested_function = match call.get("function") {
        Some(Value::Object(function)) => {
            collect_name_fields(function, "function", &mut names)?;
            Some(function)
        }
        Some(Value::String(name)) => {
            collect_name(Some(&Value::String(name.clone())), "function", &mut names)?;
            None
        }
        Some(Value::Null) | None => None,
        Some(_) => return Err("g1i function must be a tool name or object".into()),
    };

    let nested_function_call = match call.get("function_call") {
        Some(Value::Object(function_call)) => {
            collect_name_fields(function_call, "function_call", &mut names)?;
            Some(function_call)
        }
        Some(Value::Null) | None => None,
        Some(_) => return Err("g1i function_call must be an object".into()),
    };

    let nested_tool_calls = collect_tool_call_wrappers(call, &mut names)?;

    let mut argument_candidates = Vec::new();
    collect_arguments(call, "call", &mut argument_candidates);
    if shorthand_arguments.is_none() {
        collect_inline_arguments(call, "call", &mut argument_candidates);
    }
    if let Some(function) = nested_function {
        collect_arguments(function, "function", &mut argument_candidates);
        collect_inline_arguments(function, "function", &mut argument_candidates);
    }
    if let Some(function_call) = nested_function_call {
        collect_arguments(function_call, "function_call", &mut argument_candidates);
        collect_inline_arguments(function_call, "function_call", &mut argument_candidates);
    }
    for (index, tool_call) in nested_tool_calls.iter().enumerate() {
        collect_arguments(
            tool_call,
            &format!("tool_calls[{index}]"),
            &mut argument_candidates,
        );
        collect_inline_arguments(
            tool_call,
            &format!("tool_calls[{index}]"),
            &mut argument_candidates,
        );
        if let Some(function) = tool_call.get("function").and_then(Value::as_object) {
            collect_arguments(
                function,
                &format!("tool_calls[{index}].function"),
                &mut argument_candidates,
            );
            collect_inline_arguments(
                function,
                &format!("tool_calls[{index}].function"),
                &mut argument_candidates,
            );
        }
    }
    if let Some(arguments) = shorthand_arguments {
        argument_candidates.push(("shorthand".into(), arguments));
    }

    let name = if names.is_empty() {
        let inferred_arguments = infer_arguments(&argument_candidates)?;
        infer_name_from_arguments(inferred_arguments.as_ref(), tools).ok_or_else(|| {
            "g1i function call is missing name (expected name, tool_name, function.name, or tool_calls[0].function.name)"
                .to_string()
        })?
    } else {
        resolve_name(names)?
    };
    let arguments = resolve_arguments(&name, argument_candidates)?;

    Ok(G1iFunctionCall { name, arguments })
}

fn infer_arguments(candidates: &[(String, Value)]) -> Result<Option<Value>, String> {
    if candidates.is_empty() {
        return Ok(None);
    }
    let mut resolved: Option<(String, Value)> = None;
    for (field, value) in candidates {
        let decoded = decode_arguments_without_name(value.clone())
            .map_err(|error| format!("invalid {field}: {error}"))?;
        if let Some((previous_field, previous)) = &resolved {
            if previous != &decoded {
                return Err(format!(
                    "g1i function call has conflicting arguments in {previous_field} and {field}"
                ));
            }
        } else {
            resolved = Some((field.clone(), decoded));
        }
    }
    Ok(resolved.map(|(_, value)| value))
}

fn infer_name_from_arguments(arguments: Option<&Value>, tools: &[ToolSpec]) -> Option<String> {
    let arguments = normalize_arguments_for_matching(arguments?)?;
    if tools.is_empty() {
        return None;
    }

    let mut matches = tools
        .iter()
        .filter(|tool| arguments_match_tool(&arguments, &tool.arguments))
        .map(|tool| tool.name.trim())
        .filter(|name| valid_tool_name(name))
        .map(str::to_string)
        .collect::<Vec<_>>();
    matches.sort();
    matches.dedup();
    (matches.len() == 1).then(|| matches.remove(0))
}

fn normalize_arguments_for_matching(value: &Value) -> Option<Map<String, Value>> {
    let decoded = decode_argument_value(value.clone()).ok()?;
    let mut object = match decoded {
        Value::Object(object) => object,
        Value::Null => Map::new(),
        _ => return None,
    };
    for wrapper in ["args", "params", "parameters", "arguments"] {
        let Some(nested_raw) = object.remove(wrapper) else {
            continue;
        };
        match decode_argument_value(nested_raw.clone()).ok() {
            Some(Value::Object(mut nested)) => {
                for (key, value) in std::mem::take(&mut object) {
                    nested.insert(key, value);
                }
                object = nested;
                break;
            }
            _ => {
                object.insert(wrapper.into(), nested_raw);
            }
        }
    }
    if !object.contains_key("path") {
        for alias in ["file_path", "filename", "file", "filepath"] {
            if let Some(path) = object.remove(alias) {
                object.insert("path".into(), path);
                break;
            }
        }
    }
    Some(object)
}

fn arguments_match_tool(arguments: &Map<String, Value>, schema: &Value) -> bool {
    let Some(schema) = decode_tool_schema(schema) else {
        return false;
    };
    let Some(properties) = schema_properties(&schema) else {
        return false;
    };
    if properties.is_empty() || arguments.keys().any(|key| !properties.contains_key(key)) {
        return false;
    }
    if schema
        .get("required")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .any(|required| !arguments.contains_key(required))
    {
        return false;
    }
    arguments.iter().all(|(key, value)| {
        properties
            .get(key)
            .is_some_and(|property| value_matches_schema(value, property))
    })
}

fn decode_tool_schema(schema: &Value) -> Option<Value> {
    let mut decoded = schema.clone();
    for _ in 0..2 {
        let Value::String(encoded) = decoded else {
            break;
        };
        decoded = parse_strict_json_value(&encoded).ok()?;
    }
    decoded.as_object()?;
    Some(decoded)
}

fn schema_properties(schema: &Value) -> Option<&Map<String, Value>> {
    let object = schema.as_object()?;
    if let Some(properties) = object.get("properties").and_then(Value::as_object) {
        return Some(properties);
    }
    if looks_like_json_schema(object) {
        return None;
    }
    Some(object)
}

fn looks_like_json_schema(object: &Map<String, Value>) -> bool {
    object
        .get("type")
        .is_some_and(|value| value.is_string() || value.is_array())
        || object.get("required").is_some_and(Value::is_array)
        || [
            "$schema",
            "$id",
            "$ref",
            "items",
            "prefixItems",
            "contains",
            "additionalItems",
            "additionalProperties",
            "patternProperties",
            "propertyNames",
            "allOf",
            "anyOf",
            "oneOf",
            "not",
            "if",
            "then",
            "else",
            "const",
            "enum",
        ]
        .iter()
        .any(|key| object.contains_key(*key))
}

fn value_matches_schema(value: &Value, schema: &Value) -> bool {
    if let Value::String(kind) = schema {
        return value_matches_type_name(value, kind);
    }
    let Some(schema_object) = schema.as_object() else {
        return false;
    };
    if let Some(expected) = schema_object.get("const") {
        return value == expected;
    }
    if let Some(values) = schema_object.get("enum").and_then(Value::as_array) {
        if !values.iter().any(|expected| value == expected) {
            return false;
        }
    }
    if let Some(variants) = schema_object.get("allOf").and_then(Value::as_array) {
        if !variants
            .iter()
            .all(|variant| value_matches_schema(value, variant))
        {
            return false;
        }
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(variants) = schema_object.get(key).and_then(Value::as_array) {
            let matches = variants
                .iter()
                .filter(|variant| value_matches_schema(value, variant))
                .count();
            return if key == "oneOf" {
                matches == 1
            } else {
                matches > 0
            };
        }
    }
    if let Some(properties) = schema_object.get("properties").and_then(Value::as_object) {
        let Some(object) = value.as_object() else {
            return false;
        };
        if object.keys().any(|key| !properties.contains_key(key)) {
            return false;
        }
        if schema_object
            .get("required")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .any(|required| !object.contains_key(required))
        {
            return false;
        }
        return object.iter().all(|(key, value)| {
            properties
                .get(key)
                .is_some_and(|property| value_matches_schema(value, property))
        });
    }
    if let Some(items) = schema_object.get("items") {
        if let Some(array) = value.as_array() {
            if !array.iter().all(|item| value_matches_schema(item, items)) {
                return false;
            }
        } else if schema_object.get("type").is_none() {
            return false;
        }
    }
    if let Some(type_value) = schema_object.get("type") {
        if let Some(types) = type_value.as_array() {
            return types.iter().any(|kind| value_matches_type(value, kind));
        }
        return value_matches_type(value, type_value);
    }
    schema_object.contains_key("items")
}

fn value_matches_type(value: &Value, type_value: &Value) -> bool {
    type_value
        .as_str()
        .is_some_and(|kind| value_matches_type_name(value, kind))
}

fn value_matches_type_name(value: &Value, kind: &str) -> bool {
    match kind.trim().to_ascii_lowercase().as_str() {
        "string" | "str" => value.is_string(),
        "boolean" | "bool" => {
            value.is_boolean()
                || value.as_str().is_some_and(|text| {
                    matches!(
                        text.trim().to_ascii_lowercase().as_str(),
                        "true" | "false" | "1" | "0"
                    )
                })
        }
        "object" | "map" => value.is_object(),
        "array" | "list" => value.is_array(),
        "integer" | "int" => {
            value.as_i64().is_some()
                || value.as_u64().is_some()
                || value
                    .as_str()
                    .is_some_and(|text| text.trim().parse::<i64>().is_ok())
        }
        "number" | "float" => {
            value.is_number()
                || value
                    .as_str()
                    .is_some_and(|text| text.trim().parse::<f64>().ok().is_some_and(f64::is_finite))
        }
        "null" => value.is_null(),
        _ => false,
    }
}

fn looks_like_legacy_planner(call: &Map<String, Value>) -> bool {
    call.contains_key("analysis")
        || call.contains_key("plan")
        || call.contains_key("commands")
        || call.contains_key("task_complete")
}

fn collect_tool_call_wrappers<'a>(
    call: &'a Map<String, Value>,
    names: &mut Vec<(String, String)>,
) -> Result<Vec<&'a Map<String, Value>>, String> {
    let mut wrappers = Vec::new();
    if let Some(value) = call.get("tool_call") {
        let object = value
            .as_object()
            .ok_or_else(|| "g1i tool_call must be an object".to_string())?;
        collect_tool_call_name(object, "tool_call", names)?;
        wrappers.push(object);
    }
    if let Some(value) = call.get("tool_calls") {
        let entries = value
            .as_array()
            .ok_or_else(|| "g1i tool_calls must be an array".to_string())?;
        if entries.len() != 1 {
            return Err("g1i tool_calls must contain exactly one call".into());
        }
        let object = entries[0]
            .as_object()
            .ok_or_else(|| "g1i tool_calls entries must be objects".to_string())?;
        collect_tool_call_name(object, "tool_calls[0]", names)?;
        wrappers.push(object);
    }
    Ok(wrappers)
}

fn collect_tool_call_name(
    call: &Map<String, Value>,
    field: &str,
    names: &mut Vec<(String, String)>,
) -> Result<(), String> {
    collect_name_fields(call, field, names)?;
    if let Some(function) = call.get("function") {
        let function = function
            .as_object()
            .ok_or_else(|| format!("g1i {field}.function must be an object"))?;
        collect_name_fields(function, &format!("{field}.function"), names)?;
    }
    Ok(())
}

fn collect_name_fields(
    source: &Map<String, Value>,
    source_name: &str,
    names: &mut Vec<(String, String)>,
) -> Result<(), String> {
    for field in ["name", "tool_name", "command", "cmd", "tool"] {
        let label = if source_name.is_empty() {
            field.to_string()
        } else {
            format!("{source_name}.{field}")
        };
        collect_name(source.get(field), &label, names)?;
    }
    Ok(())
}

fn collect_name(
    value: Option<&Value>,
    field: &str,
    names: &mut Vec<(String, String)>,
) -> Result<(), String> {
    let Some(value) = value else {
        return Ok(());
    };
    let Value::String(name) = value else {
        return Err(format!("g1i {field} must be a string"));
    };
    let name = name.trim();
    if name.is_empty() {
        return Err(format!("g1i {field} must be non-empty"));
    }
    if !valid_tool_name(name) {
        return Err(format!("invalid g1i tool name {name:?}"));
    }
    names.push((field.to_string(), name.to_string()));
    Ok(())
}

fn resolve_name(names: Vec<(String, String)>) -> Result<String, String> {
    let Some((first_field, first_name)) = names.first() else {
        return Err(
            "g1i function call is missing name (expected name, tool_name, function.name, or tool_calls[0].function.name)"
                .into(),
        );
    };
    if let Some((field, name)) = names.iter().skip(1).find(|(_, name)| name != first_name) {
        return Err(format!(
            "g1i function call has conflicting names: {first_field}={first_name:?}, {field}={name:?}"
        ));
    }
    Ok(first_name.clone())
}

fn valid_tool_name(name: &str) -> bool {
    let mut characters = name.chars();
    let Some(first) = characters.next() else {
        return false;
    };
    (first.is_ascii_alphabetic() || first == '_')
        && characters
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | '-'))
}

fn collect_arguments(
    source: &Map<String, Value>,
    source_name: &str,
    candidates: &mut Vec<(String, Value)>,
) {
    for field in ["arguments", "args", "params", "parameters"] {
        if let Some(value) = source.get(field).filter(|value| !value.is_null()) {
            candidates.push((format!("{source_name}.{field}"), value.clone()));
        }
    }
}

fn collect_inline_arguments(
    source: &Map<String, Value>,
    source_name: &str,
    candidates: &mut Vec<(String, Value)>,
) {
    if ["arguments", "args", "params", "parameters"]
        .iter()
        .any(|field| source.get(*field).is_some_and(|value| !value.is_null()))
    {
        return;
    }
    let mut arguments = Map::new();
    for (key, value) in source {
        if !matches!(
            key.as_str(),
            "name"
                | "tool_name"
                | "command"
                | "cmd"
                | "tool"
                | "type"
                | "function"
                | "function_call"
                | "tool_call"
                | "tool_calls"
                | "id"
                | "index"
        ) && !value.is_null()
        {
            arguments.insert(key.clone(), value.clone());
        }
    }
    if !arguments.is_empty() {
        candidates.push((format!("{source_name}.inline"), Value::Object(arguments)));
    }
}

fn resolve_arguments(name: &str, candidates: Vec<(String, Value)>) -> Result<Value, String> {
    let mut resolved: Option<(String, Value)> = None;
    for (field, value) in candidates {
        let decoded =
            decode_arguments(name, value).map_err(|error| format!("invalid {field}: {error}"))?;
        if let Some((previous_field, previous)) = &resolved {
            if previous != &decoded {
                return Err(format!(
                    "g1i function call has conflicting arguments in {previous_field} and {field}"
                ));
            }
        } else {
            resolved = Some((field, decoded));
        }
    }
    Ok(resolved
        .map(|(_, value)| value)
        .unwrap_or_else(|| json!({})))
}

fn decode_arguments(name: &str, mut value: Value) -> Result<Value, String> {
    value = decode_argument_value(value)?;
    match value {
        Value::Object(_) => Ok(value),
        Value::Null => Ok(json!({})),
        Value::Array(tasks) if name == "run_agents" => Ok(json!({"tasks": tasks})),
        _ => Err("arguments must decode to a JSON object".into()),
    }
}

fn decode_arguments_without_name(value: Value) -> Result<Value, String> {
    match decode_argument_value(value)? {
        Value::Object(object) => Ok(Value::Object(object)),
        Value::Null => Ok(json!({})),
        _ => Err("arguments must decode to a JSON object".into()),
    }
}

fn decode_argument_value(mut value: Value) -> Result<Value, String> {
    for _ in 0..2 {
        let Value::String(encoded) = value else {
            break;
        };
        value = parse_strict_json_value(&encoded)
            .map_err(|error| format!("arguments JSON is invalid: {error}"))?;
    }
    Ok(value)
}

fn parse_strict_json_value(raw: &str) -> Result<Value, String> {
    let mut deserializer = serde_json::Deserializer::from_str(raw.trim());
    let StrictValue(value) = StrictValue::deserialize(&mut deserializer)
        .map_err(|error| format!("invalid g1i function JSON: {error}"))?;
    deserializer
        .end()
        .map_err(|error| format!("invalid g1i function JSON: {error}"))?;
    Ok(value)
}

struct StrictValue(Value);

impl<'de> Deserialize<'de> for StrictValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(StrictValueVisitor)
    }
}

struct StrictValueVisitor;

impl<'de> Visitor<'de> for StrictValueVisitor {
    type Value = StrictValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value without duplicate object keys")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::Number(value.into())))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::Number(value.into())))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        serde_json::Number::from_f64(value)
            .map(Value::Number)
            .map(StrictValue)
            .ok_or_else(|| E::custom("non-finite JSON number"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::String(value.to_string())))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::String(value)))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::Null))
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::Null))
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        StrictValue::deserialize(deserializer)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(StrictValue(value)) = sequence.next_element()? {
            values.push(value);
        }
        Ok(StrictValue(Value::Array(values)))
    }

    fn visit_map<A>(self, mut entries: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut seen = HashSet::new();
        let mut values = Map::new();
        while let Some(key) = entries.next_key::<String>()? {
            if !seen.insert(key.clone()) {
                return Err(de::Error::custom(format!("duplicate JSON key {key:?}")));
            }
            let StrictValue(value) = entries.next_value()?;
            values.insert(key, value);
        }
        Ok(StrictValue(Value::Object(values)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_canonical_and_observed_function_names() {
        for value in [
            json!({"name": "read_file", "arguments": {"path": "README.md"}}),
            json!({"function": "read_file", "arguments": {"path": "README.md"}}),
            json!({
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": "{\"path\":\"README.md\"}"
                }
            }),
            json!({
                "function_call": {
                    "name": "read_file",
                    "arguments": {"path": "README.md"}
                }
            }),
            json!({
                "read_file": {
                    "path": "README.md"
                }
            }),
            json!({"command": "read_file", "path": "README.md"}),
            json!({"cmd": "read_file", "args": {"path": "README.md"}}),
            json!({"tool": "read_file", "parameters": {"path": "README.md"}}),
            json!({"tool_name": "read_file", "params": {"path": "README.md"}}),
        ] {
            let call = normalize_function_call(&value).expect("normalize supported call");
            assert_eq!(call.name, "read_file");
            assert_eq!(call.arguments, json!({"path": "README.md"}));
        }
    }

    #[test]
    fn accepts_openai_tool_call_wrappers() {
        for value in [
            json!({
                "tool_call": {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": "{\"path\":\"README.md\"}"
                    }
                }
            }),
            json!({
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": "{\"path\":\"README.md\"}"
                    }
                }]
            }),
        ] {
            let call = normalize_function_call(&value).expect("normalize tool call wrapper");
            assert_eq!(call.name, "read_file");
            assert_eq!(call.arguments, json!({"path": "README.md"}));
        }
    }

    #[test]
    fn accepts_safe_argument_aliases_and_double_encoding() {
        let encoded = serde_json::to_string("{\"path\":\"README.md\"}").unwrap();
        let call = normalize_function_call(&json!({
            "function": "read_file",
            "parameters": encoded
        }))
        .expect("normalize encoded parameters");
        assert_eq!(call.arguments, json!({"path": "README.md"}));

        let agents = normalize_function_call(&json!({
            "function": "run_agents",
            "args": [{"goal": "读取 README"}, {"goal": "梳理源码入口"}]
        }))
        .expect("normalize run_agents array");
        assert_eq!(agents.arguments["tasks"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn rejects_ambiguous_or_unstructured_calls() {
        for value in [
            json!({"arguments": {"path": "."}}),
            json!({"tool": {"path": "."}}),
            json!({"id": {"path": "."}}),
            json!({"name": "", "function": "write_file", "arguments": {}}),
            json!({"name": "read_file", "function": "write_file", "arguments": {}}),
            json!({"function": "read file", "arguments": {}}),
            json!({"function": "read_file", "arguments": "README.md"}),
            json!({
                "name": "read_file",
                "arguments": {"path": "README.md"},
                "args": {"path": "Cargo.toml"}
            }),
            json!({
                "name": "read_file",
                "arguments": {"path": "README.md"},
                "function_call": {
                    "name": "write_file",
                    "arguments": {"path": "README.md"}
                }
            }),
            json!({
                "function": {
                    "name": "read_file",
                    "arguments": {"path": "README.md"}
                },
                "function_call": {
                    "name": "read_file",
                    "arguments": {"path": "Cargo.toml"}
                }
            }),
            json!({"function_call": "read_file"}),
            json!({"function_call": {"arguments": {"path": "README.md"}}}),
        ] {
            assert!(normalize_function_call(&value).is_err(), "{value}");
        }
    }

    #[test]
    fn accepts_matching_fields_across_all_observed_wrappers() {
        let call = normalize_function_call(&json!({
            "name": "read_file",
            "arguments": {"path": "README.md"},
            "function": {
                "name": "read_file",
                "arguments": "{\"path\":\"README.md\"}"
            },
            "function_call": {
                "name": "read_file",
                "parameters": {"path": "README.md"}
            }
        }))
        .expect("matching wrappers");
        assert_eq!(call.name, "read_file");
        assert_eq!(call.arguments, json!({"path": "README.md"}));
    }

    #[test]
    fn strict_json_rejects_duplicates_and_trailing_actions() {
        for raw in [
            r#"{"name":"read_file","name":"write_file","arguments":{}}"#,
            r#"{"name":"read_file","arguments":{"path":"README.md","path":"Cargo.toml"}}"#,
            r#"{"name":"read_file","arguments":"{\"path\":\"README.md\",\"path\":\"Cargo.toml\"}"}"#,
            r#"{"name":"read_file","arguments":{"path":"README.md"}} trailing"#,
            r#"{"name":"read_file","arguments":{"path":"README.md"}} {"name":"write_file","arguments":{}}"#,
            r#"{"function_call":{"name":"read_file","name":"write_file","arguments":{}}}"#,
        ] {
            assert!(parse_function_call_json(raw).is_err(), "{raw}");
        }
    }

    #[test]
    fn rejects_legacy_planner_json_with_actionable_error() {
        let error = parse_function_call_json(
            r#"{"analysis":"list the files","plan":"use ls","commands":["ls"],"task_complete":false}"#,
        )
        .expect_err("legacy planner output must not execute");
        assert!(error.contains("legacy planner JSON"));
        assert!(error.contains("TOOL_NAME"));
    }

    #[test]
    fn recovers_missing_name_only_from_one_matching_tool_schema() {
        let tools = [
            ToolSpec {
                name: "list_files".into(),
                description: "list".into(),
                arguments: json!({
                    "path": {"type": "string"},
                    "offset": {"type": "integer"},
                    "max_depth": {"type": "integer"}
                }),
            },
            ToolSpec {
                name: "read_file".into(),
                description: "read".into(),
                arguments: json!({
                    "path": {"type": "string"},
                    "offset": {"type": "integer"},
                    "max_bytes": {"type": "integer"}
                }),
            },
        ];

        let recovered =
            parse_function_call_json_with_tools(r#"{"arguments":{"max_depth":1}}"#, &tools)
                .expect("max_depth uniquely identifies list_files");
        assert_eq!(recovered.name, "list_files");
        assert_eq!(recovered.arguments, json!({"max_depth": 1}));

        let flat = parse_function_call_json_with_tools(r#"{"max_depth":1}"#, &tools)
            .expect("flat arguments also identify list_files");
        assert_eq!(flat.name, "list_files");
        assert_eq!(flat.arguments, json!({"max_depth": 1}));

        let ambiguous =
            parse_function_call_json_with_tools(r#"{"arguments":{"path":"README.md"}}"#, &tools)
                .expect_err("path alone matches both tools");
        assert!(ambiguous.contains("missing name"));

        let ambiguous_offset = parse_function_call_json_with_tools(
            r#"{"arguments":{"path":"README.md","offset":128}}"#,
            &tools,
        )
        .expect_err("path and offset match both file tools");
        assert!(ambiguous_offset.contains("missing name"));

        let empty = parse_function_call_json_with_tools(r#"{"arguments":{}}"#, &tools)
            .expect_err("empty arguments cannot identify a tool");
        assert!(empty.contains("missing name"));

        let only_list = [tools[0].clone()];
        let empty = parse_function_call_json_with_tools(r#"{"arguments":{}}"#, &only_list)
            .expect("empty arguments uniquely match the only allowed list tool");
        assert_eq!(empty.name, "list_files");
    }

    #[test]
    fn schema_recovery_decodes_encoded_and_flat_shapes() {
        let encoded = ToolSpec {
            name: "list_files".into(),
            description: "list".into(),
            arguments: Value::String(
                r#"{"type":"object","properties":{"max_depth":{"type":"integer"}},"required":["max_depth"]}"#
                    .into(),
            ),
        };
        let recovered = parse_function_call_json_with_tools(
            r#"{"arguments":{"max_depth":"2"}}"#,
            std::slice::from_ref(&encoded),
        )
        .expect("encoded schema and numeric string should match");
        assert_eq!(recovered.name, "list_files");
        assert_eq!(recovered.arguments, json!({"max_depth":"2"}));

        let flat = ToolSpec {
            name: "search".into(),
            description: "search".into(),
            arguments: json!({"case_sensitive": "boolean"}),
        };
        let recovered = parse_function_call_json_with_tools(
            r#"{"arguments":{"case_sensitive":"true"}}"#,
            std::slice::from_ref(&flat),
        )
        .expect("flat string type schema should match");
        assert_eq!(recovered.name, "search");
    }

    #[test]
    fn schema_recovery_rejects_invalid_or_ambiguous_coercions() {
        let integer = ToolSpec {
            name: "list_files".into(),
            description: "list".into(),
            arguments: json!({"max_depth": {"type": "integer"}}),
        };
        let invalid = parse_function_call_json_with_tools(
            r#"{"arguments":{"max_depth":"not-a-number"}}"#,
            std::slice::from_ref(&integer),
        )
        .expect_err("invalid numeric strings must not become empty arguments");
        assert!(invalid.contains("missing name"));

        let string = ToolSpec {
            name: "custom_tool".into(),
            description: "custom".into(),
            arguments: json!({"max_depth": {"type": "string"}}),
        };
        let ambiguous = parse_function_call_json_with_tools(
            r#"{"arguments":{"max_depth":"2"}}"#,
            &[integer, string],
        )
        .expect_err("coercible values matching two tools must remain ambiguous");
        assert!(ambiguous.contains("missing name"));
    }

    #[test]
    fn schema_recovery_accepts_union_string_or_array_values() {
        let write = ToolSpec {
            name: "write_file".into(),
            description: "write".into(),
            arguments: json!({
                "path": {"type": "string"},
                "content": {
                    "type": ["string", "array"],
                    "items": {"type": "string"}
                }
            }),
        };
        let text = parse_function_call_json_with_tools(
            r#"{"arguments":{"path":"note.txt","content":"hello"}}"#,
            std::slice::from_ref(&write),
        )
        .expect("string content should match union schema");
        assert_eq!(text.name, "write_file");

        let lines = parse_function_call_json_with_tools(
            r#"{"arguments":{"path":"note.txt","content":["hello","world"]}}"#,
            std::slice::from_ref(&write),
        )
        .expect("string array content should match union schema");
        assert_eq!(lines.name, "write_file");

        let invalid = parse_function_call_json_with_tools(
            r#"{"arguments":{"path":"note.txt","content":["hello",3]}}"#,
            std::slice::from_ref(&write),
        )
        .expect_err("array items must still honor their schema");
        assert!(invalid.contains("missing name"));
    }

    #[test]
    fn schema_recovery_rejects_malformed_schema_shapes() {
        let object_schema_without_properties = ToolSpec {
            name: "object_tool".into(),
            description: "object".into(),
            arguments: json!({"type": "object"}),
        };
        let error = parse_function_call_json_with_tools(
            r#"{"arguments":{"type":"object"}}"#,
            std::slice::from_ref(&object_schema_without_properties),
        )
        .expect_err("schema keywords must not become flat argument names");
        assert!(error.contains("missing name"));

        let malformed_union = ToolSpec {
            name: "malformed".into(),
            description: "malformed".into(),
            arguments: json!({
                "value": {"type": ["string", 7]}
            }),
        };
        let error = parse_function_call_json_with_tools(
            r#"{"arguments":{"value":42}}"#,
            std::slice::from_ref(&malformed_union),
        )
        .expect_err("non-string schema types must not match by default");
        assert!(error.contains("missing name"));
    }

    #[test]
    fn explicit_name_aliases_still_reject_conflicts() {
        for raw in [
            r#"{"name":"read_file","command":"write_file","arguments":{}}"#,
            r#"{"cmd":"read_file","tool":"search","arguments":{}}"#,
            r#"{"name":"read_file","tool_name":"write_file","arguments":{}}"#,
            r#"{"function":{"name":"read_file","command":"search","arguments":{}}}"#,
        ] {
            let error = parse_function_call_json(raw).expect_err("conflicting aliases");
            assert!(error.contains("conflicting names"), "{error}");
        }
    }
}
