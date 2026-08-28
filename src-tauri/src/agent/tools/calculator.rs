use serde::Deserialize;
use serde_json::{json, Value};

use super::{Tool, ToolError};
use crate::agent::protocol::ToolSpec;

pub struct Calculator;

impl Tool for Calculator {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "calculator".into(),
            description:
                "Evaluate a finite arithmetic expression with +, -, *, /, %, and parentheses."
                    .into(),
            arguments: json!({"expression": {"type": "string"}}),
        }
    }

    fn execute(&self, arguments: &Value) -> Result<Value, ToolError> {
        let args: Args = serde_json::from_value(arguments.clone())
            .map_err(|err| ToolError::invalid(format!("invalid calculator arguments: {err}")))?;
        let expression = args.expression.trim();
        if expression.is_empty() || expression.len() > 4096 {
            return Err(ToolError::invalid(
                "expression must contain 1 to 4096 bytes",
            ));
        }
        let result = evaluate(expression)?;
        if !result.is_finite() {
            return Err(ToolError::invalid("expression result must be finite"));
        }
        Ok(json!({
            "expression": expression,
            "result": result,
        }))
    }
}

#[derive(Deserialize)]
struct Args {
    expression: String,
}

struct Parser<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(src: &'a str) -> Self {
        Self {
            src: src.as_bytes(),
            pos: 0,
        }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.src.len() && self.src[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_ws();
        self.src.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        self.skip_ws();
        let ch = self.src.get(self.pos).copied()?;
        self.pos += 1;
        Some(ch)
    }

    fn parse_expr(&mut self) -> Result<f64, ToolError> {
        let mut value = self.parse_term()?;
        loop {
            match self.peek() {
                Some(b'+') => {
                    self.bump();
                    value += self.parse_term()?;
                }
                Some(b'-') => {
                    self.bump();
                    value -= self.parse_term()?;
                }
                _ => return Ok(value),
            }
        }
    }

    fn parse_term(&mut self) -> Result<f64, ToolError> {
        let mut value = self.parse_unary()?;
        loop {
            match self.peek() {
                Some(b'*') => {
                    self.bump();
                    value *= self.parse_unary()?;
                }
                Some(b'/') => {
                    self.bump();
                    let right = self.parse_unary()?;
                    if right == 0.0 {
                        return Err(ToolError::invalid("division by zero"));
                    }
                    value /= right;
                }
                Some(b'%') => {
                    self.bump();
                    let right = self.parse_unary()?;
                    if right == 0.0 {
                        return Err(ToolError::invalid("division by zero"));
                    }
                    value %= right;
                }
                _ => return Ok(value),
            }
        }
    }

    fn parse_unary(&mut self) -> Result<f64, ToolError> {
        match self.peek() {
            Some(b'+') => {
                self.bump();
                self.parse_unary()
            }
            Some(b'-') => {
                self.bump();
                Ok(-self.parse_unary()?)
            }
            _ => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) -> Result<f64, ToolError> {
        match self.peek() {
            Some(b'(') => {
                self.bump();
                let value = self.parse_expr()?;
                if self.bump() != Some(b')') {
                    return Err(ToolError::invalid("missing closing parenthesis"));
                }
                Ok(value)
            }
            Some(ch) if ch.is_ascii_digit() || ch == b'.' => self.parse_number(),
            _ => Err(ToolError::invalid("invalid expression")),
        }
    }

    fn parse_number(&mut self) -> Result<f64, ToolError> {
        self.skip_ws();
        let start = self.pos;
        while self.pos < self.src.len()
            && (self.src[self.pos].is_ascii_digit() || self.src[self.pos] == b'.')
        {
            self.pos += 1;
        }
        let raw = std::str::from_utf8(&self.src[start..self.pos])
            .map_err(|_| ToolError::invalid("invalid number"))?;
        raw.parse::<f64>()
            .map_err(|_| ToolError::invalid("invalid number"))
    }
}

fn evaluate(expression: &str) -> Result<f64, ToolError> {
    let mut parser = Parser::new(expression);
    let value = parser.parse_expr()?;
    parser.skip_ws();
    if parser.pos != parser.src.len() {
        return Err(ToolError::invalid(
            "expression contains trailing characters",
        ));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn multiplies() {
        let value = Calculator
            .execute(&json!({"expression": "4827 * 391"}))
            .unwrap();
        assert_eq!(value["result"], 1_887_357.0);
    }
}
