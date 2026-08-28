use super::protocol::{Message, Role};

pub fn render(messages: &[Message], thinking_fast: bool) -> Result<String, String> {
    if messages.is_empty() {
        return Err("render agent prompt: no messages".into());
    }
    let mut prompt = String::new();
    for message in messages {
        if message.content.trim().is_empty() {
            return Err(format!(
                "render agent prompt: {} message is empty",
                message.role.label().to_lowercase()
            ));
        }
        let content = if message.role == Role::Assistant {
            message.content.clone()
        } else {
            clean_message_text(&message.content)
        };
        prompt.push_str(&format!("{}: {}\n\n", message.role.label(), content));
    }
    prompt.push_str("Assistant:");
    if thinking_fast {
        prompt.push_str(" <think></think");
    }
    Ok(prompt)
}

pub fn append_assistant_prefix(prompt: &str, prefix: &str, thinking_fast: bool) -> (String, bool) {
    if thinking_fast || prefix.is_empty() {
        return (prompt.to_string(), false);
    }
    (format!("{prompt} {prefix}"), true)
}

pub fn reconstruct_output(output: &str, thinking_fast: bool) -> String {
    if output.trim_start().starts_with("<think>") {
        return output.to_string();
    }
    if thinking_fast {
        let trimmed = output.trim_start();
        if trimmed.starts_with('>') {
            let leading_len = output.len() - trimmed.len();
            format!("{}<think></think{}", &output[..leading_len], trimmed)
        } else {
            format!("<think></think>{output}")
        }
    } else {
        output.to_string()
    }
}

fn clean_message_text(text: &str) -> String {
    let mut value = text.replace("\r\n", "\n").replace('\r', "\n");
    while value.contains("\n\n") {
        value = value.replace("\n\n", "\n");
    }
    value.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::protocol::{Message, Role};

    #[test]
    fn renders_continuation_prefix() {
        let prompt = render(
            &[Message {
                role: Role::User,
                content: "hello".into(),
            }],
            false,
        )
        .unwrap();
        assert_eq!(prompt, "User: hello\n\nAssistant:");
    }

    #[test]
    fn fast_rendering_withholds_the_think_close_boundary() {
        let prompt = render(
            &[Message {
                role: Role::User,
                content: "hello".into(),
            }],
            true,
        )
        .unwrap();
        assert_eq!(prompt, "User: hello\n\nAssistant: <think></think");
        assert_eq!(reconstruct_output("answer", true), "<think></think>answer");
        assert_eq!(reconstruct_output(">answer", true), "<think></think>answer");
        assert_eq!(
            reconstruct_output("\n>answer", true),
            "\n<think></think>answer"
        );
    }
}
