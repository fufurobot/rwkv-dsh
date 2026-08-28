use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use tokio::sync::{watch, Mutex};

use crate::agent::lightning::{self, ContinueRequest, ContinueResult, LightningRoute};
use crate::cordis::{Context, Error, Plugin};

pub type LlmFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, String>> + Send + 'a>>;
pub type LlmConnectFuture<'a> = Pin<Box<dyn Future<Output = Arc<dyn Llm>> + Send + 'a>>;

pub trait Llm: Send + Sync {
    fn complete<'a>(
        &'a self,
        request: ContinueRequest,
        cancel: watch::Receiver<bool>,
        on_delta: &'a mut (dyn FnMut(&str) + Send),
    ) -> LlmFuture<'a, ContinueResult>;

    fn complete_with_phase<'a>(
        &'a self,
        request: ContinueRequest,
        _phase: &'static str,
        cancel: watch::Receiver<bool>,
        on_delta: &'a mut (dyn FnMut(&str) + Send),
    ) -> LlmFuture<'a, ContinueResult> {
        self.complete(request, cancel, on_delta)
    }

    fn complete_batch<'a>(
        &'a self,
        template: ContinueRequest,
        prompts: Vec<String>,
        cancel: watch::Receiver<bool>,
    ) -> LlmFuture<'a, Vec<ContinueResult>>;

    fn complete_batch_with_phase<'a>(
        &'a self,
        template: ContinueRequest,
        prompts: Vec<String>,
        _phase: &'static str,
        cancel: watch::Receiver<bool>,
    ) -> LlmFuture<'a, Vec<ContinueResult>> {
        self.complete_batch(template, prompts, cancel)
    }
}

pub trait LlmConnector: Send + Sync {
    fn connect<'a>(&'a self, base_url: &'a str, password: &'a str) -> LlmConnectFuture<'a>;
}

pub struct LlmService {
    connector: Arc<dyn LlmConnector>,
}

impl LlmService {
    pub fn new(connector: Arc<dyn LlmConnector>) -> Self {
        Self { connector }
    }

    pub fn connect<'a>(&'a self, base_url: &'a str, password: &'a str) -> LlmConnectFuture<'a> {
        self.connector.connect(base_url, password)
    }
}

pub struct LightningLlm {
    client: reqwest::Client,
    route: Mutex<LightningRoute>,
    password: String,
}

impl LightningLlm {
    pub async fn connect(
        client: reqwest::Client,
        base_url: &str,
        password: impl Into<String>,
    ) -> Self {
        let route = lightning::resolve_route(&client, base_url).await;
        Self {
            client,
            route: Mutex::new(route),
            password: password.into(),
        }
    }
}

impl Llm for LightningLlm {
    fn complete<'a>(
        &'a self,
        request: ContinueRequest,
        cancel: watch::Receiver<bool>,
        on_delta: &'a mut (dyn FnMut(&str) + Send),
    ) -> LlmFuture<'a, ContinueResult> {
        Box::pin(async move {
            let mut route = self.route.lock().await;
            lightning::continue_prompt(
                &self.client,
                &mut route,
                &self.password,
                request,
                cancel,
                |delta| on_delta(delta),
            )
            .await
        })
    }

    fn complete_with_phase<'a>(
        &'a self,
        request: ContinueRequest,
        phase: &'static str,
        cancel: watch::Receiver<bool>,
        on_delta: &'a mut (dyn FnMut(&str) + Send),
    ) -> LlmFuture<'a, ContinueResult> {
        Box::pin(async move {
            let mut route = self.route.lock().await;
            lightning::continue_prompt_with_phase(
                &self.client,
                &mut route,
                &self.password,
                request,
                phase,
                cancel,
                |delta| on_delta(delta),
            )
            .await
        })
    }

    fn complete_batch<'a>(
        &'a self,
        template: ContinueRequest,
        prompts: Vec<String>,
        cancel: watch::Receiver<bool>,
    ) -> LlmFuture<'a, Vec<ContinueResult>> {
        Box::pin(async move {
            let mut route = self.route.lock().await;
            lightning::continue_batch(
                &self.client,
                &mut route,
                &self.password,
                &template,
                &prompts,
                cancel,
            )
            .await
        })
    }

    fn complete_batch_with_phase<'a>(
        &'a self,
        template: ContinueRequest,
        prompts: Vec<String>,
        phase: &'static str,
        cancel: watch::Receiver<bool>,
    ) -> LlmFuture<'a, Vec<ContinueResult>> {
        Box::pin(async move {
            let mut route = self.route.lock().await;
            lightning::continue_batch_with_phase(
                &self.client,
                &mut route,
                &self.password,
                &template,
                &prompts,
                phase,
                cancel,
            )
            .await
        })
    }
}

#[derive(Clone)]
pub struct LightningFactory {
    client: reqwest::Client,
}

impl LightningFactory {
    pub fn new(client: reqwest::Client) -> Self {
        Self { client }
    }
}

impl LlmConnector for LightningFactory {
    fn connect<'a>(&'a self, base_url: &'a str, password: &'a str) -> LlmConnectFuture<'a> {
        Box::pin(async move {
            Arc::new(
                LightningLlm::connect(self.client.clone(), base_url, password.to_string()).await,
            ) as Arc<dyn Llm>
        })
    }
}

pub struct LightningPlugin {
    service: Arc<LlmService>,
}

impl LightningPlugin {
    pub fn new(factory: LightningFactory) -> Self {
        Self {
            service: Arc::new(LlmService::new(Arc::new(factory))),
        }
    }
}

impl Plugin for LightningPlugin {
    fn name(&self) -> &'static str {
        "llm/lightning"
    }

    fn apply(&self, ctx: &mut Context) -> Result<(), Error> {
        ctx.provide("llm", Arc::clone(&self.service))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::protocol::FinishReason;

    struct FakeLlm;

    impl Llm for FakeLlm {
        fn complete<'a>(
            &'a self,
            _request: ContinueRequest,
            _cancel: watch::Receiver<bool>,
            on_delta: &'a mut (dyn FnMut(&str) + Send),
        ) -> LlmFuture<'a, ContinueResult> {
            Box::pin(async move {
                on_delta("single");
                Ok(ContinueResult {
                    text: "single".into(),
                    finish_reason: FinishReason::Stop,
                })
            })
        }

        fn complete_batch<'a>(
            &'a self,
            _template: ContinueRequest,
            prompts: Vec<String>,
            _cancel: watch::Receiver<bool>,
        ) -> LlmFuture<'a, Vec<ContinueResult>> {
            Box::pin(async move {
                Ok(prompts
                    .into_iter()
                    .map(|text| ContinueResult {
                        text,
                        finish_reason: FinishReason::Stop,
                    })
                    .collect())
            })
        }
    }

    fn request() -> ContinueRequest {
        ContinueRequest {
            prompt: "prompt".into(),
            model: "model".into(),
            max_output_tokens: 32,
            stops: vec!["\n\nUser:".into()],
            temperature: 0.001,
            top_k: 50,
            top_p: 1.0,
            alpha_presence: 0.0,
            alpha_frequency: 0.0,
            alpha_decay: 0.99,
        }
    }

    #[tokio::test]
    async fn trait_is_object_safe_for_single_and_batch_calls() {
        let provider: Arc<dyn Llm> = Arc::new(FakeLlm);
        let (_cancel_tx, cancel_rx) = watch::channel(false);
        let mut streamed = String::new();
        let mut on_delta = |delta: &str| streamed.push_str(delta);
        let single = provider
            .complete_with_phase(request(), "intent", cancel_rx.clone(), &mut on_delta)
            .await
            .unwrap();
        assert_eq!(single.text, "single");
        assert_eq!(streamed, "single");

        let batch = provider
            .complete_batch_with_phase(
                request(),
                vec!["first".into(), "second".into()],
                "planner",
                cancel_rx,
            )
            .await
            .unwrap();
        assert_eq!(
            batch.into_iter().map(|item| item.text).collect::<Vec<_>>(),
            ["first", "second"]
        );
    }

    #[test]
    fn plugin_provides_and_disposes_factory_service() {
        let mut context = Context::new();
        context
            .load(LightningPlugin::new(LightningFactory::new(
                reqwest::Client::new(),
            )))
            .unwrap();
        assert!(context.service::<LlmService>("llm").is_some());
        context.unload("llm/lightning").unwrap();
        assert!(!context.has_service("llm"));
    }
}
