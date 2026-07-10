//! Natural-language → filter-tree generation backed by Groq.
//!
//! The visual filter builder used to call the Groq API directly from the browser
//! with a key baked into the frontend bundle, which exposed the key to every
//! visitor. This endpoint moves the call server-side: the system prompt is built
//! here and the Groq key is read from the server config (`BOOM_API__GROQ_API_KEY`).
//!
//! Adapted from the boom `groq-api-backend-setup` branch. That branch's per-user
//! saved-key management lives in the babamul module and is intentionally omitted
//! here — this sandbox only needs the public filter-page generation route, keyed
//! off the server's default Groq key.

use crate::api::auth::{babamul_user_from_token, AuthProvider};
use crate::api::models::response;
use crate::api::routes::babamul::groq::get_user_groq_key;
use crate::conf::AppConfig;
use crate::utils::enums::Survey;
use actix_web::{post, web, HttpRequest, HttpResponse};
use mongodb::Database;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

const GROQ_API_URL: &str = "https://api.groq.com/openai/v1/chat/completions";

/// Maximum length of a natural-language query, in characters.
const MAX_QUERY_LEN: usize = 2000;

/// Build the system prompt that instructs Groq to emit a block/condition filter
/// tree. Ported from the frontend `llmFilterAgent.ts` so the prompt (and the
/// model) are controlled server-side. Returns `None` for surveys that do not yet
/// have a natural-language field vocabulary.
fn build_system_prompt(survey: &Survey) -> Option<String> {
    // Shared output-format contract, identical across surveys.
    let format = r#"Given a natural language description of what alerts a user wants to find, generate a filter in the following JSON format.

## Output Format

Return ONLY a JSON array (no markdown, no explanation). The array contains filter blocks:

[
  {
    "id": "root-block",
    "category": "block",
    "operator": "and",
    "children": [...]
  }
]

Each child is either a **condition**:
{
  "id": "cond-1",
  "category": "condition",
  "field": "candidate.drb",
  "operator": "$gt",
  "value": 0.5
}

Or a **nested block** (for OR logic):
{
  "id": "block-2",
  "category": "block",
  "operator": "or",
  "children": [...]
}

## Available Operators
- "$eq" (equals)
- "$ne" (not equal)
- "$gt" (greater than)
- "$gte" (greater than or equal)
- "$lt" (less than)
- "$lte" (less than or equal)
- "$in" (in list)
- "$exists" (field exists, value should be true/false)"#;

    let rules = r#"## Rules
- Always generate unique IDs for each node (use cond-1, cond-2, block-2 etc.)
- Always include at least one condition
- Use "and" as the default operator for the root block
- Do NOT add any date/time/observation-window filters (e.g. candidate.jd). The time range is set separately by the user via dedicated Start JD / End JD controls.
- Return ONLY the JSON array, nothing else"#;

    let prompt = match survey {
        Survey::Ztf => format!(
            r#"You are an astronomical alert filter builder for the Zwicky Transient Facility (ZTF).

{format}

## Available Fields

### Candidate
- candidate.drb (number)
- candidate.magpsf (number)
- candidate.sgscore1 (number)
- candidate.ndethist (number)
- candidate.isdiffpos (boolean)

### Classifications
- classifications.acai_h (number)
- classifications.acai_n (number)
- classifications.acai_v (number)
- classifications.acai_o (number)
- classifications.acai_b (number)
- classifications.btsbot (number)

### Properties
- properties.rock (boolean)
- properties.star (boolean)
- properties.near_brightstar (boolean)
- properties.stationary (boolean)

### Coordinates
- coordinates.l (number)
- coordinates.b (number)

## Common ZTF Conventions
- candidate.drb: deep-learning real-bogus score (0-1). Higher = more likely real. Use > 0.5 for real alerts.
- candidate.magpsf: PSF magnitude. Brighter objects have LOWER magnitude. "brighter than 18" means < 18.
- candidate.sgscore1: star-galaxy score. 0 = galaxy, 1 = star.
- candidate.ndethist: number of spatially coincident detections. Low values = new/recent.
- candidate.isdiffpos: positive difference image detection (true for real transients).
- classifications.acai_h: ACAI "human-interesting" transient score (0-1). Higher = more likely a real transient interesting to humans. Best indicator for supernovae.
- classifications.acai_n: ACAI "nuclear" transient score (0-1). Higher = near galaxy nucleus (AGN-like).
- classifications.acai_v: ACAI "variable star" score (0-1). Higher = likely variable star.
- classifications.acai_o: ACAI "orphan" transient score (0-1). Higher = hostless/orphan transient.
- classifications.acai_b: ACAI "bogus" score (0-1). Higher = likely bogus/artifact.
- classifications.btsbot: BTS Bot real transient score (0-1). Higher = more likely a real transient.
- properties.rock: boolean. true = likely a solar system object (asteroid).
- properties.star: boolean. true = likely a star, not a transient.
- properties.near_brightstar: boolean. true = near a bright star (higher artifact risk).
- properties.stationary: boolean. true = not moving (non-asteroid).
- coordinates.l: galactic longitude in degrees.
- coordinates.b: galactic latitude in degrees. |b| < 15 = galactic plane (higher stellar contamination).

{rules}
- For "real transients" or "supernovae", include classifications.acai_h > 0.5 AND properties.rock = false AND properties.star = false
- For "bright transients", combine classifications.acai_h > 0.5 with candidate.magpsf < [threshold]
- For filtering out bogus, use classifications.acai_b < 0.5
- For avoiding the galactic plane, use coordinates.b > 15 OR coordinates.b < -15"#
        ),
        Survey::Lsst => format!(
            r#"You are an astronomical alert filter builder for the Vera C. Rubin Observatory LSST survey.

{format}

## Available Fields

### Candidate
- candidate.magpsf (number)
- candidate.reliability (number)

### Properties
- properties.rock (boolean)
- properties.star (boolean)
- properties.stationary (boolean)

### Coordinates
- coordinates.l (number)
- coordinates.b (number)

## Common LSST Conventions
- candidate.magpsf: PSF magnitude. Brighter objects have LOWER magnitude. "brighter than 22" means < 22.
- candidate.reliability: real-bogus score (0-1). Higher = more likely real. Use > 0.5 for real alerts.
- properties.rock: boolean. true = likely a solar system object (asteroid).
- properties.star: boolean. true = likely a star, not a transient.
- properties.stationary: boolean. true = not moving (non-asteroid).
- coordinates.b: galactic latitude in degrees. |b| < 15 = galactic plane (higher stellar contamination).

{rules}"#
        ),
        // No natural-language field vocabulary defined for other surveys yet.
        Survey::Decam => return None,
    };

    Some(prompt)
}

/// Extract the raw bearer token from the Authorization header, if present.
fn bearer_token(req: &HttpRequest) -> Option<&str> {
    req.headers()
        .get("Authorization")?
        .to_str()
        .ok()?
        .strip_prefix("Bearer ")
        .map(|t| t.trim())
}

/// Strip a markdown code fence (```json ... ```) if the model wrapped its output.
fn strip_code_fence(content: &str) -> &str {
    let trimmed = content.trim();
    if let Some(rest) = trimmed.strip_prefix("```") {
        // Drop an optional language tag on the first line, then the trailing fence.
        let rest = rest.strip_prefix("json").unwrap_or(rest);
        let rest = rest.trim_start_matches('\n');
        return rest.strip_suffix("```").unwrap_or(rest).trim();
    }
    trimmed
}

#[derive(Deserialize, Clone, ToSchema)]
pub struct LlmFilterRequest {
    /// Natural-language description of the alerts the user wants to find.
    pub query: String,
    /// Survey the filter targets (ZTF or LSST).
    pub survey: Survey,
}

#[derive(Serialize, ToSchema)]
pub struct LlmFilterResponse {
    /// The generated block/condition filter tree.
    pub filters: serde_json::Value,
}

/// Generate a filter tree from a natural-language query using Groq.
#[utoipa::path(
    post,
    path = "/filters/generate",
    request_body = LlmFilterRequest,
    responses(
        (status = 200, description = "Filter generated successfully", body = LlmFilterResponse),
        (status = 400, description = "Invalid request or no Groq key configured"),
        (status = 502, description = "Groq API error")
    ),
    tags = ["Filters"]
)]
#[post("/filters/generate")]
pub async fn post_generate_filter(
    req: HttpRequest,
    config: web::Data<AppConfig>,
    body: web::Json<LlmFilterRequest>,
) -> HttpResponse {
    let query = body.query.trim();
    if query.is_empty() {
        return response::bad_request("query cannot be empty");
    }
    if query.len() > MAX_QUERY_LEN {
        return response::bad_request(&format!("query cannot exceed {} characters", MAX_QUERY_LEN));
    }

    // Resolve which Groq key to use. A logged-in babamul user's own saved key
    // takes precedence; guests (and users without a saved key) fall back to the
    // server default. `used_default` drives the rate-limit messaging below.
    //
    // The DB/auth providers are pulled from the request extensions so the route
    // stays usable in tests that wire only the config.
    let mut api_key: Option<String> = None;
    let mut used_default = true;

    if let (Some(db), Some(auth), Some(token)) = (
        req.app_data::<web::Data<Database>>(),
        req.app_data::<web::Data<AuthProvider>>(),
        bearer_token(&req),
    ) {
        if let Some(user) = babamul_user_from_token(token, auth, db).await {
            if let Some(key) = get_user_groq_key(db, &config, &user.id).await {
                api_key = Some(key);
                used_default = false;
            }
        }
    }

    if api_key.is_none() {
        if let Some(key) = &config.api.groq_api_key {
            if !key.trim().is_empty() {
                api_key = Some(key.clone());
            }
        }
    }

    let api_key = match api_key {
        Some(key) => key,
        None => {
            return HttpResponse::BadRequest().json(response::ApiResponseBody::ok(
                "Natural-language filter generation requires a Groq API key. \
                 Please add your own Groq API key to continue.",
                serde_json::json!({ "requires_api_key": true }),
            ))
        }
    };

    let system_prompt = match build_system_prompt(&body.survey) {
        Some(prompt) => prompt,
        None => {
            return response::bad_request(
                "Natural-language filter generation is not supported for this survey yet.",
            )
        }
    };

    let request_body = serde_json::json!({
        "model": config.api.groq_model,
        "messages": [
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": query },
        ],
        "temperature": 0.1,
        "max_tokens": 2048,
    });

    let client = reqwest::Client::new();
    let groq_response = match client
        .post(GROQ_API_URL)
        .bearer_auth(&api_key)
        .json(&request_body)
        .send()
        .await
    {
        Ok(resp) => resp,
        Err(e) => {
            tracing::error!("Failed to reach Groq API: {}", e);
            return HttpResponse::BadGateway().json(response::ApiResponseBody::error(
                "Failed to reach the Groq API. Please try again.",
            ));
        }
    };

    let status = groq_response.status();
    let raw = groq_response.text().await.unwrap_or_default();
    if !status.is_success() {
        tracing::warn!("Groq API returned {}: {}", status, raw);
        // Surface auth/quota problems clearly, but do not echo the raw provider
        // body (it can contain key fragments).
        if status.as_u16() == 429 {
            // Rate limit hit. If the *shared default* key ran out, prompt the
            // caller to supply their own key; if it was the user's own key,
            // just ask them to retry later.
            return if used_default {
                HttpResponse::TooManyRequests().json(response::ApiResponseBody::ok(
                    "The shared Groq API key has reached its rate limit. \
                     Please add your own Groq API key to continue.",
                    serde_json::json!({ "requires_api_key": true }),
                ))
            } else {
                HttpResponse::TooManyRequests().json(response::ApiResponseBody::error(
                    "Your Groq API key has reached its rate limit. \
                     Please wait a moment and try again.",
                ))
            };
        }
        let msg = if status.as_u16() == 401 {
            if used_default {
                "The server's Groq API key was rejected. Please add your own Groq API key."
            } else {
                "Groq rejected your saved API key. Please update it in your profile."
            }
        } else {
            "Groq API error while generating the filter."
        };
        return HttpResponse::BadGateway().json(response::ApiResponseBody::error(msg));
    }

    let parsed: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!("Failed to parse Groq response envelope: {}", e);
            return HttpResponse::BadGateway().json(response::ApiResponseBody::error(
                "Malformed response from Groq.",
            ));
        }
    };

    let content = parsed
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .trim();
    if content.is_empty() {
        return HttpResponse::BadGateway().json(response::ApiResponseBody::error(
            "Empty response from the LLM.",
        ));
    }

    let json_str = strip_code_fence(content);
    let filters: serde_json::Value = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!("LLM returned non-JSON filter content: {}", e);
            return response::bad_request(
                "The LLM did not return a valid filter. Try rephrasing your description.",
            );
        }
    };

    // Basic shape validation: a non-empty array whose first element looks like a node.
    let valid = filters
        .as_array()
        .and_then(|arr| arr.first())
        .map(|first| first.get("id").is_some() && first.get("category").is_some())
        .unwrap_or(false);
    if !valid {
        return response::bad_request(
            "The LLM returned an invalid filter structure. Try rephrasing your description.",
        );
    }

    response::ok(
        "filter generated successfully",
        serde_json::json!({ "filters": filters }),
    )
}
