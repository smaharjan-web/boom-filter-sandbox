//! Per-user Groq API key storage for natural-language filter generation.
//!
//! Keys are stored **encrypted** (AES-256-GCM, using the server secret) in a
//! dedicated `babamul_groq_keys` collection keyed by babamul user id — kept out
//! of the main user document so the `BabamulUser` shape (and its many
//! initializers) is untouched.
//!
//! The public `/filters/generate` route resolves a logged-in caller's saved key
//! here via [`get_user_groq_key`], falling back to the server default key for
//! guests or users who haven't saved one.

use super::{decrypt_password, encrypt_password, BabamulUser};
use crate::api::models::response;
use crate::conf::AppConfig;
use actix_web::{delete, get, put, web, HttpResponse};
use mongodb::{bson::doc, bson::Document, Collection, Database};
use serde::Deserialize;
use utoipa::ToSchema;

const GROQ_KEYS_COLLECTION: &str = "babamul_groq_keys";

/// Look up and decrypt a babamul user's saved Groq API key, if any. Returns
/// `None` when the user has no saved key or decryption fails.
pub(crate) async fn get_user_groq_key(
    db: &Database,
    config: &AppConfig,
    user_id: &str,
) -> Option<String> {
    let collection: Collection<Document> = db.collection(GROQ_KEYS_COLLECTION);
    let doc = collection.find_one(doc! { "_id": user_id }).await.ok()??;
    let encrypted = doc.get_str("key").ok()?;
    decrypt_password(encrypted, config.api.auth.get_hashed_secret_key()).ok()
}

#[derive(Deserialize, Clone, ToSchema)]
pub struct SetGroqKeyRequest {
    /// The user's Groq API key (e.g. "gsk_...").
    pub groq_api_key: String,
}

/// Save (or replace) the authenticated user's Groq API key.
#[utoipa::path(
    put,
    path = "/babamul/profile/groq-key",
    request_body = SetGroqKeyRequest,
    responses(
        (status = 200, description = "Groq API key saved"),
        (status = 400, description = "Invalid key"),
        (status = 401, description = "Unauthorized"),
        (status = 500, description = "Internal server error")
    ),
    tags = ["Babamul"]
)]
#[put("/profile/groq-key")]
pub async fn put_groq_key(
    db: web::Data<Database>,
    config: web::Data<AppConfig>,
    current_user: Option<web::ReqData<BabamulUser>>,
    body: web::Json<SetGroqKeyRequest>,
) -> HttpResponse {
    let current_user = match current_user {
        Some(user) => user,
        None => return HttpResponse::Unauthorized().body("Unauthorized"),
    };

    let key = body.groq_api_key.trim();
    if key.is_empty() {
        return response::bad_request("Groq API key cannot be empty");
    }
    // Groq keys are short opaque strings; guard against obviously bad input.
    if key.len() > 200 {
        return response::bad_request("Groq API key is too long");
    }

    let encrypted = match encrypt_password(key, config.api.auth.get_hashed_secret_key()) {
        Ok(enc) => enc,
        Err(e) => {
            tracing::error!("Failed to encrypt Groq API key: {}", e);
            return response::internal_error("Failed to encrypt Groq API key");
        }
    };

    let collection: Collection<Document> = db.collection(GROQ_KEYS_COLLECTION);
    match collection
        .update_one(
            doc! { "_id": &current_user.id },
            doc! { "$set": { "key": encrypted } },
        )
        .upsert(true)
        .await
    {
        Ok(_) => response::ok_no_data("Groq API key saved successfully"),
        Err(e) => {
            tracing::error!("Failed to save Groq API key: {}", e);
            response::internal_error("Failed to save Groq API key")
        }
    }
}

/// Remove the authenticated user's saved Groq API key.
#[utoipa::path(
    delete,
    path = "/babamul/profile/groq-key",
    responses(
        (status = 200, description = "Groq API key removed"),
        (status = 401, description = "Unauthorized"),
        (status = 500, description = "Internal server error")
    ),
    tags = ["Babamul"]
)]
#[delete("/profile/groq-key")]
pub async fn delete_groq_key(
    db: web::Data<Database>,
    current_user: Option<web::ReqData<BabamulUser>>,
) -> HttpResponse {
    let current_user = match current_user {
        Some(user) => user,
        None => return HttpResponse::Unauthorized().body("Unauthorized"),
    };

    let collection: Collection<Document> = db.collection(GROQ_KEYS_COLLECTION);
    match collection
        .delete_one(doc! { "_id": &current_user.id })
        .await
    {
        Ok(_) => response::ok_no_data("Groq API key removed successfully"),
        Err(e) => {
            tracing::error!("Failed to remove Groq API key: {}", e);
            response::internal_error("Failed to remove Groq API key")
        }
    }
}

/// Report whether the authenticated user has a saved Groq API key, so the
/// frontend can show "using your key" vs "add a key" without exposing the key.
#[utoipa::path(
    get,
    path = "/babamul/profile/groq-key",
    responses(
        (status = 200, description = "Whether a key is saved", body = serde_json::Value),
        (status = 401, description = "Unauthorized")
    ),
    tags = ["Babamul"]
)]
#[get("/profile/groq-key")]
pub async fn get_groq_key_status(
    db: web::Data<Database>,
    current_user: Option<web::ReqData<BabamulUser>>,
) -> HttpResponse {
    let current_user = match current_user {
        Some(user) => user,
        None => return HttpResponse::Unauthorized().body("Unauthorized"),
    };

    let collection: Collection<Document> = db.collection(GROQ_KEYS_COLLECTION);
    let has_key = matches!(
        collection.find_one(doc! { "_id": &current_user.id }).await,
        Ok(Some(_))
    );

    response::ok(
        "groq key status",
        serde_json::json!({ "has_groq_api_key": has_key }),
    )
}
