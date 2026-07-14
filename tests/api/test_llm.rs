/// Tests for the natural-language filter generation route (`POST /filters/generate`).
///
/// The Groq API key is read from
/// MongoDB (the `babamul_groq_keys` collection: a user's saved key, or the
/// shared `__default__` document). These tests therefore talk to the Dockerized
/// test database via `get_test_db_api()`, matching the babamul test suite.
///
/// The query-validation path runs offline; the key-resolution paths hit Mongo;
/// the live Groq call is exercised only by the `#[ignore]`d test, which requires
/// a real key in the `GROQ_API_KEY` environment variable.
#[cfg(test)]
mod tests {
    use actix_web::http::StatusCode;
    use actix_web::{test, web, App};
    use boom::api::auth::PUBLIC_ROUTES;
    use boom::api::db::get_test_db_api;
    use boom::api::routes;
    use boom::api::routes::babamul::groq::{delete_default_groq_key, DEFAULT_GROQ_KEY_ID};
    use boom::api::test_utils::read_json_response;
    use boom::conf::AppConfig;
    use mongodb::{bson::doc, bson::Document, Collection, Database};
    use serde_json::json;

    /// Store a plaintext shared default key in Mongo (mirrors a hand-inserted
    /// `__default__` document).
    async fn set_plaintext_default(db: &Database, key: &str) {
        let collection: Collection<Document> = db.collection("babamul_groq_keys");
        collection
            .update_one(
                doc! { "_id": DEFAULT_GROQ_KEY_ID },
                doc! { "$set": { "plaintext_key": key } },
            )
            .upsert(true)
            .await
            .expect("failed to seed default groq key");
    }

    #[actix_web::test]
    async fn empty_query_is_rejected() {
        // Query validation happens before any key lookup, so no DB is needed.
        let config = AppConfig::from_test_config().expect("Failed to load test config");
        let app = test::init_service(
            App::new()
                .app_data(web::Data::new(config))
                .service(routes::llm::post_generate_filter),
        )
        .await;
        let req = test::TestRequest::post()
            .uri("/filters/generate")
            .set_json(json!({ "query": "   ", "survey": "ZTF" }))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = read_json_response(resp).await;
        assert!(
            body["message"]
                .as_str()
                .unwrap()
                .contains("query cannot be empty"),
            "unexpected body: {body}"
        );
    }

    #[actix_web::test]
    async fn missing_default_key_is_rejected() {
        // With no user key and no `__default__` document in Mongo, a guest is
        // prompted to supply their own key (machine-readable flag for the UI).
        let config = AppConfig::from_test_config().expect("Failed to load test config");
        let db = get_test_db_api().await;
        delete_default_groq_key(&db)
            .await
            .expect("failed to clear default key");

        let app = test::init_service(
            App::new()
                .app_data(web::Data::new(config))
                .app_data(web::Data::new(db))
                .service(routes::llm::post_generate_filter),
        )
        .await;
        let req = test::TestRequest::post()
            .uri("/filters/generate")
            .set_json(json!({ "query": "bright real transients", "survey": "ZTF" }))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = read_json_response(resp).await;
        assert!(
            body["message"].as_str().unwrap().contains("Groq API key"),
            "unexpected body: {body}"
        );
        assert_eq!(body["data"]["requires_api_key"], serde_json::json!(true));
    }

    #[actix_web::test]
    async fn route_is_publicly_accessible() {
        // The visual filter builder calls this unauthenticated, like the
        // filter-test endpoints, so it must be in the public allowlist.
        assert!(PUBLIC_ROUTES.contains(&"/filters/generate"));
    }

    /// Live end-to-end call against the real Groq API, with the key resolved
    /// from the Mongo `__default__` document (the production path). Ignored by
    /// default; run with a real key via:
    ///   GROQ_API_KEY=gsk_... cargo test --test test_api -- --ignored live_generation
    #[actix_web::test]
    #[ignore]
    async fn live_generation_uses_mongo_default() {
        let key = std::env::var("GROQ_API_KEY")
            .expect("set GROQ_API_KEY to run the live generation test");
        let config = AppConfig::from_test_config().expect("Failed to load test config");
        let db = get_test_db_api().await;
        set_plaintext_default(&db, &key).await;

        let app = test::init_service(
            App::new()
                .app_data(web::Data::new(config))
                .app_data(web::Data::new(db.clone()))
                .service(routes::llm::post_generate_filter),
        )
        .await;
        let req = test::TestRequest::post()
            .uri("/filters/generate")
            .set_json(json!({
                "query": "real bright supernova candidates brighter than magnitude 18",
                "survey": "ZTF"
            }))
            .to_request();
        let resp = test::call_service(&app, req).await;
        let status = resp.status();
        let body = read_json_response(resp).await;

        // Clean up the seeded key regardless of the assertion outcome.
        let _ = delete_default_groq_key(&db).await;

        assert_eq!(status, StatusCode::OK, "expected 200 from Groq: {body}");
        let filters = &body["data"]["filters"];
        assert!(filters.is_array(), "filters should be an array: {body}");
        let first = &filters[0];
        assert!(first.get("id").is_some() && first.get("category").is_some());
    }
}
