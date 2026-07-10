/// Tests for the natural-language filter generation route (`POST /filters/generate`).
///
/// These drive the real handler in-process via actix's test service, so they
/// need no database or Kafka. The validation and config-guard paths run offline;
/// the live Groq call is exercised only by the `#[ignore]`d test, which requires
/// a real key in the `GROQ_API_KEY` environment variable.
#[cfg(test)]
mod tests {
    use actix_web::http::StatusCode;
    use actix_web::{test, web, App};
    use boom::api::auth::PUBLIC_ROUTES;
    use boom::api::routes;
    use boom::api::test_utils::read_json_response;
    use boom::conf::AppConfig;
    use serde_json::json;

    /// Load the test config with the given Groq key wired in. Each test builds
    /// its own app inline from this (the actix service type is awkward to name).
    fn config_with_key(groq_api_key: Option<String>) -> AppConfig {
        let mut config = AppConfig::from_test_config().expect("Failed to load test config");
        config.api.groq_api_key = groq_api_key;
        config
    }

    #[actix_web::test]
    async fn empty_query_is_rejected() {
        let app = test::init_service(
            App::new()
                .app_data(web::Data::new(config_with_key(Some(
                    "gsk_dummy".to_string(),
                ))))
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
    async fn missing_groq_key_is_rejected() {
        let app = test::init_service(
            App::new()
                .app_data(web::Data::new(config_with_key(None)))
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
        // With no key configured and no logged-in user, the guest is prompted to
        // supply their own key (machine-readable flag for the frontend).
        assert!(
            body["message"].as_str().unwrap().contains("Groq API key"),
            "unexpected body: {body}"
        );
        assert_eq!(body["data"]["requires_api_key"], serde_json::json!(true));
    }

    #[actix_web::test]
    async fn blank_key_is_treated_as_unconfigured() {
        let app = test::init_service(
            App::new()
                .app_data(web::Data::new(config_with_key(Some("   ".to_string()))))
                .service(routes::llm::post_generate_filter),
        )
        .await;
        let req = test::TestRequest::post()
            .uri("/filters/generate")
            .set_json(json!({ "query": "bright real transients", "survey": "LSST" }))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[actix_web::test]
    async fn route_is_publicly_accessible() {
        // The visual filter builder calls this unauthenticated, like the
        // filter-test endpoints, so it must be in the public allowlist.
        assert!(PUBLIC_ROUTES.contains(&"/filters/generate"));
    }

    /// Live end-to-end call against the real Groq API. Ignored by default; run
    /// with a real key via:
    ///   GROQ_API_KEY=gsk_... cargo test --test test_api -- --ignored live_generation
    #[actix_web::test]
    #[ignore]
    async fn live_generation_returns_filter_tree() {
        let key = std::env::var("GROQ_API_KEY")
            .expect("set GROQ_API_KEY to run the live generation test");
        let app = test::init_service(
            App::new()
                .app_data(web::Data::new(config_with_key(Some(key))))
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
        assert_eq!(resp.status(), StatusCode::OK, "expected 200 from Groq");
        let body = read_json_response(resp).await;
        let filters = &body["data"]["filters"];
        assert!(filters.is_array(), "filters should be an array: {body}");
        let first = &filters[0];
        assert!(first.get("id").is_some() && first.get("category").is_some());
    }
}
