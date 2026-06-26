import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

UNIVERSAL_TECH_TAXONOMY = [
    # --- 1. CORE AI, ML & DATA SCIENCE ---
    "machine_learning_model_training", "deep_learning_neural_networks",
    "large_language_models_and_generative_ai", "llm_fine_tuning_and_alignment",
    "rag_retrieval_augmented_generation", "vector_search_and_embedding_infrastructure",
    "semantic_search_and_dense_retrieval", "model_quantization_and_inference_optimization",
    "distributed_gpu_training", "nlp_text_processing_and_transformers",
    "computer_vision_and_image_processing", "speech_recognition_and_audio_processing",
    "time_series_forecasting_and_anomaly_detection", "reinforcement_learning_systems",
    "mlops_and_model_deployment", "feature_store_engineering",
    "search_relevance_and_information_retrieval", "learning_to_rank_models",
    "recommendation_systems", "hybrid_search_retrieval_systems",
    "ml_observability_and_monitoring",

    # --- 2. DATA ENGINEERING & PIPELINES ---
    "real_time_stream_processing", "batch_data_processing",
    "data_pipeline_orchestration", "data_warehouse_architecture",
    "data_lakehouse_architecture", "relational_data_modeling_and_schema_design",
    "graph_database_architecture", "olap_analytical_query_optimization",
    "data_governance_and_data_quality", "business_intelligence_and_analytics",

    # --- 3. BACKEND & DISTRIBUTED SYSTEMS ---
    "microservices_architecture", "monolithic_architecture_refactoring",
    "api_design_and_development", "event_driven_architecture_and_message_brokers",
    "serverless_architecture", "distributed_consensus_and_state_machine_replication",
    "backend_performance_profiling_and_tuning", "low_level_memory_management",
    "concurrent_and_parallel_programming", "websocket_and_real_time_communications",

    # --- 4. DATABASE & STORAGE OPS ---
    "relational_database_administration", "nosql_database_design",
    "distributed_caching_layers", "database_sharding_and_replication_strategies",
    "query_optimization_and_index_tuning", "acid_transactions_and_isolation_levels",
    "distributed_storage_systems",

    # --- 5. FRONTEND & CLIENT-SIDE ---
    "component_based_ui_frameworks", "frontend_state_management",
    "server_side_rendering_and_hydration", "single_page_application_spa_architecture",
    "css_architecture_and_styling", "web_accessibility_compliance",
    "frontend_build_tooling", "webassembly_performance_optimization",
    "ui_ux_animation_and_microinteractions", "browser_rendering_optimization",

    # --- 6. MOBILE APP DEVELOPMENT ---
    "native_ios_development", "native_android_development",
    "cross_platform_mobile_development", "mobile_app_state_and_offline_storage",
    "mobile_ui_performance_profiling", "app_store_deployment_and_ci_cd",

    # --- 7. CLOUD PLATFORMS & INFRASTRUCTURE ---
    "public_cloud_infrastructure", "multi_cloud_and_hybrid_architecture",
    "infrastructure_as_code", "containerization",
    "container_orchestration", "cloud_networking_and_load_balancing",
    "server_provisioning_and_configuration",

    # --- 8. DEVOPS, SRE & OBSERVABILITY ---
    "ci_cd_pipeline_automation", "system_observability_and_metrics",
    "distributed_tracing", "log_aggregation_and_analysis",
    "incident_response_and_on_call_management", "site_reliability_engineering",
    "linux_system_administration_and_kernel_tuning", "chaos_engineering_and_fault_tolerance",
    "experimentation_and_ab_testing",

    # --- 9. SECURITY & COMPLIANCE ---
    "web_application_security", "identity_and_access_management",
    "cryptography_and_encryption_in_transit", "network_security_and_defense",
    "penetration_testing_and_vulnerability_scanning", "security_compliance_and_governance",
    "secure_software_development_lifecycle",

    # --- 10. LOW-LEVEL, FIRMWARE & GAME DEV ---
    "embedded_systems_and_firmware", "real_time_operating_systems_rtos",
    "game_engine_architecture", "graphics_programming",
    "physics_simulation_and_rendering", "iot_device_communication_protocols",

    # --- 11. SOFTWARE ENGINEERING PRACTICES & LEADERSHIP ---
    "test_driven_development_and_testing", "end_to_end_integration_qa_automation",
    "agile_engineering_leadership", "cross_functional_team_mentorship_and_management",
    "system_architecture_and_design", "technical_debt_management_and_refactoring",
    "open_source_contributions_and_maintainership", "product_engineering_and_feature_ownership",

    # --- 12. SPECIALIZED DOMAINS ---
    "payment_systems_and_fintech", "blockchain_and_web3_infrastructure"
]
