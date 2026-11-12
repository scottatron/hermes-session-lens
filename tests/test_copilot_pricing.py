"""Approximate Copilot and user override pricing tests."""
import unittest
from unittest.mock import patch

from dashboard import _hermes_compat as compat
from dashboard import _routes as routes


class ApproximatePricingTests(unittest.TestCase):
    def test_routes_and_custom_override(self):
        self.assertTrue(compat._is_copilot_route("github-copilot"))
        self.assertTrue(compat._is_copilot_route("https://models.github.ai/inference"))
        rates, source = compat._approx_model_rates("my-local-model", "local", settings={"model_pricing_overrides": {
            "my-*-model": {"input": 2, "output": 6, "cache_read": 1, "cache_write": 3}
        }})
        self.assertEqual(source, "user_override")
        self.assertEqual(rates, (2e-6, 6e-6, 1e-6, 3e-6))

    def test_codex_routes_and_fallback_models(self):
        for route in ("openai-codex", "https://chatgpt.com/backend-api/codex", "https://chatgpt.com/backend-api/codex/"):
            self.assertTrue(compat._is_codex_route(route))
        for model in ("gpt-6.1-sol", "gpt-6.1-sol-900k", "gpt-6-luna", "gpt-6.1-terra", "gpt-5.3-codex", "gpt-5.2-codex", "codex-mini-latest"):
            rates, source = compat._approx_model_rates(model, "openai-codex")
            self.assertEqual(source, "codex_approx", model)
            self.assertIsNotNone(rates, model)
        rates, source = compat._approx_model_rates("gpt-5.3-codex", "openai-codex", settings={"codex_pricing_mode": "off"})
        self.assertIsNone(rates)
        self.assertEqual(source, "")
        rates, source = compat._approx_model_rates("gpt-5.3-codex", "openai-codex", settings={"codex_pricing_mode": "off", "model_pricing_overrides": {"gpt-5.3-codex": [1, 2]}})
        self.assertEqual(source, "user_override")
        self.assertEqual(rates, (1e-6, 2e-6, 1e-6, 1e-6))

    def test_codex_context_pricing_modes(self):
        for status, settings, expected in (("", {}, "priced"), ("estimated", {}, "priced"), ("", {"codex_pricing_mode": "subscription"}, "included"), ("", {"codex_pricing_mode": "off"}, "included")):
            routes._context_pricing_cache.clear()
            with patch.object(routes, "_plugin_settings", return_value=settings):
                self.assertEqual(routes._context_pricing("gpt-5.3-codex", "openai-codex", "", "subscription_included", status)["kind"], expected)

    def test_standalone_fallback_without_agent_package(self):
        with patch.dict("sys.modules", {"agent": None, "agent.usage_pricing": None}):
            self.assertEqual(compat._approx_model_rates("gemini-4-carbon", "copilot")[1], "copilot_approx")
            self.assertEqual(compat._approx_model_rates("gpt-5.3-codex", "openai-codex")[1], "codex_approx")

    def test_watch_and_savings_codex_helpers(self):
        from dashboard import _watch, _savings
        with patch.object(_watch, "_codex_pricing_mode", return_value="subscription"):
            self.assertTrue(_watch._is_subscription_route("openai-codex"))
        with patch.object(_watch, "_plugin_settings", return_value={}):
            self.assertIsNotNone(_watch._call_rates("gpt-5.3-codex", "openai-codex"))
        with patch.object(_savings, "_plugin_settings", return_value={}):
            self.assertIsNotNone(_savings._mix_rates("gpt-5.3-codex", "openai-codex"))

    def test_copilot_mode_off_and_override_remains_available(self):
        rates, source = compat._approx_model_rates("gpt-4.1", "copilot", settings={"copilot_pricing_mode": "off"})
        self.assertIsNone(rates)
        self.assertEqual(source, "")
        rates, source = compat._approx_model_rates("gpt-4.1", "copilot", settings={
            "copilot_pricing_mode": "off", "model_pricing_overrides": {"gpt-4.1": [1, 2]}
        })
        self.assertEqual(source, "user_override")
        self.assertEqual(rates, (1e-6, 2e-6, 1e-6, 1e-6))

    def test_context_pricing_modes_and_override(self):
        routes._context_pricing_cache.clear()
        with patch.object(routes, "_plugin_settings", return_value={"copilot_pricing_mode": "estimated"}):
            self.assertEqual(routes._context_pricing("gpt-4.1", "copilot", "", "", "") ["kind"], "priced")
        routes._context_pricing_cache.clear()
        with patch.object(routes, "_plugin_settings", return_value={"copilot_pricing_mode": "subscription"}):
            self.assertEqual(routes._context_pricing("gpt-4.1", "copilot", "", "", "") ["kind"], "included")
        routes._context_pricing_cache.clear()
        with patch.object(routes, "_plugin_settings", return_value={"copilot_pricing_mode": "off"}):
            self.assertEqual(routes._context_pricing("gpt-4.1", "copilot", "", "", "") ["kind"], "unpriced")
        routes._context_pricing_cache.clear()
        with patch.object(routes, "_plugin_settings", return_value={"model_pricing_overrides": {"custom-x": [3, 4]}}):
            priced = routes._context_pricing("custom-x", "custom", "", "", "")
        self.assertEqual(priced["kind"], "priced")
        self.assertEqual(priced["source"], "user_override")

    def test_accounted_sessions_sql_temp_views_all_modes(self):
        import sqlite3
        db = sqlite3.connect(":memory:")
        db.executescript("""
            CREATE TABLE sessions(id TEXT, __profile TEXT, input_tokens INTEGER, output_tokens INTEGER, cache_read_tokens INTEGER, cache_write_tokens INTEGER, reasoning_tokens INTEGER, api_call_count INTEGER, actual_cost_usd REAL, estimated_cost_usd REAL, cost_status TEXT, cost_source TEXT, list_price_usd REAL, billing_provider TEXT);
            CREATE TABLE session_model_usage(session_id TEXT, __profile TEXT, billing_provider TEXT, input_tokens INTEGER, output_tokens INTEGER, cache_read_tokens INTEGER, cache_write_tokens INTEGER, reasoning_tokens INTEGER, api_call_count INTEGER, actual_cost_usd REAL, estimated_cost_usd REAL, cost_status TEXT, cost_source TEXT, list_price_usd REAL);
            INSERT INTO sessions VALUES ('s','p',0,0,0,0,0,0,0,0,'unknown',NULL,0,NULL);
            INSERT INTO session_model_usage VALUES ('s','p','copilot',10,0,0,0,0,1,0,0,'included','test',2);
        """)
        db.execute("CREATE TEMP VIEW accounted AS " + compat._accounted_sessions_sql(db))
        for mode, expected in (("estimated", "included"), ("subscription", "included"), ("off", "included")):
            row = db.execute("SELECT cost_status, billing_provider, list_price_usd FROM accounted").fetchone()
            self.assertEqual(row[0], expected)
            self.assertEqual(row[1], None)
            self.assertEqual(row[2], 2)
        db.close()

    def test_install_accounting_views_copilot_modes(self):
        import sqlite3

        schema = """
            CREATE TABLE sessions(id TEXT, model TEXT, billing_provider TEXT, billing_base_url TEXT,
                input_tokens INTEGER, output_tokens INTEGER, cache_read_tokens INTEGER, cache_write_tokens INTEGER,
                reasoning_tokens INTEGER, api_call_count INTEGER, actual_cost_usd REAL,
                estimated_cost_usd REAL, cost_status TEXT, cost_source TEXT, list_price_usd REAL);
            CREATE TABLE session_model_usage(session_id TEXT, model TEXT, billing_provider TEXT,
                billing_base_url TEXT, input_tokens INTEGER, output_tokens INTEGER, cache_read_tokens INTEGER,
                cache_write_tokens INTEGER, reasoning_tokens INTEGER, api_call_count INTEGER,
                actual_cost_usd REAL, estimated_cost_usd REAL, cost_status TEXT, cost_source TEXT, list_price_usd REAL);
            INSERT INTO sessions VALUES ('s','gemini-4-carbon','copilot',NULL,1000000,0,0,0,0,1,0,0,'unknown',NULL,0);
            INSERT INTO session_model_usage VALUES ('s','gemini-4-carbon','copilot',NULL,1000000,0,0,0,0,1,0,0,'unknown',NULL,0);
        """
        expected = (
            ({}, ("estimated", "copilot_approx", 2.0, 0.0)),
            ({"copilot_pricing_mode": "subscription"}, ("included", "copilot_approx", 0.0, 2.0)),
            ({"copilot_pricing_mode": "off"}, ("unknown", None, 0.0, 0.0)),
        )
        for settings, values in expected:
            with self.subTest(settings=settings), patch.object(compat, "_plugin_settings", return_value=settings):
                db = sqlite3.connect(":memory:")
                db.executescript(schema)
                compat._install_accounting_views(db)
                row = db.execute("SELECT cost_status, cost_source, estimated_cost_usd, list_price_usd FROM (" + compat._accounted_sessions_sql(db) + ")").fetchone()
                self.assertEqual(row, values)
                if settings.get("copilot_pricing_mode") == "subscription":
                    usage_row = db.execute("SELECT cost_status, estimated_cost_usd, list_price_usd FROM session_model_usage").fetchone()
                    self.assertEqual(usage_row, ("included", 0, 2.0))
                db.close()

    def test_install_accounting_views_codex_modes(self):
        import sqlite3
        schema = """
            CREATE TABLE sessions(id TEXT, model TEXT, billing_provider TEXT, billing_base_url TEXT, billing_mode TEXT,
                input_tokens INTEGER, output_tokens INTEGER, cache_read_tokens INTEGER, cache_write_tokens INTEGER,
                reasoning_tokens INTEGER, api_call_count INTEGER, actual_cost_usd REAL,
                estimated_cost_usd REAL, cost_status TEXT, cost_source TEXT, list_price_usd REAL);
            CREATE TABLE session_model_usage(session_id TEXT, model TEXT, billing_provider TEXT,
                billing_base_url TEXT, billing_mode TEXT, input_tokens INTEGER, output_tokens INTEGER, cache_read_tokens INTEGER,
                cache_write_tokens INTEGER, reasoning_tokens INTEGER, api_call_count INTEGER,
                actual_cost_usd REAL, estimated_cost_usd REAL, cost_status TEXT, cost_source TEXT, list_price_usd REAL);
            INSERT INTO sessions VALUES ('s','gpt-5.3-codex','openai-codex','https://chatgpt.com/backend-api/codex','subscription_included',1000000,0,0,0,0,1,0,0,'included','none',0);
            INSERT INTO session_model_usage VALUES ('s','gpt-5.3-codex','openai-codex','https://chatgpt.com/backend-api/codex','subscription_included',1000000,0,0,0,0,1,0,0,'included','none',0);
        """
        cases = (({}, ("estimated", "codex_approx", 1.75, 0.0)), ({"codex_pricing_mode": "subscription"}, ("included", "codex_approx", 0.0, 1.75)), ({"codex_pricing_mode": "off"}, ("included", "none", 0.0, 0.0)))
        for settings, expected in cases:
            with self.subTest(settings=settings), patch.object(compat, "_plugin_settings", return_value=settings):
                db = sqlite3.connect(":memory:")
                db.executescript(schema)
                compat._install_accounting_views(db)
                row = db.execute("SELECT cost_status, cost_source, estimated_cost_usd, list_price_usd FROM session_model_usage").fetchone()
                self.assertEqual(row, expected)
                db.close()

    def test_copilot_known_models_use_local_hermes_pricing(self):
        for model in ("gemini-4-carbon", "claude-opus-5.5", "claude-haiku-5.5", "gpt-4.1-2025-04-14", "gpt-6.1-sol-900k"):
            rates, source = compat._approx_model_rates(model, "copilot")
            self.assertEqual(source, "copilot_approx", model)
            self.assertIsNotNone(rates, model)


if __name__ == "__main__":
    unittest.main()
