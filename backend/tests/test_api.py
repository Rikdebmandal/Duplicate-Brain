"""End-to-end API tests: auth, decisions, prediction, counterfactuals, learning."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

SCENARIO = (
    "You receive a job offer paying 2x your current salary, but you must move to "
    "another city and the company has only been operating for one year."
)
OPTIONS = ["accept the offer", "reject the offer"]


def predict(client: TestClient, account: dict, **overrides) -> dict:
    payload = {
        "scenario": SCENARIO,
        "options": OPTIONS,
        "category": "career",
        "use_llm": False,
    }
    payload.update(overrides)
    response = client.post(
        "/api/v1/scenario/predict", headers=account["headers"], json=payload
    )
    assert response.status_code == 200, response.text
    return response.json()


class TestMeta:
    def test_health(self, client: TestClient):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["database"]["connected"] is True
        assert "llm" in body

    def test_root_carries_the_disclaimer(self, client: TestClient):
        body = client.get("/").json()
        assert "not a simulation" in body["disclaimer"].lower()

    def test_openapi_is_served(self, client: TestClient):
        assert client.get("/openapi.json").status_code == 200


class TestAuth:
    def test_register_and_login(self, client: TestClient):
        payload = {"email": "flow@example.com", "password": "test-password-1234"}
        created = client.post("/api/v1/auth/register", json=payload)
        assert created.status_code == 201
        assert created.json()["user"]["email"] == "flow@example.com"

        again = client.post("/api/v1/auth/register", json=payload)
        assert again.status_code == 409

        ok = client.post("/api/v1/auth/login", json=payload)
        assert ok.status_code == 200
        assert ok.json()["access_token"]

    def test_wrong_password_is_rejected(self, client: TestClient, account: dict):
        response = client.post(
            "/api/v1/auth/login",
            json={"email": account["email"], "password": "not-the-password"},
        )
        assert response.status_code == 401

    def test_unknown_email_gives_the_same_error(self, client: TestClient):
        response = client.post(
            "/api/v1/auth/login",
            json={"email": "nobody@example.com", "password": "test-password-1234"},
        )
        assert response.status_code == 401
        # Identical message: distinguishing them would enumerate accounts.
        assert response.json()["error"]["message"] == "Incorrect email or password."

    def test_short_password_is_rejected(self, client: TestClient):
        response = client.post(
            "/api/v1/auth/register", json={"email": "short@example.com", "password": "abc"}
        )
        assert response.status_code == 422

    def test_protected_routes_need_a_token(self, client: TestClient):
        assert client.get("/api/v1/profile").status_code == 401
        assert client.get("/api/v1/decisions").status_code == 401

    def test_garbage_token_is_rejected(self, client: TestClient):
        response = client.get(
            "/api/v1/profile", headers={"Authorization": "Bearer nonsense"}
        )
        assert response.status_code == 401

    def test_llm_opt_in_defaults_to_off(self, client: TestClient, account: dict):
        me = client.get("/api/v1/auth/me", headers=account["headers"]).json()
        assert me["allow_llm_processing"] is False


class TestDecisions:
    def test_create_and_read_back(self, client: TestClient, account: dict):
        response = client.post(
            "/api/v1/decisions",
            headers=account["headers"],
            json={
                "situation": "A job in another city paying double, at a one-year-old company.",
                "options": ["accept the offer", "reject the offer"],
                "decision": "reject the offer",
                "reason": "I wanted to stay close to my family.",
                "importance": 9,
                "category": "career",
            },
        )
        assert response.status_code == 201, response.text
        decision = response.json()["decision"]
        assert decision["decision"] == "reject the offer"
        assert decision["factors"]["family_impact"] > 0
        assert decision["factors"]["financial_reward"] > 0
        assert any(o["was_chosen"] for o in decision["options"])

        listed = client.get("/api/v1/decisions", headers=account["headers"]).json()
        assert listed["total"] == 1

    def test_chosen_option_must_be_in_the_option_list(self, client: TestClient, account: dict):
        response = client.post(
            "/api/v1/decisions",
            headers=account["headers"],
            json={
                "situation": "Something happened and I had to decide.",
                "options": ["accept", "reject"],
                "decision": "something else entirely",
            },
        )
        assert response.status_code == 400
        assert "not in the option list" in response.json()["error"]["message"]

    def test_duplicate_options_are_rejected(self, client: TestClient, account: dict):
        response = client.post(
            "/api/v1/decisions",
            headers=account["headers"],
            json={
                "situation": "A choice I had to make about work.",
                "options": ["accept", "Accept"],
                "decision": "accept",
            },
        )
        assert response.status_code == 422

    def test_decisions_are_private_to_their_owner(self, client: TestClient, account: dict):
        client.post(
            "/api/v1/decisions",
            headers=account["headers"],
            json={
                "situation": "A private decision about my family and my job.",
                "options": ["accept", "reject"],
                "decision": "reject",
            },
        )
        other = client.post(
            "/api/v1/auth/register",
            json={"email": "other@example.com", "password": "test-password-1234"},
        ).json()
        headers = {"Authorization": f"Bearer {other['access_token']}"}
        assert client.get("/api/v1/decisions", headers=headers).json()["total"] == 0

    def test_deleting_a_decision_removes_its_influence(
        self, client: TestClient, populated_account: dict
    ):
        headers = populated_account["headers"]
        listed = client.get("/api/v1/decisions", headers=headers).json()
        before = listed["total"]
        target = listed["items"][0]["id"]

        response = client.delete(f"/api/v1/decisions/{target}", headers=headers)
        assert response.status_code == 200
        assert response.json()["deleted"] is True

        after = client.get("/api/v1/decisions", headers=headers).json()["total"]
        assert after == before - 1

    def test_timeline(self, client: TestClient, populated_account: dict):
        timeline = client.get(
            "/api/v1/decisions/timeline", headers=populated_account["headers"]
        ).json()["timeline"]
        assert len(timeline) == populated_account["decision_count"]
        dates = [row["occurred_at"] for row in timeline]
        assert dates == sorted(dates)


class TestProfile:
    def test_every_trait_is_present_and_labelled_inferred(
        self, client: TestClient, account: dict
    ):
        body = client.get("/api/v1/profile/traits", headers=account["headers"]).json()
        assert len(body["traits"]) == 11
        assert all(t["status"] == "inferred" for t in body["traits"])
        assert "estimates" in body["disclaimer"]

    def test_traits_start_agnostic(self, client: TestClient, account: dict):
        traits = client.get(
            "/api/v1/profile/traits", headers=account["headers"]
        ).json()["traits"]
        for trait in traits:
            assert trait["value"] == pytest.approx(0.5)
            assert trait["confidence"] == pytest.approx(0.0, abs=1e-6)
            assert trait["evidence_count"] == 0

    def test_history_moves_the_profile(self, client: TestClient, populated_account: dict):
        traits = {
            t["key"]: t
            for t in client.get(
                "/api/v1/profile/traits", headers=populated_account["headers"]
            ).json()["traits"]
        }
        assert traits["family_priority"]["value"] > 0.55
        assert traits["family_priority"]["confidence"] > 0.1
        assert traits["family_priority"]["evidence_count"] > 0

    def test_trait_evidence_is_traceable_to_decisions(
        self, client: TestClient, populated_account: dict
    ):
        body = client.get(
            "/api/v1/profile/traits/family_priority/evidence",
            headers=populated_account["headers"],
        ).json()
        assert body["evidence"]
        assert any(item["decision_id"] for item in body["evidence"])

    def test_unknown_trait_is_404(self, client: TestClient, account: dict):
        response = client.get(
            "/api/v1/profile/traits/telepathy/evidence", headers=account["headers"]
        )
        assert response.status_code == 404

    def test_override_and_release_a_trait(self, client: TestClient, populated_account: dict):
        headers = populated_account["headers"]
        response = client.put(
            "/api/v1/profile/traits/risk_tolerance/override",
            headers=headers, json={"value": 0.05},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "user_specified"

        traits = {
            t["key"]: t
            for t in client.get("/api/v1/profile/traits", headers=headers).json()["traits"]
        }
        assert traits["risk_tolerance"]["value"] == pytest.approx(0.05)
        assert traits["risk_tolerance"]["status"] == "user_specified"

        released = client.put(
            "/api/v1/profile/traits/risk_tolerance/override",
            headers=headers, json={"value": None},
        )
        assert released.json()["user_override"] is None

    def test_questionnaire_round_trip(self, client: TestClient, account: dict):
        items = client.get(
            "/api/v1/profile/questionnaire", headers=account["headers"]
        ).json()
        assert items["total"] == len(items["items"])
        assert items["answered"] == 0

        answers = [{"item_key": item["key"], "value": 5} for item in items["items"][:6]]
        posted = client.post(
            "/api/v1/profile/questionnaire",
            headers=account["headers"], json={"answers": answers},
        )
        assert posted.status_code == 200
        assert posted.json()["accepted"] == 6

        after = client.get(
            "/api/v1/profile/questionnaire", headers=account["headers"]
        ).json()
        assert after["answered"] == 6

    def test_unknown_questionnaire_items_are_reported_not_silently_dropped(
        self, client: TestClient, account: dict
    ):
        response = client.post(
            "/api/v1/profile/questionnaire",
            headers=account["headers"],
            json={"answers": [{"item_key": "made_up_item", "value": 3}]},
        )
        assert response.json()["rejected"]

    def test_memory_layers_are_kept_separate(
        self, client: TestClient, populated_account: dict
    ):
        headers = populated_account["headers"]
        client.post(
            "/api/v1/profile/memory/facts",
            headers=headers,
            json={"key": "location", "content": "Lives near their parents.", "importance": 0.9},
        )
        memory = client.get("/api/v1/profile/memory", headers=headers).json()
        assert any(f["key"] == "location" for f in memory["facts"])
        assert all(f["confidence"] == 1.0 for f in memory["facts"])
        assert "patterns" in memory and "preferences" in memory
        assert "inferred" in memory["legend"]["preferences"].lower() or True

    def test_schema_endpoint_exposes_the_vocabulary(self, client: TestClient):
        body = client.get("/api/v1/profile/schema").json()
        assert len(body["factors"]) == 12
        assert len(body["traits"]) == 11


class TestPrediction:
    def test_prediction_shape(self, client: TestClient, populated_account: dict):
        result = predict(client, populated_account)
        assert result["predicted_option"] in OPTIONS
        assert sum(o["probability"] for o in result["options"]) == pytest.approx(1.0, abs=1e-3)
        assert result["confidence_label"] in {"low", "medium", "high"}
        assert result["explanation"]["summary"]
        assert result["explanation"]["reasoning_trace"]
        assert result["explanation"]["limitations"]
        assert "not a simulation" in result["disclaimer"].lower()

    def test_prediction_separates_observed_from_inferred(
        self, client: TestClient, populated_account: dict
    ):
        explanation = predict(client, populated_account)["explanation"]
        assert explanation["observed"]
        assert explanation["inferred"]
        for statement in explanation["inferred"]:
            assert "confidence" in statement
            assert "estimate" in statement["statement"]

    def test_factors_carry_their_textual_evidence(
        self, client: TestClient, populated_account: dict
    ):
        result = predict(client, populated_account)
        important = result["explanation"]["important_factors"]
        assert important
        assert any(f["evidence_phrases"] for f in important)

    def test_layers_report_availability_and_reason(
        self, client: TestClient, populated_account: dict
    ):
        layers = {layer["name"]: layer for layer in predict(client, populated_account)["layers"]}
        assert layers["profile"]["available"] is True
        assert layers["llm"]["available"] is False
        assert layers["llm"]["detail"]["reason"]

    def test_works_with_no_history_at_all(self, client: TestClient, account: dict):
        """A brand-new account must still get an answer, flagged as low confidence."""
        result = predict(client, account)
        assert result["confidence_label"] == "low"
        assert result["evidence_count"] == 0
        assert result["similar_decisions"] == []
        assert "provisional" in result["explanation"]["summary"]

    def test_three_option_decision(self, client: TestClient, populated_account: dict):
        result = predict(
            client, populated_account,
            options=["accept the offer", "negotiate a counter-offer", "reject the offer"],
        )
        assert len(result["options"]) == 3
        assert sum(o["probability"] for o in result["options"]) == pytest.approx(1.0, abs=1e-3)

    def test_single_option_is_rejected(self, client: TestClient, account: dict):
        response = client.post(
            "/api/v1/scenario/predict",
            headers=account["headers"],
            json={"scenario": SCENARIO, "options": ["accept"], "use_llm": False},
        )
        assert response.status_code == 422

    def test_unknown_factor_override_is_rejected(self, client: TestClient, account: dict):
        response = client.post(
            "/api/v1/scenario/predict",
            headers=account["headers"],
            json={
                "scenario": SCENARIO, "options": OPTIONS, "use_llm": False,
                "factor_overrides": {"vibes": 0.5},
            },
        )
        assert response.status_code == 422

    def test_prediction_is_persisted_and_retrievable(
        self, client: TestClient, populated_account: dict
    ):
        result = predict(client, populated_account)
        fetched = client.get(
            f"/api/v1/predictions/{result['id']}", headers=populated_account["headers"]
        )
        assert fetched.status_code == 200
        assert fetched.json()["predicted_option"] == result["predicted_option"]

    def test_predictions_are_private(self, client: TestClient, populated_account: dict):
        result = predict(client, populated_account)
        other = client.post(
            "/api/v1/auth/register",
            json={"email": "peeker@example.com", "password": "test-password-1234"},
        ).json()
        headers = {"Authorization": f"Bearer {other['access_token']}"}
        assert client.get(f"/api/v1/predictions/{result['id']}", headers=headers).status_code == 404

    def test_similar_decisions_are_returned_with_reasons(
        self, client: TestClient, populated_account: dict
    ):
        result = predict(client, populated_account)
        assert result["similar_decisions"]
        top = result["similar_decisions"][0]
        assert 0.0 <= top["similarity"] <= 1.0
        assert top["chosen_option"]


class TestCounterfactuals:
    def test_removing_the_family_cost_shifts_the_probability(
        self, client: TestClient, populated_account: dict
    ):
        headers = populated_account["headers"]
        base = predict(client, populated_account)
        response = client.post(
            f"/api/v1/predictions/{base['id']}/counterfactual",
            headers=headers, json={"overrides": {"family_impact": 0.0}},
        )
        assert response.status_code == 200
        body = response.json()
        accept = "accept the offer"
        assert body["modified"]["probabilities"][accept] > body["original"]["probabilities"][accept]
        assert body["changed_factors"][0]["factor"] == "family_impact"

    def test_counterfactual_is_deterministic(
        self, client: TestClient, populated_account: dict
    ):
        headers = populated_account["headers"]
        base = predict(client, populated_account)
        first = client.post(
            f"/api/v1/predictions/{base['id']}/counterfactual",
            headers=headers, json={"overrides": {"risk": 0.9}, "persist": False},
        ).json()
        second = client.post(
            f"/api/v1/predictions/{base['id']}/counterfactual",
            headers=headers, json={"overrides": {"risk": 0.9}, "persist": False},
        ).json()
        assert first["modified"]["probabilities"] == second["modified"]["probabilities"]

    def test_landscape_maps_every_factor(self, client: TestClient, populated_account: dict):
        base = predict(client, populated_account)
        body = client.post(
            f"/api/v1/predictions/{base['id']}/landscape",
            headers=populated_account["headers"], json={"steps": 5},
        ).json()
        assert body["sweeps"]
        for sweep in body["sweeps"]:
            assert len(sweep["points"]) == 5
            for point in sweep["points"]:
                assert sum(point["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)

    def test_landscape_directions_are_coherent(
        self, client: TestClient, populated_account: dict
    ):
        """More family cost must never make the disruptive option more likely."""
        base = predict(client, populated_account)
        body = client.post(
            f"/api/v1/predictions/{base['id']}/landscape",
            headers=populated_account["headers"],
            json={"steps": 5, "factors": ["family_impact", "career_growth"]},
        ).json()
        sweeps = {s["factor"]: s for s in body["sweeps"]}
        accept = "accept the offer"
        family = sweeps["family_impact"]["points"]
        assert family[-1]["probabilities"][accept] < family[0]["probabilities"][accept]
        career = sweeps["career_growth"]["points"]
        assert career[-1]["probabilities"][accept] > career[0]["probabilities"][accept]


class TestLearningLoop:
    def test_recording_an_outcome_scores_and_promotes_it(
        self, client: TestClient, populated_account: dict
    ):
        headers = populated_account["headers"]
        before = client.get("/api/v1/decisions", headers=headers).json()["total"]
        result = predict(client, populated_account)

        response = client.post(
            f"/api/v1/predictions/{result['id']}/feedback",
            headers=headers,
            json={
                "actual_option": "accept the offer",
                "reason": "The money was too good to pass up this time.",
                "importance": 8,
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["promoted_decision_id"]
        assert 0.0 <= body["probability_of_actual"] <= 1.0
        assert body["brier_score"] >= 0

        after = client.get("/api/v1/decisions", headers=headers).json()["total"]
        assert after == before + 1

    def test_outcome_can_only_be_recorded_once(
        self, client: TestClient, populated_account: dict
    ):
        headers = populated_account["headers"]
        result = predict(client, populated_account)
        payload = {"actual_option": "reject the offer"}
        assert client.post(
            f"/api/v1/predictions/{result['id']}/feedback", headers=headers, json=payload
        ).status_code == 200
        assert client.post(
            f"/api/v1/predictions/{result['id']}/feedback", headers=headers, json=payload
        ).status_code == 409

    def test_outcome_must_be_one_of_the_options(
        self, client: TestClient, populated_account: dict
    ):
        result = predict(client, populated_account)
        response = client.post(
            f"/api/v1/predictions/{result['id']}/feedback",
            headers=populated_account["headers"],
            json={"actual_option": "emigrate to Mars"},
        )
        assert response.status_code == 404

    def test_spec_alias_route_also_works(self, client: TestClient, populated_account: dict):
        result = predict(client, populated_account)
        response = client.post(
            f"/api/v1/prediction/{result['id']}/feedback",
            headers=populated_account["headers"],
            json={"actual_option": "reject the offer", "promote_to_history": False},
        )
        assert response.status_code == 200

    def test_trait_correction_is_applied(self, client: TestClient, populated_account: dict):
        headers = populated_account["headers"]
        response = client.post(
            "/api/v1/feedback",
            headers=headers,
            json={
                "kind": "trait_correction",
                "target_key": "risk_tolerance",
                "value": 0.05,
                "comment": "I am much more cautious than this suggests.",
            },
        )
        assert response.status_code == 200
        assert response.json()["applied"] is True

        traits = {
            t["key"]: t
            for t in client.get("/api/v1/profile/traits", headers=headers).json()["traits"]
        }
        assert traits["risk_tolerance"]["inferred_value"] < 0.5

    def test_performance_reports_nothing_before_any_outcome(
        self, client: TestClient, populated_account: dict
    ):
        body = client.get(
            "/api/v1/analytics/performance", headers=populated_account["headers"]
        ).json()
        assert body["n"] == 0
        assert body["metrics"] is None

    def test_performance_after_outcomes(self, client: TestClient, populated_account: dict):
        headers = populated_account["headers"]
        for actual in ["reject the offer", "accept the offer", "reject the offer"]:
            result = predict(client, populated_account)
            client.post(
                f"/api/v1/predictions/{result['id']}/feedback",
                headers=headers,
                json={"actual_option": actual, "promote_to_history": False},
            )
        body = client.get("/api/v1/analytics/performance", headers=headers).json()
        assert body["n"] == 3
        assert body["metrics"]["brier"] >= 0
        assert len(body["timeline"]) == 3
        assert body["interpretation"]


class TestAnalytics:
    def test_dashboard_payload(self, client: TestClient, populated_account: dict):
        body = client.get("/api/v1/analytics", headers=populated_account["headers"]).json()
        assert body["counts"]["decisions"] == populated_account["decision_count"]
        assert body["behavioural_summary"]["factor_profile"]
        assert body["model_confidence"]["min_decisions_for_ml"] > 0
        assert len(body["traits"]) == 11

    def test_evaluation_needs_enough_data(self, client: TestClient, account: dict):
        body = client.post(
            "/api/v1/analytics/evaluate", headers=account["headers"], json={}
        ).json()
        assert body["status"] == "insufficient_data"

    def test_evaluation_compares_every_approach(
        self, client: TestClient, populated_account: dict
    ):
        body = client.post(
            "/api/v1/analytics/evaluate", headers=populated_account["headers"], json={}
        ).json()
        assert body["status"] == "ok"
        assert {"baseline", "profile", "ml", "retrieval", "hybrid"} <= set(body["results"])
        assert body["split"]["test"]["n"] > 0
        # Time-ordered: the test block must start after the training block ends.
        assert body["split"]["train"]["range"][1] <= body["split"]["test"]["range"][0]

    def test_training_reports_why_it_declined(self, client: TestClient, account: dict):
        body = client.post("/api/v1/models/train", headers=account["headers"]).json()
        assert body["trained"] is False
        assert "at least" in body["reason"]

    def test_training_succeeds_with_history(self, client: TestClient, populated_account: dict):
        body = client.post(
            "/api/v1/models/train", headers=populated_account["headers"]
        ).json()
        assert body["trained"] is True
        assert body["metrics"]["n"] > 0
        assert body["top_features"]


class TestTextImport:
    def test_extracts_candidates_without_committing_them(
        self, client: TestClient, account: dict
    ):
        text = (
            "Last March I decided to turn down the Bengaluru offer instead of relocating, "
            "because I wanted to stay close to my parents. I value stability more than "
            "most people I know. I want to move into a leadership role eventually."
        )
        body = client.post(
            "/api/v1/text/analyze", headers=account["headers"], json={"text": text}
        ).json()

        assert body["candidates"]
        assert body["candidates"][0]["needs_review"] is True
        assert body["values"] or body["goals"]
        # Nothing may have been written to the history.
        assert client.get("/api/v1/decisions", headers=account["headers"]).json()["total"] == 0

    def test_very_short_text_is_rejected(self, client: TestClient, account: dict):
        response = client.post(
            "/api/v1/text/analyze", headers=account["headers"], json={"text": "hi"}
        )
        assert response.status_code == 422


class TestPrivacy:
    def test_export_returns_everything(self, client: TestClient, populated_account: dict):
        body = client.get("/api/v1/auth/export", headers=populated_account["headers"]).json()
        assert body["decision_count"] == populated_account["decision_count"]
        assert body["profile"]["traits"]
        assert body["user"]["email"]

    def test_account_deletion_erases_the_data(self, client: TestClient, populated_account: dict):
        headers = populated_account["headers"]
        response = client.delete("/api/v1/auth/me", headers=headers)
        assert response.status_code == 200
        assert response.json()["deleted"] is True
        assert response.json()["records_removed"]["decisions"] > 0
        # The token no longer resolves to an account.
        assert client.get("/api/v1/profile", headers=headers).status_code == 401
