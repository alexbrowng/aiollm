import pytest

from aiollm.models.model import Model


def test_model_creation_minimal():
    model = Model(id="gpt-4o-mini", name="GPT 4o mini", provider="OpenAI")
    assert model.id == "gpt-4o-mini"
    assert model.name == "GPT 4o mini"
    assert model.provider == "OpenAI"
    assert model.input_price is None
    assert model.output_price is None


def test_model_creation_with_prices():
    model = Model(id="gpt-4o", name="GPT 4o", provider="OpenAI", input_price=0.01, output_price=0.02)
    assert model.input_price == pytest.approx(0.01)
    assert model.output_price == pytest.approx(0.02)
