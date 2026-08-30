from models.factory import build_model


def test_build_unet1d_model():
    model = build_model("UNet1D", input_shape=(64,), use_condition=False)
    assert model.__class__.__name__ == "UNet1D"


def test_build_simple_mlp_model():
    model = build_model("SimpleMLP", input_shape=(3,), use_condition=False)
    assert model.__class__.__name__ == "SimpleMLP"


def test_build_fno1d_model_case_insensitive():
    model = build_model("fno1d", input_shape=(64,), use_condition=False)
    assert model.__class__.__name__ == "FNO1d"


def test_build_dit_model():
    model = build_model("DiT", input_shape=(16, 16, 1), use_condition=False)
    assert model.__class__.__name__ == "DiT"
