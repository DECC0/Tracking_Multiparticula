"""Tests del modelo y de la perdida: formas de salida para C canales y
upsampling s (con H != W), rango de la activacion, gradientes en todos los
parametros, red totalmente convolucional, valores limite de cada termino de
la perdida y que el conjunto aprende (la perdida baja en un batch fijo)."""
import numpy as np
import torch

from comun import ConfigTarget, espera_error
from config import ConfigCorrida, ConfigModelo, ConfigPerdida, ConfigDatos
from data.target import RejillaVoxeles
from models import crear_modelo, crear_perdida, LocalizadorCNN, Perdida, n_parametros

H, W = 24, 40


def modelo(C=1, s=2, Nz=7, **kw):
    torch.manual_seed(0)
    cfg = ConfigModelo(**{"canales_base": 8, **kw})
    return LocalizadorCNN(C, Nz, s, 100.0, cfg)


def test_formas_de_salida():
    for C in (1, 2):
        for s in (1, 2, 4):
            m = modelo(C=C, s=s).eval()
            y = m(torch.randn(3, C, H, W))
            assert y.shape == (3, 7, s * H, s * W)
    # totalmente convolucional: otra forma de entrada, sin reentrenar
    assert modelo(s=4).eval()(torch.randn(1, 1, 48, 56)).shape == (1, 7, 192, 224)


def test_crear_modelo_desde_config():
    cfg = ConfigCorrida(datos=ConfigDatos(canales=2), target=ConfigTarget(upsampling=4, dz_nm=100.0),
                        modelo=ConfigModelo(canales_base=8))
    rej = RejillaVoxeles(cfg.target, H, W, (-800.0, 800.0))
    m = crear_modelo(cfg, rej).eval()
    assert m(torch.randn(2, 2, H, W)).shape == (2, *rej.forma)
    assert m.campo_receptivo_px == 65
    espera_error(ValueError, LocalizadorCNN, 1, 7, 3, 100.0, ConfigModelo())   # s no es potencia de 2


def test_rango_de_la_activacion():
    x = 50 * torch.randn(2, 1, H, W)                             # entrada exagerada: satura la salida
    for act in ("hardtanh", "sigmoide"):
        y = modelo(activacion_salida=act).eval()(x)
        assert y.min() >= 0 and y.max() <= 100.0
    y = modelo(activacion_salida="sigmoide").eval()(torch.randn(2, 1, H, W))
    assert y.mean() < 10                                         # arranca cerca de 0, no en escala / 2


def test_gradientes_en_todos_los_parametros():
    for norm in ("batch", "grupo"):
        m = modelo(norm=norm).train()
        y = m(torch.randn(2, 1, H, W))
        y.sum().backward()
        sin_grad = [n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
        assert not sin_grad, sin_grad
    assert n_parametros(modelo()) > 0


def _rejilla_y_target():
    rej = RejillaVoxeles(ConfigTarget(upsampling=2, dz_nm=200.0, planos_margen=1), H, W, (-600.0, 600.0))
    pos = torch.tensor([[[5.3, 7.9, -120.0], [30.1, 15.2, 410.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    val = torch.tensor([[True, True], [False, False]])           # la segunda imagen no tiene emisores
    return rej, rej.target(pos, val)


def test_valores_limite_de_la_perdida():
    rej, t = _rejilla_y_target()
    p = Perdida(ConfigPerdida(terminos={"mse": 1.0, "dice": 1.0}), rej.escala)
    total, v = p(t.clone(), t)
    assert float(v["mse"]) == 0 and float(v["dice"]) < 1e-6 and float(total) < 1e-6
    # todo apagado: dice ~1 en la imagen con emisores, 0 en la vacia -> media ~0.5
    total, v = p(torch.zeros_like(t), t)
    assert abs(float(v["dice"]) - 0.5) < 0.05
    assert np.isclose(float(v["mse"]), float((t**2).mean()))
    # el total es la suma pesada
    p2 = Perdida(ConfigPerdida(terminos={"mse": 2.0, "dice": 3.0}), rej.escala)
    total2, v2 = p2(torch.zeros_like(t), t)
    assert np.isclose(float(total2), 2 * float(v2["mse"]) + 3 * float(v2["dice"]))
    # la perdida se calcula en float32 aunque la salida venga en float16
    total16, _ = p(torch.zeros_like(t).half(), t.half())
    assert total16.dtype == torch.float32
    espera_error(ValueError, Perdida, ConfigPerdida(terminos={"no_existe": 1.0}), 100.0)


def test_la_perdida_baja_en_un_batch_fijo():
    rej, t = _rejilla_y_target()
    cfg = ConfigCorrida(target=ConfigTarget(upsampling=2, dz_nm=200.0, planos_margen=1),
                        modelo=ConfigModelo(canales_base=8))
    torch.manual_seed(0)
    m = crear_modelo(cfg, rej).train()
    perdida = crear_perdida(cfg, rej)
    x = torch.randn(2, 1, H, W)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    valores = []
    for _ in range(40):
        opt.zero_grad()
        total, _ = perdida(m(x), t)
        total.backward()
        opt.step()
        valores.append(float(total.detach()))
    assert valores[-1] < 0.5 * valores[0], valores[::10]
