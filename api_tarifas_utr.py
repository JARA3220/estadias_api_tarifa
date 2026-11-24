import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

DIR_BASE = Path(__file__).resolve().parent
RUTA_VEHICULOS = DIR_BASE / "vehiculos_limpios_utr_filtrado.xlsx"
RUTA_MODELO = DIR_BASE / "modelo_mlp_utr.keras"

FRANJAS_TRAFICO = (
    (5 * 60 + 0, 7 * 60 + 29, 0.00, "Madrugada / tráfico ligero (0%)"),
    (7 * 60 + 30, 10 * 60 + 30, 0.15, "Hora pico entrada matutina (+15%)"),
    (10 * 60 + 31, 13 * 60 + 29, 0.00, "Horario valle / tráfico normal (0%)"),
    (13 * 60 + 30, 18 * 60 + 30, 0.10, "Salida y turno vespertino (+10%)"),
)

MEDIA_PCA = np.array([11.4131, 15.0905, 20.9650, 2.3100, 0.0982], dtype=np.float64)
DESV_PCA = np.array([3.2457, 2.8103, 4.1981, 0.9241, 0.0539], dtype=np.float64)
W_PCA = np.array(
    [
        [-0.0307,  0.6442, -0.7246, -0.2427],
        [-0.6901, -0.1436, -0.0535, -0.1017],
        [-0.7000, -0.0662, -0.0315, -0.0261],
        [ 0.0875, -0.5811, -0.6717,  0.4508],
        [-0.1587,  0.4714,  0.1415,  0.8526],
    ],
    dtype=np.float64,
)

df_vehiculos = pd.read_excel(RUTA_VEHICULOS)
df_vehiculos.columns = [str(c).strip() for c in df_vehiculos.columns]
df_vehiculos = df_vehiculos.dropna(subset=["marca", "modelo", "rendimiento_km_l"]).copy()
df_vehiculos["marca_norm"] = df_vehiculos["marca"].astype(str).str.strip().str.lower()
df_vehiculos["modelo_norm"] = df_vehiculos["modelo"].astype(str).str.strip().str.lower()
df_vehiculos["rendimiento_km_l"] = pd.to_numeric(df_vehiculos["rendimiento_km_l"], errors="coerce")
df_vehiculos = df_vehiculos.dropna(subset=["rendimiento_km_l"]).reset_index(drop=True)
if "anio" in df_vehiculos.columns:
    df_vehiculos["anio"] = pd.to_numeric(df_vehiculos["anio"], errors="coerce")

modelo_mlp = tf.keras.models.load_model(RUTA_MODELO)

app = FastAPI(title="API Tarifas Viajes Compartidos UTR", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SolicitudCotizacion(BaseModel):
    marca: str = Field(..., example="Nissan")
    modelo: str = Field(..., example="Versa")
    anio: Optional[int] = Field(None, example=2022)
    hora: str = Field(..., example="08:15")
    km_recorrido: float = Field(..., gt=0, example=14.0)
    tiempo_estimado: float = Field(..., gt=0, example=22.0)
    asientos: int = Field(..., ge=1, le=4, example=3)


class RespuestaCotizacion(BaseModel):
    marca: str
    modelo: str
    anio: Optional[int]
    rendimiento_km_l: float
    recargo_trafico: float
    franja_trafico: str
    tarifa_continua: float
    tarifa_mxn: int


def obtener_recargo(hora_str: str) -> tuple[float, str]:
    try:
        t = datetime.strptime(hora_str.strip(), "%H:%M")
    except ValueError:
        raise HTTPException(status_code=400, detail="Formato de hora inválido. Usa HH:MM (ej. 08:15).")

    mins = t.hour * 60 + t.minute
    for ini, fin, rec, desc in FRANJAS_TRAFICO:
        if ini <= mins <= fin:
            return rec, desc
    raise HTTPException(
        status_code=400,
        detail=f"La hora '{hora_str}' está fuera del horario de operación UTR (05:00 a 18:30).",
    )


def buscar_rendimiento(marca: str, modelo: str, anio: Optional[int]) -> tuple[str, str, Optional[int], float]:
    m_norm = marca.strip().lower()
    mod_norm = modelo.strip().lower()

    sub = df_vehiculos[(df_vehiculos["marca_norm"] == m_norm) & (df_vehiculos["modelo_norm"] == mod_norm)]
    if sub.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No se encontró el vehículo '{marca} {modelo}' en el catálogo UTR.",
        )

    anio_resuelto = anio
    if anio is not None and "anio" in sub.columns:
        sub_anio = sub[sub["anio"] == anio]
        if not sub_anio.empty:
            sub = sub_anio
        else:
            anio_resuelto = None

    marca_real = str(sub["marca"].iloc[0]).strip()
    modelo_real = str(sub["modelo"].iloc[0]).strip()
    rendimiento = round(float(sub["rendimiento_km_l"].mean()), 2)
    return marca_real, modelo_real, anio_resuelto, rendimiento


@app.get("/")
def health_check():
    return {"status": "ok", "servicio": "API Tarifas UTR", "vehiculos_cargados": len(df_vehiculos)}


@app.get("/vehiculos/marcas")
def listar_marcas():
    return {"marcas": sorted(df_vehiculos["marca"].unique(), key=str.lower)}


@app.get("/vehiculos/{marca}/modelos")
def listar_modelos(marca: str):
    sub = df_vehiculos[df_vehiculos["marca_norm"] == marca.strip().lower()]
    if sub.empty:
        raise HTTPException(status_code=404, detail=f"Marca '{marca}' no encontrada.")
    return {"marca": str(sub["marca"].iloc[0]), "modelos": sorted(sub["modelo"].unique(), key=str.lower)}


@app.post("/cotizar", response_model=RespuestaCotizacion)
def cotizar_viaje(datos: SolicitudCotizacion):
    marca_real, modelo_real, anio_resuelto, rendimiento = buscar_rendimiento(
        datos.marca, datos.modelo, datos.anio
    )
    recargo, desc_trafico = obtener_recargo(datos.hora)

    x = np.array(
        [
            rendimiento,
            float(datos.km_recorrido),
            float(datos.tiempo_estimado),
            float(datos.asientos),
            recargo,
        ],
        dtype=np.float64,
    )
    pcs = ((x - MEDIA_PCA) / DESV_PCA) @ W_PCA
    pred = float(modelo_mlp.predict(pcs.reshape(1, -1).astype(np.float32), verbose=0)[0, 0])
    tarifa_mxn = max(1, int(np.rint(pred)))

    return RespuestaCotizacion(
        marca=marca_real,
        modelo=modelo_real,
        anio=anio_resuelto,
        rendimiento_km_l=rendimiento,
        recargo_trafico=recargo,
        franja_trafico=desc_trafico,
        tarifa_continua=round(pred, 2),
        tarifa_mxn=tarifa_mxn,
    )
