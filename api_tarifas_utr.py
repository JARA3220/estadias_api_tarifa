#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
api_tarifas_utr.py
==================
API REST en FastAPI para estimar tarifas de viajes compartidos UTR.
Recibe desde la app móvil (Flutter):
- marca (str)
- modelo (str)
- anio (int | None)
- hora (str, formato "HH:MM", de 05:00 a 18:30)
- km_recorrido (float)
- tiempo_estimado (float, en minutos)
- asientos (int, 1 a 4)

Si no encuentra la marca o el modelo en el catálogo, utiliza automáticamente
el rendimiento promedio de respaldo: 10.16404059 km/L.

Devuelve únicamente:
  {"tarifa_mxn": 31}

Ejecución en Render:
    uvicorn api_tarifas_utr:app --host 0.0.0.0 --port $PORT
"""

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

RENDIMIENTO_PROMEDIO_DEFAULT = 10.16404059

FRANJAS_TRAFICO = (
    (5 * 60 + 0, 7 * 60 + 29, 0.00),    # 05:00 - 07:29 (madrugada / tráfico ligero)
    (7 * 60 + 30, 10 * 60 + 30, 0.15),  # 07:30 - 10:30 (hora pico entrada matutina)
    (10 * 60 + 31, 13 * 60 + 29, 0.00), # 10:31 - 13:29 (horario valle / tráfico normal)
    (13 * 60 + 30, 18 * 60 + 30, 0.10), # 13:30 - 18:30 (salida y turno vespertino)
)

# Proyección PCA calibrada para el modelo MLP entrenado con los 1,000 registros
# Orden de entrada x: [rendimiento_km_l, distancia_km, tiempo_estimado_min, num_pasajeros, recargo_trafico]
INTERCEPTO_PCA = np.array([-7.37980095, 0.56953824, -4.41355641, 1.15023713], dtype=np.float64)
MATRIZ_PCA = np.array(
    [
        [ 0.00946661, -0.19848883,  0.22323690,  0.07476280],  # rendimiento_km_l
        [ 0.24555475,  0.05109545,  0.01904802,  0.03617496],  # distancia_km
        [ 0.16674485,  0.01577924,  0.00749162,  0.00621934],  # tiempo_estimado_min
        [-0.09467618,  0.62886676,  0.72684502, -0.48779245],  # num_pasajeros
        [ 2.94259409, -8.74012579, -2.62376024, -15.80661040], # recargo_trafico
    ],
    dtype=np.float64,
)

# Carga única en memoria al iniciar el servidor
if not RUTA_VEHICULOS.exists():
    raise FileNotFoundError(f"No se encontró '{RUTA_VEHICULOS.name}' en {DIR_BASE}")
if not RUTA_MODELO.exists():
    raise FileNotFoundError(f"No se encontró '{RUTA_MODELO.name}' en {DIR_BASE}")

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

app = FastAPI(title="API Tarifas Viajes Compartidos UTR", version="2.0.0")

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
    tarifa_mxn: int


def obtener_recargo(hora_str: str) -> float:
    try:
        t = datetime.strptime(hora_str.strip(), "%H:%M")
    except ValueError:
        raise HTTPException(status_code=400, detail="Formato de hora inválido. Usa HH:MM (ej. 08:15).")

    mins = t.hour * 60 + t.minute
    for ini, fin, rec in FRANJAS_TRAFICO:
        if ini <= mins <= fin:
            return rec
    raise HTTPException(
        status_code=400,
        detail=f"La hora '{hora_str}' está fuera del horario de operación UTR (05:00 a 18:30).",
    )


def buscar_rendimiento(marca: str, modelo: str, anio: Optional[int]) -> float:
    m_norm = marca.strip().lower()
    mod_norm = modelo.strip().lower()

    sub = df_vehiculos[(df_vehiculos["marca_norm"] == m_norm) & (df_vehiculos["modelo_norm"] == mod_norm)]
    if sub.empty:
        return RENDIMIENTO_PROMEDIO_DEFAULT

    if anio is not None and "anio" in sub.columns:
        sub_anio = sub[sub["anio"] == anio]
        if not sub_anio.empty:
            sub = sub_anio

    return round(float(sub["rendimiento_km_l"].mean()), 2)


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/cotizar", response_model=RespuestaCotizacion)
def cotizar_viaje(datos: SolicitudCotizacion):
    rendimiento = buscar_rendimiento(datos.marca, datos.modelo, datos.anio)
    recargo = obtener_recargo(datos.hora)

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
    pcs = np.round(INTERCEPTO_PCA + x @ MATRIZ_PCA, 4)
    pred = float(modelo_mlp.predict(pcs.reshape(1, -1).astype(np.float32), verbose=0)[0, 0])
    tarifa_mxn = max(1, int(np.rint(pred)))

    return RespuestaCotizacion(tarifa_mxn=tarifa_mxn)