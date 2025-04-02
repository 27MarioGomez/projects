#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Proyecto Integral para Visualización de Datos Nucleares de la IAEA

Esta aplicación descarga y muestra datos extraídos de diversas APIs de la IAEA:
  - Datos experimentales EXFOR (reacción: n + Al-27 → α)
  - Evaluaciones ENDF, incluyendo:
      • Secciones preprocesadas
      • Rendimiento de productos de fisión (con gráfica acumulativa)
      • Datos de decaimiento radiactivo
      • Covarianza de secciones (MF33)
  - Datos IBANDL (reacciones y rendimiento)

La interfaz está diseñada para ofrecer información relevante a usuarios no técnicos,
mostrando títulos y descripciones claras, gráficos optimizados y un pie de página con la fuente.
También se incluye una visualización 3D con VTK para complementar los análisis.

Fuente de datos: API del IAEA – https://nds.iaea.org/exfor/
"""

import sys, os, time, re, json, requests, pandas as pd
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QTableWidget,
    QTableWidgetItem, QTabWidget, QLabel, QPushButton, QStatusBar
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk

# -----------------------------------------------------------------------------
# Caché interna para evitar múltiples descargas
# -----------------------------------------------------------------------------
cache = {}

def get_json(url):
    """
    Realiza una petición GET a 'url' y retorna el JSON parseado.
    Se intenta manejar respuestas en formato JSON o JSON lines.
    En caso de error (por ejemplo, 500), se imprime el error y se retorna None.
    """
    if url in cache:
        return cache[url]
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        response.raise_for_status()  # Lanza excepción en códigos 4xx/5xx
        text = response.text
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Intentar parsear línea por línea
            data_list = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    data_list.append(obj)
                except json.JSONDecodeError:
                    continue
            data = data_list if data_list else None
        cache[url] = data
        time.sleep(1.0)  # Para evitar saturar el servidor
        return data
    except requests.RequestException as e:
        print(f"Error en la petición {url}: {e}")
        return None

# =============================================================================
# Sección EXFOR: Datos experimentales (n, Al-27 → α)
# =============================================================================
def get_exfor_data():
    """
    Descarga datos EXFOR (formato JSON) del endpoint 'eesig'.
    """
    url = "https://nds.iaea.org/exfor/servlet/eesig?Projectile=n&Target=Al-27&Emission=a"
    data = get_json(url)
    if not data or not isinstance(data, dict):
        return pd.DataFrame()
    x4datasets = data.get("x4datasets", [])
    rows = []
    for ds in x4datasets:
        dataset_id = ds.get("DatasetID")
        Ei = ds.get("Ei", [])
        y = ds.get("y", [])
        dy = ds.get("dy", [])
        dEi = ds.get("dEi", [])
        for i in range(len(Ei)):
            row = {
                "DatasetID": dataset_id,
                "Energy (MeV)": float(Ei[i]) / 1e6 if Ei[i] > 1e6 else float(Ei[i]),
                "Cross Section (b)": float(y[i]) if i < len(y) else None,
                "Uncertainty (b)": float(dy[i]) if i < len(dy) else None,
                "Energy Uncertainty (MeV)": float(dEi[i]) / 1e6 if (i < len(dEi) and dEi[i] > 1e6) else (float(dEi[i]) if i < len(dEi) else None),
            }
            rows.append(row)
    return pd.DataFrame(rows)

def plot_exfor_curve(ax, df):
    """
    Grafica Energy vs Cross Section para datos EXFOR, incluyendo error bars.
    """
    if "Energy (MeV)" in df.columns and "Cross Section (b)" in df.columns:
        sorted_df = df.dropna(subset=["Energy (MeV)", "Cross Section (b)"]).sort_values("Energy (MeV)")
        x = sorted_df["Energy (MeV)"]
        y = sorted_df["Cross Section (b)"]
        yerr = sorted_df.get("Uncertainty (b)")
        if yerr is not None and not yerr.empty:
            ax.errorbar(x, y, yerr=yerr, fmt="b-o", ecolor="lightblue", capsize=3)
        else:
            ax.plot(x, y, "b-o")
        ax.set_title("Reacción Experimental (n, Al-27 → α)")
        ax.set_xlabel("Energía (MeV)")
        ax.set_ylabel("Sección Eficaz (b)")
    else:
        ax.set_title("No hay datos EXFOR disponibles")

def exfor_3d_point(row):
    """
    Mapea una fila de EXFOR a una coordenada 3D: (Energy, Cross Section, Uncertainty).
    """
    try:
        x = float(row["Energy (MeV)"])
        y = float(row["Cross Section (b)"])
        z = float(row.get("Uncertainty (b)", 0.0))
        return (x, y, z)
    except:
        return None

# =============================================================================
# Sección ENDF: Evaluaciones (Nuevos endpoints)
# =============================================================================
def get_endf_cross_section_data():
    """
    Descarga datos de secciones evaluadas preprocesadas a partir de:
    https://nds.iaea.org/exfor/e4sig?PenSectID=13657869&json
    """
    url = "https://nds.iaea.org/exfor/e4sig?PenSectID=13657869&json"
    data = get_json(url)
    if not data or not isinstance(data, dict):
        return pd.DataFrame()
    ds_arr = data.get("datasets", [])
    frames = []
    for ds in ds_arr:
        pts = ds.get("pts", [])
        rows = []
        for p in pts:
            energy_eV = float(p.get("E", 0.0))
            sigma = float(p.get("Sig", 0.0))
            dsig = float(p.get("dSig", 0.0))
            energy = energy_eV * 1e-6  # Convertir de eV a MeV
            rows.append({
                "Energy (MeV)": energy,
                "Cross Section (b)": sigma,
                "Uncertainty (b)": dsig
            })
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def plot_endf_cross_section(ax, df):
    """
    Grafica las secciones evaluadas preprocesadas.
    """
    if "Energy (MeV)" in df.columns and "Cross Section (b)" in df.columns:
        sorted_df = df.dropna(subset=["Energy (MeV)", "Cross Section (b)"]).sort_values("Energy (MeV)")
        x = sorted_df["Energy (MeV)"]
        y = sorted_df["Cross Section (b)"]
        yerr = sorted_df.get("Uncertainty (b)")
        if yerr is not None and not yerr.empty:
            ax.errorbar(x, y, yerr=yerr, fmt="r-o", ecolor="orange", capsize=3)
        else:
            ax.plot(x, y, "r-o")
        ax.set_title("Secciones Evaluadas (Preprocesadas)")
        ax.set_xlabel("Energía (MeV)")
        ax.set_ylabel("Sección (b)")
    else:
        ax.set_title("No hay datos de secciones evaluadas")

def endf_3d_point(row):
    """
    Mapea una fila de secciones evaluadas a (Energy, Cross Section, Uncertainty).
    """
    try:
        x = float(row["Energy (MeV)"])
        y = float(row["Cross Section (b)"])
        z = float(row.get("Uncertainty (b)", 0.0))
        return (x, y, z)
    except:
        return None

def get_endf_fission_yield_data():
    """
    Descarga datos de rendimiento de productos de fisión desde:
    https://nds.iaea.org/exfor/e4fy?SectID=9033651&json
    Se asume que el JSON tiene las claves "E", "Y" y "dY".
    """
    url = "https://nds.iaea.org/exfor/e4fy?SectID=9033651&json"
    data = get_json(url)
    if not data or not isinstance(data, dict):
        return pd.DataFrame()
    pts = data.get("pts", [])
    rows = []
    for p in pts:
        energy_eV = float(p.get("E", 0.0))
        yield_val = float(p.get("Y", 0.0))
        dy = float(p.get("dY", 0.0))
        energy = energy_eV * 1e-6
        rows.append({
            "Energy (MeV)": energy,
            "Yield": yield_val,
            "Uncertainty": dy
        })
    return pd.DataFrame(rows)

def plot_endf_fission_yield(ax, df):
    """
    Grafica el rendimiento de productos de fisión y su rendimiento acumulado.
    """
    if "Energy (MeV)" in df.columns and "Yield" in df.columns:
        sorted_df = df.dropna(subset=["Energy (MeV)", "Yield"]).sort_values("Energy (MeV)")
        x = sorted_df["Energy (MeV)"]
        y = sorted_df["Yield"]
        yerr = sorted_df.get("Uncertainty")
        ax.errorbar(x, y, yerr=yerr, fmt="m-o", ecolor="pink", capsize=3, label="Yield")
        # Rendimiento acumulado
        cum_y = y.cumsum()
        ax2 = ax.twinx()
        ax2.plot(x, cum_y, "k--", label="Rendimiento Acumulado")
        ax.set_title("Rendimiento de Productos de Fisión")
        ax.set_xlabel("Energía (MeV)")
        ax.set_ylabel("Yield")
        ax2.set_ylabel("Yield Acumulado")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
    else:
        ax.set_title("No hay datos de rendimiento de fisión")

def endf_fission_yield_3d_point(row):
    """
    Mapea una fila de rendimiento de fisión a (Energy, Yield, Uncertainty).
    """
    try:
        x = float(row["Energy (MeV)"])
        y = float(row["Yield"])
        z = float(row.get("Uncertainty", 0.0))
        return (x, y, z)
    except:
        return None

def get_endf_decay_data():
    """
    Descarga datos de decaimiento radiactivo desde:
    https://nds.iaea.org/exfor/e4decay?SectID=8930328&json
    Se asume que el JSON contiene "pts" con claves "T", "A" y "dA".
    """
    url = "https://nds.iaea.org/exfor/e4decay?SectID=8930328&json"
    data = get_json(url)
    if not data or not isinstance(data, dict):
        return pd.DataFrame()
    pts = data.get("pts", [])
    rows = []
    for p in pts:
        t = float(p.get("T", 0.0))
        act = float(p.get("A", 0.0))
        dA = float(p.get("dA", 0.0))
        rows.append({
            "Time (s)": t,
            "Activity": act,
            "Uncertainty": dA
        })
    return pd.DataFrame(rows)

def plot_endf_decay(ax, df):
    """
    Grafica los datos de decaimiento radiactivo.
    """
    if "Time (s)" in df.columns and "Activity" in df.columns:
        sorted_df = df.dropna(subset=["Time (s)", "Activity"]).sort_values("Time (s)")
        x = sorted_df["Time (s)"]
        y = sorted_df["Activity"]
        yerr = sorted_df.get("Uncertainty")
        ax.errorbar(x, y, yerr=yerr, fmt="c-o", ecolor="teal", capsize=3)
        ax.set_title("Datos de Decaimiento Radiactivo")
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Actividad")
    else:
        ax.set_title("No hay datos de decaimiento")

def endf_decay_3d_point(row):
    """
    Mapea una fila de decaimiento a (Time, Activity, Uncertainty).
    """
    try:
        x = float(row["Time (s)"])
        y = float(row["Activity"])
        z = float(row.get("Uncertainty", 0.0))
        return (x, y, z)
    except:
        return None

def get_endf_covariance_data():
    """
    Descarga datos de covarianza desde:
    https://nds.iaea.org/exfor/e4sig?SectID=9019998&json
    Se espera que el JSON incluya una matriz de covarianza y una lista de energías.
    """
    url = "https://nds.iaea.org/exfor/e4sig?SectID=9019998&json"
    data = get_json(url)
    if not data or not isinstance(data, dict):
        return None
    cov = data.get("covariance")
    energies = data.get("energies")
    if cov is None or energies is None:
        return None
    return cov, energies

def plot_endf_covariance(ax, cov_data):
    """
    Visualiza la matriz de covarianza utilizando imshow.
    """
    if cov_data is None:
        ax.set_title("No hay datos de covarianza")
        return
    cov, energies = cov_data
    cov = np.array(cov)
    im = ax.imshow(cov, cmap="viridis", aspect="auto", origin="lower")
    ax.set_title("Covarianza de Secciones (MF33)")
    ax.set_xlabel("Índice")
    ax.set_ylabel("Índice")
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.figure.colorbar(im, cax=cax)

# =============================================================================
# Sección IBANDL: Datos de reacciones y rendimiento
# =============================================================================
def get_ibandl_dataset(ff):
    """
    Descarga un dataset IBANDL dado un identificador 'ff'.
    Se usa 'convert=rr2mb' para c3pp0l.
    """
    if ff == "o6pp0m":
        url = "https://nds.iaea.org/exfor/ibandl?ff=o6pp0m&json"
    elif ff == "c3pp0l":
        url = "https://nds.iaea.org/exfor/ibandl?ff=c3pp0l&convert=rr2mb&json"
    else:
        url = f"https://nds.iaea.org/exfor/ibandl?ff={ff}&json"
    return get_json(url)

def parse_ibandl_data(full_data):
    """
    Parsea el JSON de IBANDL.
    Se espera una estructura con "datasets" y dentro "Data" (matriz de valores).
    """
    if not isinstance(full_data, dict):
        return pd.DataFrame()
    ds_arr = full_data.get("datasets", [])
    frames = []
    for ds in ds_arr:
        data_matrix = ds.get("Data", [])
        rows = []
        for row_vals in data_matrix:
            # Asumimos: [Energy, Angle, Sigma/Yield, Uncertainty]
            energy = float(row_vals[0]) if len(row_vals) > 0 else None
            second_val = float(row_vals[1]) if len(row_vals) > 1 else None
            sigma = float(row_vals[2]) if len(row_vals) > 2 else None
            dsig = float(row_vals[3]) if len(row_vals) > 3 else None
            # Para o6pp0m, la segunda columna es ángulo; para c3pp0l puede ser incertidumbre en E.
            angle = second_val if ds.get("file") == "o6pp0m" else None
            rows.append({
                "Year": ds.get("year"),
                "Energy (MeV)": energy / 1e3 if energy and energy > 1000 else energy,
                "Angle (deg)": angle,
                "Sigma or Yield": sigma,
                "Uncertainty": dsig
            })
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def get_two_ibandl_data():
    """
    Descarga dos datasets IBANDL: o6pp0m y c3pp0l.
    """
    frames = []
    for ff in ["o6pp0m", "c3pp0l"]:
        full = get_ibandl_dataset(ff)
        if not full:
            continue
        df = parse_ibandl_data(full)
        if not df.empty:
            df["ff"] = ff
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def plot_ibandl_scatter(ax, df):
    """
    Grafica datos IBANDL: Energy vs Sigma/Yield.
    """
    if "Energy (MeV)" in df.columns and "Sigma or Yield" in df.columns:
        sorted_df = df.dropna(subset=["Energy (MeV)", "Sigma or Yield"]).sort_values("Energy (MeV)")
        x = sorted_df["Energy (MeV)"]
        y = sorted_df["Sigma or Yield"]
        yerr = sorted_df.get("Uncertainty")
        ax.errorbar(x, y, yerr=yerr, fmt="p", color="purple", ecolor="violet", capsize=3)
        ax.set_title("Análisis IBANDL (Reacciones y Rendimiento)")
        ax.set_xlabel("Energía (MeV)")
        ax.set_ylabel("Sigma / Yield")
    else:
        ax.set_title("No hay datos IBANDL disponibles")

def ibandl_3d_point(row):
    """
    Mapea una fila IBANDL a (Energy, Sigma/Yield, Angle) (si existe).
    """
    try:
        x = float(row["Energy (MeV)"])
        y = float(row["Sigma or Yield"])
        z = float(row.get("Angle (deg)", 0.0))
        return (x, y, z)
    except:
        return None

# =============================================================================
# VTK: Visualización 3D
# =============================================================================
class VTKWidget(QVTKRenderWindowInteractor):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.renderer = vtk.vtkRenderer()
        self.GetRenderWindow().AddRenderer(self.renderer)
        self.Initialize()
        self.Start()
        self._show_default()

    def _show_default(self):
        """
        Muestra un cubo por defecto cuando no hay puntos.
        """
        self.renderer.RemoveAllViewProps()
        cube = vtk.vtkCubeSource()
        cube.SetXLength(2.0)
        cube.SetYLength(2.0)
        cube.SetZLength(2.0)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cube.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.3, 0.7, 0.9)
        self.renderer.AddActor(actor)
        self.renderer.ResetCamera()
        self.GetRenderWindow().Render()

    def show_points(self, points):
        """
        Visualiza los puntos 3D. Se crea una celda para cada punto para que sean visibles.
        """
        if not points:
            self._show_default()
            return
        vtk_pts = vtk.vtkPoints()
        vertices = vtk.vtkCellArray()
        for pt in points:
            pid = vtk_pts.InsertNextPoint(pt)
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(pid)
        poly = vtk.vtkPolyData()
        poly.SetPoints(vtk_pts)
        poly.SetVerts(vertices)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(6)
        actor.GetProperty().SetColor(1, 0, 0)
        self.renderer.RemoveAllViewProps()
        self.renderer.AddActor(actor)
        self.renderer.ResetCamera()
        self.GetRenderWindow().Render()

# =============================================================================
# Interfaz: Widget con título, descripción, tabla, gráfico Matplotlib, VTK y pie de página
# =============================================================================
class DataTabWidget(QWidget):
    def __init__(self, title, description, footer, parent=None):
        super().__init__(parent)
        self.title = title
        self.df = pd.DataFrame()
        layout = QVBoxLayout(self)
        
        self.lbl_title = QLabel(f"<h2>{title}</h2>")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_title)
        
        self.lbl_desc = QLabel(f"<i>{description}</i>")
        self.lbl_desc.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_desc)
        
        self.table = QTableWidget()
        layout.addWidget(self.table)
        
        self.canvas = MplCanvas()
        layout.addWidget(self.canvas)
        
        self.vtk_widget = VTKWidget()
        layout.addWidget(self.vtk_widget)
        
        self.lbl_footer = QLabel(f"<small>{footer}</small>")
        self.lbl_footer.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_footer)

    def load_data(self, df, plot_func=None, point_func=None):
        self.df = df.copy()
        self._fill_table(df)
        self.canvas.axes.clear()
        if plot_func and not df.empty:
            if callable(plot_func):
                plot_func(self.canvas.axes, df)
            elif isinstance(plot_func, list):
                for func in plot_func:
                    func(self.canvas.axes, df)
        else:
            self.canvas.axes.set_title("Sin datos para graficar")
        self.canvas.draw()
        if point_func and not df.empty:
            pts = []
            for _, row in df.iterrows():
                pt = point_func(row)
                if pt:
                    pts.append(pt)
            if pts:
                self.vtk_widget.show_points(pts)
            else:
                self.vtk_widget._show_default()
        else:
            self.vtk_widget._show_default()

    def _fill_table(self, df: pd.DataFrame):
        self.table.clear()
        if df.empty:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels(df.columns.tolist())
        for i in range(len(df)):
            for j, col in enumerate(df.columns):
                self.table.setItem(i, j, QTableWidgetItem(str(df.iloc[i, j])))
        self.table.resizeColumnsToContents()

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=3, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)

# =============================================================================
# Ventana Principal con Múltiples Pestañas
# =============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IAEA APIs – Análisis Integral de Datos Nucleares")
        self.resize(1200, 900)
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        self.btn_reload = QPushButton("Recargar y analizar datos")
        self.btn_reload.clicked.connect(self.load_all_data)
        main_layout.addWidget(self.btn_reload)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Definición de textos para cada pestaña (títulos, descripciones y pie de página)
        footer_exfor = "Fuente: IAEA EXFOR API – https://nds.iaea.org/exfor/"
        footer_endf = "Fuente: IAEA ENDF API – https://nds.iaea.org/exfor/"
        footer_ibandl = "Fuente: IAEA IBANDL API – https://nds.iaea.org/exfor/"

        # Pestaña 1: EXFOR
        self.tab_exfor = DataTabWidget(
            "Reacción Experimental (n, Al-27 → α) [EXFOR]",
            "Datos experimentales obtenidos a través de la API EXFOR. Se analizan energías y secciones eficaces.",
            footer_exfor
        )
        # Pestaña 2: ENDF - Secciones evaluadas preprocesadas
        self.tab_endf_sec = DataTabWidget(
            "Secciones Evaluadas (Preprocesadas) [ENDF]",
            "Datos de secciones de reacción preprocesados a partir de evaluaciones nucleares ENDF.",
            footer_endf
        )
        # Pestaña 3: ENDF - Rendimiento de productos de fisión
        self.tab_endf_fy = DataTabWidget(
            "Rendimiento de Productos de Fisión [ENDF]",
            "Análisis del rendimiento de productos de fisión basado en datos evaluados de ENDF.",
            footer_endf
        )
        # Pestaña 4: ENDF - Datos de decaimiento radiactivo
        self.tab_endf_decay = DataTabWidget(
            "Datos de Decaimiento Radiactivo [ENDF]",
            "Información sobre decaimiento radiactivo, incluyendo actividades y tiempos.",
            footer_endf
        )
        # Pestaña 5: ENDF - Covarianza de secciones (MF33)
        self.tab_endf_cov = DataTabWidget(
            "Covarianza de Secciones (MF33) [ENDF]",
            "Visualización de la matriz de covarianza obtenida de evaluaciones nucleares (MF33).",
            footer_endf
        )
        # Pestaña 6: IBANDL
        self.tab_ibandl = DataTabWidget(
            "Análisis IBANDL (Reacciones y Rendimiento)",
            "Datos IBANDL que permiten el análisis de reacciones nucleares y rendimiento de procesos.",
            footer_ibandl
        )

        self.tabs.addTab(self.tab_exfor, "EXFOR")
        self.tabs.addTab(self.tab_endf_sec, "ENDF: Secciones")
        self.tabs.addTab(self.tab_endf_fy, "ENDF: Fisión")
        self.tabs.addTab(self.tab_endf_decay, "ENDF: Decaimiento")
        self.tabs.addTab(self.tab_endf_cov, "ENDF: Covarianza")
        self.tabs.addTab(self.tab_ibandl, "IBANDL")

        self.load_all_data()

    def load_all_data(self):
        self.status_bar.showMessage("Descargando y procesando datos...", 3000)
        # EXFOR
        exfor_df = get_exfor_data()
        self.tab_exfor.load_data(
            exfor_df,
            plot_func=plot_exfor_curve,
            point_func=exfor_3d_point
        )
        # ENDF - Secciones evaluadas
        endf_sec_df = get_endf_cross_section_data()
        self.tab_endf_sec.load_data(
            endf_sec_df,
            plot_func=plot_endf_cross_section,
            point_func=endf_3d_point
        )
        # ENDF - Fisión
        endf_fy_df = get_endf_fission_yield_data()
        self.tab_endf_fy.load_data(
            endf_fy_df,
            plot_func=plot_endf_fission_yield,
            point_func=endf_fission_yield_3d_point
        )
        # ENDF - Decaimiento
        endf_decay_df = get_endf_decay_data()
        self.tab_endf_decay.load_data(
            endf_decay_df,
            plot_func=plot_endf_decay,
            point_func=endf_decay_3d_point
        )
        # ENDF - Covarianza
        cov_data = get_endf_covariance_data()
        # Para covarianza, no mostramos tabla ni VTK (se visualiza solo el plot)
        dummy_df = pd.DataFrame({"A": [0]})  # DataFrame vacío para la tabla
        self.tab_endf_cov.load_data(
            dummy_df,
            plot_func=lambda ax, df: plot_endf_covariance(ax, cov_data),
            point_func=None
        )
        # IBANDL
        ibandl_df = get_two_ibandl_data()
        self.tab_ibandl.load_data(
            ibandl_df,
            plot_func=plot_ibandl_scatter,
            point_func=ibandl_3d_point
        )
        self.status_bar.showMessage("Datos cargados.", 3000)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
