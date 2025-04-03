#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Proyecto Integral para Visualización de Datos Nucleares de la IAEA

Se muestran datos provenientes de las APIs de EXFOR, ENDF (dividido en Cross Section,
Fission Product Yield, Radioactive Decay y Covariance) e IBANDL. Cada pestaña presenta 
una tabla y dos gráficos 2D (uno principal y otro secundario). Además, se ha creado 
una nueva pestaña “3D: Covariance Surface” que muestra un gráfico 2D (heatmap) y una 
visualización 3D (superficie) de la covarianza.
Todos los datos provienen de la API del IAEA y su web.
"""

import sys, os, time, re, json, requests, pandas as pd, numpy as np
import matplotlib.pyplot as plt
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
# Caché interna y función para realizar peticiones GET
# -----------------------------------------------------------------------------
cache = {}

def get_json(url):
    if url in cache:
        return cache[url]
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        response.raise_for_status()
        text = response.text
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
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
        time.sleep(1.0)
        return data
    except requests.RequestException as e:
        print(f"Error en la petición {url}: {e}")
        return None

# -----------------------------------------------------------------------------
# Funciones de obtención de datos
# -----------------------------------------------------------------------------

# EXFOR: Se usa la URL válida y se adapta el parseo según la estructura encontrada.
def get_exfor_data():
    url = "https://nds.iaea.org/exfor/servlet/eesig?Projectile=n&Target=Al-27&Emission=a"
    data = get_json(url)
    if data is None:
        return pd.DataFrame()
    if "x4datasets" in data:
        rows = []
        for ds in data["x4datasets"]:
            dataset_id = ds.get("DatasetID", "")
            Ei = ds.get("Ei", [])
            y_arr = ds.get("y", [])
            dy_arr = ds.get("dy", [])
            for i in range(len(Ei)):
                energy = Ei[i] / 1e6 if Ei[i] is not None else None
                cross = y_arr[i] if i < len(y_arr) else None
                uncertainty = dy_arr[i] if (dy_arr and i < len(dy_arr)) else None
                rows.append({
                    "Energy (MeV)": energy,
                    "Cross Section (b)": cross,
                    "Uncertainty (b)": uncertainty,
                    "DatasetID": dataset_id
                })
        return pd.DataFrame(rows)
    else:
        content = data.get("data") if isinstance(data, dict) else None
        rows = []
        if isinstance(content, str):
            for line in content.splitlines():
                parts = re.split(r'\s+', line.strip())
                try:
                    energy = float(parts[0])
                    xs = float(parts[1])
                    row = {"Energy (MeV)": energy, "Cross Section (b)": xs}
                    if len(parts) >= 3:
                        try:
                            unc = float(parts[2])
                            row["Uncertainty (b)"] = unc
                        except:
                            row["Uncertainty (b)"] = None
                    rows.append(row)
                except:
                    continue
        return pd.DataFrame(rows)

# ENDF Cross Section
def get_endf_cross_section():
    url = "https://nds.iaea.org/exfor/e4sig?PenSectID=13657869&json"
    data = get_json(url)
    if data and "datasets" in data and len(data["datasets"]) > 0:
        ds = data["datasets"][0]
        pts = ds.get("pts", [])
        rows = []
        for pt in pts:
            row = {"Energy (eV)": pt.get("E"), "Cross Section (b)": pt.get("Sig"), "Uncertainty (b)": pt.get("dSig")}
            rows.append(row)
        df = pd.DataFrame(rows)
        for key in ["TARGET", "REACTION", "LIBRARY"]:
            if key in ds:
                df[key] = ds[key]
        return df
    return pd.DataFrame()

# ENDF Fission Product Yield
def get_endf_fission_yield():
    url = "https://nds.iaea.org/exfor/e4fy?SectID=9033651&json"
    data = get_json(url)
    if data and "datasets" in data and len(data["datasets"]) > 0:
        ds = data["datasets"][0]
        fys = ds.get("FYs", [])
        rows = []
        for fy in fys:
            row = {"ZAFP": fy.get("ZAFP"), "FPS": fy.get("FPS"), "FY": fy.get("FY"),
                   "DFY": fy.get("DFY"), "PRODUCT": fy.get("PROD")}
            rows.append(row)
        return pd.DataFrame(rows)
    return pd.DataFrame()

# ENDF Radioactive Decay Data
def get_endf_radioactive_decay():
    url = "https://nds.iaea.org/exfor/e4decay?SectID=8930328&json"
    data = get_json(url)
    if data and "DecayModes" in data:
        modes = data["DecayModes"]
        rows = []
        for mode in modes:
            row = {"Mode": mode.get("txRTYP"), "Decay Q (keV)": mode.get("DecayQ"),
                   "Branching": mode.get("Branching")}
            rows.append(row)
        df = pd.DataFrame(rows)
        for key in ["Nucleus", "T12", "Ebeta", "Egamma"]:
            if key in data:
                df[key] = data[key]
        return df
    return pd.DataFrame()

# ENDF Covariance
def get_endf_covariance():
    url = "https://nds.iaea.org/exfor/e4sig?SectID=9019998&json"
    data = get_json(url)
    if data and "datasets" in data and len(data["datasets"]) > 0:
        ds = data["datasets"][0]
        if "xArray1" in ds and "yArray1" in ds and "zArray2" in ds:
            xArr = ds["xArray1"]
            yArr = ds["yArray1"]
            zArr = ds["zArray2"]
            if zArr and isinstance(zArr[0], list):
                flat_zArr = [elem for sublist in zArr for elem in sublist]
            else:
                flat_zArr = zArr
            rows = []
            nx = len(xArr)
            ny = len(yArr)
            for i in range(nx):
                for j in range(ny):
                    index = i * ny + j
                    if index >= len(flat_zArr):
                        continue
                    rows.append({"x": xArr[i], "y": yArr[j], "covariance": flat_zArr[index]})
            return pd.DataFrame(rows)
    return pd.DataFrame()

# IBANDL: Se restaura el código que usaba el listado (lst=2) y se limita a 5 datasets.
def get_ibandl_list():
    url = "https://nds.iaea.org/exfor/ibandl?lst=2&json"
    data = get_json(url)
    if data and isinstance(data, dict) and "results" in data:
        return data["results"]
    return []

def get_ibandl_dataset(ff):
    url = f"https://nds.iaea.org/exfor/ibandl?ff={ff}&json"
    return get_json(url)

def parse_ibandl_data(full_data):
    if not isinstance(full_data, dict):
        return pd.DataFrame()
    try:
        if "Dataset" in full_data and "Data" in full_data["Dataset"]:
            return pd.DataFrame(full_data["Dataset"]["Data"])
        return pd.DataFrame()
    except Exception as e:
        print(f"Error parseando IBANDL JSON: {e}")
        return pd.DataFrame()

def get_all_ibandl_data(limit=5):
    lst = get_ibandl_list()
    frames = []
    count = 0
    for rec in lst:
        ff = rec.get("ff")
        if not ff:
            continue
        full = get_ibandl_dataset(ff)
        df = parse_ibandl_data(full)
        if not df.empty:
            df["ff"] = ff
            frames.append(df)
            count += 1
        if count >= limit:
            break
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()

# -----------------------------------------------------------------------------
# Funciones de graficado 2D (matplotlib)
# -----------------------------------------------------------------------------
def plot_exfor_data(ax, df):
    if "Year" in df.columns:
        try:
            years = pd.to_numeric(df["Year"], errors="coerce").dropna()
            if not years.empty:
                years.plot(kind="bar", ax=ax, color="orange", edgecolor="black")
                ax.set_title("EXFOR: Distribución de Años")
                ax.set_xlabel("Índice")
                ax.set_ylabel("Año")
            else:
                ax.set_title("Sin valores numéricos en 'Year'")
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")
    elif "Energy (MeV)" in df.columns and "Cross Section (b)" in df.columns:
        sorted_df = df.dropna(subset=["Energy (MeV)", "Cross Section (b)"]).sort_values("Energy (MeV)")
        ax.plot(sorted_df["Energy (MeV)"], sorted_df["Cross Section (b)"], "b-o")
        ax.set_title("EXFOR: Energy vs Cross Section")
        ax.set_xlabel("Energy (MeV)")
        ax.set_ylabel("Cross Section (b)")
    else:
        ax.set_title("Datos EXFOR no disponibles")

def plot_exfor_data_log(ax, df):
    if "Energy (MeV)" in df.columns and "Cross Section (b)" in df.columns:
        sorted_df = df.dropna(subset=["Energy (MeV)", "Cross Section (b)"]).sort_values("Energy (MeV)")
        ax.plot(sorted_df["Energy (MeV)"], sorted_df["Cross Section (b)"], "m-o")
        ax.set_xscale("log")
        ax.set_title("EXFOR (Log Scale)")
        ax.set_xlabel("Energy (MeV)")
        ax.set_ylabel("Cross Section (b)")
    else:
        ax.set_title("Sin datos para plot log EXFOR")

def plot_endf_cross_section(ax, df):
    if "Energy (eV)" in df.columns and "Cross Section (b)" in df.columns:
        ax.plot(df["Energy (eV)"]/1e6, df["Cross Section (b)"], "r-o")
        ax.set_title("ENDF Cross Section: PB-204 (n,g)")
        ax.set_xlabel("Energy (MeV)")
        ax.set_ylabel("Cross Section (b)")
    else:
        ax.set_title("Datos ENDF Cross Section no disponibles")

def plot_endf_cross_section_log(ax, df):
    if "Energy (eV)" in df.columns and "Cross Section (b)" in df.columns:
        ax.plot(df["Energy (eV)"]/1e6, df["Cross Section (b)"], "c-o")
        ax.set_xscale("log")
        ax.set_title("ENDF Cross Section (Log Scale)")
        ax.set_xlabel("Energy (MeV)")
        ax.set_ylabel("Cross Section (b)")
    else:
        ax.set_title("Sin datos para plot log ENDF CS")

def plot_endf_fission_yield(ax, df):
    label_col = "PRODUCT" if "PRODUCT" in df.columns else "ZAFP"
    if label_col in df.columns and "FY" in df.columns:
        ax.bar(df[label_col], df["FY"], yerr=df["DFY"], color="orange")
        ax.set_title("ENDF Fission Product Yield: U-238")
        ax.set_xlabel(label_col)
        ax.set_ylabel("Fission Yield")
        ax.tick_params(axis='x', rotation=45)
    else:
        ax.set_title("Datos de Fission Product Yield no disponibles")

def plot_endf_fission_yield_secondary(ax, df):
    if "PRODUCT" in df.columns and "FY" in df.columns:
        sorted_df = df.sort_values("PRODUCT")
        ax.barh(sorted_df["PRODUCT"], sorted_df["FY"], xerr=sorted_df["DFY"], color="skyblue")
        ax.set_title("Fission Yield (Horizontal)")
        ax.set_xlabel("Fission Yield")
        ax.set_ylabel("Producto")
    else:
        ax.set_title("Sin datos para plot secundario FY")

def plot_endf_radioactive_decay(ax, df):
    if "Mode" in df.columns and "Branching" in df.columns:
        ax.bar(df["Mode"], df["Branching"], color="green")
        ax.set_title("ENDF Radioactive Decay: N-16")
        ax.set_xlabel("Modo de Decaimiento")
        ax.set_ylabel("Branching Ratio")
        for i, row in df.iterrows():
            ax.text(i, row["Branching"], f'{row["Decay Q (keV)"]:.1f}', ha="center", va="bottom", fontsize=8)
    else:
        ax.set_title("Datos de Radioactive Decay no disponibles")

def plot_endf_radioactive_decay_secondary(ax, df):
    if "Decay Q (keV)" in df.columns and "Branching" in df.columns:
        ax.scatter(df["Decay Q (keV)"], df["Branching"], color="purple")
        ax.set_title("Decay Q vs Branching")
        ax.set_xlabel("Decay Q (keV)")
        ax.set_ylabel("Branching Ratio")
    else:
        ax.set_title("Sin datos para plot secundario decay")

def plot_covariance(ax, df):
    if "x" in df.columns and "y" in df.columns and "covariance" in df.columns:
        sc = ax.scatter(df["x"], df["y"], c=df["covariance"], cmap="viridis")
        ax.set_title("ENDF Covariance: PB-204")
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Energy (eV)")
        ax.get_figure().colorbar(sc, ax=ax)
    else:
        ax.set_title("Datos de Covariance no disponibles")

def plot_covariance_secondary(ax, df):
    if "x" in df.columns and "y" in df.columns and "covariance" in df.columns:
        tri = ax.tricontourf(df["x"], df["y"], df["covariance"], cmap="viridis")
        ax.set_title("Covariance Contour")
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Energy (eV)")
        ax.get_figure().colorbar(tri, ax=ax)
    else:
        ax.set_title("Sin datos para plot secundario covariance")

def plot_ibandl_scatter(ax, df):
    if "Energy (MeV)" in df.columns and "Sigma or Yield" in df.columns:
        data_ok = df.dropna(subset=["Energy (MeV)", "Sigma or Yield"])
        if not data_ok.empty:
            ax.scatter(data_ok["Energy (MeV)"], data_ok["Sigma or Yield"], color="purple")
            ax.set_title("IBANDL: Energy vs Sigma/Yield")
            ax.set_xlabel("Energy (MeV)")
            ax.set_ylabel("Sigma/Yield")
        else:
            ax.set_title("Datos IBANDL sin valores válidos")
    else:
        ax.set_title("Datos IBANDL no disponibles")

def plot_ibandl_scatter_secondary(ax, df):
    if "Energy (MeV)" in df.columns and "Sigma or Yield" in df.columns:
        data_ok = df.dropna(subset=["Energy (MeV)", "Sigma or Yield"])
        if not data_ok.empty:
            ax.scatter(data_ok["Energy (MeV)"], data_ok["Sigma or Yield"], marker="^", color="brown")
            ax.set_title("IBANDL Secondary")
            ax.set_xlabel("Energy (MeV)")
            ax.set_ylabel("Sigma/Yield")
        else:
            ax.set_title("Sin datos IBANDL secundario")
    else:
        ax.set_title("Sin datos IBANDL secundario")

# -----------------------------------------------------------------------------
# Funciones para extraer puntos 3D (usadas solo en la pestaña 3D si se requiere)
# -----------------------------------------------------------------------------
def exfor_3d_point(row):
    try:
        x = float(row["Energy (MeV)"])
        y = float(row["Cross Section (b)"])
        return (x, y, float(row.get("Uncertainty (b)", 0.0)))
    except:
        return None

def endf_cross_section_3d_point(row):
    try:
        x = float(row["Energy (eV)"])/1e6
        y = float(row["Cross Section (b)"])
        return (x, y, float(row.get("Uncertainty (b)", 0.0)))
    except:
        return None

def endf_fission_yield_3d_point(row):
    try:
        x = float(row.name)
        y = float(row["FY"])
        return (x, y, float(row.get("DFY", 0.0)))
    except:
        return None

def endf_radioactive_decay_3d_point(row):
    try:
        x = float(row["Decay Q (keV)"])
        y = float(row["Branching"])
        return (x, y, 0.0)
    except:
        return None

def covariance_3d_point(row):
    try:
        x = float(row["x"])
        y = float(row["y"])
        return (x, y, float(row["covariance"]))
    except:
        return None

def ibandl_3d_point(row):
    try:
        x = float(row["Energy (MeV)"])
        y = float(row["Sigma or Yield"])
        return (x, y, 0.0)
    except:
        return None

# -----------------------------------------------------------------------------
# Clases para la interfaz: Matplotlib y VTK
# -----------------------------------------------------------------------------
class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax1 = fig.add_subplot(211)
        self.ax2 = fig.add_subplot(212)
        super().__init__(fig)

class VTKWidget(QVTKRenderWindowInteractor):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.renderer = vtk.vtkRenderer()
        self.GetRenderWindow().AddRenderer(self.renderer)
        self.Initialize()
        self.Start()
        self._show_default()

    def _show_default(self):
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
        try:
            if not points:
                self._show_default()
                return
            vtk_pts = vtk.vtkPoints()
            for pt in points:
                vtk_pts.InsertNextPoint(pt)
            poly = vtk.vtkPolyData()
            poly.SetPoints(vtk_pts)
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
        except Exception as e:
            print("Error rendering VTK points:", e)
            self._show_default()

# -----------------------------------------------------------------------------
# Widget de Datos con tabla y gráficos (matplotlib); opción de incluir VTK
# -----------------------------------------------------------------------------
class DataTabWidget(QWidget):
    def __init__(self, title, with_vtk=True, parent=None):
        super().__init__(parent)
        self.title = title
        self.df = pd.DataFrame()
        layout = QVBoxLayout(self)
        self.lbl_title = QLabel(f"<h2>{title}</h2>")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_title)
        self.table = QTableWidget()
        layout.addWidget(self.table)
        self.canvas = MplCanvas()
        layout.addWidget(self.canvas)
        if with_vtk:
            self.vtk_widget = VTKWidget()
            self.vtk_widget.setMinimumHeight(300)
            layout.addWidget(self.vtk_widget)
        else:
            self.vtk_widget = None
        self.lbl_comment = QLabel("Todos los datos provienen de la API del IAEA y su web.")
        self.lbl_comment.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_comment)

    def load_data(self, df, plot_func=None, secondary_plot_func=None, point_func=None):
        self.df = df.copy()
        self._fill_table(df)
        self.canvas.ax1.clear()
        self.canvas.ax2.clear()
        if plot_func and not df.empty:
            if callable(plot_func):
                plot_func(self.canvas.ax1, df)
            elif isinstance(plot_func, list):
                plot_func[0](self.canvas.ax1, df)
        else:
            self.canvas.ax1.set_title("Sin datos para graficar")
        if secondary_plot_func and not df.empty:
            if callable(secondary_plot_func):
                secondary_plot_func(self.canvas.ax2, df)
            elif isinstance(secondary_plot_func, list):
                secondary_plot_func[0](self.canvas.ax2, df)
        else:
            self.canvas.ax2.set_title("Sin datos para plot secundario")
        self.canvas.draw()
        if self.vtk_widget is not None and point_func and not df.empty:
            pts = []
            for _, row in df.iterrows():
                pt = point_func(row)
                if pt:
                    pts.append(pt)
            self.vtk_widget.show_points(pts)

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

# -----------------------------------------------------------------------------
# Widget para la pestaña 3D (solo VTK y un canvas 2D adicional)
# -----------------------------------------------------------------------------
class ThreeDTabWidget(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        layout = QVBoxLayout(self)
        self.lbl_title = QLabel(f"<h2>{title}</h2>")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_title)
        self.canvas = MplCanvas(width=5, height=3, dpi=100)
        layout.addWidget(self.canvas)
        self.vtk_widget = VTKWidget()
        self.vtk_widget.setMinimumHeight(400)
        layout.addWidget(self.vtk_widget)
        self.lbl_comment = QLabel("Todos los datos provienen de la API del IAEA y su web.")
        self.lbl_comment.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_comment)

    def load_data(self, df):
        self.canvas.ax1.clear()
        if "x" in df.columns and "y" in df.columns and "covariance" in df.columns:
            sc = self.canvas.ax1.scatter(df["x"], df["y"], c=df["covariance"], cmap="viridis")
            self.canvas.ax1.set_title("Covariance Heatmap")
            self.canvas.ax1.set_xlabel("Energy (eV)")
            self.canvas.ax1.set_ylabel("Energy (eV)")
            self.canvas.ax1.get_figure().colorbar(sc, ax=self.canvas.ax1)
        else:
            self.canvas.ax1.set_title("Sin datos de Covariance")
        self.canvas.draw()

        # Crear superficie 3D a partir de los datos de covariance
        sgrid = create_covariance_vtk_surface(df)
        self.vtk_widget.renderer.RemoveAllViewProps()
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputData(sgrid)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        self.vtk_widget.renderer.AddActor(actor)
        self.vtk_widget.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

def create_covariance_vtk_surface(df):
    x_unique = np.sort(df["x"].unique())
    y_unique = np.sort(df["y"].unique())
    nx = len(x_unique)
    ny = len(y_unique)
    sgrid = vtk.vtkStructuredGrid()
    sgrid.SetDimensions(nx, ny, 1)
    points = vtk.vtkPoints()
    cov_dict = {(row["x"], row["y"]): row["covariance"] for idx, row in df.iterrows()}
    for i in range(nx):
        for j in range(ny):
            x_val = x_unique[i]
            y_val = y_unique[j]
            cov = cov_dict.get((x_val, y_val), 0.0)
            points.InsertNextPoint(x_val, y_val, cov)
    sgrid.SetPoints(points)
    return sgrid

# -----------------------------------------------------------------------------
# Ventana Principal
# -----------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visualización de Datos IAEA – Proyecto Integral")
        self.resize(1300, 900)
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

        # Crear pestañas sin VTK (solo gráficos 2D)
        self.tab_exfor = DataTabWidget("EXFOR: Al-27 (n,a)", with_vtk=False)
        self.tab_endf_cs = DataTabWidget("ENDF Cross Section: PB-204 (n,g)", with_vtk=False)
        self.tab_endf_fy = DataTabWidget("ENDF Fission Product Yield: U-238", with_vtk=False)
        self.tab_endf_decay = DataTabWidget("ENDF Radioactive Decay: N-16", with_vtk=False)
        self.tab_endf_cov = DataTabWidget("ENDF Covariance: PB-204", with_vtk=False)
        self.tab_ibandl = DataTabWidget("IBANDL: Datos originales", with_vtk=False)
        # Nueva pestaña 3D para la superficie de covariance
        self.tab_3d = ThreeDTabWidget("3D: Covariance Surface")

        self.tabs.addTab(self.tab_exfor, "EXFOR")
        self.tabs.addTab(self.tab_endf_cs, "ENDF - Cross Section")
        self.tabs.addTab(self.tab_endf_fy, "ENDF - Fission Yield")
        self.tabs.addTab(self.tab_endf_decay, "ENDF - Radioactive Decay")
        self.tabs.addTab(self.tab_endf_cov, "ENDF - Covariance")
        self.tabs.addTab(self.tab_ibandl, "IBANDL")
        self.tabs.addTab(self.tab_3d, "3D")

        self.load_all_data()

    def load_all_data(self):
        self.status_bar.showMessage("Descargando y procesando datos...", 3000)
        # EXFOR
        exfor_df = get_exfor_data()
        self.tab_exfor.load_data(exfor_df, plot_func=plot_exfor_data, secondary_plot_func=plot_exfor_data_log, point_func=exfor_3d_point)
        # ENDF Cross Section
        endf_cs_df = get_endf_cross_section()
        self.tab_endf_cs.load_data(endf_cs_df, plot_func=plot_endf_cross_section, secondary_plot_func=plot_endf_cross_section_log, point_func=endf_cross_section_3d_point)
        # ENDF Fission Product Yield
        endf_fy_df = get_endf_fission_yield()
        self.tab_endf_fy.load_data(endf_fy_df, plot_func=plot_endf_fission_yield, secondary_plot_func=plot_endf_fission_yield_secondary, point_func=endf_fission_yield_3d_point)
        # ENDF Radioactive Decay
        endf_decay_df = get_endf_radioactive_decay()
        self.tab_endf_decay.load_data(endf_decay_df, plot_func=plot_endf_radioactive_decay, secondary_plot_func=plot_endf_radioactive_decay_secondary, point_func=endf_radioactive_decay_3d_point)
        # ENDF Covariance
        endf_cov_df = get_endf_covariance()
        self.tab_endf_cov.load_data(endf_cov_df, plot_func=plot_covariance, secondary_plot_func=plot_covariance_secondary, point_func=covariance_3d_point)
        # IBANDL
        ibandl_df = get_all_ibandl_data(limit=5)
        self.tab_ibandl.load_data(ibandl_df, plot_func=plot_ibandl_scatter, secondary_plot_func=plot_ibandl_scatter_secondary, point_func=ibandl_3d_point)
        # Pestaña 3D: Usar los datos de covariance para visualizar la superficie 3D
        self.tab_3d.load_data(endf_cov_df)

        self.status_bar.showMessage("Datos cargados.", 3000)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
