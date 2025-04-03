#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive Nuclear Data Visualization Project

This application retrieves and visualizes nuclear data from IAEA APIs, including datasets from EXFOR and ENDF (Cross Section, Fission Product Yield, Radioactive Decay, and Covariance).

The user interface presents the data in multiple tabs, each displaying a data table along with two Matplotlib graphs. Additionally, a dedicated 3D tab shows a covariance surface visualization using VTK, complete with a scalar bar for color interpretation.

The design emphasizes clarity and readability, with adjustable spacing and optimized layout for an improved user experience.
"""

import sys, os, time, re, json, requests, pandas as pd, numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # to control tick frequency in log-scale plots
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QTableWidget,
    QTableWidgetItem, QTabWidget, QLabel, QPushButton, QStatusBar,
    QScrollArea
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk

# -----------------------------------------------------------------------------
# Internal cache and function to perform GET requests
# -----------------------------------------------------------------------------
cache = {}

def get_json(url):
    """
    Performs a GET request to the specified URL and returns the parsed JSON result.
    Caches responses to avoid redundant requests.
    """
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
        print(f"Error in request {url}: {e}")
        return None

# -----------------------------------------------------------------------------
# Data retrieval functions
# -----------------------------------------------------------------------------

def get_exfor_data():
    """
    Retrieves EXFOR data for the reaction (n, Al-27 -> a).
    """
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

def get_endf_cross_section():
    """
    Retrieves cross section data for PB-204 (n,g) from ENDF.
    """
    url = "https://nds.iaea.org/exfor/e4sig?PenSectID=13657869&json"
    data = get_json(url)
    if data and "datasets" in data and len(data["datasets"]) > 0:
        ds = data["datasets"][0]
        pts = ds.get("pts", [])
        rows = []
        for pt in pts:
            row = {
                "Energy (eV)": pt.get("E"),
                "Cross Section (b)": pt.get("Sig"),
                "Uncertainty (b)": pt.get("dSig")
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        for key in ["TARGET", "REACTION", "LIBRARY"]:
            if key in ds:
                df[key] = ds[key]
        return df
    return pd.DataFrame()

def get_endf_fission_yield():
    """
    Retrieves fission product yield data for U-238 from ENDF.
    """
    url = "https://nds.iaea.org/exfor/e4fy?SectID=9033651&json"
    data = get_json(url)
    if data and "datasets" in data and len(data["datasets"]) > 0:
        ds = data["datasets"][0]
        fys = ds.get("FYs", [])
        rows = []
        for fy in fys:
            row = {
                "ZAFP": fy.get("ZAFP"),
                "FPS": fy.get("FPS"),
                "FY": fy.get("FY"),
                "DFY": fy.get("DFY"),
                "PRODUCT": fy.get("PROD")
            }
            rows.append(row)
        return pd.DataFrame(rows)
    return pd.DataFrame()

def get_endf_radioactive_decay():
    """
    Retrieves radioactive decay data for N-16 from ENDF.
    """
    url = "https://nds.iaea.org/exfor/e4decay?SectID=8930328&json"
    data = get_json(url)
    if data and "DecayModes" in data:
        modes = data["DecayModes"]
        rows = []
        for mode in modes:
            row = {
                "Mode": mode.get("txRTYP"),
                "Decay Q (keV)": mode.get("DecayQ"),
                "Branching": mode.get("Branching")
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        for key in ["Nucleus", "T12", "Ebeta", "Egamma"]:
            if key in data:
                df[key] = data[key]
        return df
    return pd.DataFrame()

def get_endf_covariance():
    """
    Retrieves the covariance matrix for PB-204 (n,g) from ENDF.
    """
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
                    rows.append({
                        "x": xArr[i],
                        "y": yArr[j],
                        "covariance": flat_zArr[index]
                    })
            return pd.DataFrame(rows)
    return pd.DataFrame()

# -----------------------------------------------------------------------------
# 2D Plotting Functions (Matplotlib)
# -----------------------------------------------------------------------------

def plot_exfor_data(ax, df):
    """
    Primary EXFOR plot: Energy vs Cross Section (or Year distribution if available).
    """
    if "Year" in df.columns:
        try:
            years = pd.to_numeric(df["Year"], errors="coerce").dropna()
            if not years.empty:
                years.plot(kind="bar", ax=ax, color="orange", edgecolor="black")
                ax.set_title("EXFOR: Year Distribution")
                ax.set_xlabel("Index")
                ax.set_ylabel("Year")
            else:
                ax.set_title("No numeric values in 'Year'")
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")
    elif "Energy (MeV)" in df.columns and "Cross Section (b)" in df.columns:
        sorted_df = df.dropna(subset=["Energy (MeV)", "Cross Section (b)"]).sort_values("Energy (MeV)")
        ax.plot(sorted_df["Energy (MeV)"], sorted_df["Cross Section (b)"], "b-o")
        ax.set_title("EXFOR: Energy vs Cross Section")
        ax.set_xlabel("Energy (MeV)")
        ax.set_ylabel("Cross Section (b)")
    else:
        ax.set_title("EXFOR data not available")

def plot_exfor_data_log(ax, df):
    """
    Secondary EXFOR plot: Uncertainty vs Energy (log scale) with controlled tick labels.
    """
    if "Energy (MeV)" in df.columns and "Uncertainty (b)" in df.columns:
        sorted_df = df.dropna(subset=["Energy (MeV)", "Uncertainty (b)"]).sort_values("Energy (MeV)")
        ax.plot(sorted_df["Energy (MeV)"], sorted_df["Uncertainty (b)"], "m-o")
        ax.set_xscale("log")
        ax.set_title("EXFOR: Uncertainty vs Energy (Log Scale)")
        ax.set_xlabel("Energy (MeV)")
        ax.set_ylabel("Uncertainty (b)")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_fontsize(8)
    else:
        ax.set_title("No data for EXFOR log plot")

def plot_endf_cross_section(ax, df):
    """
    Primary ENDF Cross Section plot: PB-204 (n,g) using error bars to represent uncertainties.
    """
    if "Energy (eV)" in df.columns and "Cross Section (b)" in df.columns:
        x = df["Energy (eV)"] / 1e6
        y = df["Cross Section (b)"]
        yerr = df["Uncertainty (b)"] if "Uncertainty (b)" in df.columns else None
        ax.errorbar(x, y, yerr=yerr, fmt="o-", color="red", ecolor="black", capsize=2)
        ax.set_title("ENDF Cross Section: PB-204 (n,g) with Error Bars")
        ax.set_xlabel("Energy (MeV)")
        ax.set_ylabel("Cross Section (b)")
    else:
        ax.set_title("ENDF Cross Section data not available")

def plot_endf_cross_section_log(ax, df):
    """
    Secondary ENDF Cross Section plot: Log scale representation.
    """
    if "Energy (eV)" in df.columns and "Cross Section (b)" in df.columns:
        ax.plot(df["Energy (eV)"]/1e6, df["Cross Section (b)"], "c-o")
        ax.set_xscale("log")
        ax.set_title("ENDF Cross Section (Log Scale)")
        ax.set_xlabel("Energy (MeV)")
        ax.set_ylabel("Cross Section (b)")
    else:
        ax.set_title("No data for ENDF Cross Section log plot")

def plot_endf_fission_yield(ax, df):
    """
    Primary ENDF Fission Yield plot: Top 20 products (vertical bar chart).
    """
    label_col = "PRODUCT" if "PRODUCT" in df.columns else "ZAFP"
    if label_col in df.columns and "FY" in df.columns:
        top20 = df.sort_values("FY", ascending=False).head(20)
        ax.bar(top20[label_col], top20["FY"], yerr=top20["DFY"], color="orange")
        ax.set_title("ENDF Fission Product Yield: U-238 (Top 20)")
        ax.set_xlabel(label_col)
        ax.set_ylabel("Fission Yield")
        ax.tick_params(axis='x', rotation=45)
    else:
        ax.set_title("ENDF Fission Product Yield data not available")

def plot_endf_fission_yield_secondary(ax, df):
    """
    Secondary ENDF Fission Yield plot: Top 10 products in a horizontal bar chart.
    """
    if "PRODUCT" in df.columns and "FY" in df.columns:
        top10 = df.sort_values("FY", ascending=False).head(10)
        ax.barh(top10["PRODUCT"], top10["FY"], xerr=top10["DFY"], color="skyblue")
        ax.set_title("Top 10 Fission Yield")
        ax.set_xlabel("Fission Yield")
        ax.set_ylabel("Product")
        ax.invert_yaxis()  # Ensures the highest yield is at the top
    else:
        ax.set_title("No data for ENDF Fission Yield secondary plot")

def plot_endf_radioactive_decay(ax, df):
    """
    Primary ENDF Radioactive Decay plot: N-16.
    """
    if "Mode" in df.columns and "Branching" in df.columns:
        ax.bar(df["Mode"], df["Branching"], color="green")
        ax.set_title("ENDF Radioactive Decay: N-16")
        ax.set_xlabel("Decay Mode")
        ax.set_ylabel("Branching Ratio")
        for i, row in df.iterrows():
            dq = row.get("Decay Q (keV)", None)
            if dq is not None:
                ax.text(i, row["Branching"], f'{dq:.1f} keV', ha="center", va="bottom", fontsize=8)
    else:
        ax.set_title("ENDF Radioactive Decay data not available")

def plot_endf_radioactive_decay_secondary(ax, df):
    """
    Secondary ENDF Radioactive Decay display: explanatory text.
    """
    if "Decay Q (keV)" in df.columns and "Branching" in df.columns:
        ax.text(0.5, 0.5,
                ("This chart shows the decay Q-values and branching ratios for N-16. "
                 "The primary decay mode is Beta-, with a nearly unity branching ratio."),
                ha="center", va="center", fontsize=10, wrap=True)
        ax.axis('off')
    else:
        ax.set_title("No data for ENDF Radioactive Decay secondary display")

def plot_covariance(ax, df):
    """
    Primary Covariance plot: scatter plot with colors representing covariance values.
    """
    if "x" in df.columns and "y" in df.columns and "covariance" in df.columns:
        sc = ax.scatter(df["x"], df["y"], c=df["covariance"], cmap="viridis")
        ax.set_title("ENDF Covariance: PB-204")
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Energy (eV)")
        fig = ax.get_figure()
        fig.colorbar(sc, ax=ax)
    else:
        ax.set_title("ENDF Covariance data not available")

def plot_covariance_secondary(ax, df):
    """
    Secondary Covariance plot: contour plot of covariance.
    """
    if "x" in df.columns and "y" in df.columns and "covariance" in df.columns:
        tri = ax.tricontourf(df["x"], df["y"], df["covariance"], cmap="viridis")
        ax.set_title("Covariance Contour")
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Energy (eV)")
        fig = ax.get_figure()
        fig.colorbar(tri, ax=ax)
    else:
        ax.set_title("No data for ENDF Covariance secondary plot")

# -----------------------------------------------------------------------------
# Classes for the GUI: Matplotlib Canvas and VTK Widget
# -----------------------------------------------------------------------------
class MplCanvas(FigureCanvasQTAgg):
    """
    Canvas containing two subplots with adjusted vertical spacing.
    """
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        # Increase vertical spacing to avoid overlapping of titles and labels
        self.fig.subplots_adjust(hspace=0.8)
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        super().__init__(self.fig)

class VTKWidget(QVTKRenderWindowInteractor):
    """
    VTK widget that displays a default 3D object if no covariance data is loaded.
    """
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

# -----------------------------------------------------------------------------
# DataTabWidget: Data table with two Matplotlib graphs and footer
# -----------------------------------------------------------------------------
class DataTabWidget(QWidget):
    """
    Each tab displays:
      - A large title with increased font size and a compact header container.
      - A data table with a limited height.
      - Two Matplotlib graphs with optimized spacing.
      - A footer with IAEA API reference.
    """
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        self.df = pd.DataFrame()

        # Scroll area to allow content to be scrollable if it exceeds the window size
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)

        main_container = QWidget()
        layout = QVBoxLayout(main_container)

        # Title with larger font but reduced container height
        self.lbl_title = QLabel(f"<h2>{title}</h2>")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        self.lbl_title.setStyleSheet("font-size: 16px; margin: 5px;")
        self.lbl_title.setMaximumHeight(40)
        layout.addWidget(self.lbl_title)

        # Data table with limited height
        self.table = QTableWidget()
        self.table.setMaximumHeight(110)
        layout.addWidget(self.table)

        # Canvas with two graphs
        self.canvas = MplCanvas()
        layout.addWidget(self.canvas)

        # Footer with reference text (container height reduced)
        self.lbl_comment = QLabel("Data extracted from the IAEA API – https://nds.iaea.org/")
        self.lbl_comment.setAlignment(Qt.AlignCenter)
        self.lbl_comment.setMaximumHeight(25)
        layout.addWidget(self.lbl_comment)

        scroll.setWidget(main_container)
        outer_layout = QVBoxLayout(self)
        outer_layout.addWidget(scroll)

    def load_data(self, df, plot_func=None, secondary_plot_func=None):
        """
        Loads the DataFrame into the table and generates two graphs using the provided plotting functions.
        """
        self.df = df.copy()
        self._fill_table(df)

        self.canvas.ax1.clear()
        self.canvas.ax2.clear()

        if plot_func and not df.empty:
            plot_func(self.canvas.ax1, df)
        else:
            self.canvas.ax1.set_title("No data available for primary plot")

        if secondary_plot_func and not df.empty:
            secondary_plot_func(self.canvas.ax2, df)
        else:
            self.canvas.ax2.set_title("No data available for secondary plot")

        self.canvas.draw()

    def _fill_table(self, df: pd.DataFrame):
        """
        Fills the QTableWidget with the contents of the DataFrame.
        """
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
# ThreeDTabWidget: 3D Covariance Surface visualization
# -----------------------------------------------------------------------------
class ThreeDTabWidget(QWidget):
    """
    3D tab displays a VTKWidget with the covariance surface. A scalar bar is added for color interpretation.
    """
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title

        layout = QVBoxLayout(self)
        self.lbl_title = QLabel(f"<h2>{title}</h2>")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        self.lbl_title.setStyleSheet("font-size: 16px; margin: 5px;")
        self.lbl_title.setMaximumHeight(40)
        layout.addWidget(self.lbl_title)

        self.vtk_widget = VTKWidget()
        self.vtk_widget.setMinimumHeight(600)
        layout.addWidget(self.vtk_widget)

        self.lbl_comment = QLabel("Data extracted from the IAEA API – https://nds.iaea.org/")
        self.lbl_comment.setAlignment(Qt.AlignCenter)
        self.lbl_comment.setMaximumHeight(25)
        layout.addWidget(self.lbl_comment)

    def load_data(self, df):
        """
        Loads the 3D covariance surface using the provided DataFrame.
        """
        self.vtk_widget.renderer.RemoveAllViewProps()

        if df.empty or "x" not in df.columns or "y" not in df.columns or "covariance" not in df.columns:
            self.vtk_widget._show_default()
            return

        grid_output = create_covariance_vtk_surface(df)

        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputData(grid_output)
        cmin = df["covariance"].min()
        cmax = df["covariance"].max()

        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        lut.Build()

        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(cmin, cmax)
        mapper.ScalarVisibilityOn()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        self.vtk_widget.renderer.AddActor(actor)

        scalarBar = vtk.vtkScalarBarActor()
        scalarBar.SetLookupTable(lut)
        scalarBar.SetTitle("Covariance")
        scalarBar.SetNumberOfLabels(5)
        self.vtk_widget.renderer.AddActor2D(scalarBar)

        self.vtk_widget.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

def create_covariance_vtk_surface(df):
    """
    Constructs a StructuredGrid and applies WarpScalar to the covariance data,
    creating a 3D surface where the Z-axis represents the covariance values.
    """
    x_unique = np.sort(df["x"].unique())
    y_unique = np.sort(df["y"].unique())
    nx = len(x_unique)
    ny = len(y_unique)

    cov_dict = {}
    for idx, row in df.iterrows():
        cov_dict[(row["x"], row["y"])] = row["covariance"]

    sgrid = vtk.vtkStructuredGrid()
    sgrid.SetDimensions(nx, ny, 1)

    points = vtk.vtkPoints()
    scalars = vtk.vtkFloatArray()
    scalars.SetNumberOfComponents(1)
    scalars.SetName("Covariance")

    for i in range(nx):
        for j in range(ny):
            x_val = x_unique[i]
            y_val = y_unique[j]
            cov = cov_dict.get((x_val, y_val), 0.0)
            points.InsertNextPoint(x_val, y_val, cov)
            scalars.InsertNextValue(cov)

    sgrid.SetPoints(points)
    sgrid.GetPointData().SetScalars(scalars)

    warp = vtk.vtkWarpScalar()
    warp.SetInputData(sgrid)
    warp.SetScaleFactor(0.1)
    warp.Update()

    return warp.GetOutput()

# -----------------------------------------------------------------------------
# Main application window
# -----------------------------------------------------------------------------
class MainWindow(QMainWindow):
    """
    Main window of the application with tabs for:
      - EXFOR
      - ENDF Cross Section
      - ENDF Fission Yield
      - ENDF Radioactive Decay
      - ENDF Covariance
      - 3D Covariance Surface
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nuclear Data Visualization – IAEA APIs")
        self.resize(1300, 900)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        self.btn_reload = QPushButton("Reload and Analyze Data")
        self.btn_reload.clicked.connect(self.load_all_data)
        main_layout.addWidget(self.btn_reload)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Define tabs
        self.tab_exfor = DataTabWidget("EXFOR: Al-27 (n,a)")
        self.tab_endf_cs = DataTabWidget("ENDF Cross Section: PB-204 (n,g)")
        self.tab_endf_fy = DataTabWidget("ENDF Fission Product Yield: U-238")
        self.tab_endf_decay = DataTabWidget("ENDF Radioactive Decay: N-16")
        self.tab_endf_cov = DataTabWidget("ENDF Covariance: PB-204")
        self.tab_3d = ThreeDTabWidget("3D Covariance Surface")

        self.tabs.addTab(self.tab_exfor, "EXFOR")
        self.tabs.addTab(self.tab_endf_cs, "ENDF - Cross Section")
        self.tabs.addTab(self.tab_endf_fy, "ENDF - Fission Yield")
        self.tabs.addTab(self.tab_endf_decay, "ENDF - Radioactive Decay")
        self.tabs.addTab(self.tab_endf_cov, "ENDF - Covariance")
        self.tabs.addTab(self.tab_3d, "3D")

        self.load_all_data()

    def load_all_data(self):
        """
        Downloads and displays the data in each tab.
        """
        self.status_bar.showMessage("Downloading and processing data...", 3000)

        # EXFOR
        exfor_df = get_exfor_data()
        self.tab_exfor.load_data(
            exfor_df,
            plot_func=plot_exfor_data,
            secondary_plot_func=plot_exfor_data_log
        )

        # ENDF Cross Section
        endf_cs_df = get_endf_cross_section()
        self.tab_endf_cs.load_data(
            endf_cs_df,
            plot_func=plot_endf_cross_section,
            secondary_plot_func=plot_endf_cross_section_log
        )

        # ENDF Fission Product Yield
        endf_fy_df = get_endf_fission_yield()
        self.tab_endf_fy.load_data(
            endf_fy_df,
            plot_func=plot_endf_fission_yield,
            secondary_plot_func=plot_endf_fission_yield_secondary
        )

        # ENDF Radioactive Decay
        endf_decay_df = get_endf_radioactive_decay()
        self.tab_endf_decay.load_data(
            endf_decay_df,
            plot_func=plot_endf_radioactive_decay,
            secondary_plot_func=plot_endf_radioactive_decay_secondary
        )

        # ENDF Covariance
        endf_cov_df = get_endf_covariance()
        self.tab_endf_cov.load_data(
            endf_cov_df,
            plot_func=plot_covariance,
            secondary_plot_func=plot_covariance_secondary
        )

        # 3D Covariance Surface
        self.tab_3d.load_data(endf_cov_df)

        self.status_bar.showMessage("Data loaded.", 3000)

def main():
    """
    Main entry point of the application.
    """
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
