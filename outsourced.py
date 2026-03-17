"""
### English

Docstring for capstone project of the course Data Scientist (IHK), file: outsourced.py

The objective of this final project is to develop a machine leanring model that bridges the gap between civil engineering domain knowledge and data science expertise, applying the course content to earn the respective certificate. The primary focus lies on demonstrating the practical application of the concepts taught throughout the course.

Due to limited information regarding the data generation process in the cited sources [1, 2], the partly restricted access to the referenced literature, and the time constraints for an extensive literature review, this project operates within a relatively low information density. The model developed herein is intended solely for academic demonstration pruposes within the scope of this final project and must not be used for any other applications. For real-world implementation, it would be mandatory to define operational boundaries, perform rigorous validation tests, and ensure full transparency regarding potential biases.

Accordingly, the author assumes no liability for any damages resulting from improper use or reliance on the model's outputs. Use of the model and any results derived from it is strictly at the users' own risk.

The following section contains several helper functions to maintain a 'clean' notebook.

---

### Deutsch

Docstring für Abschlussprojekt des Kurses Data Scientist (IHK), Datei: outsourced.py

Das Ziel dieses Abschlussprojekts soll die Entwicklung eines Machine-Learning Modells sein, das das Domänenwissen eines Bauingenieurs mit dem eines Data Scientists verbindet und die Kursinhalte aufgreift, um das zugehörige Zertifikat zu erlangen. Der Fokus liegt also klar darin, die Kursinhalte aufzugreifen und anzuwenden.
Da die Quellen [1, 2] wenig Informationen zu den Daten selbst bzw. deren Zustandekommen enthalten, die referenzierte Literatur teilweise nicht frei zugänglich ist und auch die Zeit für eine extensive Literaturrecherche zu knapp bemessen ist, wird mit dieser vergleichsweise geringen Informationsdichte gearbeitet. Das hier erarbeitete Modell dient lediglich akademischen Demonstrationszwecken im Rahmen des Abschlussprojekts für den genannten Kurs und darf nicht anderweitig eingesetzt werden. Für den realen Einsatz müssten mindestens die Anwendungsgrenzen erarbeitet, Validierungsversuche durchgeführt und jeglicher Bias transparent herausgestellt werden.
Der Autor übernimmt daher keine Haftung für Schäden, die aus einer unsachgemäßen Verwendung oder dem Vertrauen auf die Modellergebnisse resultieren. Die Nutzung des Modells wie auch damit erzielte Ergebnisse erfolgen auf eigene Gefahr.

Nachfolgend werden einige Hilfsfunktionen aufgeführt, um das Notebook selbst "sauberer" zu halten.
"""


# ---------------------------------------------
# region     IMPORT OF LIBRARIES
# ---------------------------------------------

# general
import os
import re

# types
from typing import cast, Literal, List, Dict, Any, Optional

import numpy.typing as npt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.text import Text
from matplotlib.projections.polar import PolarAxes
from matplotlib.colors import Colormap
# from matplotlib.legend import Legend

from plotly.graph_objects import Figure as PlotlyFigure # type: ignore


# math and data
import math
import numpy as np
import pandas as pd  # type: ignore
from dataclasses import dataclass

# data science
from sklearn.base import BaseEstimator, RegressorMixin  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.manifold import TSNE  # type: ignore
from sklearn.cluster import HDBSCAN  # type: ignore
from sklearn.model_selection import KFold, cross_validate  # type: ignore
from sklearn.metrics import make_scorer  # type: ignore
import xgboost as xgb  # type: ignore
import optuna  # type: ignore

# visualization
import shap  # type: ignore
import seaborn as sns  # type: ignore
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import HTML, display # type: ignore
from adjustText import adjust_text  # type: ignore
from contextlib import redirect_stdout  # to silence adjustText output
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from plotly.io import show # type: ignore
from cycler import cycler


# endregion IMPORT OF LIBRARIES


# ---------------------------------------------
# region      TYPE DEFINITIONS
# ---------------------------------------------

type InitialRealParams = Literal[
    "beam_number",
    "beam_length__m",
    "concrete_area__mm2",
    "concrete_cover__mm",
    "steel_area__mm2",
    "frp_area__mm2",
    "insulation_thickness__mm",
    "insulation_depth__mm",
    "concrete_compr_strength__MPa",
    "steel_yield_strength__MPa",
    "steel_youngs_modulus__MPa",
    "frp_tens_strength__MPa",
    "frp_youngs_modulus__MPa",
    "transition_temp_glass__deg",
    "insulation_thermal_conductivity__W_per_mK",
    "insulation_specific_heat_cap__J_per_degm3",
    "total_load__kN",
    "load_ratio__percent",
    "test_end_criterion",
    "fire_resistance__min",
    "deflection_at_failure__mm",
]
"""
type definition of the initial column names after loading the real dataset (for autocomplete only)
"""

type InitialSynthParams = Literal[
    "beam_number",
    "beam_length__mm",
    "concrete_area__mm2",
    "concrete_cover__mm",
    "steel_area__mm2",
    "frp_area__mm2",
    "insulation_thickness__mm",
    "insulation_depth__mm",
    "concrete_compr_strength__MPa",
    "steel_yield_strength__MPa",
    "steel_youngs_modulus__MPa",
    "frp_tens_strength__MPa",
    "frp_youngs_modulus__MPa",
    "transition_temp_glass__deg",
    "insulation_thermal_conductivity__W_per_mmdeg",
    "insulation_specific_heat_cap__J_per_degmm3",
    "total_load__kN",
    "load_ratio__percent",
    "test_end_criterion",
    "fire_resistance__min",
    "deflection_at_failure__mm",
    "limit_state",
    "initial_capacity__kNm",
    "final_capacity__kNm",
]
"""
type definition of the initial column names after loading the synthetic dataset (for autocomplete only)
"""

type ProjectParams = str | InitialRealParams | InitialSynthParams

# endregion


# ---------------------------------------------
# region    CLASSES
# ---------------------------------------------


@dataclass
class ColorTheme:
    """
    Klasse zur Verwaltung und automatischen Anwendung von Matplotlib-Themes.
    """

    # Zwingend erforderlich
    mode: Literal["light", "dark"]
    color_cycle: List[str]
    """Es müssen mindestens 3 Farben als Hex-String enthalten sein, bspw. ['#333333', '#AAAAAA', '#EEEEEE']"""

    # Optionale Overrides (wenn "", greifen die mode-Defaults)
    fig_bg: str = ""
    ax_bg: str = ""
    title_color: str = ""
    text_color: str = ""
    grid_color: str = ""

    # Semantische Farben
    primary_color: str = ""
    secondary_color: str = ""
    tertiary_color: str = ""
    outlier_color: str = "#E74C3C"  # Default: Ein warnendes Rot/Orange

    grid_linestyle: str = "--"

    grid_alpha: float = 0.3
    primary_alpha: float = 0.5
    secondary_alpha: float = 0.4

    def __post_init__(self):
        """Wird nach __init__ aufgerufen. Setzt intelligente Defaults basierend auf dem Modus."""

        if self.mode == "dark":
            # Defaults für Dark Mode
            self.fig_bg = self.fig_bg or "#121212"
            self.ax_bg = self.ax_bg or "#121212"
            self.title_color = self.title_color or "#FFFFFF"
            self.text_color = self.text_color or "#CFCFCF"
            self.grid_color = self.grid_color or "#444444"

        else:
            # Defaults für Light Mode
            self.fig_bg = self.fig_bg or "#FFFFFF"
            self.ax_bg = self.ax_bg or "#FFFFFF"
            self.title_color = (
                self.title_color or "#111111"
            )  # Fix: Vorher #FFFFFF
            self.text_color = self.text_color or "#333333"
            self.grid_color = self.grid_color or "#E0E0E0"

        # Wenn keine Akzentfarbe definiert wurde, nehmen wir die leuchtendste (oft die erste)
        self.primary_color = self.primary_color or self.color_cycle[0]
        self.secondary_color = self.secondary_color or self.color_cycle[1]
        self.tertiary_color = self.tertiary_color or self.color_cycle[2]

    def get_cont_palette(
        self, n_colors: Optional[int] = None, as_cmap: bool = False
    ):
        """
        Gibt je nach Bedarf eine Liste von Farben oder eine Colormap zurück.
        """
        if as_cmap:
            return LinearSegmentedColormap.from_list(
                "custom_cont", self.color_cycle
            )

        if n_colors is not None:
            return sns.color_palette(self.color_cycle, n_colors)

        return self.color_cycle

    def to_rc_dict(self) -> Dict[str, Any]:
        """Übersetzt die Klassenattribute in ein umfassendes Matplotlib rcParams Dictionary."""
        return {
            # === Figure & Layout ===
            "figure.facecolor": self.fig_bg,
            "figure.edgecolor": self.fig_bg,
            "savefig.facecolor": self.fig_bg,
            "savefig.edgecolor": self.fig_bg,
            "figure.dpi": 150,  # Schärfere Darstellung in Notebooks
            "figure.autolayout": True,  # tightLayout wird angewendet, wenn True (nicht kompatibel mit constrained_layout)
            "figure.constrained_layout.use": False,  # Verhindert abgeschnittene Labels/Titel, wenn True
            # === Achsen & Hintergrund ===
            "axes.facecolor": self.ax_bg,
            "axes.edgecolor": self.grid_color,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,  # Gitterlinien immer hinter den Graphen
            # === Grid ===
            "axes.grid": True,
            "grid.color": self.grid_color,
            "grid.linestyle": self.grid_linestyle,
            "grid.alpha": self.grid_alpha,
            "grid.linewidth": 0.8,
            # === Texte & Typografie ===
            "font.family": "sans-serif",
            "font.size": 11,
            "text.color": self.text_color,
            "axes.labelcolor": self.text_color,
            "axes.titlecolor": self.title_color,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.titlepad": 12.0,
            "axes.labelsize": 11,
            "axes.labelpad": 6.0,
            # === Ticks ===
            "xtick.color": self.text_color,
            "ytick.color": self.text_color,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            # === Farbzyklus (Lines & Scatter) ===
            "axes.prop_cycle": cycler(color=self.color_cycle),
            # === Line Plots ===
            "lines.linewidth": 2.5,
            "lines.markersize": 7,
            "lines.solid_capstyle": "round",  # Macht Linienenden weicher
            "lines.solid_joinstyle": "round",
            # === Patches (Barplots, Histograms, Pie Charts) ===
            "patch.linewidth": 1.0,
            "patch.edgecolor": self.fig_bg,  # Trennt aneinandergrenzende Bars optisch
            "patch.force_edgecolor": True,
            "patch.antialiased": True,
            "hist.bins": "auto",
            # === Scatter Plots ===
            "scatter.edgecolors": "none",
            "scatter.marker": "o",
            # === Boxplots ===
            "boxplot.showmeans": False,
            "boxplot.boxprops.color": self.text_color,
            "boxplot.boxprops.linewidth": 1.5,
            "boxplot.whiskerprops.color": self.text_color,
            "boxplot.whiskerprops.linewidth": 1.5,
            "boxplot.capprops.color": self.text_color,
            "boxplot.capprops.linewidth": 1.5,
            "boxplot.medianprops.color": self.primary_color,  # Median sticht hervor
            "boxplot.medianprops.linewidth": 2.0,
            "boxplot.flierprops.color": self.outlier_color,
            "boxplot.flierprops.markeredgecolor": self.outlier_color,
            "boxplot.flierprops.markerfacecolor": "none",
            # === Errorbars ===
            "errorbar.capsize": 3.0,
            # === Legend ===
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.facecolor": self.fig_bg,
            "legend.edgecolor": self.grid_color,
            "legend.fontsize": 10,
            "legend.title_fontsize": 11,
            "legend.borderpad": 0.5,
            "legend.labelspacing": 0.4,
        }

    def apply(self):
        """Wendet das Theme global auf Matplotlib an."""
        mpl.rcParams.update(self.to_rc_dict())


class CustomSplitter:
    def __init__(self, n_splits: int = 5, random_state: int = 1234):
        self.kfold = KFold(n_splits, shuffle=True, random_state=random_state)

    def split(self, X: pd.DataFrame, y=None, groups=None):
        """
        Erstellt Indices, mit denen die Gesamtdaten (synth + real) in Trainings- und Testset aufgeteilt werden,
        sodass im Test-Set nur Realdaten vorhanden sind, bei denen das Versuchsende aufgrund Versagen der
        Balken festgestellt wurde. Im Trainingsset landen die verbleibenden Beobachtungen. Es ist sicherzustellen,
        dass die Indizes beim Konkatenieren von synth + real eindeutig sind.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Der konkatenierte Datensatz (synth + real). Vorausgesetzt sind die Spalten `is_real` und `test_end_criterion`

        y : array-like of shape (n_samples,)
            Hier nicht verwendet

        groups : array-like of shape (n_samples,), default=None
            Hier nicht verwendet

        Yields
        ------
        train_idx : ndarray
            The training set indices for that split.

        test_idx : ndarray
            The testing set indices for that split.
        """

        # Positionen der synthetischen Daten (alle fürs Training)
        train_synth_pos = np.where([X["is_real"] == 0])[0]

        # Positionen der realen Daten fürs Training (alle Versuche, die ohne Versagen beendet wurden)
        train_real_pos = np.where(
            (X["is_real"] == 1) & (X["test_end_criterion"] == 1)
        )[0]

        # Positionen der realen Daten zum Splitten
        prelim_real_pos = np.where(
            (X["is_real"] == 1) & (X["test_end_criterion"] == 0)
        )[0]

        # Schleife über KFolds-Splits der Realdaten
        for train_ind, test_ind in self.kfold.split(prelim_real_pos):
            add_train_real_pos = prelim_real_pos[train_ind]
            test_real_pos = prelim_real_pos[test_ind]

            if len(test_real_pos) == 0:
                print(
                    "HINWEIS: Es waren keine Testdaten im aktuellen KFold-Split. Er wird übersprungen."
                )
                continue

            # Realdaten müssen an den Anfang, damit sie bspw. bei der Learning-Curve in den Samples auch immer mit dabei sind
            final_train_pos = np.concatenate(
                [train_real_pos, add_train_real_pos, train_synth_pos]
            )
            yield final_train_pos, test_real_pos

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.kfold.get_n_splits(X, y, groups)


class OptimizedFireXGB(BaseEstimator, RegressorMixin):
    """
    Ein scikit-learn kompatibler Wrapper für das beste XGBoost-Modell
    aus der Optuna-Studie. Übernimmt automatisch Feature Selection
    und Monotone Constraints basierend auf den best_params.
    """

    def __init__(
        self,
        params: dict,
        verbose: bool = False,
    ):

        self.verbose = verbose
        self.params = params
        self.model: xgb.XGBRegressor | None = None
        self.selected_features_: list[str] | None = None

        self.must_have_feats = [
            "load_ratio__percent",
            "stress_proxy__Pa",
            "ax_dist__m",
            "reinforcement_ratio",
            "section_modulus__m3",
            "thermal_resistance__m2K_per_W",
        ]

        self.mono_constraints = {
            "steel_yield_strength__Pa": 1,
            "frp_tens_strength__Pa": 1,
            "load_ratio__percent": -1,
            "section_modulus__m3": 1,
            "stress_proxy__Pa": -1,
            "width__m": 1,
            "insulation_ratio__1_per_m": 1,
            "section_factor__1_per_m": -1,
            "slenderness__1_per_m": -1,
            "reinforcement_ratio": 1,
            "ax_dist__m": 1,
            "thermal_resistance__m2K_per_W": 1,
            "thermal_diffusivity__m2_per_s": -1,
        }

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: npt.ArrayLike | None = None,
    ):
        """
        Trainiert das Modell.
        X: pd.DataFrame mit allen potenziellen Features
        y: pd.Series mit den Targets (bereits markiert für Zensur!)
        sample_weights: pd.Series oder np.array mit den Gewichten
        """
        # 1. Feature Selection ausführen
        self.selected_features_ = self.must_have_feats.copy()
        for feat in X.columns:
            if feat not in self.must_have_feats and self.params.get(
                f"use_{feat}", False
            ):
                self.selected_features_.append(feat)

        X_train_selected = X[self.selected_features_]

        # 2. Monotone Constraints filtern
        selected_mc = None
        if self.params.get("use_constraints", False):
            selected_mc = {
                f: self.mono_constraints[f]
                for f in self.selected_features_
                if f in self.mono_constraints
            }

        # 3. XGBoost Parameter zusammenbauen
        xgb_params = {
            "objective": custom_mse_objective,
            "tree_method": "hist",
            "device": "gpu",
            "monotone_constraints": selected_mc,
            "base_score": 60,
            "eval_metric": "rmse",
            "max_depth": self.params["max_depth"],
            "learning_rate": self.params["learning_rate"],
            "n_estimators": self.params["n_estimators"],
            "reg_lambda": self.params["reg_lambda"],
            "reg_alpha": self.params["reg_alpha"],
            "colsample_bytree": self.params["colsample_bytree"],
            "subsample": self.params["subsample"],
        }

        # 4. Modell instanziieren und trainieren
        self.model = xgb.XGBRegressor(**xgb_params)
        self.model.fit(
            X_train_selected,
            y,
            sample_weight=sample_weight,
            verbose=self.verbose,
        )

        # Scikit-learn Konvention: fit() gibt das Objekt selbst zurück
        return self

    def predict(self, X):
        """
        Macht Vorhersagen.
        X: pd.DataFrame mit allen Features. Die Klasse filtert automatisch die richtigen heraus.
        """
        # Scikit-learn Best Practice: Prüfen ob fit() schon aufgerufen wurde
        if self.model is None or self.selected_features_ is None:
            raise ValueError(
                "Das Modell muss zuerst mit .fit() trainiert werden!"
            )

        # Nur die Features nutzen, die beim Training ausgewählt wurden
        X_test_selected = X[self.selected_features_]

        return self.model.predict(X_test_selected)


# endregion CLASSES


# ---------------------------------------------
# region    CONSTANTS
# ---------------------------------------------

DATASET_FILE = "Dataset_FireResistanceofFRP-StrengthenedBeams_PB_Ver6.0.xlsx"

END_CRITERIONS = ["Versagen", "Ende Brand"]
"""
list of end criteria
"""

COL_NAMES_R: list[InitialRealParams] = [
    "beam_number",
    "beam_length__m",  # NOTE: here the naming differs from the other dataset
    "concrete_area__mm2",
    "concrete_cover__mm",
    "steel_area__mm2",
    "frp_area__mm2",
    "insulation_thickness__mm",
    "insulation_depth__mm",
    "concrete_compr_strength__MPa",
    "steel_yield_strength__MPa",
    "steel_youngs_modulus__MPa",
    "frp_tens_strength__MPa",
    "frp_youngs_modulus__MPa",
    "transition_temp_glass__deg",
    "insulation_thermal_conductivity__W_per_mK",  # NOTE: here the naming differs from the other dataset
    "insulation_specific_heat_cap__J_per_degm3",  # NOTE: here the naming differs from the other dataset
    "total_load__kN",
    "load_ratio__percent",
    "test_end_criterion",
    "fire_resistance__min",
    "deflection_at_failure__mm",
]
"""
list of new column names in order of the original headers from sheet `01_FireTestData` (real data)
"""

COL_NAMES_S: list[InitialSynthParams] = [
    "beam_number",
    "beam_length__mm",  # NOTE: here the naming differs from the other dataset
    "concrete_area__mm2",
    "concrete_cover__mm",
    "steel_area__mm2",
    "frp_area__mm2",
    "insulation_thickness__mm",
    "insulation_depth__mm",
    "concrete_compr_strength__MPa",
    "steel_yield_strength__MPa",
    "steel_youngs_modulus__MPa",
    "frp_tens_strength__MPa",
    "frp_youngs_modulus__MPa",
    "transition_temp_glass__deg",
    "insulation_thermal_conductivity__W_per_mmdeg",  # NOTE: here the naming differs from the other dataset
    "insulation_specific_heat_cap__J_per_degmm3",  # NOTE: here the naming differs from the other dataset
    "total_load__kN",
    "load_ratio__percent",
    "test_end_criterion",
    "fire_resistance__min",
    "deflection_at_failure__mm",
    # NOTE: additional columns for synth. dataset
    "limit_state",
    "initial_capacity__kNm",
    "final_capacity__kNm",
]
"""
list of new column names in order of the original headers from sheet `02_NumericaModelData` (synth. data)
"""

CBF_THEME = ColorTheme(
    mode="dark",
    # Hintergrund-Overrides
    fig_bg="#141C2B",
    ax_bg="#141C2B",
    
    # Text- und Gitterfarben
    title_color="#FFFFFF",
    text_color="#E2E8F0",
    grid_color="#474747",
    
    # Standard-Farbzyklus
    color_cycle=[
        "#5CA073",
        "#F5BE49",
        "#E09351",
        "#CE9070",
        "#B75347",
        "#6D2F20",
        "#224B5E",
    ],
    
    # Semantische Farben
    primary_color="#F5BE49",
    secondary_color="#224B5E",
    tertiary_color="#6D2F20",
    outlier_color="#E05F92",
    
    grid_alpha=0.5,
    grid_linestyle="--",
)


ACADIA_MIDNIGHT = ColorTheme(
    mode="dark",
    color_cycle=[
        "#EADB6EFF",
        "#306CAFFF",
        "#164555FF",
        "#5ACC8CFF",
        "#6090A4FF",
        "#ECC0A1FF",
        "#583B5DFF",
    ],
    outlier_color="#984136FF",
    fig_bg="#000F1DFF",
    ax_bg="#000F1DFF",
    title_color="#FFF2D7FF",
)

REDS_MIDNIGHT = ColorTheme(
    mode="dark",
    color_cycle=[
        "#C26A7AFF",
        "#ECC0A1FF",
        "#6A3034FF",
        "#984136FF",
        "#FF9677FF",
        "#F0F0E4FF",
    ],
    outlier_color="#79A3B3FF",
    fig_bg="#000F1DFF",
    ax_bg="#000F1DFF",
    title_color="#F0F0E4FF",
)


VIBRANT_MIDNIGHT = ColorTheme(
    mode="dark",
    color_cycle=[
        "#79A3B3FF",
        "#FF9677FF",
        "#FFCDA7FF",
        "#164555FF",
        "#EADB6EFF",
        "#456671FF",
    ],
    outlier_color="#C26A7AFF",
    fig_bg="#000F1DFF",
    ax_bg="#000F1DFF",
    title_color="#F0F0E4FF",
)

# endregion CONSTANTS


# ---------------------------------------------
# region    DATA LOADING
# ---------------------------------------------


def check_for_dataset():
    """
    checks for the neccessary excel file
    """
    if not os.path.exists(DATASET_FILE):
        print(
            "Dataset not found, please download it an place it next to the notebook"
        )
        print("https://data.mendeley.com/datasets/3c2szhbdn5/6")


def load_real_data() -> pd.DataFrame:
    # Versuchsdaten laden
    df_real = pd.read_excel(
        io=DATASET_FILE,
        sheet_name="01_FireTestData",
        names=COL_NAMES_R,
        skiprows=1,
        nrows=50,
    )
    return df_real


def load_synth_data() -> pd.DataFrame:
    # synthetische Daten laden
    df_synth = pd.read_excel(
        io=DATASET_FILE,
        sheet_name="02_NumericaModelData",
        names=COL_NAMES_S,
        skiprows=1,
        nrows=21384,
    )
    return df_synth


# endregion DATA LOADING


# ---------------------------------------------
# region    EXPLORATORY DATA ANALYSIS
# ---------------------------------------------

def make_safe_filename(title: str):
    safe_title = title.lower().strip()
    umlaute = {'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'ß': 'ss'}
    for k, v in umlaute.items():
        safe_title = safe_title.replace(k, v)
    
    safe_title = safe_title.replace(' ', '-')
    safe_title = re.sub(r'[^a-z0-9_\.\-\+]', '', safe_title)
    # safe_title = re.sub(r'_+', '_', safe_title)
    return safe_title


def save_and_load_img(title: str):
    safe_title = make_safe_filename(title)
    filename = f"plots/{safe_title}.png"
    plt.savefig(filename)
    plt.close() # Inline-Output unterdrücken
    # Bild explizit mit Alt-Text wieder anzeigen
    bild_html = f'<img src="{filename}" alt="{title}">'
    display(HTML(bild_html))


def get_value_counts(df: pd.DataFrame, col: ProjectParams) -> pd.DataFrame:
    df_count = df[col].value_counts().to_frame(name="Anzahl")
    df_count["Anteil"] = (
        df_count["Anzahl"] / df_count["Anzahl"].sum() * 100
    ).round(2)
    return df_count


def get_translation(value: str):
    """
    translates the technical terms to German

    :param value: the value to be translated
    :type value: str
    """
    names_dict = {
        "beam_number": "Balkennummer",
        "beam_length": "Balkenlänge",
        "concrete_area": "Betonfläche",
        "concrete_cover": "Betondeckung",
        "steel_area": "Bewehrungsfläche",
        "frp_area": "FRP-Fläche",
        "insulation_thickness": "Dicke der Dämmung",
        "insulation_depth": "Höhe der Dämmung",
        "concrete_compr_strength": "Betonfestigkeit",
        "steel_yield_strength": "Streckgrenze Bewehrung",
        "steel_youngs_modulus": "E-Modul Bewehrung",
        "frp_tens_strength": "Zugfestigkeit FRP",
        "frp_youngs_modulus": "E-Modul FRP",
        "transition_temp_glass": "Glasübergangstemperatur",
        "insulation_thermal_conductivity": "Wärmeleitfähigkeit Dämmung",
        "insulation_specific_heat_cap": "Spez. Wärmekapazität Dämmung",
        "total_load": "Gesamtlast",
        "load_ratio": "Lastverhältnis",
        "test_end_criterion": "Grund Versuchsende",
        "fire_resistance": "Feuerwiderstandsdauer",
        "deflection_at_failure": "Enddurchbiegung",
        "limit_state": "Grund Versuchsende (Literal)",
        "initial_capacity": "Ausgangstragfähigkeit",
        "final_capacity": "Endtragfähigkeit",
        "width": "Balkenbreite",
        "height": "Balkenhöhe",
        "reinforcement_ratio": "Bewehrungsgrad",
        "thermal_resistance": "Wärmedurchlasswiderstand Dämmung",
        "thermal_diffusivity": "Temperaturleitfähigkeit Dämmung",
        "slenderness": "Schlankheit",
        "stress_proxy": "Ersatzspannung",
        "insulation_ratio": "Ersatzverhältnis seitl. Dämmung",
        "ax_dist": "Achsabstand Bewehrung",
        "section_modulus": "Widerstandsmoment",
        "is_real": "Aus Realdatensatz (ja/nein)",
        "section_factor": "Profilfaktor",
    }
    splt = value.split("__")[0]
    if splt in names_dict.keys():
        return names_dict[splt]
    else:
        return value


def plot_distribution(
    df: pd.DataFrame,
    theme: ColorTheme = ACADIA_MIDNIGHT,
    fig_pre: str = "Abbildung",
    fig_suf: str = ":",
    title: str = "Verteilung der Parameter (Hist + KDE)",
    hue_col: str | None = None,
    legend_texts: list[str] | None = None,
    num_cols: int = 3,
):
    """
    Plots the distribution of all parameters in dataframe `df`. The first subplot is empty and is used for the legend.

    :param df: the dataframe of the data to be plotted
    :type df: pd.DataFrame

    :param theme: The ColorTheme instance to access semantic colors if needed.
    :type theme: ColorTheme

    :param fig_pre: prefix for numbering and referencing the plots, e.g. **"Figure 1."** + plot number + suffix (default: "Abbildung")
    :type fig_pre: str

    :param fig_suf: suffix for numbering, e.g. prefix + plot number + **":"** (default: ":")
    :type fig_suf: str

    :param title: title of the whole figure (default: "Verteilung der Parameter (Hist + KDE)")
    :type title: str

    :param hue_col: the name of the column, which should be used for the hue (default: "test_end_criterion")
    :type hue_col: str | None

    :param legend_texts: the entries for the legend (default: `END_CRITERIONS`)
    :type legend_texts: list[str] | None

    :param num_cols: number of columns in the plot (default: 3)
    :type num_cols: int
    """

    hue_col = hue_col or "test_end_criterion"
    # Fallback für END_CRITERIONS, falls nicht global definiert
    legend_texts = legend_texts or ["Versagen", "Ende Brand"]

    df_plt = df.select_dtypes(include="number")

    all_cols = df_plt.columns.tolist()
    len_cols = len(df_plt.columns) + 1  # +1 für Legenden-Plot
    num_rows = math.ceil(len_cols / num_cols)

    # Figure erstellen (Hintergrundfarbe wird durch rcParams gesteuert)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 4))

    # Sup-Titel (Schriftfarbe wird durch rcParams gesteuert)
    fig.suptitle(title, fontsize=20, y=1.02, color=theme.title_color)

    axes_flat = cast(list[Axes], axes.flatten())

    # --- Dummy-Plot zur Platzierung der zentralen Legende ---
    ax_lgnd = axes_flat[0]

    # Der Histplot nutzt automatisch den axes.prop_cycle aus dem Theme
    hnd = sns.histplot(
        data=df_plt,
        x=all_cols[0],
        hue=hue_col,
        kde=True,
        ax=ax_lgnd,
        alpha=theme.primary_alpha,
        element="step",
    )

    lgnd = hnd.legend_
    if lgnd is not None:
        handles = cast(list[Artist], lgnd.legend_handles)
        labels = (
            legend_texts
            if legend_texts and len(legend_texts) == len(handles)
            else [t.get_text() for t in lgnd.get_texts()]
        )

        ax_lgnd.clear()
        ax_lgnd.axis("off")

        # Neue Legende aufbauen
        ax_lgnd.legend(
            handles,
            labels,
            loc="upper left",
            title=get_translation(
                hue_col
            ),  # Setzt get_translation-Funktion voraus
            fontsize=11,
            frameon=False,
        )

        # Das Styling der Legendentexte und des Rahmens passiert automatisch
        # über die Matplotlib-Defaults (rcParams)!

    else:
        # Fallback falls oben nichts gefunden wurde
        ax_lgnd.axis("off")

    ax_lgnd.set_title(
        "Legende für Plots",
        loc="left",
        fontweight="bold",
    )

    # --- Schleife über alle Spalten ---
    for i, col in enumerate(all_cols):
        ax = axes_flat[i + 1]

        # Plot erstellen
        hnd = sns.histplot(
            data=df_plt,
            x=col,
            hue=hue_col,
            kde=True,
            alpha=theme.secondary_alpha,
            ax=ax,
            element="step",
            legend=False,
        )

        ax.set_title(
            f"{fig_pre}{i + 1}{fig_suf} {get_translation(col)}",
            loc="left",
            fontsize=12,
            color=theme.text_color,
        )
        ax.set_xlabel(col)
        ax.set_ylabel("Anzahl")

    # --- Leere Plots verstecken ---
    for j in range(i + 2, num_rows * num_cols):  # i + 2 für extra legenden plot
        axes_flat[j].axis("off")

    # plt.tight_layout()
    # plt.show() # removed for file saving and using IPython display
    save_and_load_img(fig_pre + "x: " + title)


def scatter_data(
    df: pd.DataFrame,
    y: ProjectParams = "fire_resistance__min",
    fig_pre: str = "Abbildung",
    fig_suf: str = ":",
    title: str = "Scatterplots der Parameter",
    theme: ColorTheme = ACADIA_MIDNIGHT,
    hue_col: str | None = None,
    style_col: str | None = None,
    markers: dict[int | str, str] | None = None,
    palette: list[str] | str | Colormap | None = None,
    legend_texts: list[str] | None = None,
    num_cols: int = 3,
):
    """
    Creates Scatterplots for all parameters in datafram `df`. The first subplot is reserved for the legend.

    :param df: the dataframe to be plotted
    :type df: pd.DataFrame

    :param y: name of the target columns (default: "fire_resistance__min")
    :type y: str

    :param fig_pre: prefix for numbering and referencing the plots, e.g. **"Figure 1."** + plot number + suffix (default: "Abbildung")
    :type fig_pre: str

    :param fig_pre: suffix for numbering, e.g. prefix + plot number + **":"** (default: ":")
    :type fig_pre: str

    :param title: title of the whole figure (default: "Verteilung der Parameter (Hist + KDE)")
    :type title: str

    :param hue_col: the name of the column, which should be used for the hue (default: "load_ratio__percent")
    :type hue_col: str | None

    :param style_col: the name of the column, which should be used for the style symbols (default: None)
    :type style_col: str | None

    :param markers: markers as dictionary for style_col (default: None)
    :type markers: dict[int|str, str] | None

    :param palette: color palette (default: "viridis")
    :type palette: str

    :param legend_texts: the entries for the legend (default: `END_CRITERIONS`)
    :type legend_texts: list[str] | None

    :param num_cols: number of columns in the plot (default: 3)
    :type num_cols: int
    """

    hue_col = hue_col or "load_ratio__percent"
    legend_texts = legend_texts or []

    df_plt = df.select_dtypes(include="number")

    all_cols = df_plt.columns.tolist()
    len_cols = len(df_plt.columns) + 1  # +1 für Legenden-Plot
    num_rows = math.ceil(len_cols / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 4))
    fig.suptitle(title, fontsize=20, fontweight="bold", y=1.0)

    axes_flat = cast(list[Axes], axes.flatten())

    # Dummy-Plot zur Platzierung einer einzigen Legende für alle anderen Plots
    ax_lgnd = axes_flat[0]

    palette = palette or theme.get_cont_palette(df_plt[hue_col].nunique()) # type: ignore

    sns.scatterplot(
        data=df_plt,
        x=all_cols[0],
        y=y,
        hue=hue_col,
        style=style_col,
        markers=markers,
        ax=ax_lgnd,
        palette=palette,
        alpha=0.75,
        edgecolor=None,
        s=60,
    )

    handles, labels = ax_lgnd.get_legend_handles_labels()

    if legend_texts and len(legend_texts) <= len(labels):
        labels = legend_texts

    ax_lgnd.clear()  # Plotinhalt löschen
    ax_lgnd.axis("off")  # Achsen verstecken

    if handles:
        ax_lgnd.legend(
            handles,
            labels,
            loc="upper left",
            title=get_translation(hue_col),
            fontsize=12,
            frameon=False,
        )

        ax_lgnd.set_title(
            "Legende für Plots",
            loc="left",
            fontweight="bold",
        )

    for i, col in enumerate(all_cols):
        ax = axes_flat[i + 1]

        sns.scatterplot(
            data=df_plt,
            x=col,
            y=y,
            hue=hue_col,
            style=style_col,
            markers=markers,
            ax=ax,
            palette=palette,
            alpha=0.75,
            edgecolor=None,
            s=60,
            legend=False,
        )
        ax.set_title(
            f"{fig_pre}{i + 1}{fig_suf} {get_translation(col)}",
            loc="left",
            fontsize=14,
        )

        ax.set_xlabel(col)
        ax.set_ylabel(y)

    for j in range(i + 2, num_rows * num_cols):  # i + 2 für extra legenden plot
        axes_flat[j].axis("off")

    # plt.tight_layout()
    # plt.show()
    save_and_load_img(fig_pre + "x: " + title)


def plot_zscore(
    df: pd.DataFrame,
    label_col: ProjectParams = "beam_number",
    title: str = "Z-Score Analyse & Outlier Identifikation",
    plot_type: Literal["box", "violin"] = "box",
    width: float = 0.5,
):
    """
    Plots the z-scores of the parameters as boxplot or violinplot for outlier detection
    split into two vertical subplots for better readability.

    :param df: the dataframe
    :type df: pd.DataFrame

    :param label_col: the column to use for labeling the datapoints
    :type label_col: str

    :param title: the figure title
    :type title: str

    :param plot_type: the type of plot to use
    :type plot_type: Literal["box", "violin"]
    """

    # Daten vorbereiten
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()

    scaled_array = scaler.fit_transform(df[numeric_cols])
    df_scaled = pd.DataFrame(scaled_array, columns=numeric_cols, index=df.index)

    beam_labels = df[label_col]

    # Spalten für die zwei Subplots aufteilen (Hälfte oben, Hälfte unten)
    mid_idx = len(numeric_cols) // 2 + (len(numeric_cols) % 2)
    cols_split = [numeric_cols[:mid_idx], numeric_cols[mid_idx:]]

    # Figure mit zwei vertikalen Subplots erstellen (sharey sorgt für gleiche Y-Achsen)
    fig, axes = cast(
        tuple[Figure, tuple[Axes, Axes]],
        plt.subplots(nrows=2, ncols=1, figsize=(15, 12), sharey=True),
    )

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    # in einer Schleife durch beide Subplots iterieren
    for ax, cols in zip(axes, cols_split):
        df_sub = df_scaled[cols]

        # --- Plots zeichnen ---
        if plot_type == "box":
            sns.boxplot(
                data=df_sub,
                fliersize=0,
                linewidth=1.5,
                ax=ax,
                # alpha=0.6,
                width=width,
            )
            sns.stripplot(
                data=df_sub,
                size=3,
                # alpha=0.5,
                jitter=0.15,
                ax=ax,
            )
        else:
            sns.violinplot(
                data=df_sub,
                orient="v",
                inner="quart",
                ax=ax,
                # alpha=0.6,
                width=width,
            )

        # Nulllinie (Mittelwert) deutlich markieren
        # ax.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.6)

        all_texts = []

        # --- Ausreißer identifizieren und Labels sammeln ---
        for plot_col, col in enumerate(cols):
            Q1 = df_sub[col].quantile(0.25)
            Q3 = df_sub[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_mask = (df_sub[col] < lower_bound) | (
                df_sub[col] > upper_bound
            )
            outliers = df_sub[col][outlier_mask]

            # Nur beschriften, wenn Dataset nicht zu riesig ist
            if len(df) < 1000:
                for idx, value in outliers.items():
                    txt = ax.text(
                        x=plot_col,
                        y=value,
                        s=str(beam_labels.at[idx]),
                        fontsize=12,
                        color="black",
                    )
                    all_texts.append(txt)

        # Automatische Text-Anpassung pro Subplot
        try:
            if 0 < len(all_texts) < 1000:
                with open(os.devnull, "w") as f:
                    with redirect_stdout(f):
                        adjust_text(
                            all_texts,
                            ax=ax,
                            arrowprops=dict(
                                arrowstyle="-", color="gray", lw=0.5
                            ),
                            only_move={"points": "y", "text": "xy"},
                            time_lim=3,
                        )
        except Exception as e:
            print(
                f"Fehler bei adjust_text in Spalten {cols[0]} bis {cols[-1]}: {e}"
            )

        # Achsen und Gitter
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=14)
        ax.set_ylabel("Z-Score")

        # dezente Gitterlinien
        ax.yaxis.grid(True)
        ax.xaxis.grid(False)

    # plt.tight_layout()
    # Platz für den suptitle lassen, damit er nicht mit dem ersten Plot kollidiert
    plt.subplots_adjust(top=0.93)
    # plt.show()
    save_and_load_img(title)


def draw_curved_label(
    ax_cart, text, polar_angle, radius, scale=0.035, color="white"
):
    # inspiriert von https://github.com/rougier/scientific-visualization-book/blob/master/code/scales-projections/text-polar.py
    path = TextPath((0, 0), text, size=10)
    path.vertices.flags.writeable = True
    V = path.vertices

    # 1. Text zentrieren
    xmin, xmax = V[:, 0].min(), V[:, 0].max()
    ymin, ymax = V[:, 1].min(), V[:, 1].max()
    V -= (xmin + xmax) / 2, (ymin + ymax) / 2

    # 2. Text skalieren
    V *= scale

    # 3. Winkel für kartesisches System umrechnen (da dein Plot bei 12 Uhr startet)
    cart_angle = np.pi / 2 - polar_angle

    # 4. Text in der unteren Plot-Hälfte drehen, damit er gut lesbar bleibt
    clean_angle = np.degrees(polar_angle) % 360
    if 90 < clean_angle < 270:
        V[:, 0] *= -1
        V[:, 1] *= -1

    # 5. Rougiers Transformation
    for i in range(len(V)):
        # x-Koordinate des Textes als Bogenmaß (Winkel-Offset) interpretieren
        a = cart_angle - V[i, 0] / radius
        r_current = radius + V[i, 1]

        # Neue kartesische Position berechnen
        V[i, 0] = r_current * np.cos(a)
        V[i, 1] = r_current * np.sin(a)

    patch = PathPatch(
        path, facecolor=color, edgecolor="none", linewidth=0, zorder=10
    )
    ax_cart.add_artist(patch)


def plot_radial_zscore(
    df: pd.DataFrame,
    theme: ColorTheme = ACADIA_MIDNIGHT,
    title: str = "Radialer Z-Score für Outlier Analyse",
    label_col: str = "beam_number",
    sort_by: Literal["Q1", "Q2", "Q3", "IQR"] = "Q1",
    ascending: bool = False,
    point_size: float = 15.0,
    rot_offset: float = 7.0,
    export_file: str | None = None,
    use_curved_text: bool = False,
    text_scale: float = 0.08,
    use_text_offset: bool = False,
):
    """
    Plots the z-scores of the parameters in a radial fashion for outlier detection

    :param df: the dataframe
    :type df: pd.DataFrame

    :param theme: the ColorTheme instance for semantic colors and backgrounds
    :type theme: ColorTheme

    :param title: the figure title
    :type title: str

    :param label_col: the column to use for labeling the datapoints
    :type label_col: str

    :param sort_by: the quantity to sort by (default: "Q1")
    :type sort_by: Literal["Q1", "Q2", "Q3", "IQR"]

    :param ascending: sort ascending (default: False)
    :type ascending: bool

    :param point_size: size of scatterd data points (default: 15.0)
    :type point_size: float

    :param rot_offset: offset in degrees the labels are rotated by from their tangetial orientation (default: 7.0)
    :type rot_offset: float

    :param export_file: optional path to save the plot as svg
    :type export_file: str | None
    """
    # Dynamische Farben aus Matplotlib ziehen (die vom Theme gesetzt wurden)
    text_c = cast(str, theme.text_color)
    bg_c = cast(str, theme.ax_bg)
    grid_c = cast(str, theme.grid_color)

    # 1. Datenvorbereitung (Z-Scores berechnen)
    df1 = df.copy()
    df1["is_outlier"] = False

    numeric_cols = df1.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df1[numeric_cols]), columns=numeric_cols
    )

    num_vars = len(numeric_cols)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    width = (2 * np.pi / num_vars) * 0.9

    stats = df_scaled.describe(percentiles=[0.25, 0.50, 0.75]).T
    stats["IQR"] = stats["75%"] - stats["25%"]

    sort_map = {"Q1": "25%", "Q2": "50%", "Q3": "75%", "IQR": "IQR"}
    sorted_cols = stats.sort_values(
        sort_map[sort_by], ascending=ascending
    ).index.tolist()
    df_scaled = df_scaled[sorted_cols]

    max_z = math.ceil(df_scaled.max().max())
    min_z = math.floor(df_scaled.min().min()) - 0.5

    # neu:
    # Figure anlegen
    data_radius = max_z - min_z
    text_padding = 2.0

    fig = plt.figure(figsize=(12, 12))

    # polar axis
    ax = cast(PolarAxes, fig.add_subplot(1, 1, 1, projection="polar"))

    # axis for curved labels
    ax_text = fig.add_subplot(1, 1, 1, frameon=False)
    
    # --- GRENZEN UND ABSTÄNDE DEFINIEREN ---
    ax.set_ylim(min_z, max_z + text_padding)
    ax.set_rorigin(min_z)
    

    # Radien für den Text definieren
    radial_span = max_z - min_z
    label_base_radius = radial_span + 0.6  # Erste Ebene: Knapper Abstand zum äußeren Grid
    stagger_offset = 0.8                   # Versatz für zweite Ebene (entspricht ca. einer Texthöhe)
    
    # 3. Leinwand der Text-Achse groß genug machen, damit nichts abgeschnitten wird
    max_visual_radius = data_radius + text_padding
    
    ax_text.set_xlim(-max_visual_radius, max_visual_radius)
    ax_text.set_ylim(-max_visual_radius, max_visual_radius)
    ax_text.set_aspect("equal")
    ax_text.set_xticks([])
    ax_text.set_yticks([])

    # alt:
    # _, ax = cast(
    #     tuple[Figure, PolarAxes],
    #     plt.subplots(figsize=(12, 12), subplot_kw={"projection": "polar"}),
    # )

    # Start bei "12 Uhr"
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2)


    zones = [
        (-6, -4, "#1b1b1b", 0.1),
        (-4, -2, "#424242", 0.1),
        (-2, 2, "#8b8b8b", 0.1),
        (2, 4, "#424242", 0.1),
        (4, 6, "#1b1b1b", 0.1),
    ]

    # radiale Füllbereiche
    angles_range = np.linspace(0, 2 * np.pi, 200)
    for r_start, r_end, color, alpha in zones:
        ax.fill_between(
            angles_range, r_start, r_end, color=color, alpha=alpha, zorder=0
        )

    all_texts = []

    for i, col in enumerate(sorted_cols):
        angle = angles[i]
        data = df_scaled[col]
        s = stats.loc[col]

        # Whiskers (1.5 * IQR)
        lower_whisker = max(data.min(), s["25%"] - 1.5 * s["IQR"])
        upper_whisker = min(data.max(), s["75%"] + 1.5 * s["IQR"])

        # --- Zeichnen ---

        # IQR Box als Kreissegment
        ax.bar(
            angle,
            s["IQR"],
            width=width,
            bottom=s["25%"],
            color=theme.primary_color,  # Semantische Farbe!
            alpha=theme.primary_alpha,
            edgecolor=bg_c,  # Trennung durch Hintergrundfarbe
            linewidth=1,
            zorder=3,
        )

        # Median als markante Linie
        t = np.linspace(angle - width / 2, angle + width / 2, 10)
        r = np.full_like(t, s["50%"])
        ax.plot(t, r, color=text_c, lw=1.5, zorder=4)

        # Whisker ends
        t = np.linspace(angle - width / 2 * 0.3, angle + width / 2 * 0.3, 10)
        r = np.full_like(t, lower_whisker)
        ax.plot(t, r, color=text_c, lw=1, zorder=4)

        r = np.full_like(t, upper_whisker)
        ax.plot(t, r, color=text_c, lw=1, zorder=4)

        # Whiskers als radiale Linie
        ax.plot(
            [angle, angle],
            [lower_whisker, upper_whisker],
            color=text_c,
            lw=1,
            zorder=4,
        )

        # Einzelne Datenpunkte (Stripplot radial)
        jitter = np.random.uniform(-0.05, 0.05, size=len(data))
        outlier_mask = (data < s["25%"] - 1.5 * s["IQR"]) | (
            data > s["75%"] + 1.5 * s["IQR"]
        )

        # Normale Punkte
        ax.scatter(
            angle + jitter[~outlier_mask],
            data[~outlier_mask],
            color=theme.secondary_color,  # Semantische Farbe!
            alpha=theme.secondary_alpha,
            s=point_size,
            edgecolors="none",
            zorder=1,
            rasterized=True,
        )

        # Outlier Punkte
        ax.scatter(
            angle + jitter[outlier_mask],
            data[outlier_mask],
            color=theme.outlier_color,  # Semantische Farbe!
            alpha=0.8,
            s=point_size,
            edgecolors="none",
            zorder=1,
            rasterized=True,
        )

        if not use_curved_text:
            # --- TANGENTIALE BESCHRIFTUNG ---
            angle_deg = np.degrees(angle)
            rotation = -angle_deg

            clean_angle = angle_deg % 360
            if 90 < clean_angle < 270:
                final_rot = rotation + 180
                va = "top"
            else:
                final_rot = rotation
                va = "bottom"

            ax.text(
                angle,
                label_base_radius,
                get_translation(col),  # Setzt get_translation voraus
                rotation=final_rot + rot_offset,
                rotation_mode="anchor",
                ha="center",
                va=va,
                fontsize=10,
            )

        else:
            # --- CURVED LABEL ---
            label_text = get_translation(col)
            current_radius = label_base_radius + (i % 2) * stagger_offset * use_text_offset

            draw_curved_label(
                ax_cart=ax_text,  # Wir zeichnen auf die unsichtbare Achse!
                text=label_text,
                polar_angle=angle,
                radius=current_radius,
                scale=text_scale,  # Hier kannst du die Schriftgröße global steuern
                color=text_c,
            )

        # 5. Outlier Labels
        outlier_positions = np.where(outlier_mask)[0]

        for i_pos in outlier_positions:
            idx_orig = data.index[i_pos]
            df1.loc[idx_orig, "is_outlier"] = True

            if label_col and len(df1) < 500:
                val = data.iloc[i_pos]
                label = str(df1.loc[idx_orig, label_col])
                jittered_angle = angle + jitter[i_pos]
                txt = ax.text(
                    jittered_angle,
                    val,
                    label,
                    fontsize=9,
                    style="italic",
                    color=theme.tertiary_color,
                )
                all_texts.append(txt)

    # --- Optik & Feinschliff ---

    ax.spines["polar"].set_visible(False)

    # Grid explizit auf Theme-Farben zwingen (Polar Plots sind da manchmal störrisch)
    gridlines_x = ax.xaxis.get_gridlines()
    for line in gridlines_x:
        line.set_color(grid_c)
        line.set_linestyle(theme.grid_linestyle)
        line.set_alpha(theme.grid_alpha)

    ax.set_xticks(angles)
    ax.set_xticklabels(["" for _ in sorted_cols], fontsize=11)

    gridlines_y = ax.yaxis.get_gridlines()
    for line in gridlines_y:
        line.set_color(grid_c)
        line.set_linestyle(theme.grid_linestyle)
        line.set_alpha(theme.grid_alpha)

    # Ringe für Z-Scores
    ax.set_rlabel_position(180 / num_vars)
    ax.set_yticks([-6, -4, -2, 0, 2, 4, 6])

    ytick_labels = cast(
        list[Text],
        ax.set_yticklabels(
            ["-6", "-4", "-2", "0", "2", "4", "6"],
            fontsize=10,
        ),
    )

    # Halo-Effekt dynamisch mit bg_c
    for lbl in ytick_labels:
        lbl.set_path_effects(
            [path_effects.withStroke(linewidth=3, foreground=bg_c)]
        )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=30)

    # Textüberlappungen verhindern
    if all_texts:
        adjust_text(
            all_texts,
            ax=ax,
            arrowprops=dict(
                arrowstyle="->", color=theme.tertiary_color, lw=0.5
            ),
        )

    # plt.tight_layout()

    # if export_file is not None:
    #     plt.savefig(
    #         export_file,
    #         format="svg",
    #         bbox_inches="tight",
    #         transparent=True,
    #     )

    # plt.show()
    save_and_load_img(title)

    return df1["is_outlier"]


def plot_heatmap(
    df: pd.DataFrame,
    title: str = "Korrelationsmatrix für reale Daten",
    theme: ColorTheme = ACADIA_MIDNIGHT,
    palette: str | LinearSegmentedColormap | None = None,
    target_col: str = "fire_resistance__min",
):
    """
    plots a triangular heatmap with x-axis labels on the diagonal

    :param df: the dataframe
    :type df: pd.DataFrame

    :param title: the title of thje figure
    :type title: str

    """

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)
        numeric_cols.append(target_col)

    corr = df[numeric_cols].corr()
    # Maske für das obere Dreieck
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    labels = [get_translation(lbl) for lbl in corr.columns]

    _, ax = plt.subplots(figsize=(12, 12))

    colors = cast(
        list[str],
        [theme.primary_color, theme.secondary_color, theme.tertiary_color],
    )
    palette = palette or LinearSegmentedColormap.from_list(
        "my_gradient", colors
    )

    # 1. Heatmap erstellen
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        cmap=palette,
        fmt=".2f",
        linewidths=1.0,
        linecolor=theme.ax_bg,
        annot_kws={"size": 9},
        cbar_kws={"shrink": 0.8, "orientation": "horizontal", "pad": 0.01},
        xticklabels=False,
        yticklabels=labels,
        ax=ax,
        square=True,
    )

    # 2. DIE LÖSUNG FÜR DIE GRIDLINES:
    # Wir schalten das Standard-Grid komplett aus.
    # Die Linien zwischen den Zellen kommen nur durch 'linewidths' in sns.heatmap.
    ax.grid(False)

    # Rahmen (Spines) entfernen, damit rechts oben alles leer ist
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # 3. Diagonale Labels (tangential)
    for i, label in enumerate(labels):
        if i == 0 or i == (len(labels) - 1):
            continue

        ax.text(
            x=i + 0.5,
            y=i,
            s="← " + label,
            rotation=45,
            ha="left",
            va="bottom",
            fontsize=11,
        )

    # 4. Y-Achse stylen
    ax.set_yticklabels(labels, fontsize=11, rotation=0)
    ax.tick_params(axis="both")

    # 5. Colorbar (Legende) Texte anpassen
    # Die Colorbar ist in ax.collections[0].colorbar erreichbar
    cbar = ax.collections[0].colorbar
    if cbar:
        cbar.ax.tick_params(labelsize=10)

    plt.title(
        title,
        pad=60,
        fontsize=16,
        fontweight="bold",
    )

    # plt.tight_layout()
    # plt.show()
    save_and_load_img(title)


def tsne_plot(
    df: pd.DataFrame,
    hue_col: str | list[str],
    theme: ColorTheme = ACADIA_MIDNIGHT,
    style_tuple: tuple[str, dict[int | str, str]] | None = None,
    label_col: str = "beam_number",
    target_col: str = "fire_resistance__min",
    title: str = "t-SNE Visualisierung",
    palette: list[str] | str | None = None,
    perplexity: int = 10,
    sizes: tuple[int, int] = (50, 250),
    do_plot: bool = False,
    fig_pre: str = "Abbildung",
    fig_suf: str = ":",
    edgecolor: str = "none",
):
    """
    Performs t-SNE dimensionality reduction and plots the 2D embedding.

    :param df: The input dataframe.
    :type df: pd.DataFrame

    :param hue_col: Column name(s) to be used for color grouping.
    :type hue_col: str | list[str]

    :param theme: The ColorTheme instance to access semantic colors and dynamic palettes.
    :type theme: ColorTheme

    :param style_tuple: Tuple defining the style column and its marker mapping.
    :type style_tuple: tuple[str, dict[int | str, str]]

    :param label_col: Column containing labels for individual data points.
    :type label_col: str

    :param target_col: Target column to drop prior to dimensionality reduction.
    :type target_col: str

    :param title: Title of the whole figure.
    :type title: str

    :param palette: Optional manual override for the color palette.
    :type palette: list[str] | str | None

    :param perplexity: The perplexity metric for the t-SNE algorithm.
    :type perplexity: int

    :param sizes: Minimum and maximum bubble sizes based on max absolute z-score.
    :type sizes: tuple[int, int]

    :param do_plot: Boolean flag to enable or disable plotting.
    :type do_plot: bool

    :param fig_pre: Prefix for plot numbering.
    :type fig_pre: str

    :param fig_suf: Suffix for plot numbering.
    :type fig_suf: str
    """

    if style_tuple is None:
        style_tuple = (
        "test_end_criterion",
        {0: "X", 1: "o"},
    )

    style_col = style_tuple[0]
    markers = style_tuple[1]

    df_numeric = df.select_dtypes(include="number").drop(
        [target_col], axis=1, errors="ignore"
    )
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df_numeric)

    # Max Z-Score zur Größenberechnung der Bubbles
    max_z_per_row = np.abs(scaled_array).max(axis=1)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=1234,
    )
    embedded_array = tsne.fit_transform(scaled_array)

    if do_plot:
        hue_cols = [hue_col] if isinstance(hue_col, str) else hue_col

        for i, hue in enumerate(hue_cols):
            df_plot = pd.DataFrame(
                {
                    "tsne_1": embedded_array[:, 0],
                    "tsne_2": embedded_array[:, 1],
                    "max_z_score": max_z_per_row,
                    hue: df[hue],
                    style_col: df[style_col],
                    label_col: df[label_col],
                },
                index=df.index,
            )

            # --- PALETTEN-LOGIK OPTIMIERT ---

            # Falls manuell eine Palette übergeben wurde, hat diese Vorrang
            if palette is not None:
                palette_param = palette
            else:
                # 1. Prüfen, ob die Daten numerisch sind
                is_numeric = pd.api.types.is_numeric_dtype(df[hue])

                # Bestimmung der Anzahl einzigartiger Werte für die aktuelle Spalte 'hue'
                # Wichtig: cast zu int, um Series-Fehler zu vermeiden
                n_unique = int(df[hue].nunique())

                # 2. Palette dynamisch vom Theme anfordern
                # Wenn numerisch UND viele Werte vorhanden sind, nutzen wir eine Map.
                # Bei wenigen Werten (auch numerisch) ist eine Liste sicherer für Seaborn.
                if is_numeric and n_unique > 10:
                    palette_param = theme.get_cont_palette(as_cmap=True)
                else:
                    palette_param = theme.get_cont_palette(n_colors=n_unique)

            _, ax = plt.subplots(figsize=(12, 9))

            # Scatterplot zeichnen
            sns.scatterplot(
                data=df_plot,
                x="tsne_1",
                y="tsne_2",
                hue=hue,
                size="max_z_score",
                style=style_col,
                markers=markers,
                sizes=sizes,
                palette=palette_param,
                ax=ax,
                alpha=0.8,
                edgecolor=edgecolor,
            )

            # Labels
            texts = []
            if len(df) < 1000:
                for j, row in df_plot.iterrows():
                    txt = ax.text(
                        row["tsne_1"],
                        row["tsne_2"],
                        str(row[label_col]),
                        fontsize=9,
                        alpha=0.9,
                        color=theme.tertiary_color,
                    )
                    texts.append(txt)

            if texts:
                adjust_text(
                    texts,
                    ax=ax,
                    force_explode=(0.2, 1.0),
                    arrowprops=dict(
                        arrowstyle="-",
                        lw=0.5,
                        color=theme.tertiary_color,  # Pfeile in Textfarbe
                    ),
                )

            # Achsen & Titel
            ax.set_title(
                f"{fig_pre} {i + 1}{fig_suf} {title} ({hue})",
                fontsize=14,
                fontweight="bold",
            )
            ax.set_xlabel("t-SNE Dimension 1")
            ax.set_ylabel("t-SNE Dimension 2")

            # Legende optimieren
            leg = ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                frameon=False,  # Entfernt den harten Rahmen für einen cleaneren Look
            )

            # Marker in der Legende voll deckend und groß machen
            if leg:
                for lh in leg.legend_handles:
                    if isinstance(lh, Artist):
                        if hasattr(lh, "set_alpha"):
                            lh.set_alpha(1.0)
                        if hasattr(lh, "set_sizes"):
                            lh.set_sizes([100]) # type: ignore

            # plt.tight_layout()
            # plt.show()
            save_and_load_img(f"{fig_pre} {i + 1}{fig_suf} {title} ({hue})")

    return embedded_array


def specific_tsne_plot(
    df: pd.DataFrame,
    title: str = "t-SNE Visualisierung",
    theme: ColorTheme = ACADIA_MIDNIGHT,
    fig_pre: str = "Abbildung",
    perplexity: int = 50,
    sizes=(20, 300),
    palette: list[str] | str | None = None,
    hue_cols: list[str] | None = None,
    do_plot: bool = False,
    edgecolor: str = "none",
):
    """
    TODO: create docstring
    """

    cols = df.columns.tolist()
    to_drop = [
        "deflection_at_failure__mm",
        "limit_state",
        "initial_capacity__kNm",
        "final_capacity__kNm",
        "is_real",
    ]

    do_drop = [d for d in to_drop if d in cols]
    df_tsne = df.drop(columns=do_drop)

    # Gruppen für Farbe (dazu Berechnung der Schlankheit, falls noch nicht vorhanden)

    frp_unit = "m2"
    if "frp_area__mm2" in cols:
        frp_unit = "mm2"

    len_unit = "m"
    if "beam_length__mm" in cols:
        len_unit = "mm"

    area_unit = "m2"
    if "concrete_area__mm2" in cols:
        area_unit = "mm2"

    if "slenderness__1_per_m" not in cols:
        length_in_m = df_tsne[f"beam_length__{len_unit}"]
        if len_unit == "mm":
            length_in_m = length_in_m * 1e-3

        area_in_m2 = df_tsne[f"concrete_area__{area_unit}"]
        if area_unit == "mm2":
            area_in_m2 = area_in_m2 * 1e-6

        df_tsne["slenderness__1_per_m"] = length_in_m / area_in_m2

    # Gruppen für Style (anhand von Bedingungen für das
    # Vorhandensein von FRP und des Versuchendes)
    tsne_groups = [
        "mit FRP, Versagen",
        "mit FRP, Brandende",
        "ohne FRP, Versagen",
        "ohne FRP, Brandende",
    ]

    markers: dict[int | str, str] = {
        k: m for (k, m) in zip(tsne_groups, ["P", "o", "X", "s"])
    }

    cond1 = (df_tsne[f"frp_area__{frp_unit}"] > 0) & (
        df_tsne["test_end_criterion"] == 0
    )
    cond2 = (df_tsne[f"frp_area__{frp_unit}"] > 0) & (
        df_tsne["test_end_criterion"] != 0
    )
    cond3 = (df_tsne[f"frp_area__{frp_unit}"] <= 0) & (
        df_tsne["test_end_criterion"] == 0
    )
    cond4 = (df_tsne[f"frp_area__{frp_unit}"] <= 0) & (
        df_tsne["test_end_criterion"] != 0
    )

    df_tsne["FRP_und_Versuchsende"] = np.select(
        [cond1, cond2, cond3, cond4], tsne_groups, default=""
    )

    # t-SNE Plots
    hue_cols = (
        hue_cols or (df_tsne.select_dtypes(include="number")).columns.tolist()
    )
    embedding = tsne_plot(
        df_tsne,
        hue_col=hue_cols,
        style_tuple=("FRP_und_Versuchsende", markers),
        palette=palette,
        title=title,
        theme=theme,
        perplexity=perplexity,
        sizes=sizes,
        fig_pre=fig_pre,
        do_plot=do_plot,
        edgecolor=edgecolor,
    )

    return embedding


def cluster_analysis_plot(
    df: pd.DataFrame,
    tsne_embedding: np.ndarray,
    min_cluster_size: int = 10,
    min_samples: int = 20,
    palette: list[str] | str | Colormap | list[tuple] | None = None,
    title="Automatische Gruppenbildung via HDBSCAN",
    hue: str | None = "cluster",
    style: str | None = None,
    markers: dict | None = None,
    theme: ColorTheme = ACADIA_MIDNIGHT,
):
    """
    Performs HDBSCAN clustering on provided t-SNE embeddings and visualizes the results with labeled centroids.
    Returns a summary table of main features (DataFrame)

    :param df: the input dataframe
    :type df: pd.DataFrame

    :param tsne_embedding: the 2D array containing the t-SNE coordinates
    :type tsne_embedding: npt.NDArray

    :param min_cluster_size: the minimum number of samples in a group for that group to be considered a cluster
    :type min_cluster_size: int

    :param min_samples: the number of samples in a neighborhood for a point to be considered a core point
    :type min_samples: int

    :param palette: the seaborn color palette to use for the clusters
    :type palette: str

    :param title: the title of the figure
    :type title: str

    :param hue: the column name to use for color encoding (defaults to the generated "cluster" column)
    :type hue: str | None

    :param style: the column name to use for marker style encoding
    :type style: str | None

    :param markers: a dictionary mapping values to marker shapes
    :type markers: dict | None

    :return: a summary table of main features
    :rtype: pd.DataFrame
    """
    df_cluster = df.copy()
    df_cluster["tsne_1"] = tsne_embedding[:, 0]
    df_cluster["tsne_2"] = tsne_embedding[:, 1]

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples
    )
    df_cluster["cluster"] = clusterer.fit_predict(
        df_cluster[["tsne_1", "tsne_2"]]
    )

    _, ax = plt.subplots(figsize=(12, 10))

    noise_data = df_cluster[df_cluster["cluster"] == -1].copy()
    noise_data["cluster_label"] = "Rauschen"

    normal_data = df_cluster[df_cluster["cluster"] != -1]
    palette = palette or theme.get_cont_palette(normal_data[hue].nunique())

    # Rauschen (Cluster -1) in Outlier-Farbe zeichnen
    sns.scatterplot(
        data=noise_data,
        x="tsne_1",
        y="tsne_2",
        hue="cluster_label",
        alpha=0.8,
        style=style,
        markers=markers,
        ax=ax,
        palette=[theme.outlier_color],
        edgecolor="#333333",
    )

    # Echte Cluster
    sns.scatterplot(
        data=normal_data,
        x="tsne_1",
        y="tsne_2",
        hue=hue,
        palette=palette,
        style=style,
        markers=markers,
        ax=ax,
        alpha=0.8,
        edgecolor="#333333",
    )

    all_texts = []
    for cluster in sorted(df_cluster["cluster"].unique()):
        if cluster == -1:
            continue

        centroid = df_cluster[df_cluster["cluster"] == cluster][
            ["tsne_1", "tsne_2"]
        ].mean()

        # Bbox im Theme-Style
        txt = ax.text(
            centroid["tsne_1"],
            centroid["tsne_2"],
            str(cluster),
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(
                alpha=0.8, boxstyle="round,pad=0.3", color=theme.tertiary_color
            ),
        )
        all_texts.append(txt)

    adjust_text(
        all_texts,
        ax=ax,
        arrowprops=dict(arrowstyle="->", lw=0.5),
    )

    for lbl in ax.get_xticklabels():
        lbl.set_color(theme.text_color)
    for lbl in ax.get_yticklabels():
        lbl.set_color(theme.text_color)

    ax.tick_params(axis="both")

    ax.set_title(title, fontsize=15, fontweight="bold")
    leg = ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    for t in leg.get_texts():
        t.set_color(theme.text_color)

    # plt.show()
    save_and_load_img(title)
    print("Aufgrund von Parallelisierung kann es zu abweichender Clusterbildung kommen. Die Beschreibung passt daher eventuell nicht 1:1 zu den angezeigten Daten.")

    # create summary table
    df_summary = create_summary_table(
        df_cluster,
        "cluster",
        df_cluster.select_dtypes(include="number")
        .drop(
            [
                "tsne_1",
                "tsne_2",
                "cluster",
                "test_end_criterion",
                "fire_resistance__min",
            ],
            axis=1,
        )
        .columns.tolist(),
        style="is_real",
    )

    return df_summary


def create_summary_table(
    df: pd.DataFrame,
    cluster_col: str,
    feature_cols: list[str],
    hue: str | None = None,
    style: str | None = None,
):
    """
    generates a summary table for clustered data, including point counts, percentages, optional categorical distributions, and the top distinguishing features per cluster

    :param df: the input dataframe containing the data and cluster labels
    :type df: pd.DataFrame

    :param cluster_col: the name of the column containing the cluster assignments
    :type cluster_col: str

    :param feature_cols: a list of feature column names used to calculate the cluster characteristics and importance
    :type feature_cols: list[str]

    :param hue: an optional categorical column name to calculate within-cluster distributions
    :type hue: str

    :param style: an optional second categorical column name to calculate within-cluster distributions
    :type style: str

    :return: a formatted summary dataframe containing cluster statistics and top features
    :rtype: pd.DataFrame
    """

    # Basis-Statistiken: Anzahl und Anteil pro Cluster
    summary = df.groupby(cluster_col).size().reset_index(name="Anzahl_Punkte")
    summary["Anteil_Prozent"] = (
        summary["Anzahl_Punkte"] / len(df) * 100
    ).round(1)

    # Globale Statistiken für Feature-Importance
    global_means = df[feature_cols].mean()
    global_stds = df[feature_cols].std()

    # Listen für die neuen Spalten
    top_char_list = []
    hue_dist_list = []
    style_dist_list = []

    # Iteration über alle Cluster
    for cluster in summary[cluster_col]:
        cluster_data = df[df[cluster_col] == cluster]

        # --- A. Verteilung für HUE berechnen ---
        if hue and hue in df.columns:
            # Zählt die Vorkommnisse der Werte in der Hue-Spalte für diesen Cluster
            counts = cluster_data[hue].value_counts().sort_index()
            # Formatiert z.B. zu "0: 120, 1: 5"
            dist_str = ", ".join([f"{k}={v}" for k, v in counts.items()])
            hue_dist_list.append(dist_str)
        else:
            hue_dist_list.append("-")

        # --- B. Verteilung für STYLE berechnen ---
        if style and style in df.columns:
            # Zählt die Vorkommnisse der Werte in der Style-Spalte
            counts = cluster_data[style].value_counts().sort_index()
            dist_str = ", ".join([f"{k}={v}" for k, v in counts.items()])
            style_dist_list.append(dist_str)
        else:
            style_dist_list.append("-")

        # --- C. Feature Importance (Hauptmerkmale) ---
        if cluster == -1:
            top_char_list.append("potentiell Rauschen / Ausreißer")
            continue

        importance = (
            cluster_data[feature_cols].mean() - global_means
        ) / global_stds

        # Die n wichtigsten Merkmale finden
        n = 5
        top_n = importance.abs().sort_values(ascending=False).head(n)

        desc = []
        for feat in top_n.index:
            # Richtungspfeil
            direction = "↑" if importance[feat] > 0 else "↓"
            # Optional: Übersetzung nutzen, falls vorhanden, sonst Spaltenname
            # name = get_translation(feat)
            name = feat
            desc.append(f"{name} {direction}")

        top_char_list.append(", ".join(desc))

    # Spalten zum DataFrame hinzufügen
    if hue:
        summary[f"Verteilung ({hue})"] = hue_dist_list
    if style:
        summary[f"Verteilung ({style})"] = style_dist_list

    summary["Hauptmerkmale"] = top_char_list

    # Spaltenreihenfolge aufräumen
    # Basis -> Hue/Style -> Merkmale
    cols = [cluster_col, "Anzahl_Punkte", "Anteil_Prozent"]
    if hue:
        cols.append(f"Verteilung ({hue})")
    if style:
        cols.append(f"Verteilung ({style})")
    cols.append("Hauptmerkmale")

    return summary[cols]


# endregion EXPLORATORY DATA ANALYSIS


# ---------------------------------------------
# region    PREPROCESSING
# ---------------------------------------------


def change_units(df: pd.DataFrame):
    """
    cahnge units to SI-units (m, kg, s) for the parameter-renamed dataframes `f_real` and `df_synth`

    :param df: the dataframe, whose columns already got renamed with `COL_NAMES_R` and `COL_NAMES_S` after loading the excel-file
    :type df: pd.DataFrame
    """

    for col in df.columns.tolist():
        if col.endswith("__mm"):
            # mm zu m
            df[col] /= 1e3
            df.rename(columns={col: col.replace("__mm", "__m")}, inplace=True)
        elif col.endswith("__mm2"):
            # mm² zu m²
            df[col] /= 1e6
            df.rename(columns={col: col.replace("__mm2", "__m2")}, inplace=True)
        elif col.endswith("__MPa"):
            # MPa zu Pa
            df[col] *= 1e6
            df.rename(columns={col: col.replace("__MPa", "__Pa")}, inplace=True)
        elif col.endswith("__deg"):
            # °C zu K
            df[col] += 273.15
            df.rename(columns={col: col.replace("__deg", "__K")}, inplace=True)
        elif col.endswith("__J_per_degm3"):
            # REAL: nur umbenennen, da es sich hier typischerweise um eine
            # Temperaturdifferenz handelt und nicht um eine Absoluttemperatur
            df.rename(
                columns={col: col.replace("__J_per_degm3", "__J_per_Km3")},
                inplace=True,
            )
        elif col.endswith("__W_per_mmdeg"):
            # SYNTH: nur umbenennen, da es sich hier einerseits um
            # Temperaturdifferenzen handelt und nicht um Absoluttemperaturen
            # und andererseits, da die Auswertung mit df.describe() zeigt,
            # dass die Daten im gleichen Bereich liegen wie bei den Realdaten,
            # die aber bereits W/mK als Einheit haben. Das Paper gibt auch W/mK an.
            df.rename(
                columns={col: col.replace("__W_per_mmdeg", "__W_per_mK")},
                inplace=True,
            )
        elif col.endswith("__J_per_degmm3"):
            # SYNTH: nur umbenennen, da es sich hier einerseits um
            # Temperaturdifferenzen handelt und nicht um Absoluttemperaturen
            # und andererseits, da die Auswertung mit df.describe() zeigt,
            # dass die Daten im gleichen Bereich liegen wie bei den Realdaten,
            # die aber bereits J/Km³ als Einheit haben. Das Paper gibt auch J/Km³ an.
            df.rename(
                columns={col: col.replace("__J_per_degmm3", "__J_per_Km3")},
                inplace=True,
            )
        elif col.endswith("__kN"):
            # kN zu N
            df[col] *= 1e3
            df.rename(columns={col: col.replace("__kN", "__N")}, inplace=True)
        elif col.endswith("__kNm"):
            # kNm zu Nm
            df[col] *= 1e3
            df.rename(columns={col: col.replace("__kNm", "__Nm")}, inplace=True)
        elif col.endswith("__min"):
            # Die Zielvariable wird mit keinem anderem Wert verrechnet und ist der einzige
            # Parameter mit einer Zeiteinheit. Daher werden die Minuten nicht in Sekunden
            # überführt, da die typischen Feuerwiderstandsklassen ebenfalls in Minuten angegeben sind.
            pass


def engineer_new_params(df: pd.DataFrame):
    """
    engineering of new features

    :param df: the combined dataframe (synth + real) after data cleaning
    :type df: pd.DataFrame
    """

    # BREITE UND HÖHE
    # -----------------

    # Die Daten (Breite, Höhe) wurden aus den angegebenen Intervallen im Paper und
    # üblichen QS-Verhältnissen sowie über eine KI-Rechereche (Gemini) rekonstruiert,
    # wobei die Ergebnisse plausibel erschienen. Dennoch besteht hier die größte
    # NOTE: Gefahr eines neuen Bias
    area_dict = {
        # Für Laborversuche
        0.012000: (0.100, 0.120),
        0.045000: (0.150, 0.300),
        0.060000: (0.200, 0.300),
        0.060800: (0.152, 0.400),
        # 0.090000: (0.200, 0.450), # Die synthetischen Daten haben zunächst Präferenz. Die Versuchsdaten werden nachträglich angepasst
        # 0.100000: (0.200, 0.500), # Die synthetischen Daten haben zunächst Präferenz. Die Versuchsdaten werden nachträglich angepasst
        0.103124: (0.254, 0.406),
        # Hier wird der T-Balken vereinfacht als Rechteck-QS dargestellt
        0.125730: (0.254, 0.406),
        # Für synth. Daten
        # 0.012000: (0.100, 0.120), # NOTE: identisch zu Realdaten
        0.014400: (0.120, 0.120),
        0.015000: (0.100, 0.150),
        0.018000: (0.120, 0.150),
        0.022500: (0.150, 0.150),
        0.030000: (0.150, 0.200),
        0.037500: (0.150, 0.250),
        0.040000: (0.200, 0.200),
        0.050000: (0.200, 0.250),
        # 0.060000: (0.200, 0.300), # NOTE: identisch zu Realdaten
        0.062500: (0.250, 0.250),
        0.075000: (0.250, 0.300),
        0.087500: (0.250, 0.350),
        # NOTE: hier scheinen die Realdaten sinnvoller, es wird aber trotzdem unterschieden
        0.090000: (0.300, 0.300),
        # NOTE: das Paper sagt als max. Höhe 450 mm, weswegen hier zwischen
        0.100000: (0.250, 0.400),
        0.105000: (0.300, 0.350),
        0.120000: (0.300, 0.400),
        # NOTE: quadr. QS wird hier akzeptiert, da sonst keine sinnvolle Kombination in den Grenzen des Papers möglich
        #       real und synth. Daten unterschieden wird, siehe den Kommentar oben bei Fläche 0.100000
        0.122500: (0.350, 0.350),
        0.135000: (0.300, 0.450),
        0.140000: (0.350, 0.400),
        0.157500: (0.350, 0.450),
    }

    df["width__m"] = df["concrete_area__m2"].apply(lambda x: area_dict[x][0])
    df["height__m"] = df["concrete_area__m2"].apply(lambda x: area_dict[x][1])

    # Für Fläche 0.09 m² der Laborversuche
    real_mask = (df["is_real"] == 1) & (df["concrete_area__m2"] == 0.09)
    df.loc[real_mask, "width__m"] = 0.2
    df.loc[real_mask, "height__m"] = 0.45

    # Für Fläche 0.10 m² der Laborversuche
    real_mask = (df["is_real"] == 1) & (df["concrete_area__m2"] == 0.1)
    df.loc[real_mask, "width__m"] = 0.2
    df.loc[real_mask, "height__m"] = 0.5

    # WIDERSTANDSMOMENT W
    # ---------------------
    df["section_modulus__m3"] = (df["width__m"] * (df["height__m"] ** 2)) / 6

    # ERSATZSPANNUNG
    # ----------------
    df["stress_proxy__Pa"] = (
        df["beam_length__m"] * df["total_load__N"] / df["section_modulus__m3"]
    )

    # # RELATIVE SEITLICHE ABDECKUNG DURCH DÄMMUNG # NOTE: wird ersetzt durch Profilfaktor
    # # --------------------------------------------
    # # zusätzlich geteilt durch die Breite, da vor allem bei geringen Breiten der Effekt der seitlichen Dämmung spürbar wird (schnellere Durchwärmung)
    # df["insulation_ratio__1_per_m"] = (
    #     df["insulation_depth__m"] / df["height__m"]
    # ) / df["width__m"]

    # # PROFILFAKTOR
    # # ------------
    # # Ein Profilfaktor von 0 bedeutet komplett gedämmt, großer Profilfaktor bedeutet schnellere Durchwärmung
    # NOTE: Der Profilfaktor wurde nicht eingeführt, da t-SNE und DBSCAN extrem viele Cluster erzeugt haben, oder
    # bei anderen Eingangswerten für weniger Cluster die Realdaten fast alle als Rauschen / Ausreißer galten.
    # Daher wurde zurück auf "insulation_ratio__1_per_m" gegangen.

    exposed_width = np.where(
        df["insulation_thickness__m"] == 0, df["width__m"], 0
    )  # gesamte Breite, wenn keine Dämmung, sonst 0

    exposed_height = 2 * (df["height__m"] - df["insulation_depth__m"])

    df["section_factor__1_per_m"] = (exposed_width + exposed_height) / df[
        "concrete_area__m2"
    ]

    # BIEGESCHLANKHEIT
    # ------------------
    df["slenderness__1_per_m"] = df["beam_length__m"] / df["height__m"]

    # BEWEHRUNGSGRAD
    # ----------------
    df["reinforcement_ratio"] = df["steel_area__m2"] / df["concrete_area__m2"]

    # ACHSABSTAND DER BEWEHRUNG
    # ---------------------------
    # der Achsabstand wird mit im Durchschnitt 2,5 Stäben berechnet, da das Paper angibt, dass 2 oder 3 Stäbe angesetzt wurden
    # der Achsabstand ist Betondeckung plus rechnerischer Radius der Bewehrung
    df["ax_dist__m"] = df["concrete_cover__m"] + np.sqrt(
        df["steel_area__m2"] / (2.5 * np.pi)
    )

    # WÄRMEDURCHLASSWIDERSTAND
    # --------------------------
    df["thermal_resistance__m2K_per_W"] = np.where(
        df["insulation_thermal_conductivity__W_per_mK"] != 0,
        df["insulation_thickness__m"]
        / df["insulation_thermal_conductivity__W_per_mK"],
        0,
    )

    # TEMPERATURLEITFÄHIGKEIT
    # -------------------------
    # (Auf 0 im Nenner achten, da hier 0 heißt, es gibt keine Dämmung)
    df["thermal_diffusivity__m2_per_s"] = np.where(
        df["insulation_specific_heat_cap__J_per_Km3"] != 0,
        df["insulation_thermal_conductivity__W_per_mK"]
        / df["insulation_specific_heat_cap__J_per_Km3"],
        0,
    )

    print(
        """
Folgende neue Parameter wurden neu angelegt:
--------------------------------------------
width__m, height__m, section_modulus__m3, stress_proxy__Pa, 
section_factor__1_per_m, slenderness__1_per_m, reinforcement_ratio, 
ax_dist__m, thermal_resistance__m2K_per_W, thermal_diffusivity__m2_per_s
"""
    )


# endregion PREPROCESSING


# ---------------------------------------------
# region    MODEL SETUP
# ---------------------------------------------


def custom_mse_objective(y_true, y_pred, sample_weight=None):
    """
    Custom Objective für zensierte Regression.
    Erweitert um sample_weight Unterstützung.
    """
    y = y_true.copy()

    # Zensur-Logik (Negatives Vorzeichen = Zensiert)
    is_censored = y < 0
    y_abs = np.abs(y)

    # Residuum (Vorhersage - Label)
    diff = y_pred - y_abs

    # Basis Gradient und Hessian
    grad = 2 * diff
    hess = 2 * np.ones_like(diff)

    # Zensur-Maskierung: Kein Fehler wenn zensiert UND Vorhersage > Label
    mask_no_penalty = is_censored & (diff > 0)
    grad[mask_no_penalty] = 0
    hess[mask_no_penalty] = 0

    # --- WICHTIG: Gewichte anwenden ---
    if sample_weight is not None:
        grad *= sample_weight
        hess *= sample_weight

    return grad, hess


def custom_scorer(y_true, y_pred, metric: Literal["RMSE", "MAE", "R2"]):
    # als numpy Array
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    # Rechtszensierte Daten sind negativ definiert
    is_censored = y_true_arr < 0
    y_true_abs = np.abs(y_true_arr)
    # Fehler ist pred - true, wobei hier y_true strikt positiv sein muss
    diff_pred_true = y_pred_arr - y_true_abs
    # Wenn die Zielvariable rechtszensiert ist und höher vorhergesagt wurde,
    # dann keine penalty
    mask_no_penalty = is_censored & (diff_pred_true > 0)
    diff_pred_true[mask_no_penalty] = 0

    if metric == "RMSE":
        return np.sqrt(np.mean(diff_pred_true**2))

    elif metric == "MAE":
        return np.mean(np.abs(diff_pred_true))

    elif metric == "R2":
        rss = np.sum(diff_pred_true**2)  # residual sum of squares
        tss = np.sum(
            (y_true_abs - np.mean(y_true_abs)) ** 2
        )  # total sum of squares
        if tss == 0:
            return 0.0
        return 1 - (rss / tss)
    else:
        raise ValueError(
            "custom_scorer: Es werden nur RMSE, MAE und R2 für den Parameter `metric` akzeptiert!"
        )


def dynamic_weights(
    df: pd.DataFrame,
    weight_real: float,
    weight_failure: float,
    weight_censored: float,
):
    weights = np.ones(len(df))
    weights[df["is_real"] == 1] *= weight_real  # Gewichtung für Realdaten
    weights[df["test_end_criterion"] == 0] *= (
        weight_failure  # Gewichtung für Versagen
    )
    weights[df["test_end_criterion"] == 1] *= (
        weight_censored  # Gewichtung für Rechtszensur
    )
    return weights


def create_objective(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splitter: CustomSplitter,
    must_have_feats: list[str],
):

    # FEATURE SELECTION FÜR OPTUNA
    optional_feats = [feat for feat in X.columns if feat not in must_have_feats]

    # METRIKEN
    metrics = ["RMSE", "MAE", "R2"]
    arr_gib = [False, False, True]

    def objective(trial: optuna.Trial) -> float:

        # custom scorer dictionary
        scoring = {
            mtrc: make_scorer(custom_scorer, greater_is_better=gib, metric=mtrc)
            for (mtrc, gib) in zip(metrics, arr_gib)
        }

        # auch Gewichte variieren, um die besten zu finden
        weight_real = trial.suggest_float("weight_real", 1.0, 20.0)
        weight_failure = trial.suggest_float("weight_failure", 1.0, 10.0)
        weight_censored = trial.suggest_float("weight_censored", 0.01, 1.1)
        dyn_weights = dynamic_weights(
            df, weight_real, weight_failure, weight_censored
        )

        # Maske für das custom objective (zensierte y werden als negativ markiert)
        y_marked = y.copy()
        mask_censored = df["test_end_criterion"] == 1
        y_marked.loc[mask_censored] *= -1

        # alle Parameter für diesen Trial sammeln (inkl. Features und Constraints)
        trial_params = {
            "max_depth": trial.suggest_int("max_depth", 2, 4),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05),
            "n_estimators": trial.suggest_int(
                "n_estimators", 300, 1500, log=True
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1.0, 800.0, log=True
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", 0.0001, 800.0, log=True
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 0.8
            ),
            "subsample": trial.suggest_float("subsample", 0.5, 0.8),
            "use_constraints": trial.suggest_categorical(
                "use_constraints", [True, False]
            ),
        }

        # die optionalen Features als boolean in das Dict packen
        for feat in optional_feats:
            trial_params[f"use_{feat}"] = trial.suggest_categorical(
                f"use_{feat}", [True, False]
            )

        # Modell mit den Trial-Parametern instanziieren (unsere smarte Klasse!)
        model = OptimizedFireXGB(params=trial_params)

        # cross validation
        cv_results = cross_validate(
            estimator=model,
            X=X,
            y=y_marked,
            cv=cv_splitter.split(df),
            scoring=scoring,
            return_train_score=True,
            params={"sample_weight": dyn_weights}, # type: ignore
        )

        # Metadaten für Analyse speichern
        trial.set_user_attr("active_features", model.selected_features_)

        for m in metrics:
            # erstellt bspw. Attribute test_RMSE, train_RMSE und gap_RMSE
            sgn = 1 if m == "R2" else -1
            test_metric = sgn * cv_results[f"test_{m}"].mean()
            train_metric = sgn * cv_results[f"train_{m}"].mean()
            trial.set_user_attr(f"test_{m}", test_metric)
            trial.set_user_attr(f"train_{m}", train_metric)
            trial.set_user_attr(f"gap_{m}", test_metric - train_metric)

        return cast(float, trial.user_attrs.get("test_RMSE"))

    return objective


# endregion MODEL SETUP


# ---------------------------------------------
# region    EVALUATION
# ---------------------------------------------


def plot_hyperparameter_importance(study: optuna.Study, theme: ColorTheme):

    fig = cast(PlotlyFigure, optuna.visualization.plot_param_importances(study))

    # Layout anpassen (Hintergrund, Titel, Text und Gitterlinien)
    fig.update_layout(
        height=800,
        bargap=0.25,

        title={
            "text": ("Abbildung 4-1: Hyperparameter Importances"),
            "font": {"color": theme.title_color, "size": 20}
        },
        paper_bgcolor=theme.fig_bg,
        plot_bgcolor=theme.ax_bg,
        font={"color": theme.text_color},
        
        xaxis=dict(
            gridcolor=theme.grid_color,
            zerolinecolor=theme.grid_color,
            tickcolor=theme.text_color
        ),
        
        yaxis=dict(
            gridcolor=theme.grid_color,
            zerolinecolor=theme.grid_color,
            tickcolor=theme.text_color,
            ticklabelstandoff=15
        )
    )

    # Balken (Traces) anpassen
    fig.update_traces(
        marker_color=theme.primary_color,
        textfont=dict(color=theme.text_color)
    )

    show(fig)


def plot_feature_importance(best_model: OptimizedFireXGB, best_params, df_final: pd.DataFrame, X: pd.DataFrame, y: pd.Series, theme: ColorTheme):
    
    # Gewichte für das finale Training berechnen
    final_weights = dynamic_weights(
        df_final, 
        weight_real=best_params["weight_real"], 
        weight_failure=best_params["weight_failure"], 
        weight_censored=best_params["weight_censored"]
    )

    # Finales Training auf dem gesamten Datensatz (oder deinem ausgewählten Trainings-Split)
    y_marked = y.copy()
    mask_censored = df_final["test_end_criterion"] == 1
    y_marked.loc[mask_censored] *= -1

    best_model.fit(X, y_marked, sample_weight=final_weights)

    # für pylance
    if best_model.model is None or best_model.selected_features_ is None:
        raise ValueError(
            "Das Modell muss zuerst mit .fit() trainiert werden!"
        )
    
    # Feature Importances und Namen aus der Klasse extrahieren
    importances = best_model.model.feature_importances_
    features = best_model.selected_features_

    # In einen Pandas DataFrame packen und absteigend sortieren
    df_importance = pd.DataFrame({
        "Feature": [get_translation(f) for f in features],
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    # Konsolenausgabe: Welche Features hat L1 auf 0 gesetzt?
    zero_features = df_importance[df_importance['Importance'] == 0.0]
    if not zero_features.empty:
        print(f"!!! L1-Regularisierung hat {len(zero_features)} Features komplett eliminiert !!!")
        print(zero_features['Feature'].tolist())
        print("-" * 50)

    # print(df_importance.to_string(index=False))

    # Visueller Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(
        df_importance,
        x="Importance", 
        y="Feature",
        color=theme.primary_color
    )
    plt.title(title:="Abbildung 4-2: Feature Importances für bestes Optuna Modell", fontsize=14)
    plt.xlabel("Relative Wichtigkeit (0 bis 1)", fontsize=12)
    plt.ylabel("Feature Name", fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    # plt.show()
    save_and_load_img(title)



def plot_shap_beeswarm(
    shap_explanation: shap.Explanation,
    theme: ColorTheme,
    col_arr: list[str],
    title="Globale Einfluss-Analyse",
    max_display=12,
):
    """
    Erstellt einen SHAP-Beeswarm-Plot in reinem Matplotlib.
    Nutzt einen eigenen Stacking-Algorithmus, um die perfekte, geordnete 
    "Bauch-Form" (wie in der originalen SHAP-Library) zu erzeugen, 
    ohne dass sich die Reihen bei großen Datenmengen überlappen.
    """
    # 1. DATEN EXTRAHIEREN
    shap_values = np.asarray(shap_explanation.values)
    features = np.asarray(shap_explanation.data)
    feature_names = shap_explanation.feature_names

    # Fallback, falls keine Feature-Namen übergeben wurden
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(shap_values.shape[1])]

    # Wichtigkeit berechnen und Indizes sortieren (unwichtigste zuerst für Plot von unten nach oben)
    importances = np.mean(np.abs(shap_values), axis=0)
    sorted_indices = np.argsort(importances)[-max_display:]

    # 2. FARBSKALA DEFINIEREN
    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_shap", col_arr
    )

    # 3. FIGURE SETUP
    fig, ax = plt.subplots(figsize=(12, 8))
    y_ticks = []
    y_labels = []

    # 4. SCHWARM (BEESWARM) STAPELN & PLOTTEN
    for pos, idx in enumerate(sorted_indices):
        shaps = shap_values[:, idx]
        feats = features[:, idx]

        # Feature-Werte normalisieren [0, 1] für die Farbdarstellung
        try:
            feats = feats.astype(float)
            f_min, f_max = np.nanmin(feats), np.nanmax(feats)
            if f_max > f_min:
                feats_norm = (feats - f_min) / (f_max - f_min)
            else:
                feats_norm = np.zeros_like(feats)
        except Exception:
            feats_norm = np.zeros_like(shaps)

        # --- ROBUSTER STACKING-ALGORITHMUS ---
        # 1. Spanne und Punktbreite (Topf-Größe) berechnen
        span = np.nanmax(shaps) - np.nanmin(shaps)
        dot_width = span / 150.0 if span > 0 else 0.01  # Feines Binning (150 Töpfe pro Feature)

        # 2. Sortieren nach X-Wert (SHAP-Wert)
        sort_idx = np.argsort(shaps)
        shaps_sorted = shaps[sort_idx]
        # feats_norm_sorted = feats_norm[sort_idx]
        
        # Array für die Y-Koordinaten (wichtig: als float initialisieren)
        y_pos = np.zeros_like(shaps, dtype=float)
        
        # 3. Punkte in Töpfe (Bins) werfen
        bins = np.round(shaps_sorted / dot_width)
        
        # 4. Maximale Höhe eines Schwarms berechnen
        # Ein Schwarm darf maximal 0.4 Einheiten nach oben/unten wachsen, 
        # damit er nicht in das benachbarte Feature (Abstand = 1.0) ragt.
        max_swarm_height = 0.4
        
        unique_bins, counts = np.unique(bins, return_counts=True)
        max_count = counts.max() if len(counts) > 0 else 1
        
        # Dynamisches Spacing: Verhindert Überlappungen bei tausenden Punkten
        y_spacing = max_swarm_height / (max_count / 2 + 1)
        # Deckeln auf max. 0.05, falls es nur sehr wenige Punkte gibt
        y_spacing = min(y_spacing, 0.05) 

        # 5. Stapeln ausführen
        current_bin = None
        current_count = 0
        
        for i, b in enumerate(bins):
            if b != current_bin:
                current_bin = b
                current_count = 0
            
            # Abwechselnd nach oben und unten stapeln (0, 1, -1, 2, -2...)
            # Ganzzahldivision (//), um saubere Layer zu garantieren
            if current_count % 2 == 0:
                y_offset = (current_count // 2) * y_spacing
            else:
                y_offset = -((current_count + 1) // 2) * y_spacing
                
            y_pos[sort_idx[i]] = pos + y_offset
            current_count += 1
        # -------------------------------------

        # Scatter-Plot für das aktuelle Feature
        ax.scatter(
            shaps,
            y_pos,
            c=feats_norm,
            cmap=custom_cmap,
            vmin=0, vmax=1,
            s=12,             # Punktgröße
            alpha=0.7,        # Leichte Transparenz
            edgecolors='none' # Wichtig für dunkle Themes
        )

        y_ticks.append(pos)
        y_labels.append(get_translation(feature_names[idx]))

    # 5. COLORBAR (Manuell platziert für volle Kontrolle)
    cbar_ax = fig.add_axes((0.92, 0.2, 0.03, 0.6)) 
    cbar = plt.colorbar(ax.collections[0], cax=cbar_ax)
    
    # Beschriftung der Colorbar anpassen
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Niedrig", "Hoch"], rotation=90, va="center")
    cbar.set_label("Merkmalswert", color=theme.text_color, labelpad=10, fontsize=10, fontweight="bold")
    cbar.ax.tick_params(colors=theme.text_color, size=0, labelsize=10)

    # 6. ACHSEN & DESIGN
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=11, fontweight="bold")
    ax.set_xlabel("SHAP Wert (Einfluss auf Vorhersage)", color=theme.text_color, fontsize=11)
    
    # Vertikale Nulllinie zur Orientierung
    ax.axvline(0, color=theme.grid_color, alpha=0.7, lw=1.5, zorder=0)

    # Rahmen und Ticks auf das Theme abstimmen
    ax.tick_params(axis="both", colors=theme.text_color)
    for spine in ["left", "top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(theme.grid_color)

    ax.set_title(title, color=theme.text_color, fontsize=14, pad=20, loc="left")

    # Layout des Haupt-Plots anpassen, damit die Colorbar rechts nicht abgeschnitten wird
    fig.set_layout_engine('none')
    plt.subplots_adjust(left=0.25, right=0.88, top=0.9, bottom=0.1)

    # plt.show()
    save_and_load_img(title)

    return fig, ax



# endregion EVALUATION