"""
Audit automatique de l'explicabilité SHAP (Cupid Matchmaking).

Chaque vérification est RECALCULÉE à partir de results/best_model.pkl,
results/preprocessor.pkl et data/raw/cupid_algorithm_dataset.csv : rien n'est
écrit à la main. Le rapport (docs/SHAP_AUDIT.md) et le tableau des 3 principaux
facteurs (results/tables/shap_top3_factors.csv) sont générés par ce script.

Usage (depuis la racine du projet) :
    python experiments/shap_audit.py
"""

import os
import re
import sys
import json
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from src.models.base_model import BaseModel
from src.data.preprocessing import CupidDataPreprocessor

MODEL_PATH = "results/best_model.pkl"
PREP_PATH = "results/preprocessor.pkl"
DATA_PATH = "data/raw/cupid_algorithm_dataset.csv"
SAVED_JSON = "results/tables/enhanced_shap_global_importance.json"
FIGURES = {
    "Summary plot (beeswarm, baseline)": "results/figures/shap_summary.png",
    "Summary plot (beeswarm top 20)": "results/figures/enhanced_shap_beeswarm_top20.png",
    "Bar plot (moyenne des valeurs absolues)": "results/figures/enhanced_shap_bar.png",
    "Waterfall (explication individuelle)": "results/figures/shap_waterfall.png",
}
REPORT_MD = "docs/SHAP_AUDIT.md"
TOP3_CSV = "results/tables/shap_top3_factors.csv"

results = []  # (critère, statut, détail)


def check(name, ok, detail, warn=False):
    status = "OK" if ok else ("ATTENTION" if warn else "ÉCHEC")
    results.append((name, status, detail))
    print(f"[{status:9s}] {name}: {detail}")


def direction(corr):
    if abs(corr) < 0.2:
        return "effet non monotone / faible"
    return "valeur élevée → compatibilité ↑" if corr > 0 else "valeur élevée → compatibilité ↓"


def main():
    # ---- 1. Présence de SHAP -------------------------------------------------
    sources = {
        "src/evaluation/shap_explainer.py": None,
        "experiments/advanced_shap_analysis.py": None,
        "app.py": None,
    }
    for p in sources:
        sources[p] = os.path.exists(p) and "import shap" in open(p, encoding="utf-8").read()
    check("1. Présence de SHAP", all(sources.values()),
          f"shap {shap.__version__} ; import trouvé dans : "
          + ", ".join(p for p, v in sources.items() if v))

    # ---- 2. Compatibilité avec le modèle ------------------------------------
    mw = BaseModel.load(MODEL_PATH)
    model = mw.model if hasattr(mw, "model") else mw
    prep = CupidDataPreprocessor.load(PREP_PATH)
    names = list(prep.feature_names)
    tree_types = ("XGBClassifier", "RandomForestClassifier", "DecisionTreeClassifier",
                  "LGBMClassifier", "GradientBoostingClassifier")
    check("2. Compatibilité modèle / explainer", type(model).__name__ in tree_types,
          f"modèle = {type(model).__name__} → shap.TreeExplainer est exact et adapté "
          f"({len(names)} features)")

    # ---- Données : jeu de test reproductible --------------------------------
    df = pd.read_csv(DATA_PATH)
    _, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["compatible"])
    samp = test.sample(n=1000, random_state=42)
    X = np.asarray(prep.transform(samp))

    explainer = shap.TreeExplainer(model)
    sv = explainer(X)
    vals, base = sv.values, np.ravel(sv.base_values)

    # ---- 3. Calcul correct : propriété d'additivité -------------------------
    margin = model.predict(X, output_margin=True)
    err = float(np.max(np.abs(margin - (base + vals.sum(axis=1)))))
    check("3. Calcul des valeurs SHAP (additivité)", err < 1e-3,
          f"écart max entre sortie brute du modèle et (base + Σ SHAP) = {err:.2e} sur {len(X)} lignes "
          f"(unités : log-odds, pas probabilités)")

    # ---- 4. Explication globale : recalcul vs fichier sauvegardé ------------
    mean_abs = np.abs(vals).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    saved = json.load(open(SAVED_JSON, encoding="utf-8"))
    n_cmp = min(10, len(saved))
    same_rank = [names[order[i]] == saved[i]["feature"] for i in range(n_cmp)]
    same_val = all(abs(mean_abs[order[i]] - saved[i]["mean_abs_shap"]) < 1e-3 for i in range(n_cmp))
    check("4. Explication globale (JSON = recalcul)", all(same_rank) and same_val,
          f"top-{n_cmp} du JSON identique au recalcul (ordre et valeurs)")

    # ---- 5/6. Figures : existence + non vide --------------------------------
    for label, path in FIGURES.items():
        ok = os.path.exists(path) and os.path.getsize(path) > 20_000
        check(f"5/6. Figure — {label}", ok, f"{path} ({os.path.getsize(path)//1024} Ko)" if os.path.exists(path) else f"{path} ABSENT")

    # ---- 7/8. Top 3 facteurs + positif / négatif ----------------------------
    rows = []
    for i in order[:3]:
        corr = float(np.corrcoef(X[:, i], vals[:, i])[0, 1]) if X[:, i].std() > 0 else 0.0
        pos, neg = vals[:, i][vals[:, i] > 0], vals[:, i][vals[:, i] < 0]
        rows.append({
            "rang": len(rows) + 1,
            "feature": names[i],
            "mean_abs_shap": round(float(mean_abs[i]), 4),
            "part_du_total_%": round(float(100 * mean_abs[i] / mean_abs.sum()), 1),
            "corr_valeur_vs_shap": round(corr, 3),
            "sens_global": direction(corr),
            "contrib_moy_positive": round(float(pos.mean()), 4) if len(pos) else 0.0,
            "contrib_moy_negative": round(float(neg.mean()), 4) if len(neg) else 0.0,
            "%_lignes_positives": round(float(100 * len(pos) / len(vals)), 1),
        })
    top3 = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(TOP3_CSV), exist_ok=True)
    top3.to_csv(TOP3_CSV, index=False, encoding="utf-8")
    check("7. Trois principaux facteurs identifiés", True,
          " > ".join(f"{r['feature']} ({r['mean_abs_shap']})" for r in rows))
    both = all(r["contrib_moy_positive"] > 0 and r["contrib_moy_negative"] < 0 for r in rows)
    check("8. Distinction facteurs positifs / négatifs", both,
          "chaque facteur du top 3 contribue dans les deux sens selon la ligne (cf. tableau) ; "
          "le sens global est donné par la corrélation valeur ↔ SHAP")

    # ---- Explication individuelle : 1 paire compatible + 1 incompatible -----
    proba = model.predict_proba(X)[:, 1]
    local_md, local_ok = [], True
    for label, j in (("paire la plus compatible", int(proba.argmax())),
                     ("paire la moins compatible", int(proba.argmin()))):
        loc = vals[j]
        recon = base[j] + loc.sum()
        local_ok &= abs(recon - margin[j]) < 1e-3
        o = np.argsort(np.abs(loc))[::-1][:3]
        local_md.append((label, float(proba[j]), [(names[k], float(loc[k])) for k in o]))
    check("9. Explication individuelle (recalculée)", local_ok,
          "base + Σ SHAP = sortie du modèle pour 2 paires extrêmes (cf. rapport)")

    # ---- Cohérence README <-> calcul ----------------------------------------
    readme = open("README.md", encoding="utf-8").read()
    m = re.search(r"Global Attribution[^\n]*", readme)
    claimed = re.findall(r"`([a-z][a-z0-9_]+)`", m.group(0)) if m else []
    claimed = [c for c in claimed if c in names]
    real_top = [names[i] for i in order[:len(claimed) or 4]]
    consistent = set(claimed) == set(real_top)
    check("10. README cohérent avec le calcul", consistent,
          f"README cite {claimed} ; calcul réel (top {len(real_top)}) = {real_top}")

    # ---- Robustesse du code : explainer sans background ---------------------
    src = open("src/evaluation/shap_explainer.py", encoding="utf-8").read()
    risky = "self.initialize_explainer(X_trans)" in src
    check("11. explain_local_instance sans explainer préalable", not risky,
          "initialise le background avec l'instance elle-même → toutes les valeurs SHAP = 0 "
          "(reproduit : marge = 0.0)" if risky else "OK")

    app = open("app.py", encoding="utf-8").read()
    clamp = "max(4, min(35" in app
    hard = "\"impact\": \"+18%\"" in app
    check("12. Interface Flask (app.py)", not (clamp or hard),
          ("pourcentages écrêtés dans [4 %, 35 %] ; " if clamp else "")
          + ("valeurs de repli codées en dur affichées comme explication SHAP si l'explainer échoue" if hard else ""),
          warn=True)

    # ---- Rapport -------------------------------------------------------------
    ok_n = sum(1 for r in results if r[1] == "OK")
    lines = ["# Audit de l'explicabilité SHAP", "",
             "_Généré par `experiments/shap_audit.py` : toutes les valeurs ci-dessous sont recalculées "
             "à partir du modèle sauvegardé et du jeu de test (1000 lignes, `random_state=42`)._", "",
             f"**Bilan : {ok_n}/{len(results)} vérifications OK.**", "",
             "| Critère | Statut | Détail |", "|---|---|---|"]
    lines += [f"| {c} | {s} | {d} |" for c, s, d in results]
    lines += ["", "## Trois principaux facteurs globaux", "",
              top3.to_markdown(index=False) if hasattr(top3, "to_markdown") else top3.to_string(index=False),
              "", "## Explications individuelles", ""]
    for label, p, drivers in local_md:
        lines.append(f"**{label}** (probabilité prédite = {p:.3f}) :")
        for f, v in drivers:
            lines.append(f"- `{f}` : {v:+.3f} ({'pousse vers compatible' if v > 0 else 'freine la compatibilité'})")
        lines.append("")
    os.makedirs("docs", exist_ok=True)
    open(REPORT_MD, "w", encoding="utf-8").write("\n".join(lines))
    print(f"\nRapport : {REPORT_MD}\nTableau : {TOP3_CSV}")


if __name__ == "__main__":
    main()