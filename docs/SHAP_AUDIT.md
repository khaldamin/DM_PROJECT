# Audit de l'explicabilité SHAP

_Généré par `experiments/shap_audit.py` : toutes les valeurs ci-dessous sont recalculées à partir du modèle sauvegardé et du jeu de test (1000 lignes, `random_state=42`)._

**Bilan : 14/14 vérifications OK.**

| Critère | Statut | Détail |
|---|---|---|
| 1. Présence de SHAP | OK | shap 0.54.1 ; import trouvé dans : src/evaluation/shap_explainer.py, experiments/advanced_shap_analysis.py, app.py |
| 2. Compatibilité modèle / explainer | OK | modèle = XGBClassifier → shap.TreeExplainer est exact et adapté (73 features) |
| 3. Calcul des valeurs SHAP (additivité) | OK | écart max entre sortie brute du modèle et (base + Σ SHAP) = 2.62e-06 sur 1000 lignes (unités : log-odds, pas probabilités) |
| 4. Explication globale (JSON = recalcul) | OK | top-10 du JSON identique au recalcul (ordre et valeurs) |
| 5/6. Figure — Summary plot (beeswarm, baseline) | OK | results/figures/shap_summary.png (404 Ko) |
| 5/6. Figure — Summary plot (beeswarm top 20) | OK | results/figures/enhanced_shap_beeswarm_top20.png (441 Ko) |
| 5/6. Figure — Bar plot (moyenne des valeurs absolues) | OK | results/figures/enhanced_shap_bar.png (207 Ko) |
| 5/6. Figure — Waterfall (explication individuelle) | OK | results/figures/shap_waterfall.png (236 Ko) |
| 7. Trois principaux facteurs identifiés | OK | same_love_language (0.6251) > avg_trait_diff (0.5144) > diff_career_ambition (0.3489) |
| 8. Distinction facteurs positifs / négatifs | OK | chaque facteur du top 3 contribue dans les deux sens selon la ligne (cf. tableau) ; le sens global est donné par la corrélation valeur ↔ SHAP |
| 9. Explication individuelle (recalculée) | OK | base + Σ SHAP = sortie du modèle pour 2 paires extrêmes (cf. rapport) |
| 10. README cohérent avec le calcul | OK | README cite ['same_love_language', 'avg_trait_diff', 'diff_career_ambition'] ; calcul réel (top 3) = ['same_love_language', 'avg_trait_diff', 'diff_career_ambition'] |
| 11. explain_local_instance sans explainer préalable | OK | OK |
| 12. Interface Flask (app.py) | OK |  |

## Trois principaux facteurs globaux

|   rang | feature              |   mean_abs_shap |   part_du_total_% |   corr_valeur_vs_shap | sens_global                     |   contrib_moy_positive |   contrib_moy_negative |   %_lignes_positives |
|-------:|:---------------------|----------------:|------------------:|----------------------:|:--------------------------------|-----------------------:|-----------------------:|---------------------:|
|      1 | same_love_language   |          0.6251 |              17.7 |                 0.998 | valeur élevée → compatibilité ↑ |                 1.5978 |                -0.3895 |                 19.5 |
|      2 | avg_trait_diff       |          0.5144 |              14.6 |                -0.991 | valeur élevée → compatibilité ↓ |                 0.4742 |                -0.5577 |                 51.8 |
|      3 | diff_career_ambition |          0.3489 |               9.9 |                -0.955 | valeur élevée → compatibilité ↓ |                 0.3427 |                -0.3551 |                 49.4 |

## Explications individuelles

**paire la plus compatible** (probabilité prédite = 0.996) :
- `same_love_language` : +1.647 (pousse vers compatible)
- `avg_trait_diff` : +1.175 (pousse vers compatible)
- `diff_career_ambition` : +0.975 (pousse vers compatible)

**paire la moins compatible** (probabilité prédite = 0.005) :
- `avg_trait_diff` : -1.423 (freine la compatibilité)
- `diff_career_ambition` : -1.106 (freine la compatibilité)
- `same_love_language` : -0.431 (freine la compatibilité)