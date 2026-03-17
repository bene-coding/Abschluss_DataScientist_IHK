# Vorhersage der Feuerwiderstandsdauer von faserverstärkten Betonträgern
## Abschlussprojekt zum Kurs Data Scientist 2025-2 der IHK Rhein-Neckar (kondensierte Fassung)
**Kurslaufzeit** vom 27.11.2025 bis zum 14.02.2026, berufsbegleitend  
**Bearbeitungszeit**: 2 Wochen  
**Ausarbeitung** von [Benedikt Pöschl](https://www.linkedin.com/in/benepoeschl)  
**Nutzung** nur im Rahmen des Kurses  

**Quellen**:  
[1] 📄 **Paper**: [Dataset on fire resistance analysis of FRP-strengthened concrete beams (Bhatt et al., 2024)](https://doi.org/10.1016/j.dib.2024.110031)  
[2] 📊 **Daten**: [Bhatt, Pratik (2023), “Fire Resistance of FRP-Strengthened Beams”, Mendeley Data, V6](https://data.mendeley.com/datasets/3c2szhbdn5/6)
## Aufgabenstellung des Abschlussprojekts
Gefordert war der eigenständige Aufbau einer vollständigen Data-Science-Pipeline mit einem Businesscase im Hintergrund.
Die Arbeitsschritte sollten alles von der EDA und Data Preprocessing (Umgang mit fehlenden Werten, Outlier Detection) über Feature Engineering und Modellauswahl bis hin zum Hyperparameter-Tuning und einer Kreuzvalidierung (Cross-Validation) sowie abschließender Evaluation beinhalten. Empfohlen wurden hierfür etablierte Standard-Szenarien wie die Vorhersage von Immobilienpreisen (Housing Prices) oder Kundenabwanderung (Churn Prediction).  

Anstatt auf bereinigte Standard-Datensätze zurückzugreifen, wurde für diese Arbeit bewusst ein hochkomplexer, praxisnaher Use-Case aus dem Bauingenieurwesen gewählt: Die Vorhersage der Feuerwiderstandsdauer von faserverstärkten Betonträgern.  

Üblicherweise spricht man beim Arbeitsaufwand von einer Aufteilung von etwa 80% Preprocessing und 20% für den Rest. Insbesondere im gewählten Kontext sehe ich die notwendige Aufteilung als 45% domain-driven Preprocessing, 20% Modell-Architektur und 35% Evaluierung und Plausibilisierung.  
## Business Case
Im konstruktiven Brandschutz entscheidet die Feuerwiderstandsdauer von tragenden Bauteilen im Ernstfall über Menschenleben. Die traditionelle Bemessung von Stahlbetonbalken stützt sich auf konservative Tabellenwerte oder auf zeitintensive numerische Simulationen. Während Erstere bisweilen unwirtschaftlich sein können, erfordern Letztere tiefgreifendes Expertenwissen und hohe Zeit- und Rechenkapazitäten.  

Die Entwicklung präziser Vorhersagemodelle steht im konstruktiven Brandschutz vor der klassischen Hürde des "Small Data" im Ingenieurwesen, unter anderem da Brandversuche kostenbedingt auf das Nötigste beschränkt werden. In diesem Projekt stehen 50 reale Versuchsdatensätze einer Masse von mehr als 20.000 synthetischen Simulationsdaten gegenüber. Weiterhin sind Laborversuche oft "rechtszensiert". Das bedeutet viele Tests enden planmäßig bevor das untersuchte Bauteil tatsächlich versagt, was herkömmliche Regressionsmodelle systematisch verzerrt.  

Wie die minutengenaue Vorhersage in dieser Ausarbeitung zeigt, haben Machine-Learning Modelle das Potential, um mit entsprechenden Daten durch die Vorhersagen die Versuchsplanung zu unterstützen, Versuche kosteneffizienter zu gestalten oder langfristig als ergänzendes Werkzeug in der Bemessung eingesetzt zu werden.
### Disclaimer
Ziel dieses Abschlussprojekts soll die Entwicklung eines Machine-Learning Modells sein, das das Domänenwissen eines Bauingenieurs mit dem eines Data Scientists verbindet und die Kursinhalte aufgreift. Der Fokus liegt somit nicht vorrangig auf der ingenieurmäßigen Bewertung bzw. Interpretation der Ergebnisse.

Da die Quellen [1, 2] wenig Informationen zu den Daten selbst bzw. deren Zustandekommen enthalten, die referenzierte Literatur teilweise nicht frei zugänglich ist und auch der Bearbeitungszeitraum für eine extensive Literaturrecherche zu knapp bemessen ist, wird mit dieser vergleichsweise geringen Informationsdichte gearbeitet. Das hier erarbeitete Modell dient lediglich akademischen Demonstrationszwecken im Rahmen des Abschlussprojekts für den genannten Kurs und darf nicht anderweitig eingesetzt werden. Für den realen Einsatz müssten mindestens die Anwendungsgrenzen erarbeitet, Validierungsversuche durchgeführt und jeglicher Bias transparent herausgestellt werden.  

Der Autor übernimmt daher keine Haftung für Schäden, die aus einer unsachgemäßen Verwendung oder dem Vertrauen auf die Modellergebnisse resultieren. Die Nutzung des Modells wie auch damit erzielte Ergebnisse erfolgen auf eigene Gefahr.
## Inhalte
- Data Challenge:
    - Nichtlineare Korrelationen
    - Imbalanced Data
    - Rechtszensur
    - Vorhandener Bias:
        - Selection & Sampling
        - Measurement
        - Design / Confounding
        - Simulation / Model
        - Feature

- Lösungsstrategie:
    - Umfassende EDA (Exploratory Data Analysis)
    - Preprocessing
        - Data Cleaning
        - Feature Engineering und Reverse Engineering
        - Feature Reduction
    - Modell Architektur mit XGBoost:
        - Custom Splits & Sample Weights (für Imbalanced Data)
        - Custom Objective Function (für Rechtszensur)
        - Monotonic Constraints (für Overfitting)
    - Optimierung:
        - Optuna (Hyperparameter Tuning & automatische Feature Selection)
    - Evaluierung & Interpretation:
        - Custom Scoring
        - SHAP-Analyse

- Fazit
