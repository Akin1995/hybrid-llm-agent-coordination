# Hybrid LLM Agent Coordination

Eine resiliente Multi-Agenten-Simulation in einer 2D-Grid-Welt: Mehrere Drohnen kooperieren, um einen flüchtenden Dieb zu finden und zu fangen – mit probabilistischen Beliefs, Team-Kommunikation, Rollenplanung und optionaler LLM-Strategie.

## Inhaltsverzeichnis
- [Projektüberblick](#projektüberblick)
- [Features](#features)
- [Architektur im Überblick](#architektur-im-überblick)
- [Wie die Simulation funktioniert](#wie-die-simulation-funktioniert)
- [Rollen, BDI und Koordination](#rollen-bdi-und-koordination)
- [Resilienz-Konzept (LLM + Fallback)](#resilienz-konzept-llm--fallback)
- [Voraussetzungen](#voraussetzungen)
- [Installation](#installation)
- [Schnellstart](#schnellstart)
- [Konfiguration der Simulation](#konfiguration-der-simulation)
- [OpenAI/LLM-Integration](#openaillm-integration)
- [Ausgaben und Artefakte](#ausgaben-und-artefakte)
- [Typische Experimente](#typische-experimente)
- [Bekannte Grenzen](#bekannte-grenzen)
- [Projektstruktur](#projektstruktur)
- [FAQ](#faq)

## Projektüberblick

Dieses Projekt simuliert ein Team aus Drohnen, das in einer dynamischen Umgebung nach einem Dieb sucht. Die Kernidee:

1. **Jede Drohne führt lokale Wahrnehmung und Belief-Updates durch** (inkl. Wahrscheinlichkeitskarte für die Diebsposition).
2. **Drohnen tauschen Sichtungen als Nachrichten aus**.
3. Ein zentraler Planungsprozess erzeugt Team-Assignments (Rolle + Ziel pro Drohne):
   - bevorzugt über ein LLM,
   - robust abgesichert über Heuristiken/Fallback, falls das LLM ausfällt oder unbrauchbare Pläne liefert.
4. Die Agenten bewegen sich kollisionsarm im Raster, während statische und dynamische Hindernisse berücksichtigt werden.

Das Ergebnis ist eine kombinierte Architektur aus klassischer AI (BFS, Heuristiken, probabilistische Suche, Rollenlogik) und optionaler LLM-Entscheidungsunterstützung.

## Features

- **Dynamische Grid-Welt** mit statischen und beweglichen Hindernissen.
- **Mehrere Drohnen** mit:
  - begrenzter Sicht,
  - BDI-nahem Zustand (*Beliefs, Desires, Intentions*),
  - Rollenverhalten (`intercept`, `contain`, `investigate`, `search`, `reposition`).
- **Dieb-Agent** mit eigener Sicht und Ausweichverhalten.
- **Probabilistisches Tracking** (Diffusion + Ausschluss sichtbarer Freiflächen + Team-Aggregation).
- **Kommunikation im Team** (`THIEF_SPOTTED`-Nachrichten).
- **Resiliente Planung**:
  - LLM-gestützte Strategie (optional),
  - strukturierte JSON-Ausgabe mit Schema,
  - Validierung/Safety-Layer,
  - Fallback-/Degraded-Mode,
  - Cached-Plan-Reuse und Cooldown nach wiederholten LLM-Fehlern.
- **Visualisierung** pro Zeitschritt (PNG-Frames) und optional als GIF.

## Architektur im Überblick

### Hauptkomponenten

- `GridWorld`
  - verwaltet Grenzen, Passierbarkeit, Nachbarn, Hindernisse,
  - aktualisiert dynamische Hindernisse.
- `Drone`
  - Wahrnehmung + Belief-Update,
  - Nachrichteneingang,
  - Rollen-/Ziel-Intention,
  - lokale Bewegungsentscheidung via BFS + Heuristik.
- `Thief`
  - bewegt sich zufällig ohne Sichtkontakt,
  - weicht sichtbaren Drohnen aus,
  - besitzt Ausdauer-/Regenerationsmechanik.
- Planungslogik
  - erzeugt Team-Zuweisungen (LLM-hybrid oder Fallback),
  - validiert Rollen/Ziele,
  - verwaltet Plan-Speicher, Replanning-Horizont, Degraded-State.
- Rendering
  - erzeugt Frames mit Drohnen, Dieb, Hindernissen, Sichtbereichen und Hotspots.

### Datenfluss pro Simulationsschritt

1. Dynamische Hindernisse werden bewegt.
2. Drohnen beobachten die Welt und senden ggf. Sichtungsnachrichten.
3. Nachrichten werden im Team verteilt und Beliefs aktualisiert.
4. Alte Sichtungen verfallen über Zeit (Decay).
5. Team-Planung entscheidet Rollen/Ziele (LLM oder Fallback).
6. Drohnen bewegen sich koordiniert und fair aufgelöst.
7. Alle aktiven Diebe ziehen.
8. Capture-Bedingungen werden pro Dieb geprüft; gefangene Diebe werden dynamisch aus der Simulation entfernt.
9. Frame wird gezeichnet.

## Wie die Simulation funktioniert

### Suchlogik über Wahrscheinlichkeiten

- Jede Drohne hält eine `thief_probability_map`.
- Die Karte wird pro Tick diffusiv fortgeschrieben (inkl. „wait“-Option des Diebs).
- Zellen im aktuellen Sichtfeld ohne Dieb werden auf 0 gesetzt.
- Bei direkter Sichtung wird die Verteilung auf die gesichtete Position konzentriert.
- Teamweit werden Karten aggregiert, um Hotspots und Entropie abzuleiten.

### Capture-Regeln

Ein Fang zählt, wenn mindestens eine Bedingung erfüllt ist:

1. Eine Drohne steht auf der Diebszelle.
2. Kantenkreuzung: Drohne und Dieb tauschen im selben Tick die Positionen.
3. Die aktuelle Diebszelle plus alle legalen Folgezellen sind durch Drohnen blockiert.

## Rollen, BDI und Koordination

### BDI-nahe Zustände in den Drohnen

- **Beliefs**: letzte Sichtung, geschätzte Diebsposition, Wahrscheinlichkeitskarte, Team-Reports usw.
- **Desires**: z. B. `capture_thief`, `investigate_hotspot`, `search_thief`.
- **Intentions**: konkrete Rolle und Ziel pro Drohne.

### Rollen

- `intercept`: direkte Annäherung/Verfolgung.
- `contain`: Fluchtwege blockieren.
- `investigate`: Hotspots/letzte Sichtungen prüfen.
- `search`: systematische Sektorsuche.
- `reposition`: aus ungünstiger/gestauter Lage neu positionieren.

## Resilienz-Konzept (LLM + Fallback)

Die zentrale Stärke des Projekts ist, dass das LLM **nicht** zum Single Point of Failure wird.

### Wenn LLM aktiv ist

- Es wird ein strukturierter Teamzustand erstellt.
- Das LLM liefert einen Plan im JSON-Schema.
- Der Plan wird validiert (Rollenmenge, freie Ziele im Grid, Zielkollisionen vermeiden).

### Bei LLM-Problemen

- Ungültige oder fehlende Antworten führen nicht zum Abbruch.
- Ein vorhandener, zuletzt gültiger Plan kann kurzfristig weiter genutzt werden.
- Danach greift ein deterministischer Fallback-Planer.
- Nach mehreren Fehlern wird LLM temporär per Cooldown deaktiviert.

## Voraussetzungen

- Python **3.10+** empfohlen
- Abhängigkeiten aus `requirements.txt`

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

## Schnellstart

> Hinweis: Das aktuelle Skript startet im `__main__`-Block mehrere Runs in einer Schleife (`seed` 30–59) und erzeugt jeweils Frames + GIF.

```bash
python drone_ai_agents_resilient.py
```

Je nach Grid-Größe, Anzahl Runs und GIF-Erzeugung kann das länger dauern und viel Speicherplatz verbrauchen.

## Konfiguration der Simulation

Die zentrale Funktion ist:

```python
run_dynamic_simulation_llm(
    width=40,
    height=40,
    static_obstacle_ratio=0.05,
    dynamic_obstacle_ratio=0.05,
    num_drones=6,
    num_thieves=1,
    sight_radius_drone=5,
    sight_radius_thief=4,
    max_steps=250,
    output_dir="frames",
    save_gif="dynamic_simulation_llm.gif",
    use_llm=True,
    llm_api_key=None,
    llm_model="gpt-5.4-mini",
    llm_structured=True,
    seed=None,
)
```

### Wichtige Parameter

- `num_drones`: Teamgröße.
- `num_thieves`: Anzahl initialer Diebe (werden auf eindeutigen freien Zellen gespawnt).
- `sight_radius_drone` / `sight_radius_thief`: Sichtweite von Drohnen/Dieb.
- `static_obstacle_ratio` / `dynamic_obstacle_ratio`: Dichte der Hindernisse.
- `max_steps`: maximale Laufzeit pro Simulation.
- `use_llm`: aktiviert/deaktiviert LLM-Planung.
- `llm_structured`: nutzt JSON-Schema-Ausgabe für robustere Parsing-/Validierungspfad.
- `seed`: reproduzierbare Runs.

### Mehrere Diebe & dynamische Entfernung

- Jeder Dieb besitzt eine stabile `id`.
- Sichtungen/Nachrichten enthalten die `thief_id`, damit Zuordnungen bei mehreren Dieben eindeutig bleiben.
- Wird ein Dieb gefangen, wird er sofort aus der aktiven Liste entfernt; die Simulation läuft mit verbleibenden Dieben weiter, bis alle gefangen sind oder `max_steps` erreicht ist.

## OpenAI/LLM-Integration

Die LLM-Funktion ist optional. Ohne API-Key funktioniert das Projekt weiterhin über Heuristik/Fallback.

### API-Key setzen

```bash
export OPENAI_API_KEY="dein_key"
```

oder direkt als Argument (`llm_api_key=...`) übergeben.

### Modelle

Standardmäßig wird im Code ein Modellname wie `gpt-5.4-mini` verwendet. Passe diesen Namen bei Bedarf an die in deiner Umgebung verfügbaren Modelle an.

## Ausgaben und Artefakte

Während der Laufzeit entstehen:

- PNG-Frames pro Schritt im jeweiligen `output_dir` (`frame_000.png`, ...)
- optional ein GIF pro Run
- Konsolenmetriken, z. B.:
  - Anzahl erfolgreicher LLM-Planungen,
  - Anzahl Fallback-Nutzungen,
  - Anzahl Cached-Plan-Reuses,
  - Anzahl Schritte im Degraded Mode,
  - Gesamtzahl gesendeter Nachrichten,
  - Schritt der ersten Sichtung.

## Typische Experimente

- **LLM vs. Heuristik vergleichen**: gleiche Seeds mit `use_llm=True/False` laufen lassen.
- **Robustheit testen**: API-Key weglassen und prüfen, ob Fallback stabil weiterläuft.
- **Skalierung**: `num_drones`, `width/height`, Hindernisdichte erhöhen.
- **Informationsunsicherheit**: Sichtweiten variieren und Auswirkungen auf Entropie/Fangzeit beobachten.

## Bekannte Grenzen

- Kein echtes Multi-Processing/Realtime-System: alles läuft in einem Simulationsloop.
- Kein physikalisches Drohnenmodell (Akkumodell, Flugphysik, Kommunikationlatenz etc.).
- Heuristiken sind bewusst einfach und gut nachvollziehbar, nicht global optimal.
- Der aktuelle `__main__`-Block ist experimentorientiert (Batch-Runs) statt CLI-basiert.

## Projektstruktur

```text
.
├── drone_ai_agents_resilient.py   # Simulation, Planung, Visualisierung
└── requirements.txt               # Python-Abhängigkeiten
```

## FAQ

### Läuft das Projekt ohne OpenAI-Zugang?
Ja. Das LLM ist optional. Ohne Key wird automatisch auf robuste Fallback-Logik gesetzt.

### Warum werden so viele Bilder erzeugt?
Die Simulation rendert standardmäßig pro Schritt ein PNG und daraus optional ein GIF. Für schnelle Tests kannst du `max_steps` reduzieren oder `save_gif=None` setzen.

### Wie mache ich Runs reproduzierbar?
Setze `seed` auf einen festen Wert. So bleiben Initialisierung und Zufallsschritte zwischen Runs vergleichbar.

---
