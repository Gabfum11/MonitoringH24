# MonitoringH24

Sistema di monitoraggio continuo e non invasivo per anziani in ambiente domestico. Produce diari clinici narrativi a più livelli temporali tramite un Vision Language Model (VLM) eseguito localmente.

## Cosa fa

Una telecamera Xiaomi Smart Camera C500 trasmette il flusso video in tempo reale sulla rete locale. Il sistema cattura periodicamente questo flusso e lo analizza tramite **Gemma 4 26B** (VLM locale via LM Studio) per produrre osservazioni in linguaggio naturale sullo stato della persona: presenza, postura, attività in corso, stabilità nei movimenti.

La frequenza di osservazione è adattiva: aumenta in presenza di movimento, si riduce quando la scena è stabile, per risparmiare risorse computazionali.

### Rilevamento anomalie
Il sistema segnala automaticamente:
- Assenza prolungata della persona (>30 minuti nelle ore diurne)
- Cadute o instabilità grave
- Difficoltà motorie evidenti
- Presenza di estranei
- Cambiamenti ambientali rilevanti (oggetti caduti, ostacoli)

### Test clinici
Tramite bot Telegram è possibile avviare su richiesta:
- **TUG** (Timed Up and Go)
- **STS** (Five Times Sit-to-Stand)

Durante i test viene attivato **MediaPipe** per la stima della posa in tempo reale: rileva le coordinate articolari, lo stato posturale, la velocità di spostamento e i tempi di esecuzione. I risultati vengono salvati nel database e integrati nei diari.

### Diari e report
- **Ogni ora**: sintesi delle osservazioni dell'ora trascorsa
- **A mezzanotte**: diario giornaliero organizzato per fasce orarie (mattina, pomeriggio, sera/notte)
- **Ogni lunedì**: report settimanale sull'andamento della mobilità
- **Primo del mese**: report mensile con confronto rispetto al mese precedente
- **1 gennaio**: report annuale con evoluzione trimestrale e abitudini stagionali

### Interfaccia Telegram
I familiari possono interrogare il sistema in linguaggio naturale:
- "Come sta adesso?"
- "Cosa ha fatto stamattina?"
- "Ci sono stati problemi ieri pomeriggio?"
- "È migliorata rispetto alla settimana scorsa?"

Le domande vengono classificate automaticamente e il sistema recupera i dati più rilevanti: osservazioni recenti, file storici per date esplcite, o ricerca semantica sull'archivio tramite **RAG** (Retrieval-Augmented Generation) con ChromaDB e BGE-M3.

Comandi disponibili: `/stato`, `/ieri`, `/alert`, `/tug`, `/sts`

## Architettura

| Componente | Ruolo |
|---|---|
| `Monitor.py` | Loop principale di monitoraggio |
| `Capture.py` | Cattura frame e rilevamento cambiamenti |
| `Observer.py` | Logica di osservazione, intervallo adattivo, alert |
| `Vlm_calls.py` | Client per Gemma via LM Studio |
| `Diary_generator.py` | Generazione riepiloghi e diari |
| `TestRunner.py` | Esecuzione test TUG e STS |
| `PoseDetector.py` | Wrapper MediaPipe |
| `Telegram.py` | Bot Telegram per i familiari |
| `Rag.py` | Indicizzazione e ricerca semantica |
| `Database_manager.py` | Persistenza risultati test clinici |

## Requisiti
- Python 3.11
- LM Studio con Gemma 4 27B in esecuzione locale
- Telecamera Xiaomi C500 sulla rete locale
- Token bot Telegram

## Installazione
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python Monitor.py
```
