"""
Client per il Vision Language Model.

Gestisce le chiamate a LM Studio (API compatibile OpenAI):
- Chiamate con immagini (singole o sequenze)
- Chiamate solo testo (per sintesi e diari)
- Contesto orario per guidare l'analisi clinica
"""

import requests
from datetime import datetime


class VLMClient:
    def __init__(self, model="gemma-4-26b-a4b-it", lmstudio_url="http://localhost:1234"):
        """
        Args:
            model: nome del modello su LM Studio
            lmstudio_url: URL del server LM Studio
        """
        self.model = model
        self.lmstudio_url = lmstudio_url

        # Prompt di sistema per le osservazioni
        self.system_prompt = (
            "Sei un assistente per il monitoraggio domiciliare di una persona anziana.\n\n"
            "Nell'ambiente possono essere presenti una o più persone, "
            "concentrati sulla persona anziana se visibile.\n"
            "Analizza l'immagine e descrivi SOLO informazioni rilevanti dal punto di vista clinico.\n\n"
            "In ogni risposta, valuta sempre:\n"
            "- Presenza o assenza della persona\n"
            "- Postura (seduta, in piedi, sdraiata)\n"
            "- Attività in corso\n"
            "- Stabilità e movimento (normale, lento, incerto)\n"
            "- Se la persona è assistita da qualcuno nell'alzarsi, camminare o altre azioni\n"
            "- Possibili segnali di rischio (caduta, immobilità prolungata, difficoltà)\n\n"
            "Se nelle osservazioni precedenti la persona era in una posizione diversa, "
            "segnala il cambiamento.\n\n"
            "Scrivi in italiano, massimo 2-3 frasi, stile cartella clinica.\n"
            "Sii oggettivo e preciso.\n"
            "NON descrivere dettagli irrilevanti (arredamento, luce, ecc.) "
            "a meno che non siano importanti per la sicurezza.\n"
            "Se la persona non è visibile, dichiaralo chiaramente."
        )

    # =========================================
    # CONTESTO ORARIO
    # =========================================
    def get_time_context(self):
        """Contesto clinico basato sull'ora del giorno.
        
        Guida il VLM a interpretare la scena in modo diverso
        a seconda della fascia oraria (notte vs giorno, pasto vs riposo).
        """
        hour = datetime.now().hour
        if 23 <= hour or hour < 6:
            return (
                "È notte. La persona dovrebbe essere a riposo. "
                "Un'alzata breve può indicare un bisogno fisiologico, "
                "ma attività prolungata o instabilità sono potenzialmente anomale."
            )
        elif 6 <= hour < 8:
            return "È mattina presto, orario tipico del risveglio. Valuta stabilità nei movimenti."
        elif 8 <= hour < 12:
            return "È mattina. Valuta continuità delle attività e mobilità."
        elif 12 <= hour < 14:
            return "È ora di pranzo. Verifica presenza e autonomia."
        elif 14 <= hour < 18:
            return "È pomeriggio. Osserva eventuale sonnolenza o difficoltà nei movimenti."
        elif 18 <= hour < 20:
            return "È ora di cena. Valuta attività e autonomia."
        elif 20 <= hour < 23:
            return "È sera, orario pre-riposo. Osserva stabilità nei movimenti."
        return ""

    # =========================================
    # CHIAMATA CON IMMAGINI
    # =========================================
    def call_with_images(self, images_b64, context_messages=None, max_tokens=200, prompt_text=None):
        """Invia una o più immagini a LM Studio con contesto.
        
        Args:
            images_b64: stringa base64 (singola) o lista di stringhe (sequenza)
            context_messages: contesto conversazionale (lista di messaggi)
            max_tokens: token massimi per la risposta
            prompt_text: testo personalizzato (default: genera automaticamente)
        """
        messages = [{"role": "system", "content": self.system_prompt}]

        if context_messages:
            messages.extend(context_messages)

        now = datetime.now().strftime("%H:%M:%S")
        time_ctx = self.get_time_context()

        # Costruisci il contenuto con una o più immagini
        current_max=max_tokens
        if isinstance(images_b64, list) and len(images_b64) > 1:
            current_max=350
            if prompt_text is None:
                prompt_text = (
                    f"Ore {now}. {time_ctx} "
                    f"Questa è una sequenza di {len(images_b64)} frame consecutivi "
                    f"catturati in pochi secondi. Descrivi l'azione in corso: "
                    f"cosa sta facendo la persona? Come si muove? "
                    f"Noti difficoltà, instabilità o qualcosa di rilevante?"
                )
            content = [{"type": "text", "text": prompt_text}]
            for img in images_b64:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img}"}
                })
        else:
            if prompt_text is None:
                prompt_text = f"Ore {now}. {time_ctx} Descrivi cosa vedi."
            content = [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{images_b64}"}}
            ]

        messages.append({"role": "user", "content": content})

        try:
            response = requests.post(
                f"{self.lmstudio_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": current_max,
                    "temperature": 0.3
                },
                timeout=60 
            )
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            else:
                print(f"[VLM] Errore: {response.status_code}")
                return None
        except Exception as e:
            print(f"[VLM] Errore connessione: {e}")
            return None

    # =========================================
    # CHIAMATA SOLO TESTO
    # =========================================
    def call_text(self, prompt, system=None, max_tokens=800):
        """Chiamata solo testo per sintesi orarie, diari e report.
        
        Args:
            prompt: testo del prompt
            system: prompt di sistema (opzionale, override del default)
            max_tokens: token massimi per la risposta
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = requests.post(
                f"{self.lmstudio_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.4
                },
                timeout=120
            )
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            else:
                print(f"[VLM] Errore: {response.status_code}")
                return None
        except Exception as e:
            print(f"[VLM] Errore: {e}")
            return None