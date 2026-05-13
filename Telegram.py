"""
Bot Telegram per il monitoraggio domiciliare.

Permette ai familiari di interrogare il sistema tramite Telegram:
- "Cosa ha fatto alle 9?"
- "Come sta oggi?"
- "Ci sono stati problemi?"
- "Mandami il diario di ieri"

Usa Gemma (LM Studio) per rispondere basandosi sulle osservazioni reali.

Uso:
    python telegram_bot.py

Richiede:
    pip install python-telegram-bot

Configurazione:
    Crea un bot con @BotFather su Telegram e ottieni il token.
    Imposta il token come variabile d'ambiente:
        export TELEGRAM_BOT_TOKEN="il_tuo_token"

    Oppure passalo come argomento:
        python telegram_bot.py --token "il_tuo_token"
"""

import re
import json
import asyncio
from datetime import date, timedelta
from pathlib import Path
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from Database_manager import DatabaseManager

GIORNI_IT = {
    "lunedì": 0, "lunedi": 0,
    "martedì": 1, "martedi": 1,
    "mercoledì": 2, "mercoledi": 2,
    "giovedì": 3, "giovedi": 3,
    "venerdì": 4, "venerdi": 4,
    "sabato": 5,
    "domenica": 6,
}

MESI_IT = {
    "gennaio": 1, "febbraio": 2, "marzo": 3, "aprile": 4,
    "maggio": 5, "giugno": 6, "luglio": 7, "agosto": 8,
    "settembre": 9, "ottobre": 10, "novembre": 11, "dicembre": 12
}

MESI_NOMI = {
    1: "Gennaio", 2: "Febbraio", 3: "Marzo", 4: "Aprile",
    5: "Maggio", 6: "Giugno", 7: "Luglio", 8: "Agosto",
    9: "Settembre", 10: "Ottobre", 11: "Novembre", 12: "Dicembre"
}



class MonitorBot:
    def __init__(self, token, vlm_client, data_dir="diari", test_runner=None, allowed_ids=None):
        self.token = token
        self.vlm = vlm_client
        self.data_dir = Path(data_dir)
        self.test_runner = test_runner
        self.allowed_ids = set(allowed_ids) if allowed_ids else set()
        self.db = DatabaseManager()

    async def _check_auth(self, update: Update) -> bool:
        if not self.allowed_ids:
            return True
        if update.effective_user.id not in self.allowed_ids:
            await update.message.reply_text("Non sei autorizzato ad usare questo bot.")
            return False
        return True

    # =========================================
    # LETTURA DATI
    # =========================================
    def _format_test_history(self, test_type="TUG", n=6):
        """Legge gli ultimi N test dal DB e restituisce un confronto velocità leggibile."""
        end = date.today().isoformat()
        start = (date.today() - timedelta(days=90)).isoformat()

        if test_type == "TUG":
            results = self.db.get_tug_results(start, end)
        else:
            results = self.db.get_sts_results(start, end)

        if not results:
            return None

        recent = results[-n:]
        lines = [f"STORICO TEST {test_type} (ultimi {len(recent)} test):"]

        for i, r in enumerate(recent):
            d = r['date']
            t = r['total_time']
            speed = r.get('avg_speed_px_s', 0)

            if i == 0:
                lines.append(f"- {d}: {t:.1f}s")
            else:
                prev_speed = recent[i - 1].get('avg_speed_px_s', 0)
                if prev_speed and speed:
                    delta_pct = ((speed - prev_speed) / prev_speed) * 100
                    if delta_pct > 5:
                        andamento = f"più veloce del {delta_pct:.0f}% rispetto al precedente"
                    elif delta_pct < -5:
                        andamento = f"più lento del {abs(delta_pct):.0f}% rispetto al precedente"
                    else:
                        andamento = "prestazione simile al precedente"
                    lines.append(f"- {d}: {t:.1f}s ({andamento})")
                else:
                    lines.append(f"- {d}: {t:.1f}s")

        if len(recent) >= 2:
            first_speed = recent[0].get('avg_speed_px_s', 0)
            last_speed = recent[-1].get('avg_speed_px_s', 0)
            if first_speed and last_speed:
                overall_pct = ((last_speed - first_speed) / first_speed) * 100
                if overall_pct > 5:
                    lines.append(f"Tendenza generale: miglioramento del {overall_pct:.0f}%")
                elif overall_pct < -5:
                    lines.append(f"Tendenza generale: peggioramento del {abs(overall_pct):.0f}%")
                else:
                    lines.append("Tendenza generale: prestazioni stabili")

        return "\n".join(lines)

    def _get_today_observations(self):
        return self._get_observations_for_date(date.today())

    def _get_observations_for_date(self, target_date):
        """Legge osservazioni e riepiloghi orari da data.json per una data specifica."""
        d = target_date
        path = self.data_dir / str(d.year) / f"{d.month:02d}" / d.isoformat() / "data.json"
        if not path.exists():
            return None, None
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("observations", []), data.get("hourly_summaries", [])

    def _extract_referenced_dates(self, query):
        """Estrae date referenziate nella query (ieri, lunedì, 3 maggio, ecc.)."""
        today = date.today()
        q = query.lower()
        dates = []

        if "altro ieri" in q:
            dates.append(today - timedelta(days=2))
        elif "ieri" in q:
            dates.append(today - timedelta(days=1))

        if "settimana scorsa" in q:
            # Lunedì della settimana scorsa → domenica
            monday_last = today - timedelta(days=today.weekday() + 7)
            dates += [monday_last + timedelta(days=i) for i in range(7)]

        for nome, weekday in GIORNI_IT.items():
            if nome in q:
                days_back = (today.weekday() - weekday) % 7
                if days_back == 0:
                    days_back = 7
                dates.append(today - timedelta(days=days_back))

        for nome_mese, mese_num in MESI_IT.items():
            m = re.search(rf"\b(\d{{1,2}})\s+{nome_mese}\b", q)
            if m:
                giorno = int(m.group(1))
                year = today.year if mese_num <= today.month else today.year - 1
                try:
                    dates.append(date(year, mese_num, giorno))
                except ValueError:
                    pass

        return sorted({d for d in dates if d < today}, reverse=True)

    def _get_diary(self, target_date=None):
        """Legge il diario di una data specifica."""
        if target_date is None:
            target_date = date.today()
        d = target_date
        path = self.data_dir / str(d.year) / f"{d.month:02d}" / d.isoformat() / "diario.txt"
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        return None

    def _build_context_for_query(self, query):
        """Costruisce il contesto rilevante per la domanda del familiare."""
        today = date.today()
        context = ""

        # Dati di oggi
        observations, hourly_summaries = self._get_today_observations()
        if hourly_summaries:
            context += "RIEPILOGHI ORARI DI OGGI:\n"
            for s in sorted(hourly_summaries, key=lambda x: x['hour']):
                context += f"[{s['hour_label']}] {s['summary']}\n\n"
        if observations:
            context += "ULTIME OSSERVAZIONI DI OGGI:\n"
            for o in observations[-20:]:
                tag = " [ALERT]" if o.get('type') == 'alert' else (" [CONFRONTO]" if o.get('type') == 'confronto' else "")
                context += f"- {o['time']}{tag}: {o['description']}\n"

        # Date referenziate nella domanda (max 3 per non sovraccaricare il contesto)
        ref_dates = self._extract_referenced_dates(query)[:3]
        for ref_date in ref_dates:
            label = f"{ref_date.day} {MESI_NOMI[ref_date.month]} {ref_date.year}"
            obs, summaries = self._get_observations_for_date(ref_date)
            if summaries:
                context += f"\nRIEPILOGHI ORARI DEL {label}:\n"
                for s in sorted(summaries, key=lambda x: x['hour']):
                    context += f"[{s['hour_label']}] {s['summary']}\n\n"
            elif obs:
                context += f"\nOSSERVAZIONI DEL {label}:\n"
                for o in obs[-15:]:
                    tag = " [ALERT]" if o.get('type') == 'alert' else ""
                    context += f"- {o['time']}{tag}: {o['description']}\n"
            else:
                diary = self._get_diary(ref_date)
                if diary:
                    context += f"\nDIARIO DEL {label}:\n{diary}\n"

        # Storico test clinici se la domanda li menziona
        q = query.lower()
        if any(k in q for k in ["tug", "test", "cammin", "velocit", "mobilit"]):
            tug_history = self._format_test_history("TUG")
            if tug_history:
                context += f"\n{tug_history}\n"
        if any(k in q for k in ["sts", "sit", "alzat", "sedut", "ripetiz"]):
            sts_history = self._format_test_history("STS")
            if sts_history:
                context += f"\n{sts_history}\n"

        # Fallback se non c'è nulla
        if not context:
            yesterday_diary = self._get_diary(today - timedelta(days=1))
            if yesterday_diary:
                context = f"DIARIO DI IERI:\n{yesterday_diary}\n"
            else:
                context = "Nessun dato disponibile per oggi o ieri."

        return context

    # =========================================
    # RISPOSTE VLM
    # =========================================
    def _answer_query(self, query):
        """Usa Gemma per rispondere alla domanda basandosi sui dati reali."""
        context = self._build_context_for_query(query)

        prompt = (
            f"Sei un assistente per il monitoraggio domiciliare di una persona anziana. "
            f"Un familiare ti chiede informazioni.\n\n"
            f"DATI DISPONIBILI:\n{context}\n\n"
            f"DOMANDA DEL FAMILIARE: {query}\n\n"
            f"Rispondi in italiano in modo chiaro. "
            f"Basati SOLO sui dati forniti, non inventare. "
            f"Se non hai dati sufficienti per rispondere, dillo chiaramente. "
            f"Sii conciso (3-5 frasi massimo)."
        )

        response = self.vlm.call_text(
            prompt,
            system=(
                "Sei un assistente domiciliare che comunica con i familiari. "
                "Rispondi in modo semplice e comprensibile, evitando gergo medico. "
                "Fornisci informazioni utili basate sui dati, ma non fare diagnosi o previsioni. "
            ),
            max_tokens=500
        )

        return response or "Mi dispiace, non riesco a rispondere in questo momento. Riprova tra poco."

    # =========================================
    # HANDLER TELEGRAM
    # =========================================
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Messaggio di benvenuto."""
        if not await self._check_auth(update):
            return
        await update.message.reply_text(
            "Ciao! Sono il bot di monitoraggio domiciliare.\n\n"
            "Puoi chiedermi cose come:\n"
            "• Come sta oggi?\n"
            "• Cosa ha fatto stamattina?\n"
            "• Ci sono stati problemi?\n"
            "• Cosa faceva alle 15?\n\n"
            "Comandi:\n"
            "/stato - Stato attuale\n"
            "/diario - Diario di oggi\n"
            "/ieri - Diario di ieri\n"
            "/alert - Alert della giornata\n"
            "/tug - Avvia test TUG\n"
            "/sts - Avvia test STS"
        )

    async def cmd_stato(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Stato attuale basato sulle ultime osservazioni."""
        if not await self._check_auth(update):
            return
        observations, _ = self._get_today_observations()
        if not observations:
            await update.message.reply_text("Nessuna osservazione disponibile per oggi.")
            return

        last = observations[-1]
        n_obs = len(observations)
        alerts = [o for o in observations if o.get('type') == 'alert']

        msg = f" Stato alle {last['time']}:\n{last['description']}\n\n"
        msg += f"Osservazioni oggi: {n_obs}\n"
        if alerts:
            msg += f" Alert: {len(alerts)}"
        else:
            msg += "Nessun alert"

        await update.message.reply_text(msg) #permette di inviare un altro mesaggio mentre il sistema sta elaborando la risposta precedente, evitando blocchi o ritardi e migliorando l'esperienza utente.

    async def cmd_diario(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Invia il diario di oggi."""
        if not await self._check_auth(update):
            return
        diary = self._get_diary()
        if diary:
            # Telegram ha un limite di 4096 caratteri per messaggio
            if len(diary) > 4000:
                parts = [diary[i:i+4000] for i in range(0, len(diary), 4000)]
                for part in parts:
                    await update.message.reply_text(part)
            else:
                await update.message.reply_text(diary)
        else:
            await update.message.reply_text(
                "Il diario di oggi non è ancora stato generato.\n"
                "Viene creato automaticamente a mezzanotte."
            )

    async def cmd_ieri(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Invia il diario di ieri."""
        if not await self._check_auth(update):
            return
        diary = self._get_diary(date.today() - timedelta(days=1))
        if diary:
            if len(diary) > 4000:
                parts = [diary[i:i+4000] for i in range(0, len(diary), 4000)]
                for part in parts:
                    await update.message.reply_text(part)
            else:
                await update.message.reply_text(diary)
        else:
            await update.message.reply_text("Nessun diario disponibile per ieri.")

    async def cmd_alert(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Mostra gli alert della giornata."""
        if not await self._check_auth(update):
            return
        observations, _ = self._get_today_observations()
        if not observations:
            await update.message.reply_text("Nessun dato per oggi.")
            return

        alerts = [o for o in observations if o.get('type') == 'alert']
        if not alerts:
            await update.message.reply_text("Nessun alert oggi.")
        else:
            msg = f" {len(alerts)} alert oggi:\n\n"
            for a in alerts:
                msg += f"• {a['time']}: {a['description']}\n\n"
            await update.message.reply_text(msg)

    async def cmd_tug(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Avvia un test TUG."""
        if not await self._check_auth(update):
            return
        if not self.test_runner:
            await update.message.reply_text("Test runner non disponibile.")
            return
        await update.message.reply_text("Avvio test TUG... La persona deve essere visibile.")

        result = await asyncio.to_thread(self.test_runner.run_tug)

        if result:
            await update.message.reply_text(
                f"TUG completato!\n"
                f"Tempo: {result['total_time']:.1f}s"
            )
        else:
            await update.message.reply_text("Test non completato.")

    async def cmd_sts(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Avvia un test STS."""
        if not await self._check_auth(update):
            return
        if not self.test_runner:
            await update.message.reply_text("Test runner non disponibile.")
            return
        await update.message.reply_text("Avvio test STS... La persona deve essere visibile.")

        result = await asyncio.to_thread(self.test_runner.run_sts)

        if result:
            await update.message.reply_text(
                f"STS completato!\n"
                f"Ripetizioni: {result['reps_completed']}\n"
                f"Tempo: {result['total_time']:.1f}s"
            )
        else:
            await update.message.reply_text("Test non completato.")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Gestisce messaggi liberi — interroga Gemma."""
        if not await self._check_auth(update):
            return
        query = update.message.text
        await update.message.reply_text("Cerco nei dati...")

        # Eseguito in un thread separato per non bloccare il loop asyncio
        response = await asyncio.to_thread(self._answer_query, query)
        await update.message.reply_text(response)

    # =========================================
    # AVVIO BOT
    # =========================================
    def run(self):
        """Avvia il bot Telegram."""
        async def start():
            app = ApplicationBuilder().token(self.token).build()

            app.add_handler(CommandHandler("start", self.cmd_start))
            app.add_handler(CommandHandler("stato", self.cmd_stato))
            app.add_handler(CommandHandler("diario", self.cmd_diario))
            app.add_handler(CommandHandler("ieri", self.cmd_ieri))
            app.add_handler(CommandHandler("alert", self.cmd_alert))
            app.add_handler(CommandHandler("tug", self.cmd_tug))
            app.add_handler(CommandHandler("sts", self.cmd_sts))
            app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

            print(f"{'='*60}")
            print(f"Telegram Bot — Monitoraggio Domiciliare")
            print(f"  VLM:    {self.vlm.model}")
            print(f"  Server: {self.vlm.lmstudio_url}")
            print(f"  Dati:   {self.data_dir}")
            print(f"{'='*60}")
            print("Bot avviato. In attesa di messaggi...\n")

            await app.initialize()
            await app.start()
            await app.updater.start_polling()

            try:
                while True:
                    await asyncio.sleep(1)
            finally:
                await app.updater.stop()
                await app.stop()
                await app.shutdown()

        asyncio.run(start())
