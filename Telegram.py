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

import os
import json
import asyncio
import argparse
from datetime import date, timedelta
from pathlib import Path
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# Riusa il client VLM esistente
from Vlm_calls import VLMClient


class MonitorBot:
    def __init__(self, token, vlm_client, data_dir="diari", test_runner=None, allowed_ids=None):
        self.token = token
        self.vlm = vlm_client
        self.data_dir = Path(data_dir)
        self.test_runner = test_runner
        self.allowed_ids = set(allowed_ids) if allowed_ids else set()

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
    def _get_today_observations(self):
        """Legge le osservazioni di oggi dal data.json."""
        today = date.today().isoformat()
        d = date.today()
        path = self.data_dir / str(d.year) / f"{d.month:02d}" / today / "data.json"
        if not path.exists():
            return None, None
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("observations", []), data.get("hourly_summaries", [])

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
        observations, hourly_summaries = self._get_today_observations()

        context = ""

        # Sintesi orarie (quadro generale)
        if hourly_summaries:
            context += "RIEPILOGHI ORARI DI OGGI:\n"
            for s in sorted(hourly_summaries, key=lambda x: x['hour']):
                context += f"[{s['hour_label']}] {s['summary']}\n\n"

        # Osservazioni recenti (ultime 20 per dettaglio)
        if observations:
            recent = observations[-20:]
            context += "ULTIME OSSERVAZIONI:\n"
            for o in recent:
                obs_type = o.get('type', 'singolo')
                tag = ""
                if obs_type == "alert":
                    tag = " [ALERT]"
                elif obs_type == "confronto":
                    tag = " [CONFRONTO]"
                context += f"- {o['time']}{tag}: {o['description']}\n"

        # Se non ci sono dati oggi, prova il diario di ieri
        if not context:
            yesterday_diary = self._get_diary(date.today() - timedelta(days=1))
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

        await update.message.reply_text(msg)

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


def main():
    parser = argparse.ArgumentParser(description="Telegram Bot Monitoraggio")
    parser.add_argument("--token", default=os.environ.get("TELEGRAM_BOT_TOKEN"),
                        help="Token del bot Telegram (o usa TELEGRAM_BOT_TOKEN env)")
    parser.add_argument("--model", default="gemma-4-26b-a4b-it")
    parser.add_argument("--url", default="http://localhost:1234")
    parser.add_argument("--data-dir", default="diari")
    parser.add_argument("--allowed-ids", nargs="*", type=int, default=None,
                        help="User ID Telegram autorizzati (es. --allowed-ids 123456 789012)")
    args = parser.parse_args()

    if not args.token:
        print("Errore: token Telegram non fornito.")
        print("Usa --token oppure imposta TELEGRAM_BOT_TOKEN:")
        print("  export TELEGRAM_BOT_TOKEN='il_tuo_token'")
        return

    vlm = VLMClient(model=args.model, lmstudio_url=args.url)
    bot = MonitorBot(token=args.token, vlm_client=vlm, data_dir=args.data_dir,
                     allowed_ids=args.allowed_ids)
    bot.run()


if __name__ == "__main__":
    main()
