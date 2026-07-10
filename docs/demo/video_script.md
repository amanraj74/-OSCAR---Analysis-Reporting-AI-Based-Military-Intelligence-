# OSCAR — Demo Video Script (5 minutes)

> **Target audience:** BSERC internship evaluators + defense/intelligence analysts  
> **Format:** Screen recording with voiceover  
> **Tool:** OBS Studio / Loom / QuickTime screen recording  
> **Length:** ~5 minutes  

---

## Setup (5 minutes before recording)

```bash
# Ensure demo data is fresh
python scripts/seed_demo.py

# Launch dashboard
streamlit run dashboard/app.py
```

Make sure the dashboard shows non-trivial data (events: ~3K, articles: 200, entities: 44, anomalies: 5).

---

## Script

### [0:00 — 0:30] **Intro: What is OSCAR?**

> "Hi, I'm [your name], and this is OSCAR — Open-Source Conflict Analysis and Reporting.
> It's an AI-Based Military Intelligence and Threat Analysis Dashboard built as part of
> the BSERC Def-Space Summer Internship 2026.
>
> OSCAR fuses three open data sources — GDELT global events, NewsAPI news, and Reddit RSS —
> with NLP entity extraction, sentiment scoring, and machine learning forecasting —
> to help defense analysts answer 'what's happening, where, and how is the discourse shifting'.
>
> Let's take a quick tour."

### [0:30 — 1:30] **Home Page**

> "This is the home page. We can see the system has ingested 2,638 GDELT events, 200 news
> and Reddit articles, extracted 44 named entities, and flagged 5 anomalies — all in
> the last 30 days, automatically, without any manual intervention.
>
> If we open the sidebar — events, articles, entities, anomalies — at a glance.
> The system status shows we're in development mode with SQLite backend."

### [1:30 — 2:30] **Map Page**

> "The Map page shows a world choropleth of average sentiment across countries.
> Green = positive tone, red = negative. We can switch between sentiment, event count,
> and conflict events.
>
> Below the choropleth, we see a bubble plot of geo-located GDELT events,
> colored by Goldstein scale — a measure of cooperation vs conflict.
>
> The Top Countries panel ranks regions by event count or conflict events.
> We see Russia leading in events, which is consistent with the ongoing conflict."

### [2:30 — 3:30] **Sentiment Page**

> "The Sentiment page shows time-series trends. The first chart tracks average
> compound sentiment from articles over the last 30 days. The orange line is
> the mean, the green line is the median. We can see sentiment dipping around
> mid-June — likely correlated with major escalations.
>
> The second chart shows the share of positive vs negative articles over time.
> Below, we have top positive and negative articles — clicking any of them
> opens the original source."

### [3:30 — 4:30] **Entities + Forecast + Alerts**

> "The Entities page shows trending orgs, weapons, and locations. We see 'Wagner Group'
> as the top MILITARY_ORG with 50+ mentions, 'F-16' as the top weapon, 'Kyiv' as a top location.
>
> The Forecast page shows per-region 7-day forecasts. For UKR, the median forecast
> is around 40 events per day with 95% confidence bands. Production escalation
> model metrics are shown below.
>
> The Alerts page shows the live anomaly feed. Five recent anomalies flagged —
> critical (red), high (amber), medium (blue) — with severity scores. The Russia
> event-count spike is critical at 0.85 severity."

### [4:30 — 5:00] **About + Closing**

> "The About page documents our methodology, data sources, and ethics.
> Everything is open-source — open data, open libraries, open license.
>
> Thanks for watching!"

---

## Recording tips

- **Use 1920×1080** resolution; zoom Streamlit to fill screen.
- **Mute notifications** (Discord, Slack, Windows toast).
- **Use a clean dark theme** if available (`STREAMLIT_THEME_BASE=dark`).
- **Pre-warm cache** — open each page once before recording to avoid spinner flashes.
- **Speak slowly**; pause 2 seconds between major transitions.
- **Add chapter markers** if uploading to YouTube.

## Files to generate

- `docs/demo/demo.mp4` — final video
- `docs/demo/screenshot-*.png` — backup screenshots if video upload fails

## Upload

- **YouTube (unlisted)** — best for sharing in BSERC submission form
- **Google Drive** — alternative
- **GitHub repo** — embed in README + CHANGELOG

---

**Total runtime: ~5 min. Plan for 30 min to record + edit + upload.**