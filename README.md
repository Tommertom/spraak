# Spraak

**Spraak** is a lightweight, browser-based voice recorder and transcription app powered by Google's Gemini AI. Record audio directly from your microphone, and Spraak will transcribe your speech to text in seconds — no server required.

🌐 **Live demo:** [https://gpx-go-78518.web.app/](https://gpx-go-78518.web.app/)

## Features

- 🎙️ **In-browser recording** — captures audio from your microphone using the MediaRecorder API
- 🤖 **AI transcription** — sends recordings to Google Gemini for accurate speech-to-text conversion
- 📋 **One-click copy** — copies the transcript to your clipboard instantly
- 🔒 **Private by default** — your Gemini API key is stored only in your browser's `localStorage`
- 📱 **Progressive Web App (PWA)** — installable on desktop and mobile, works offline after first load
- 🚫 **No build step** — a single `index.html` file; open it and go

## Prerequisites

- A modern browser with support for the MediaRecorder API and Web Audio API (Chrome, Edge, Firefox, Safari 14.1+)
- A free [Google Gemini API key](https://makersuite.google.com/app/apikey)

## Getting Started

### 1. Set your API key

Click the **⚙** (gear) button in the top-right corner and choose **Set API Key**. Enter your Gemini API key when prompted. The key is saved in `localStorage` and never leaves your browser except when calling the Gemini API directly.

### 2. Record

Click **Start Recording** and speak. The status indicator in the header changes from idle (amber) to live (green) while recording is active.

### 3. Stop and transcribe

Click **Stop Recording**. Spraak converts the captured audio to WAV format and sends it to the Gemini API. The transcribed text appears in the text area below the controls.

### 4. Use the transcript

- **Copy** — copies all text to the clipboard
- **Clear Text** — empties the text area so you can start fresh

## How It Works

1. The browser captures audio via `navigator.mediaDevices.getUserMedia` and buffers it with `MediaRecorder`.
2. When recording stops, the raw audio blob (WebM/MP4 from the browser) is decoded with the Web Audio API and re-encoded as a WAV file in pure JavaScript.
3. The WAV data is Base64-encoded and sent as an inline payload to the Gemini `generateContent` endpoint.
4. Gemini returns a plain-text transcript, which is appended to the text area.

The service worker (`sw.js`) caches the app shell so Spraak loads instantly and works without a network connection after the first visit.

## Running Locally

No build tools are required. Open `index.html` directly in a browser:

```bash
# With Python's built-in HTTP server (avoids some browser security restrictions on file:// URLs):
python3 -m http.server 8080
# Then open http://localhost:8080 in your browser
```

## Deployment

Because browsers require a secure context to access the microphone, Spraak **must be served over HTTPS**. Deploy the files (`index.html`, `sw.js`, `manifest.webmanifest`, and the `icons/` folder) to any static web host that provides HTTPS — for example GitHub Pages, Netlify, Vercel, Cloudflare Pages, or your own server with a TLS certificate.

> ⚠️ Serving over plain `http://` (except `localhost`) will cause the browser to block microphone access.

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Vanilla HTML / CSS / JavaScript |
| AI | Google Gemini API (`gemini-3-flash-preview`) |
| Offline | Service Worker + Cache API |
| PWA | Web App Manifest |
| CI/CD | GitHub Actions + Firebase Hosting |

## License

This project does not currently include a license file. Please contact the repository owner for usage terms.
