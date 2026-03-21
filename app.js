      import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js';

      "use strict";

      const STORAGE_KEY = "geminiApiKey";
      const THEME_STORAGE_KEY = "themePreference";
      const COPY_HISTORY_STORAGE_KEY = "copyHistory";
      const FILLER_FILTER_STORAGE_KEY = "fillerFilter";
      const LOCAL_LLM_STORAGE_KEY = "localLlm";
      const MAX_COPY_HISTORY_ITEMS = 50;
      const GEMINI_MODEL = "gemini-3-flash-preview";
      const GEMINI_URL = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent`;
      const LOCAL_LLM_MODEL = "Xenova/whisper-tiny";

      const els = {
        message: document.getElementById("message"),
        recordBtn: document.getElementById("recordBtn"),
        copyBtn: document.getElementById("copyBtn"),
        clearBtn: document.getElementById("clearBtn"),
        setKeyBtn: document.getElementById("setKeyBtn"),
        clearKeyBtn: document.getElementById("clearKeyBtn"),
        reloadAppBtn: document.getElementById("reloadAppBtn"),
        transcript: document.getElementById("transcript"),
        menuBtn: document.getElementById("menuBtn"),
        themeBtn: document.getElementById("themeBtn"),
        filterBtn: document.getElementById("filterBtn"),
        localLlmBtn: document.getElementById("localLlmBtn"),
        menuPopup: document.getElementById("menuPopup"),
        copyToast: document.getElementById("copyToast"),
        historyList: document.getElementById("historyList"),
      };

      let apiKey = localStorage.getItem(STORAGE_KEY) || "";
      let mediaRecorder = null;
      let mediaStream = null;
      let isRecording = false;
      let recordedChunks = [];
      let copyToastTimer = null;
      let copyHistory = [];
      let theme = localStorage.getItem(THEME_STORAGE_KEY) || "dark";
      let fillerFilter = localStorage.getItem(FILLER_FILTER_STORAGE_KEY) === "true";
      let localLlm = localStorage.getItem(LOCAL_LLM_STORAGE_KEY) === "true";
      let localPipeline = null;
      let localModelLoading = false;
      let wakeLock = null;

      function applyTheme(nextTheme) {
        theme = nextTheme === "light" ? "light" : "dark";
        document.documentElement.setAttribute("data-theme", theme);
        const lightMode = theme === "light";
        els.themeBtn.textContent = lightMode ? "☾" : "☀";
        els.themeBtn.setAttribute(
          "aria-label",
          lightMode ? "Switch to dark mode" : "Switch to light mode",
        );
        els.themeBtn.title = lightMode
          ? "Switch to dark mode"
          : "Switch to light mode";
      }

      function toggleTheme() {
        applyTheme(theme === "dark" ? "light" : "dark");
        localStorage.setItem(THEME_STORAGE_KEY, theme);
      }

      function applyFillerFilter(enabled) {
        fillerFilter = !!enabled;
        if (fillerFilter) {
          els.filterBtn.classList.remove("btn-secondary");
        } else {
          els.filterBtn.classList.add("btn-secondary");
        }
        els.filterBtn.setAttribute(
          "aria-label",
          fillerFilter ? "Disable filler word filter" : "Enable filler word filter",
        );
        els.filterBtn.title = fillerFilter
          ? "Disable filler word filter"
          : "Enable filler word filter";
      }

      function toggleFillerFilter() {
        applyFillerFilter(!fillerFilter);
        localStorage.setItem(FILLER_FILTER_STORAGE_KEY, fillerFilter);
      }

      function applyLocalLlm(enabled) {
        localLlm = !!enabled;
        if (localLlm) {
          els.localLlmBtn.classList.remove("btn-secondary");
        } else {
          els.localLlmBtn.classList.add("btn-secondary");
        }
        els.localLlmBtn.setAttribute(
          "aria-label",
          localLlm ? "Disable local transcription" : "Enable local transcription",
        );
        els.localLlmBtn.title = localLlm
          ? "Disable local transcription"
          : "Enable local transcription";
      }

      async function toggleLocalLlm() {
        const newState = !localLlm;
        applyLocalLlm(newState);
        localStorage.setItem(LOCAL_LLM_STORAGE_KEY, localLlm);
        if (localLlm && !localPipeline && !localModelLoading) {
          try {
            await loadLocalModel();
          } catch (err) {
            console.error(err);
            setMessage(
              "Failed to load local model: " + (err.message || "Unknown error"),
              "error",
            );
            applyLocalLlm(false);
            localStorage.setItem(LOCAL_LLM_STORAGE_KEY, false);
          }
        } else if (!localLlm) {
          if (apiKey) {
            setMessage("Ready. Click Start Recording.", "info");
          } else {
            setMessage("Please set your Gemini API key to start transcribing.", "warning");
          }
        }
      }

      function setMessage(text, type = "info") {
        const colorMap = {
          info: "var(--muted)",
          success: "var(--good)",
          warning: "var(--warn)",
          error: "var(--bad)",
        };
        els.message.style.color = colorMap[type] || colorMap.info;
        els.message.textContent = text;
      }

      function updateRecordUI() {
        els.recordBtn.textContent = isRecording
          ? "Stop Recording"
          : "Start Recording";
      }

      function promptForApiKey() {
        const entered = window.prompt(
          "Enter your Gemini API key:",
          "",
        );
        if (entered && entered.trim()) {
          apiKey = entered.trim();
          localStorage.setItem(STORAGE_KEY, apiKey);
          setMessage("API key saved locally in your browser.", "success");
          return true;
        }
        setMessage("A Gemini API key is required.", "warning");
        return false;
      }

      function ensureApiKey() {
        if (apiKey) return true;
        return promptForApiKey();
      }

      function normalizeMimeType(mimeType) {
        return String(mimeType || "")
          .split(";")[0]
          .trim()
          .toLowerCase();
      }

      function writeAsciiString(view, offset, text) {
        for (let i = 0; i < text.length; i += 1) {
          view.setUint8(offset + i, text.charCodeAt(i));
        }
      }

      function audioBufferToWavBlob(audioBuffer) {
        const inputChannelCount = audioBuffer.numberOfChannels || 1;
        const sampleRate = audioBuffer.sampleRate || 16000;
        const frameCount = audioBuffer.length;
        const monoData = new Float32Array(frameCount);

        for (let channel = 0; channel < inputChannelCount; channel += 1) {
          const channelData = audioBuffer.getChannelData(channel);
          for (let i = 0; i < frameCount; i += 1) {
            monoData[i] += channelData[i] / inputChannelCount;
          }
        }

        const bytesPerSample = 2;
        const blockAlign = bytesPerSample;
        const byteRate = sampleRate * blockAlign;
        const dataSize = frameCount * bytesPerSample;
        const wavBuffer = new ArrayBuffer(44 + dataSize);
        const view = new DataView(wavBuffer);

        writeAsciiString(view, 0, "RIFF");
        view.setUint32(4, 36 + dataSize, true);
        writeAsciiString(view, 8, "WAVE");
        writeAsciiString(view, 12, "fmt ");
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, byteRate, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, 16, true);
        writeAsciiString(view, 36, "data");
        view.setUint32(40, dataSize, true);

        let offset = 44;
        for (let i = 0; i < frameCount; i += 1) {
          const sample = Math.max(-1, Math.min(1, monoData[i]));
          view.setInt16(
            offset,
            sample < 0 ? sample * 0x8000 : sample * 0x7fff,
            true,
          );
          offset += 2;
        }

        return new Blob([wavBuffer], { type: "audio/wav" });
      }

      async function convertToWavBlob(blob) {
        const AudioContextCtor =
          window.AudioContext || window.webkitAudioContext;
        if (!AudioContextCtor) {
          throw new Error("Browser audio conversion is not supported.");
        }
        const audioBuffer = await blob.arrayBuffer();
        const audioContext = new AudioContextCtor();
        try {
          const decoded = await audioContext.decodeAudioData(
            audioBuffer.slice(0),
          );
          return audioBufferToWavBlob(decoded);
        } finally {
          await audioContext.close();
        }
      }

      async function prepareGeminiAudioBlob(blob) {
        const mimeType = normalizeMimeType(blob.type);
        if (mimeType === "audio/wav") return blob;
        return convertToWavBlob(blob);
      }

      async function blobToBase64(blob) {
        const buffer = await blob.arrayBuffer();
        let binary = "";
        const bytes = new Uint8Array(buffer);
        for (let i = 0; i < bytes.length; i += 1) {
          binary += String.fromCharCode(bytes[i]);
        }
        return btoa(binary);
      }

      async function audioToFloat32(blob) {
        const AudioContextCtor =
          window.AudioContext || window.webkitAudioContext;
        if (!AudioContextCtor) {
          throw new Error("Browser audio conversion is not supported.");
        }
        const audioContext = new AudioContextCtor();
        try {
          const arrayBuffer = await blob.arrayBuffer();
          const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
          const sampleRate = audioBuffer.sampleRate;
          const inputChannelCount = audioBuffer.numberOfChannels;
          const frameCount = audioBuffer.length;
          const monoData = new Float32Array(frameCount);
          for (let ch = 0; ch < inputChannelCount; ch += 1) {
            const channelData = audioBuffer.getChannelData(ch);
            for (let i = 0; i < frameCount; i += 1) {
              monoData[i] += channelData[i] / inputChannelCount;
            }
          }
          return { data: monoData, sampleRate };
        } finally {
          await audioContext.close();
        }
      }

      async function loadLocalModel() {
        if (localPipeline) return localPipeline;
        localModelLoading = true;
        const wasDisabled = els.recordBtn.disabled;
        if (!isRecording) els.recordBtn.disabled = true;
        try {
          setMessage("Downloading local model (first-time only)…", "info");
          const pipe = await pipeline(
            "automatic-speech-recognition",
            LOCAL_LLM_MODEL,
            {
              progress_callback: (progress) => {
                if (progress.status === "downloading") {
                  const pct = progress.progress
                    ? Math.round(progress.progress)
                    : 0;
                  setMessage(`Downloading model: ${pct}%`, "info");
                } else if (progress.status === "loading") {
                  setMessage("Loading model into memory…", "info");
                }
              },
            },
          );
          localPipeline = pipe;
          setMessage("Local model ready. Click Start Recording.", "success");
          return localPipeline;
        } finally {
          localModelLoading = false;
          if (!isRecording && !wasDisabled) els.recordBtn.disabled = false;
        }
      }

      async function transcribeLocally(audioBlob) {
        const pipe = await loadLocalModel();
        const { data, sampleRate } = await audioToFloat32(audioBlob);
        let result;
        try {
          result = await pipe(data, { sampling_rate: sampleRate });
        } catch (err) {
          throw new Error(
            "Local transcription failed: " + (err?.message || err?.toString() || "Unknown error"),
          );
        }
        let text = (result.text || "").trim();
        if (fillerFilter) {
          text = text
            .replace(/\b(um|uh|ah|hmm|er|erm|uhm)\b[,.]?\s*/gi, " ")
            .replace(/\s+/g, " ")
            .trim();
        }
        return text;
      }

      function extractText(data) {
        const parts = data?.candidates?.[0]?.content?.parts;
        if (!Array.isArray(parts)) return "";
        return parts
          .map((part) => part?.text || "")
          .join("")
          .trim();
      }

      function appendTranscript(text) {
        if (!text) return;
        const prefix = els.transcript.value.trim() ? "\n" : "";
        els.transcript.value += `${prefix}${text}`;
        els.transcript.scrollTop = els.transcript.scrollHeight;
      }

      function clearRecordedAudioBuffer() {
        recordedChunks = [];
      }

      function showCopyToast() {
        if (copyToastTimer) clearTimeout(copyToastTimer);
        els.copyToast.classList.add("show");
        copyToastTimer = setTimeout(() => {
          els.copyToast.classList.remove("show");
        }, 1200);
      }
      function loadCopyHistory() {
        try {
          const parsed = JSON.parse(
            localStorage.getItem(COPY_HISTORY_STORAGE_KEY) || "[]",
          );
          if (!Array.isArray(parsed)) return [];
          return parsed
            .filter(
              (item) =>
                item &&
                typeof item.text === "string" &&
                typeof item.createdAt === "string",
            )
            .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
        } catch {
          return [];
        }
      }

      function saveCopyHistory() {
        localStorage.setItem(
          COPY_HISTORY_STORAGE_KEY,
          JSON.stringify(copyHistory),
        );
      }

      function renderCopyHistory() {
        els.historyList.textContent = "";
        if (!copyHistory.length) {
          const item = document.createElement("li");
          item.textContent = "No copied messages yet.";
          els.historyList.appendChild(item);
          return;
        }

        copyHistory.forEach((entry, index) => {
          const item = document.createElement("li");
          const selectBtn = document.createElement("button");
          selectBtn.type = "button";
          selectBtn.className = "btn-secondary history-select";
          selectBtn.textContent = entry.text;
          selectBtn.addEventListener("click", () => {
            selectCopyHistoryMessage(entry.text);
          });
          const deleteBtn = document.createElement("button");
          deleteBtn.type = "button";
          deleteBtn.className = "btn-danger history-delete";
          deleteBtn.innerHTML =
            '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4h6v2"/></svg>';
          deleteBtn.title = "Delete";
          deleteBtn.setAttribute("aria-label", "Delete this copied message");
          deleteBtn.addEventListener("click", () => {
            deleteCopyHistoryMessage(index);
          });
          const row = document.createElement("div");
          row.className = "history-row";
          row.appendChild(selectBtn);
          row.appendChild(deleteBtn);
          item.appendChild(row);
          const time = document.createElement("time");
          time.dateTime = entry.createdAt;
          time.textContent = new Date(entry.createdAt).toLocaleString();
          item.appendChild(time);
          els.historyList.appendChild(item);
        });
      }

      function addToCopyHistory(text) {
        copyHistory.unshift({ text, createdAt: new Date().toISOString() });
        if (copyHistory.length > MAX_COPY_HISTORY_ITEMS) {
          copyHistory = copyHistory.slice(0, MAX_COPY_HISTORY_ITEMS);
        }
        saveCopyHistory();
        renderCopyHistory();
      }

      async function transcribeChunk(audioBlob) {
        if (!apiKey) throw new Error("Missing API key.");

        const preparedAudioBlob = await prepareGeminiAudioBlob(audioBlob);
        const base64Audio = await blobToBase64(preparedAudioBlob);
        const transcribePrompt = fillerFilter
          ? "Transcribe this full recording. Return only meaningful spoken words as plain text. Remove filler sounds and words such as 'um', 'uh', 'ah', 'hmm', 'er', and other non-meaningful sounds or noises. If there is no clear speech, return an empty string."
          : "Transcribe this full recording. Return only spoken words as plain text. If there is no clear speech, return an empty string.";
        const payload = {
          contents: [
            {
              role: "user",
              parts: [
                {
                  text: transcribePrompt,
                },
                {
                  inlineData: {
                    mimeType: preparedAudioBlob.type || "audio/wav",
                    data: base64Audio,
                  },
                },
              ],
            },
          ],
          generationConfig: {
            temperature: 0,
          },
        };

        const response = await fetch(
          `${GEMINI_URL}?key=${encodeURIComponent(apiKey)}`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          },
        );

        if (!response.ok) {
          const errText = await response.text();
          throw new Error(
            `Gemini request failed (${response.status}): ${errText.slice(0, 180)}`,
          );
        }

        const data = await response.json();
        return extractText(data);
      }

      async function requestWakeLock() {
        if (!("wakeLock" in navigator) || wakeLock) return;
        try {
          const newWakeLock = await navigator.wakeLock.request("screen");
          if (wakeLock) {
            // A concurrent call acquired a lock while we were awaiting; release this one.
            try {
              await newWakeLock.release();
            } catch (releaseErr) {
              console.warn("Stale wake lock release failed:", releaseErr);
            }
            return;
          }
          wakeLock = newWakeLock;
          newWakeLock.addEventListener("release", () => {
            if (wakeLock === newWakeLock) {
              wakeLock = null;
            }
          });
        } catch (err) {
          console.warn("Wake lock request failed:", err);
        }
      }

      async function releaseWakeLock() {
        if (wakeLock) {
          try {
            await wakeLock.release();
          } catch (err) {
            console.warn("Wake lock release failed:", err);
          }
          wakeLock = null;
        }
      }

      async function startRecording() {
        if (!localLlm && !ensureApiKey()) return;
        if (localLlm && localModelLoading) {
          setMessage("Model is still loading, please wait…", "warning");
          return;
        }
        if (localLlm && !localPipeline) {
          try {
            await loadLocalModel();
          } catch (err) {
            console.error(err);
            setMessage(
              "Failed to load local model: " + (err.message || "Unknown error"),
              "error",
            );
            return;
          }
        }

        if (
          !navigator.mediaDevices?.getUserMedia ||
          typeof MediaRecorder === "undefined"
        ) {
          setMessage(
            "This browser does not support microphone recording.",
            "error",
          );
          return;
        }

        try {
          mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: true,
          });
          mediaRecorder = new MediaRecorder(mediaStream);
          recordedChunks = [];
          const useLocalLlm = localLlm;

          mediaRecorder.ondataavailable = (event) => {
            if (event.data && event.data.size > 0) {
              recordedChunks.push(event.data);
            }
          };

          mediaRecorder.onerror = (event) => {
            const detail = event?.error?.message || "";
            setMessage(
              "Microphone recorder error occurred." + (detail ? " " + detail : ""),
              "error",
            );
            isRecording = false;
            updateRecordUI();
            releaseWakeLock();
          };

          mediaRecorder.onstop = async () => {
            if (mediaStream) {
              mediaStream.getTracks().forEach((track) => track.stop());
              mediaStream = null;
            }
            const chunks = recordedChunks;
            clearRecordedAudioBuffer();
            const mimeType = mediaRecorder?.mimeType || "audio/webm";
            mediaRecorder = null;

            if (!chunks.length) {
              setMessage(
                "Recording stopped with no audio captured.",
                "warning",
              );
              return;
            }

            setMessage("Transcribing recording...", "info");
            try {
              const completeRecording = new Blob(chunks, { type: mimeType });
              const transcriptText = useLocalLlm
                ? await transcribeLocally(completeRecording)
                : await transcribeChunk(completeRecording);
              if (transcriptText) {
                appendTranscript(transcriptText);
                setMessage("Transcription completed.", "success");
              } else {
                setMessage("No speech detected in recording.", "warning");
              }
            } catch (err) {
              console.error(err);
              setMessage(err.message || "Transcription failed.", "error");
            }
          };

          mediaRecorder.start();
          isRecording = true;
          updateRecordUI();
          setMessage("Recording...", "success");
          await requestWakeLock();
          if (!isRecording) {
            await releaseWakeLock();
          }
        } catch (err) {
          console.error(err);
          setMessage(
            "Could not access microphone. Please allow permission.",
            "error",
          );
        }
      }

      async function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== "inactive") {
          mediaRecorder.stop();
        } else {
          mediaRecorder = null;
        }
        isRecording = false;
        updateRecordUI();
        await releaseWakeLock();
      }

      async function copyTranscript(addToHistory = true) {
        const text = els.transcript.value;
        if (!text.trim()) {
          setMessage("Nothing to copy yet.", "warning");
          return;
        }

        try {
          await navigator.clipboard.writeText(text);
          setMessage("Copied transcript to clipboard.", "success");
          showCopyToast();
        } catch {
          els.transcript.focus();
          els.transcript.select();
          document.execCommand("copy");
          els.transcript.setSelectionRange(
            els.transcript.value.length,
            els.transcript.value.length,
          );
          setMessage("Copied transcript to clipboard.", "success");
          showCopyToast();
        }
        if (addToHistory) addToCopyHistory(text);
      }

      async function selectCopyHistoryMessage(text) {
        els.transcript.value = text;
        els.transcript.focus();
        els.transcript.setSelectionRange(text.length, text.length);
        await copyTranscript(false);
      }

      function deleteCopyHistoryMessage(index) {
        if (index < 0 || index >= copyHistory.length) return;
        if (!window.confirm("Delete this copied message?")) return;
        copyHistory.splice(index, 1);
        saveCopyHistory();
        renderCopyHistory();
        setMessage("Deleted copied message.", "info");
      }

      function clearTranscript() {
        if (!window.confirm("Are you sure you want to clear the text?")) return;
        els.transcript.value = "";
        setMessage("Transcript cleared.", "info");
      }

      function clearApiKey() {
        apiKey = "";
        localStorage.removeItem(STORAGE_KEY);
        setMessage("API key cleared.", "warning");
        if (isRecording) stopRecording();
      }

      async function reloadApp() {
        closeMenu();
        if (!window.confirm("Reload the app and clear all caches? Your API key and history will be kept.")) return;
        try {
          if ("caches" in window) {
            const keys = await caches.keys();
            await Promise.all(keys.map((key) => caches.delete(key)));
          }
        } catch (err) {
          console.error("Error clearing caches before reload:", err);
        }
        window.location.href = window.location.href;
      }

      function closeMenu() {
        els.menuPopup.hidden = true;
      }

      els.recordBtn.addEventListener("click", () => {
        if (isRecording) stopRecording();
        else startRecording();
      });
      els.copyBtn.addEventListener("click", copyTranscript);
      els.clearBtn.addEventListener("click", clearTranscript);
      els.setKeyBtn.addEventListener("click", () => {
        promptForApiKey();
        closeMenu();
      });
      els.clearKeyBtn.addEventListener("click", () => {
        clearApiKey();
        closeMenu();
      });
      els.reloadAppBtn.addEventListener("click", () => reloadApp());
      els.menuBtn.addEventListener("click", (event) => {
        event.stopPropagation();
        els.menuPopup.hidden = !els.menuPopup.hidden;
      });
      els.themeBtn.addEventListener("click", toggleTheme);
      els.filterBtn.addEventListener("click", toggleFillerFilter);
      els.localLlmBtn.addEventListener("click", toggleLocalLlm);
      document.addEventListener("click", (event) => {
        if (els.menuPopup.hidden) return;
        if (
          !els.menuPopup.contains(event.target) &&
          event.target !== els.menuBtn
        ) {
          closeMenu();
        }
      });
      els.menuPopup.addEventListener("click", (event) =>
        event.stopPropagation(),
      );
      document.addEventListener("visibilitychange", async () => {
        if (document.visibilityState === "visible" && isRecording && !wakeLock) {
          await requestWakeLock();
        }
      });

      updateRecordUI();
      applyTheme(theme);
      applyFillerFilter(fillerFilter);
      applyLocalLlm(localLlm);
      copyHistory = loadCopyHistory();
      renderCopyHistory();
      if (localLlm) {
        loadLocalModel().catch((err) => {
          console.error(err);
          setMessage(
            "Failed to load local model: " + (err.message || "Unknown error"),
            "error",
          );
          applyLocalLlm(false);
          localStorage.setItem(LOCAL_LLM_STORAGE_KEY, false);
        });
      } else if (!apiKey) {
        setMessage(
          "Please set your Gemini API key to start transcribing.",
          "warning",
        );
        promptForApiKey();
      } else {
        setMessage("Ready. Click Start Recording.", "info");
      }

      window.addEventListener("unhandledrejection", (event) => {
        console.error("Unhandled promise rejection:", event.reason);
        const msg =
          event.reason?.message ||
          (typeof event.reason === "string" ? event.reason : null) ||
          "An unexpected error occurred.";
        setMessage("Error: " + msg, "error");
        resetAfterError();
        event.preventDefault();
      });

      window.addEventListener("error", (event) => {
        console.error("Uncaught error:", event.error || event.message);
        const msg =
          event.error?.message || event.message || "An unexpected error occurred.";
        setMessage("Error: " + msg, "error");
        resetAfterError();
        event.preventDefault();
      });

      function resetAfterError() {
        if (isRecording) {
          isRecording = false;
          updateRecordUI();
          releaseWakeLock();
        }
        els.recordBtn.disabled = false;
      }
