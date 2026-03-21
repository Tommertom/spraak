#!/usr/bin/env node
/**
 * transcribe.mjs — Node.js test script for local Whisper transcription.
 *
 * Mirrors the loadLocalModel() and transcribeLocally() logic from app.js,
 * using the same model (Xenova/whisper-tiny) and the same
 * @huggingface/transformers library.
 *
 * Usage:
 *   npm install
 *   node transcribe.mjs <audio-file.wav>
 *
 * The audio file must be a WAV file (PCM 8/16/24/32-bit or IEEE float).
 * Record one with the web app or any audio tool, e.g.:
 *   sox input.mp3 -r 16000 -c 1 output.wav
 */

import { pipeline } from "@huggingface/transformers";
import { readFileSync } from "fs";
import { resolve } from "path";

// Same model constant as app.js
const LOCAL_LLM_MODEL = "Xenova/whisper-tiny";

/**
 * Parse a WAV file buffer and return { data: Float32Array, sampleRate }.
 *
 * This is the Node.js equivalent of audioToFloat32() in app.js, which uses
 * the browser's AudioContext.decodeAudioData(). Both functions produce a
 * mono Float32Array that can be passed directly to the Whisper pipeline.
 *
 * Supports PCM (audioFormat=1) and IEEE float (audioFormat=3) WAV files
 * with 8, 16, 24, or 32 bits per sample, any number of channels.
 */
function wavToFloat32(buffer) {
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);

  const str4 = (offset) =>
    String.fromCharCode(
      buffer[offset],
      buffer[offset + 1],
      buffer[offset + 2],
      buffer[offset + 3],
    );

  if (str4(0) !== "RIFF") throw new Error("Not a valid WAV file (missing RIFF header)");
  if (str4(8) !== "WAVE") throw new Error("Not a valid WAV file (missing WAVE identifier)");

  let audioFormat, numChannels, sampleRate, bitsPerSample;
  let dataOffset, dataSize;
  let offset = 12;

  while (offset + 8 <= buffer.byteLength) {
    const chunkId = str4(offset);
    const chunkSize = view.getUint32(offset + 4, true);

    if (chunkId === "fmt ") {
      audioFormat = view.getUint16(offset + 8, true);
      numChannels = view.getUint16(offset + 10, true);
      sampleRate = view.getUint32(offset + 12, true);
      bitsPerSample = view.getUint16(offset + 22, true);
    } else if (chunkId === "data") {
      dataOffset = offset + 8;
      dataSize = chunkSize;
      break; // data chunk found; stop scanning
    }

    offset += 8 + chunkSize;
    // WAV chunks are word-aligned (pad to even byte boundary)
    if (chunkSize & 1) offset += 1;
  }

  if (!sampleRate) throw new Error("Could not find WAV fmt chunk");
  if (dataOffset === undefined) throw new Error("Could not find WAV data chunk");
  if (audioFormat !== 1 && audioFormat !== 3) {
    throw new Error(
      `Unsupported WAV audio format: ${audioFormat} (only PCM=1 and IEEE float=3 are supported)`,
    );
  }

  const bytesPerSample = bitsPerSample / 8;
  const numSamples = Math.floor(dataSize / (numChannels * bytesPerSample));
  // Produce a mono Float32Array — same approach as audioToFloat32() in app.js
  const monoData = new Float32Array(numSamples);

  for (let i = 0; i < numSamples; i++) {
    let sum = 0;
    for (let ch = 0; ch < numChannels; ch++) {
      const byteOffset = dataOffset + (i * numChannels + ch) * bytesPerSample;
      if (audioFormat === 3) {
        // IEEE 754 float32
        sum += view.getFloat32(byteOffset, true);
      } else if (bitsPerSample === 32) {
        sum += view.getInt32(byteOffset, true) / 0x80000000;
      } else if (bitsPerSample === 24) {
        const b0 = buffer[byteOffset];
        const b1 = buffer[byteOffset + 1];
        const b2 = buffer[byteOffset + 2];
        const int24 = (b2 << 16) | (b1 << 8) | b0;
        sum += (int24 >= 0x800000 ? int24 - 0x1000000 : int24) / 0x800000;
      } else if (bitsPerSample === 16) {
        sum += view.getInt16(byteOffset, true) / 0x8000;
      } else if (bitsPerSample === 8) {
        // 8-bit WAV is unsigned
        sum += (buffer[byteOffset] - 128) / 128;
      }
    }
    monoData[i] = numChannels > 1 ? sum / numChannels : sum;
  }

  return { data: monoData, sampleRate };
}

// ─── loadLocalModel ──────────────────────────────────────────────────────────
// Mirrors loadLocalModel() in app.js, adapted for Node.js console output.

let localPipeline = null;

async function loadLocalModel() {
  if (localPipeline) return localPipeline;

  console.log("Downloading local model (first-time only)…");
  const pipe = await pipeline(
    "automatic-speech-recognition",
    LOCAL_LLM_MODEL,
    {
      progress_callback: (progress) => {
        if (progress.status === "downloading") {
          const pct = progress.progress ? Math.round(progress.progress) : 0;
          process.stdout.write(`\rDownloading model: ${pct}%   `);
        } else if (progress.status === "loading") {
          process.stdout.write("\nLoading model into memory…\n");
        }
      },
    },
  );

  process.stdout.write("\n");
  console.log("Local model ready.");
  localPipeline = pipe;
  return pipe;
}

// ─── transcribeLocally ───────────────────────────────────────────────────────
// Mirrors transcribeLocally() in app.js. Accepts a path to a WAV file instead
// of a Blob (Node.js has no Blob/FileReader), and uses wavToFloat32() instead
// of audioToFloat32() for the same conversion result.

async function transcribeLocally(audioPath) {
  const pipe = await loadLocalModel();

  const buffer = readFileSync(resolve(audioPath));
  const { data, sampleRate } = wavToFloat32(buffer);

  console.log(
    `Transcribing "${audioPath}" (${sampleRate} Hz, ${data.length} samples)…`,
  );

  let result;
  try {
    result = await pipe(data, { sampling_rate: sampleRate });
  } catch (err) {
    throw new Error(
      "Local transcription failed: " +
        (err?.message || err?.toString() || "Unknown error"),
    );
  }

  return (result.text || "").trim();
}

// ─── CLI entry point ─────────────────────────────────────────────────────────

const audioFile = process.argv[2];

if (!audioFile) {
  console.error("Usage: node transcribe.mjs <audio-file.wav>");
  process.exit(1);
}

transcribeLocally(audioFile)
  .then((text) => {
    console.log("\nTranscription:");
    console.log(text);
  })
  .catch((err) => {
    console.error("Error:", err.message);
    process.exit(1);
  });
