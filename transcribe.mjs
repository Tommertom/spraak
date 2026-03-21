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
 * The audio file must be a WAV file with one of the following formats:
 *   - PCM (audioFormat=1): 8, 16, 24, or 32 bits per sample
 *   - IEEE float (audioFormat=3): 32 bits per sample
 *   - WAVE_FORMAT_EXTENSIBLE (audioFormat=65534/0xFFFE): with PCM or
 *     IEEE-float subformat GUID
 *
 * Note: compressed formats (MP3, AAC, etc.) are not supported.
 * Record a WAV with the web app, or convert with e.g.:
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
 * Supported WAV formats:
 *   - PCM (audioFormat=1): 8, 16, 24, or 32 bits per sample
 *   - IEEE float (audioFormat=3): 32 bits per sample
 *   - WAVE_FORMAT_EXTENSIBLE (audioFormat=0xFFFE): PCM or IEEE-float subformat
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

  // Subformat GUIDs for WAVE_FORMAT_EXTENSIBLE (first 4 bytes, little-endian)
  const KSDATAFORMAT_SUBTYPE_PCM = 1;
  const KSDATAFORMAT_SUBTYPE_IEEE_FLOAT = 3;

  let audioFormat, numChannels, sampleRate, bitsPerSample;
  let dataOffset, dataSize;
  let offset = 12;

  // Scan ALL chunks so that fmt/data can appear in any order.
  while (offset + 8 <= buffer.byteLength) {
    const chunkId = str4(offset);
    const chunkSize = view.getUint32(offset + 4, true);

    // Guard against a corrupt chunkSize that would read past the buffer.
    if (offset + 8 + chunkSize > buffer.byteLength) {
      throw new Error(`WAV chunk "${chunkId}" at offset ${offset} extends past end of file`);
    }

    if (chunkId === "fmt ") {
      audioFormat = view.getUint16(offset + 8, true);
      numChannels = view.getUint16(offset + 10, true);
      sampleRate = view.getUint32(offset + 12, true);
      bitsPerSample = view.getUint16(offset + 22, true);

      // WAVE_FORMAT_EXTENSIBLE: resolve the real subformat from the GUID.
      // The 16-byte SubFormat GUID starts 24 bytes into the fmt data body
      // (after audioFormat[2] + numChannels[2] + sampleRate[4] + byteRate[4]
      //  + blockAlign[2] + bitsPerSample[2] + cbSize[2] + validBits[2] +
      //  channelMask[4] = 24 bytes), i.e. at byte offset +32 from the chunk
      // ID. The first 2 bytes of the GUID hold the format code (same as a
      // plain audioFormat: 1=PCM, 3=IEEE float).
      if (audioFormat === 0xfffe && chunkSize >= 40) {
        audioFormat = view.getUint16(offset + 32, true);
        // Map the subformat codes back to the canonical format identifiers
        if (audioFormat !== KSDATAFORMAT_SUBTYPE_PCM && audioFormat !== KSDATAFORMAT_SUBTYPE_IEEE_FLOAT) {
          throw new Error(
            `Unsupported WAVE_FORMAT_EXTENSIBLE subformat: 0x${audioFormat.toString(16).padStart(4, "0")}` +
              " (only PCM and IEEE float subtypes are supported)",
          );
        }
      }
    } else if (chunkId === "data") {
      dataOffset = offset + 8;
      dataSize = chunkSize;
      // Guard: the declared data size must not exceed the actual buffer.
      if (dataOffset + dataSize > buffer.byteLength) {
        throw new Error("WAV data chunk size extends past end of file");
      }
    }

    offset += 8 + chunkSize;
    // WAV chunks are word-aligned (pad to even byte boundary)
    if (chunkSize & 1) offset += 1;
  }

  if (sampleRate === undefined) throw new Error("Could not find WAV fmt chunk");
  if (dataOffset === undefined) throw new Error("Could not find WAV data chunk");

  if (audioFormat !== 1 && audioFormat !== 3) {
    throw new Error(
      `Unsupported WAV audio format: 0x${audioFormat.toString(16).padStart(4, "0")}` +
        " (supported: PCM=1, IEEE float=3, WAVE_FORMAT_EXTENSIBLE=0xFFFE with PCM/float subformat)",
    );
  }

  // Validate bit depth
  if (audioFormat === 3 && bitsPerSample !== 32) {
    throw new Error(
      `Unsupported bit depth for IEEE float WAV: ${bitsPerSample} (only 32-bit float is supported)`,
    );
  }
  if (audioFormat === 1 && ![8, 16, 24, 32].includes(bitsPerSample)) {
    throw new Error(
      `Unsupported bit depth for PCM WAV: ${bitsPerSample} (supported: 8, 16, 24, 32)`,
    );
  }
  if (numChannels < 1) {
    throw new Error(`Invalid WAV channel count: ${numChannels}`);
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
        // 8-bit WAV is unsigned (validated above to only reach 8/16/24/32)
        sum += (buffer[byteOffset] - 128) / 128;
      } else {
        throw new Error(`Unexpected bit depth in decode loop: ${bitsPerSample}`);
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
