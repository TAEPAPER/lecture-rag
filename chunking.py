"""Group short Whisper segments into retrieval-sized chunks.

Whisper segments are ~3 seconds (half a sentence), too small to carry meaning
on their own. We merge consecutive segments into chunks of up to `max_chars`
characters, overlapping by `overlap` segments so that an explanation sitting
on a chunk boundary is not cut in half.
"""

import json
import sys


def chunk_segments(segments: list[dict], max_chars: int = 500, overlap: int = 2) -> list[dict]:
    """Group consecutive Whisper segments into chunks.

    Returns: [{"id": int, "start": float, "end": float, "text": str}, ...]
    """
    # Drop empty segments (defensive: a bad transcription produced hundreds of these).
    segs = [s for s in segments if s["text"].strip()]
    chunks = []
    i = 0  # index of the first segment in the current chunk

    while i < len(segs):
        # Grow the chunk [i, j) one segment at a time until the next one would overflow.
        j, length = i, 0
        while j < len(segs):
            seg_len = len(segs[j]["text"].strip())
            # Always take at least one segment, even if it alone exceeds max_chars.
            if j > i and length + seg_len > max_chars:
                break
            length += seg_len + 1  # +1 for the joining space
            j += 1

        window = segs[i:j]
        chunks.append({
            "id": len(chunks),
            "start": window[0]["start"],  # timestamp of the first segment
            "end": window[-1]["end"],     # timestamp of the last segment
            "text": " ".join(s["text"].strip() for s in window),
        })

        if j >= len(segs):
            break
        # Next chunk starts `overlap` segments before this one ended.
        # max(..., i + 1) guarantees progress even if overlap >= chunk size.
        i = max(j - overlap, i + 1)

    return chunks


def fmt_time(sec: float) -> str:
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/transcripts/ch03_mlp_backprop.json"
    segments = json.load(open(path))["segments"]
    chunks = chunk_segments(segments, max_chars=500, overlap=2)

    lengths = [len(c["text"]) for c in chunks]
    durations = [c["end"] - c["start"] for c in chunks]
    print(f"{len(segments)} segments -> {len(chunks)} chunks")
    print(f"chars: avg {sum(lengths) / len(lengths):.0f}, min {min(lengths)}, max {max(lengths)}")
    print(f"duration: avg {sum(durations) / len(durations):.0f}s")
    print()
    for c in chunks[:3]:
        print(f"[#{c['id']} {fmt_time(c['start'])} ~ {fmt_time(c['end'])}] {c['text']}\n")
