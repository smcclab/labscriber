import os
from pathlib import Path


def validate_token(hf_token: str) -> bool:
    """Attempt to load pyannote pipeline to verify the token and model acceptance.

    Returns True if token is valid and model terms have been accepted.
    """
    try:
        import whisperx
        # Use CPU for validation to avoid GPU setup overhead
        whisperx.DiarizationPipeline(use_auth_token=hf_token, device="cpu")
        return True
    except Exception as exc:
        error = str(exc).lower()
        if "401" in error or "unauthorized" in error or "invalid" in error:
            print("  Error: Invalid token or model terms not accepted.")
        elif "403" in error or "forbidden" in error or "gated" in error:
            print("  Error: Model access not granted. Have you accepted the terms at both URLs?")
        else:
            print(f"  Error: {exc}")
        return False


def run_setup(working_dir: Path) -> None:
    """Interactive HuggingFace token setup.

    1. Check if HF_TOKEN is already set.
    2. Print model acceptance URLs.
    3. Prompt for token.
    4. Write to .env.
    5. Validate token by loading pyannote pipeline.
    """
    from dotenv import dotenv_values

    env_path = working_dir / ".env"

    # Check for existing token
    existing = os.environ.get("HF_TOKEN") or dotenv_values(env_path).get("HF_TOKEN")
    if existing:
        print(f"HF_TOKEN is already set in {env_path}.")
        answer = input("Re-enter a new token? [y/N]: ").strip().lower()
        if answer != "y":
            print("Setup skipped. Run 'labscriber download-models' to proceed.")
            return

    print("\n--- labscriber HuggingFace Setup ---\n")
    print("You need a free HuggingFace account with access to two gated models.")
    print("Step 1: Visit each URL below while logged in and click 'Agree':")
    print()
    print("  https://huggingface.co/pyannote/speaker-diarization-3.1")
    print("  https://huggingface.co/pyannote/segmentation-3.0")
    print()
    print("Step 2: Create a read-only access token at:")
    print("  https://huggingface.co/settings/tokens")
    print()
    token = input("Paste your HuggingFace token here: ").strip()

    if not token:
        print("No token entered. Aborting.")
        return

    if not token.startswith("hf_"):
        print("Warning: token usually starts with 'hf_'. Proceeding anyway.")

    print("\nValidating token (downloads ~500 MB on first run — this may take a few minutes)...")
    if not validate_token(token):
        print("Setup failed. Correct the issues above and re-run 'labscriber setup'.")
        return

    # Write to .env
    env_path.write_text(f"HF_TOKEN={token}\n", encoding="utf-8")
    print(f"\n✓ Token saved to {env_path}")
    print("Next step: run 'labscriber download-models' to pre-download all models.")
